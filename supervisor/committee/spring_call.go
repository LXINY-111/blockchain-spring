package committee

import (
	"blockEmulator/params"
	"blockEmulator/utils"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"time"
)

type SpringInferInput struct {
	Address string    `json:"address"`
	Related string    `json:"related"`
	State   []float64 `json:"state"`
	Shards  int       `json:"shards"`
}

type SpringInferOutput struct {
	Shard      int     `json:"shard"`
	Source     string  `json:"source"`
	Confidence float64 `json:"confidence"`
}

type SpringDecisionRecord struct {
	TimeUnixNano int64     `json:"time_unix_nano"`
	Address      string    `json:"address"`
	Related      string    `json:"related"`
	Shard        uint64    `json:"shard"`
	Source       string    `json:"source"`
	StateDim     int       `json:"state_dim"`
	InferCostUs  int64     `json:"infer_cost_us"`
	State        []float64 `json:"state"`
}

// springChooseShardPPO 是版本 3 的入口。
// 你原来的 springChooseShard(addr, related) 仍然保留，作为启发式兜底。
func (rthm *RelayCommitteeModule) springChooseShardPPO(addr utils.Address, related utils.Address) uint64 {
	state := rthm.springBuildState(related)

	start := time.Now()
	sid, source, ok := rthm.springCallPython(string(addr), string(related), state)
	inferCostUs := time.Since(start).Microseconds()

	if ok {
		rthm.springAppendDecisionRecord(string(addr), string(related), sid, source, state, inferCostUs)
		rthm.sl.Slog.Printf(
			"[SPRING PPO] addr=%s related=%s shard=%d source=%s cost_us=%d\n",
			addr,
			related,
			sid,
			source,
			inferCostUs,
		)
		return sid
	}

	// Python 推理失败时，退回版本 1 的启发式策略。
	fallbackSid := rthm.springChooseShard(addr, related)
	rthm.springAppendDecisionRecord(
		string(addr),
		string(related),
		fallbackSid,
		"go_heuristic_fallback",
		state,
		inferCostUs,
	)

	rthm.sl.Slog.Printf(
		"[SPRING PPO FALLBACK] addr=%s related=%s shard=%d cost_us=%d\n",
		addr,
		related,
		fallbackSid,
		inferCostUs,
	)

	return fallbackSid
}

func (rthm *RelayCommitteeModule) springCallPython(addr string, related string, state []float64) (uint64, string, bool) {
	if len(state) == 0 {
		return 0, "empty_state", false
	}

	if err := os.MkdirAll("spring_io", os.ModePerm); err != nil {
		rthm.sl.Slog.Printf("[SPRING PPO] mkdir spring_io failed: %v\n", err)
		return 0, "mkdir_failed", false
	}

	reqID := time.Now().UnixNano()
	inputPath := filepath.Join("spring_io", fmt.Sprintf("state_%d.json", reqID))
	outputPath := filepath.Join("spring_io", fmt.Sprintf("action_%d.json", reqID))

	input := SpringInferInput{
		Address: addr,
		Related: related,
		State:   state,
		Shards:  params.ShardNum,
	}

	inputBytes, err := json.Marshal(input)
	if err != nil {
		rthm.sl.Slog.Printf("[SPRING PPO] marshal input failed: %v\n", err)
		return 0, "marshal_failed", false
	}

	if err := os.WriteFile(inputPath, inputBytes, 0644); err != nil {
		rthm.sl.Slog.Printf("[SPRING PPO] write input failed: %v\n", err)
		return 0, "write_input_failed", false
	}

	pythonCmd := os.Getenv("SPRING_PYTHON")
	if pythonCmd == "" {
		if runtime.GOOS == "windows" {
			pythonCmd = "python"
		} else {
			pythonCmd = "python3"
		}
	}

	cmd := exec.Command(
		pythonCmd,
		"spring_lite/infer.py",
		"--state",
		inputPath,
		"--out",
		outputPath,
	)

	out, err := cmd.CombinedOutput()
	if err != nil {
		rthm.sl.Slog.Printf("[SPRING PPO] python infer failed: %v, output=%s\n", err, string(out))
		return 0, "python_failed", false
	}

	outputBytes, err := os.ReadFile(outputPath)
	if err != nil {
		rthm.sl.Slog.Printf("[SPRING PPO] read output failed: %v\n", err)
		return 0, "read_output_failed", false
	}

	var output SpringInferOutput
	if err := json.Unmarshal(outputBytes, &output); err != nil {
		rthm.sl.Slog.Printf("[SPRING PPO] unmarshal output failed: %v, raw=%s\n", err, string(outputBytes))
		return 0, "unmarshal_output_failed", false
	}

	if output.Shard < 0 || output.Shard >= params.ShardNum {
		rthm.sl.Slog.Printf("[SPRING PPO] invalid shard=%d\n", output.Shard)
		return 0, "invalid_shard", false
	}

	source := output.Source
	if source == "" {
		source = "python_ppo"
	}

	return uint64(output.Shard), source, true
}

func (rthm *RelayCommitteeModule) springAppendDecisionRecord(
	addr string,
	related string,
	shard uint64,
	source string,
	state []float64,
	inferCostUs int64,
) {
	if err := os.MkdirAll("spring_io", os.ModePerm); err != nil {
		return
	}

	record := SpringDecisionRecord{
		TimeUnixNano: time.Now().UnixNano(),
		Address:      addr,
		Related:      related,
		Shard:        shard,
		Source:       source,
		StateDim:     len(state),
		InferCostUs:  inferCostUs,
		State:        state,
	}

	b, err := json.Marshal(record)
	if err != nil {
		return
	}

	path := filepath.Join("spring_io", "decision_records.jsonl")
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return
	}
	defer f.Close()

	_, _ = f.Write(append(b, '\n'))
}
