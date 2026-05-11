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
	"sync/atomic"
	"time"
)

type SpringBatchInferItem struct {
	Address string    `json:"address"`
	Related string    `json:"related"`
	State   []float64 `json:"state"`
}

type SpringBatchInferInput struct {
	Shards int                    `json:"shards"`
	Items  []SpringBatchInferItem `json:"items"`
}

type SpringBatchInferResult struct {
	Address    string  `json:"address"`
	Related    string  `json:"related"`
	Shard      int     `json:"shard"`
	Source     string  `json:"source"`
	Confidence float64 `json:"confidence"`
}

type SpringBatchInferOutput struct {
	Items []SpringBatchInferResult `json:"items"`
}

type SpringDecisionRecord struct {
	TimeUnixNano int64     `json:"time_unix_nano"`
	Address      string    `json:"address"`
	Related      string    `json:"related"`
	Shard        uint64    `json:"shard"`
	Source       string    `json:"source"`
	Confidence   float64   `json:"confidence"`
	StateDim     int       `json:"state_dim"`
	InferCostUs  int64     `json:"infer_cost_us"`
	BatchSize    int       `json:"batch_size"`
	State        []float64 `json:"state"`
}

// springChooseShardPPO 保留为单地址调试入口。
// 正式 TxBatch 运行时会走 springCallPythonBatch。
func (rthm *RelayCommitteeModule) springChooseShardPPO(addr utils.Address, related utils.Address) uint64 {
	state := rthm.springBuildState(related)

	items := []SpringBatchInferItem{
		{
			Address: string(addr),
			Related: string(related),
			State:   state,
		},
	}

	results, inferCostUs, ok := rthm.springCallPythonBatch(items)
	if ok && len(results) == 1 {
		output := results[0]
		if output.Shard >= 0 && output.Shard < params.ShardNum {
			sid := uint64(output.Shard)
			source := output.Source
			if source == "" {
				source = "python_ppo"
			}

			rthm.springAppendDecisionRecord(
				string(addr),
				string(related),
				sid,
				source,
				output.Confidence,
				state,
				inferCostUs,
				1,
			)

			rthm.sl.Slog.Printf(
				"[SPRING PPO] addr=%s related=%s shard=%d source=%s confidence=%.6f cost_us=%d\n",
				addr, related, sid, source, output.Confidence, inferCostUs,
			)
			return sid
		}
	}

	fallbackSid := rthm.springChooseShard(addr, related)

	rthm.springAppendDecisionRecord(
		string(addr),
		string(related),
		fallbackSid,
		"go_heuristic_fallback",
		0.0,
		state,
		inferCostUs,
		1,
	)

	rthm.sl.Slog.Printf(
		"[SPRING PPO FALLBACK] addr=%s related=%s shard=%d cost_us=%d\n",
		addr, related, fallbackSid, inferCostUs,
	)

	return fallbackSid
}

func (rthm *RelayCommitteeModule) springCallPythonBatch(items []SpringBatchInferItem) ([]SpringBatchInferResult, int64, bool) {
	if len(items) == 0 {
		return []SpringBatchInferResult{}, 0, true
	}

	for _, item := range items {
		if len(item.State) == 0 {
			return nil, 0, false
		}
	}

	if err := os.MkdirAll("spring_io", os.ModePerm); err != nil {
		rthm.sl.Slog.Printf("[SPRING PPO BATCH] mkdir spring_io failed: %v\n", err)
		return nil, 0, false
	}

	reqID := atomic.AddUint64(&rthm.springActionSeq, 1)

	inputPath := filepath.Join("spring_io", fmt.Sprintf("batch_state_%d.json", reqID))
	outputPath := filepath.Join("spring_io", fmt.Sprintf("batch_action_%d.json", reqID))

	input := SpringBatchInferInput{
		Shards: params.ShardNum,
		Items:  items,
	}

	inputBytes, err := json.Marshal(input)
	if err != nil {
		rthm.sl.Slog.Printf("[SPRING PPO BATCH] marshal input failed: %v\n", err)
		return nil, 0, false
	}

	if err := os.WriteFile(inputPath, inputBytes, 0644); err != nil {
		rthm.sl.Slog.Printf("[SPRING PPO BATCH] write input failed: %v\n", err)
		return nil, 0, false
	}

	pythonCmd := os.Getenv("SPRING_PYTHON")
	if pythonCmd == "" {
		if runtime.GOOS == "windows" {
			pythonCmd = "python"
		} else {
			pythonCmd = "python3"
		}
	}

	start := time.Now()

	cmd := exec.Command(
		pythonCmd,
		filepath.Join("spring_lite", "infer_batch.py"),
		"--state",
		inputPath,
		"--out",
		outputPath,
	)

	out, err := cmd.CombinedOutput()
	inferCostUs := time.Since(start).Microseconds()

	if err != nil {
		rthm.sl.Slog.Printf(
			"[SPRING PPO BATCH] python infer failed: %v, output=%s\n",
			err,
			string(out),
		)
		return nil, inferCostUs, false
	}

	outputBytes, err := os.ReadFile(outputPath)
	if err != nil {
		rthm.sl.Slog.Printf("[SPRING PPO BATCH] read output failed: %v\n", err)
		return nil, inferCostUs, false
	}

	var output SpringBatchInferOutput
	if err := json.Unmarshal(outputBytes, &output); err != nil {
		rthm.sl.Slog.Printf(
			"[SPRING PPO BATCH] unmarshal output failed: %v, raw=%s\n",
			err,
			string(outputBytes),
		)
		return nil, inferCostUs, false
	}

	rthm.sl.Slog.Printf(
		"[SPRING PPO BATCH] batch_id=%d items=%d results=%d cost_us=%d\n",
		reqID,
		len(items),
		len(output.Items),
		inferCostUs,
	)

	return output.Items, inferCostUs, true
}

func (rthm *RelayCommitteeModule) springAppendDecisionRecord(
	addr string,
	related string,
	shard uint64,
	source string,
	confidence float64,
	state []float64,
	inferCostUs int64,
	batchSize int,
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
		Confidence:   confidence,
		StateDim:     len(state),
		InferCostUs:  inferCostUs,
		BatchSize:    batchSize,
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
