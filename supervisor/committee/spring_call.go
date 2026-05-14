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

	BatchID uint64  `json:"batch_id"`
	LogProb float64 `json:"log_prob"`
	Value   float64 `json:"value"`
}

type SpringBatchInferOutput struct {
	Items []SpringBatchInferResult `json:"items"`
}

type SpringDecisionRecord struct {
	TimeUnixNano int64     `json:"time_unix_nano"`
	BatchID      uint64    `json:"batch_id"`
	Address      string    `json:"address"`
	Related      string    `json:"related"`
	Shard        uint64    `json:"shard"`
	Source       string    `json:"source"`
	Confidence   float64   `json:"confidence"`
	LogProb      float64   `json:"log_prob"`
	Value        float64   `json:"value"`
	StateDim     int       `json:"state_dim"`
	InferCostUs  int64     `json:"infer_cost_us"`
	BatchSize    int       `json:"batch_size"`
	State        []float64 `json:"state"`
}

type SpringOnlineTrainResult struct {
	Ok                     bool               `json:"ok"`
	TimeUnixNano           int64              `json:"time_unix_nano"`
	BatchID                uint64             `json:"batch_id"`
	FeedbackEpoch          int                `json:"feedback_epoch"`
	Shards                 int                `json:"shards"`
	NumActions             int                `json:"num_actions"`
	Skipped                int                `json:"skipped"`
	Reward                 float64            `json:"reward"`
	Done                   bool               `json:"done"`
	ActionHist             []int              `json:"action_hist"`
	CrossRate              float64            `json:"cross_rate"`
	NormalizedLoadVariance float64            `json:"normalized_load_variance"`
	TotalTx                int                `json:"total_tx"`
	TotalInner             int                `json:"total_inner"`
	TotalRelay1            int                `json:"total_relay1"`
	TotalRelay2            int                `json:"total_relay2"`
	InputPath              string             `json:"input_path"`
	ModelPath              string             `json:"model_path"`
	LogPath                string             `json:"log_path"`
	LossInfo               map[string]float64 `json:"loss_info"`
	ModelSource            string             `json:"model_source"`
	Message                string             `json:"message"`
	OnlineUpdateCount      int                `json:"online_update_count"`
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
				output.BatchID,
				string(addr),
				string(related),
				sid,
				source,
				output.Confidence,
				output.LogProb,
				output.Value,
				state,
				inferCostUs,
				1,
			)

			rthm.sl.Slog.Printf(
				"[SPRING PPO] batch_id=%d addr=%s related=%s shard=%d source=%s confidence=%.6f log_prob=%.6f value=%.6f cost_us=%d\n",
				output.BatchID,
				addr,
				related,
				sid,
				source,
				output.Confidence,
				output.LogProb,
				output.Value,
				inferCostUs,
			)

			return sid
		}
	}

	fallbackSid := rthm.springChooseShard(addr, related)

	rthm.springAppendDecisionRecord(
		0,
		string(addr),
		string(related),
		fallbackSid,
		"go_heuristic_fallback",
		0.0,
		0.0,
		0.0,
		state,
		inferCostUs,
		1,
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

	args := []string{
		filepath.Join("spring_lite", "infer_batch.py"),
		"--state",
		inputPath,
		"--out",
		outputPath,
	}

	// SpringOnlineTrain = 1：在线训练阶段，使用 --sample，让 PPO 按策略概率采样。
	// SpringOnlineTrain = 0：验证阶段，不采样，使用最大概率动作。
	// 在线训练阶段：采样 + 更新模型。
	// 验证采样阶段：采样但不更新模型，用于观察概率策略本身效果。
	//SpringOnlineTrain=0, SpringEvalSample=1：采样，但 committee_relay.go 里不会调用 update_online.py。
	//SpringOnlineTrain=0, SpringEvalSample=0：不采样，使用最大概率动作
	if params.SpringMode == 2 &&
		(params.SpringOnlineTrain == 1 || params.SpringEvalSample == 1) {
		args = append(args, "--sample")
	}

	cmd := exec.Command(pythonCmd, args...)

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

	for i := range output.Items {
		if output.Items[i].BatchID == 0 {
			output.Items[i].BatchID = reqID
		}
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
	batchID uint64,
	addr string,
	related string,
	shard uint64,
	source string,
	confidence float64,
	logProb float64,
	value float64,
	state []float64,
	inferCostUs int64,
	batchSize int,
) {
	if err := os.MkdirAll("spring_io", os.ModePerm); err != nil {
		return
	}

	record := SpringDecisionRecord{
		TimeUnixNano: time.Now().UnixNano(),
		BatchID:      batchID,
		Address:      addr,
		Related:      related,
		Shard:        shard,
		Source:       source,
		Confidence:   confidence,
		LogProb:      logProb,
		Value:        value,
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

// springCallPythonOnlineUpdate 调用 spring_lite/update_online.py，
// 根据 online_update_batch_x_epoch_y.json 真正更新 PPO 模型。
// 第一版直接同步调用，先验证训练闭环能跑通。
func (rthm *RelayCommitteeModule) springCallPythonOnlineUpdate(input SpringOnlineUpdateInput) bool {
	if input.BatchID == 0 {
		return false
	}

	inputPath := filepath.Join(
		"spring_io",
		fmt.Sprintf(
			"online_update_batch_%d_epoch_%d.json",
			input.BatchID,
			input.FeedbackEpoch,
		),
	)

	if _, err := os.Stat(inputPath); err != nil {
		rthm.sl.Slog.Printf(
			"[SPRING ONLINE UPDATE] skip batch_id=%d epoch=%d reason=input_not_found path=%s err=%v\n",
			input.BatchID,
			input.FeedbackEpoch,
			inputPath,
			err,
		)
		return false
	}

	pythonCmd := os.Getenv("SPRING_PYTHON")
	if pythonCmd == "" {
		if runtime.GOOS == "windows" {
			pythonCmd = "python"
		} else {
			pythonCmd = "python3"
		}
	}

	modelPath := filepath.Join("spring_lite", "checkpoints", "spring_ppo.pt")
	logPath := filepath.Join("spring_io", "online_train_log.jsonl")

	start := time.Now()

	cmd := exec.Command(
		pythonCmd,
		filepath.Join("spring_lite", "update_online.py"),
		"--input",
		inputPath,
		"--model",
		modelPath,
		"--log",
		logPath,
	)

	out, err := cmd.CombinedOutput()
	costUs := time.Since(start).Microseconds()

	if err != nil {
		rthm.sl.Slog.Printf(
			"[SPRING ONLINE UPDATE] python failed batch_id=%d epoch=%d cost_us=%d err=%v output=%s\n",
			input.BatchID,
			input.FeedbackEpoch,
			costUs,
			err,
			string(out),
		)
		return false
	}

	var result SpringOnlineTrainResult
	if err := json.Unmarshal(out, &result); err != nil {
		rthm.sl.Slog.Printf(
			"[SPRING ONLINE UPDATE] parse result failed batch_id=%d epoch=%d cost_us=%d err=%v raw=%s\n",
			input.BatchID,
			input.FeedbackEpoch,
			costUs,
			err,
			string(out),
		)
		return false
	}

	if !result.Ok {
		rthm.sl.Slog.Printf(
			"[SPRING ONLINE UPDATE] skip batch_id=%d epoch=%d actions=%d reward=%.6f message=%s cost_us=%d\n",
			result.BatchID,
			result.FeedbackEpoch,
			result.NumActions,
			result.Reward,
			result.Message,
			costUs,
		)
		return false
	}

	loss := 0.0
	policyLoss := 0.0
	valueLoss := 0.0
	entropy := 0.0

	if result.LossInfo != nil {
		loss = result.LossInfo["loss"]
		policyLoss = result.LossInfo["policy_loss"]
		valueLoss = result.LossInfo["value_loss"]
		entropy = result.LossInfo["entropy"]
	}

	rthm.sl.Slog.Printf(
		"[SPRING ONLINE UPDATE] ok batch_id=%d epoch=%d actions=%d reward=%.6f crossRate=%.6f normVar=%.6f loss=%.6f policy=%.6f value=%.6f entropy=%.6f update_count=%d source=%s cost_us=%d\n",
		result.BatchID,
		result.FeedbackEpoch,
		result.NumActions,
		result.Reward,
		result.CrossRate,
		result.NormalizedLoadVariance,
		loss,
		policyLoss,
		valueLoss,
		entropy,
		result.OnlineUpdateCount,
		result.ModelSource,
		costUs,
	)

	return true
}
