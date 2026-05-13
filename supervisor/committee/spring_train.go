package committee

import (
	"blockEmulator/params"
	"blockEmulator/utils"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"time"
)

type SpringFeedbackRewardRecord struct {
	TimeUnixNano int64 `json:"time_unix_nano"`

	Epoch int `json:"epoch"`

	TotalTx     int `json:"total_tx"`
	TotalInner  int `json:"total_inner"`
	TotalRelay1 int `json:"total_relay1"`
	TotalRelay2 int `json:"total_relay2"`

	Loads  []int `json:"loads"`
	Inners []int `json:"inners"`
	Relay1 []int `json:"relay1"`
	Relay2 []int `json:"relay2"`

	CrossRate              float64 `json:"cross_rate"`
	RawLoadVariance        float64 `json:"raw_load_variance"`
	NormalizedLoadVariance float64 `json:"normalized_load_variance"`
	Reward                 float64 `json:"reward"`

	Lambda float64 `json:"lambda"`
}

type SpringTrainAction struct {
	BatchID uint64    `json:"batch_id"`
	Address string    `json:"address"`
	Related string    `json:"related"`
	State   []float64 `json:"state"`
	Action  int       `json:"action"`
	LogProb float64   `json:"log_prob"`
	Value   float64   `json:"value"`
}

type SpringTrainBatch struct {
	BatchID uint64              `json:"batch_id"`
	Actions []SpringTrainAction `json:"actions"`
}

type SpringOnlineUpdateInput struct {
	TimeUnixNano int64 `json:"time_unix_nano"`

	BatchID       uint64 `json:"batch_id"`
	FeedbackEpoch int    `json:"feedback_epoch"`
	Shards        int    `json:"shards"`

	Actions    []SpringTrainAction `json:"actions"`
	Reward     float64             `json:"reward"`
	NextStates [][]float64         `json:"next_states"`
	Done       bool                `json:"done"`

	CrossRate              float64 `json:"cross_rate"`
	NormalizedLoadVariance float64 `json:"normalized_load_variance"`
	TotalTx                int     `json:"total_tx"`
	TotalInner             int     `json:"total_inner"`
	TotalRelay1            int     `json:"total_relay1"`
	TotalRelay2            int     `json:"total_relay2"`
}

// springBuildFeedbackRewardRecord 根据一个 epoch 中所有 shard 的真实出块反馈计算 reward。
// 第一版 reward：
// reward = λ * (1 - crossRate) - (1 - λ) * normalizedLoadVariance
//
// 注意：
// 1. crossRate 只用 Relay1 计算，不把 Relay2 重复算入跨片率。
// 2. 如果这个 epoch 只有 Relay2，没有 inner 和 Relay1，说明只是跨片第二阶段收尾，跳过，不生成 reward。
// 3. 不做显式负载保护，只通过 reward 惩罚负载不均衡。
func (rthm *RelayCommitteeModule) springBuildFeedbackRewardRecord(
	epoch int,
	shardStats map[uint64]SpringBlockStat,
) (SpringFeedbackRewardRecord, bool) {
	lambda := 0.5
	eps := 1e-8

	loads := make([]int, params.ShardNum)
	inners := make([]int, params.ShardNum)
	relay1s := make([]int, params.ShardNum)
	relay2s := make([]int, params.ShardNum)

	totalTx := 0
	totalInner := 0
	totalRelay1 := 0
	totalRelay2 := 0

	for sid := 0; sid < params.ShardNum; sid++ {
		stat := shardStats[uint64(sid)]

		loads[sid] = stat.NumTx
		inners[sid] = stat.InnerTx
		relay1s[sid] = stat.Relay1Tx
		relay2s[sid] = stat.Relay2Tx

		totalTx += stat.NumTx
		totalInner += stat.InnerTx
		totalRelay1 += stat.Relay1Tx
		totalRelay2 += stat.Relay2Tx
	}

	// 全空 epoch 跳过。
	if totalTx == 0 && totalInner == 0 && totalRelay1 == 0 && totalRelay2 == 0 {
		return SpringFeedbackRewardRecord{}, false
	}

	// 关键修复：
	// 如果这个 epoch 只有 Relay2，没有 inner 和 Relay1，
	// 说明它不是一个新的原始交易批次反馈，而是跨片第二阶段收尾。
	// 这种 epoch 不能给 PPO 正奖励，否则会误导训练。
	decisionRelatedTx := totalInner + totalRelay1
	if decisionRelatedTx == 0 {
		return SpringFeedbackRewardRecord{}, false
	}

	crossRate := float64(totalRelay1) / float64(decisionRelatedTx)

	avgLoad := float64(totalTx) / float64(params.ShardNum)
	rawVar := 0.0
	for _, load := range loads {
		diff := float64(load) - avgLoad
		rawVar += diff * diff
	}
	rawVar = rawVar / float64(params.ShardNum)

	normVar := rawVar / (avgLoad*avgLoad + eps)

	// 压缩到 [0,1)，避免方差太大导致 reward 爆炸。
	normVar = normVar / (1.0 + normVar)

	reward := lambda*(1.0-crossRate) - (1.0-lambda)*normVar

	if math.IsNaN(reward) || math.IsInf(reward, 0) {
		reward = 0.0
	}
	if math.IsNaN(crossRate) || math.IsInf(crossRate, 0) {
		crossRate = 0.0
	}
	if math.IsNaN(normVar) || math.IsInf(normVar, 0) {
		normVar = 0.0
	}

	record := SpringFeedbackRewardRecord{
		TimeUnixNano: time.Now().UnixNano(),

		Epoch: epoch,

		TotalTx:     totalTx,
		TotalInner:  totalInner,
		TotalRelay1: totalRelay1,
		TotalRelay2: totalRelay2,

		Loads:  loads,
		Inners: inners,
		Relay1: relay1s,
		Relay2: relay2s,

		CrossRate:              crossRate,
		RawLoadVariance:        rawVar,
		NormalizedLoadVariance: normVar,
		Reward:                 reward,

		Lambda: lambda,
	}

	return record, true
}

func (rthm *RelayCommitteeModule) springAppendFeedbackRecord(record SpringFeedbackRewardRecord) {
	if err := os.MkdirAll("spring_io", os.ModePerm); err != nil {
		return
	}

	b, err := json.Marshal(record)
	if err != nil {
		return
	}

	path := filepath.Join("spring_io", "feedback_records.jsonl")
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return
	}
	defer f.Close()

	_, _ = f.Write(append(b, '\n'))
}

// 注意：这个函数默认在 rthm.springLock 已经加锁时调用。
// 不要在这里重复加锁，避免死锁。
func (rthm *RelayCommitteeModule) springEnqueueTrainActionsLocked(batchID uint64, actions []SpringTrainAction) {
	if batchID == 0 || len(actions) == 0 {
		return
	}

	newBatch := SpringTrainBatch{
		BatchID: batchID,
		Actions: actions,
	}

	// 防御：如果最后一个 pending batch 和当前 batchID 一样，
	// 说明重复入队了，直接用更完整的 actions 覆盖它。
	// 正常情况下这个分支不会触发。
	if len(rthm.springPendingTrainBatches) > 0 {
		lastIdx := len(rthm.springPendingTrainBatches) - 1
		last := rthm.springPendingTrainBatches[lastIdx]

		if last.BatchID == batchID {
			if len(actions) >= len(last.Actions) {
				rthm.springPendingTrainBatches[lastIdx] = newBatch
			}

			rthm.sl.Slog.Printf(
				"[SPRING ONLINE ACTIONS] replace batch_id=%d actions=%d pending_batches=%d\n",
				batchID,
				len(actions),
				len(rthm.springPendingTrainBatches),
			)
			return
		}
	}

	rthm.springPendingTrainBatches = append(rthm.springPendingTrainBatches, newBatch)

	rthm.sl.Slog.Printf(
		"[SPRING ONLINE ACTIONS] enqueue batch_id=%d actions=%d pending_batches=%d\n",
		batchID,
		len(actions),
		len(rthm.springPendingTrainBatches),
	)
}

// 注意：这个函数默认在 rthm.springLock 已经加锁时调用。
// 它把最早的 PPO 动作批次和当前 reward 绑定，生成一个 online_update 输入。
func (rthm *RelayCommitteeModule) springBuildOnlineUpdateInputLocked(
	rewardRecord SpringFeedbackRewardRecord,
) (SpringOnlineUpdateInput, bool) {
	if len(rthm.springPendingTrainBatches) == 0 {
		rthm.sl.Slog.Printf(
			"[SPRING ONLINE UPDATE SKIP] epoch=%d reason=no_pending_train_batch reward=%.6f\n",
			rewardRecord.Epoch,
			rewardRecord.Reward,
		)
		return SpringOnlineUpdateInput{}, false
	}

	batch := rthm.springPendingTrainBatches[0]
	rthm.springPendingTrainBatches = rthm.springPendingTrainBatches[1:]

	if len(batch.Actions) == 0 {
		return SpringOnlineUpdateInput{}, false
	}

	nextStates := make([][]float64, 0, len(batch.Actions))
	for _, action := range batch.Actions {
		nextState := rthm.springBuildState(utils.Address(action.Related))
		nextStates = append(nextStates, nextState)
	}

	input := SpringOnlineUpdateInput{
		TimeUnixNano: time.Now().UnixNano(),

		BatchID:       batch.BatchID,
		FeedbackEpoch: rewardRecord.Epoch,
		Shards:        params.ShardNum,

		Actions:    batch.Actions,
		Reward:     rewardRecord.Reward,
		NextStates: nextStates,

		// 第一版先把每个 TxBatch 当成一个独立训练片段。
		// 后续真正连续在线训练时，再改成只有最后一批 done=true。
		Done: true,

		CrossRate:              rewardRecord.CrossRate,
		NormalizedLoadVariance: rewardRecord.NormalizedLoadVariance,
		TotalTx:                rewardRecord.TotalTx,
		TotalInner:             rewardRecord.TotalInner,
		TotalRelay1:            rewardRecord.TotalRelay1,
		TotalRelay2:            rewardRecord.TotalRelay2,
	}

	return input, true
}

func (rthm *RelayCommitteeModule) springWriteOnlineUpdateInput(input SpringOnlineUpdateInput) {
	if err := os.MkdirAll("spring_io", os.ModePerm); err != nil {
		return
	}

	b, err := json.Marshal(input)
	if err != nil {
		rthm.sl.Slog.Printf("[SPRING ONLINE UPDATE FILE] marshal failed: %v\n", err)
		return
	}

	path := filepath.Join(
		"spring_io",
		"online_update_batch_"+uint64ToString(input.BatchID)+"_epoch_"+intToString(input.FeedbackEpoch)+".json",
	)

	if err := os.WriteFile(path, b, 0644); err != nil {
		rthm.sl.Slog.Printf("[SPRING ONLINE UPDATE FILE] write failed: %v\n", err)
		return
	}

	rthm.sl.Slog.Printf(
		"[SPRING ONLINE UPDATE FILE] batch_id=%d epoch=%d actions=%d reward=%.6f crossRate=%.6f normVar=%.6f file=%s\n",
		input.BatchID,
		input.FeedbackEpoch,
		len(input.Actions),
		input.Reward,
		input.CrossRate,
		input.NormalizedLoadVariance,
		path,
	)
}

func uint64ToString(v uint64) string {
	return strconv.FormatUint(v, 10)
}

func intToString(v int) string {
	return strconv.FormatInt(int64(v), 10)
}
