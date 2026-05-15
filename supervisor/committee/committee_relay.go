package committee

import (
	"blockEmulator/core"
	"blockEmulator/message"
	"blockEmulator/networks"
	"blockEmulator/params"
	"blockEmulator/supervisor/signal"
	"blockEmulator/supervisor/supervisor_log"
	"blockEmulator/utils"
	"encoding/csv"
	"encoding/json"
	"io"
	"log"
	"math/big"
	"os"
	"time"

	"sync"
)

type SpringBlockStat struct {
	NumTx    int `json:"num_tx"`
	InnerTx  int `json:"inner_tx"`
	Relay1Tx int `json:"relay1_tx"`
	Relay2Tx int `json:"relay2_tx"`
	CrossTx  int `json:"cross_tx"`
}

type RelayCommitteeModule struct {
	csvPath      string
	dataTotalNum int
	nowDataNum   int
	batchDataNum int

	IpNodeTable map[uint64]map[uint64]string
	sl          *supervisor_log.SupervisorLog
	Ss          *signal.StopSignal

	// SPRING: Supervisor 临时代替 A-Shard 保存全局地址放置表
	springLock      sync.Mutex
	springAddrShard map[string]uint64
	springShardLoad []int

	// SPRING: 保存最近若干个块的每分片交易数和跨片交易数
	// 后续 PPO 状态向量要用
	springStats map[uint64][]SpringBlockStat

	// SPRING 在线训练第一步：按 epoch 收集真实区块反馈，用于计算 reward
	springEpochFeedback map[int]map[uint64]SpringBlockStat
	springRewardedEpoch map[int]bool

	// SPRING 在线训练：等待和真实区块 reward 匹配的 PPO 动作批次
	springPendingTrainBatches []SpringTrainBatch

	// SPRING: 新地址放置动作编号，用于生成 action_1.json、action_2.json
	springActionSeq uint64
}

func NewRelayCommitteeModule(Ip_nodeTable map[uint64]map[uint64]string, Ss *signal.StopSignal, slog *supervisor_log.SupervisorLog, csvFilePath string, dataNum, batchNum int) *RelayCommitteeModule {
	springStats := make(map[uint64][]SpringBlockStat)
	for sid := uint64(0); sid < uint64(params.ShardNum); sid++ {
		springStats[sid] = make([]SpringBlockStat, 0, 5)
	}
	return &RelayCommitteeModule{
		csvPath:      csvFilePath,
		dataTotalNum: dataNum,
		batchDataNum: batchNum,
		nowDataNum:   0,
		IpNodeTable:  Ip_nodeTable,
		Ss:           Ss,
		sl:           slog,

		springAddrShard:     make(map[string]uint64),
		springShardLoad:     make([]int, params.ShardNum),
		springStats:         springStats,
		springEpochFeedback: make(map[int]map[uint64]SpringBlockStat),
		springRewardedEpoch: make(map[int]bool),
	}
}

// transfrom, data to transaction
// check whether it is a legal txs meesage. if so, read txs and put it into the txlist
func data2tx(data []string, nonce uint64) (*core.Transaction, bool) {
	if data[6] == "0" && data[7] == "0" && len(data[3]) > 16 && len(data[4]) > 16 && data[3] != data[4] {
		val, ok := new(big.Int).SetString(data[8], 10)
		if !ok {
			log.Panic("new int failed\n")
		}
		tx := core.NewTransaction(data[3][2:], data[4][2:], val, nonce, time.Now())
		return tx, true
	}
	return &core.Transaction{}, false
}

func (rthm *RelayCommitteeModule) HandleOtherMessage([]byte) {}

// SPRING: 判断地址是否已经被放置；如果没有，就根据 SpringMode 给它分片
func (rthm *RelayCommitteeModule) springEnsurePlaced(
	addr utils.Address,
	related utils.Address,
	batchPlacement map[string]uint64,
) uint64 {
	if sid, ok := rthm.springAddrShard[string(addr)]; ok {
		return sid
	}

	var sid uint64

	switch params.SpringMode {
	case 1:
		// SPRING-Heuristic：只使用 Go 里的启发式规则，不调用 Python
		sid = rthm.springChooseShard(addr, related)

	case 2:
		// SPRING-PPO：调用 Python PPO；如果 Python 失败，springChooseShardPPO 内部会自动回退到启发式
		sid = rthm.springChooseShardPPO(addr, related)

	default:
		// SpringMode = 0 或其他非法值：退化为原始 Hash 放置
		sid = uint64(utils.Addr2Shard(addr))
	}

	rthm.springAddrShard[string(addr)] = sid
	rthm.springShardLoad[sid]++
	batchPlacement[string(addr)] = sid

	rthm.sl.Slog.Printf(
		"[SPRING PLACE] mode=%d addr=%s shard=%d related=%s totalPlaced=%d\n",
		params.SpringMode,
		addr,
		sid,
		related,
		len(rthm.springAddrShard),
	)

	return sid
}

// SPRING 第一版简单策略：
// 1. 如果 related 地址已经有分片，优先放到 related 的分片，降低跨片交易
// 2. 同时考虑当前放置负载，避免所有新地址都堆到一个分片
// 3. 如果没有 related，则退化为负载最小 + 哈希打破平局
func (rthm *RelayCommitteeModule) springChooseShard(
	addr utils.Address,
	related utils.Address,
) uint64 {
	relatedSid := -1
	if related != "" {
		if sid, ok := rthm.springAddrShard[string(related)]; ok {
			relatedSid = int(sid)
		}
	}

	hashSid := uint64(utils.Addr2Shard(addr))

	bestSid := uint64(0)
	bestScore := -1 << 60

	for sid := 0; sid < params.ShardNum; sid++ {
		score := 0

		// 交互关系分：如果新地址的交易对手在这个分片，强烈倾向于放一起
		if sid == relatedSid {
			score += 1000
		}

		// 负载惩罚：分片已经放置的地址越多，分数越低
		score -= rthm.springShardLoad[sid]

		// 哈希结果只作为平局打破项
		if uint64(sid) == hashSid {
			score += 1
		}

		if score > bestScore {
			bestScore = score
			bestSid = uint64(sid)
		}
	}

	return bestSid
}

// SPRING: 对一批交易提前完成新地址放置
func (rthm *RelayCommitteeModule) springPreparePlacement(
	txlist []*core.Transaction,
) map[string]uint64 {
	rthm.springLock.Lock()
	defer rthm.springLock.Unlock()

	batchPlacement := make(map[string]uint64)

	switch params.SpringMode {
	case 0:
		// 原始 Hash Relay，不做 SPRING 放置。
		return batchPlacement

	case 1:
		for _, tx := range txlist {
			rthm.springEnsurePlaced(tx.Sender, tx.Recipient, batchPlacement)
			rthm.springEnsurePlaced(tx.Recipient, tx.Sender, batchPlacement)
		}
		rthm.springFillTouchedPlacement(txlist, batchPlacement)
		return batchPlacement

	case 2:
		rthm.springPreparePlacementPPOBatch(txlist, batchPlacement)
		rthm.springFillTouchedPlacement(txlist, batchPlacement)
		return batchPlacement

	default:
		// 非法模式兜底：当作 Hash 放置，但仍同步 PlacementMap，避免空映射导致异常。
		for _, tx := range txlist {
			rthm.springEnsurePlaced(tx.Sender, tx.Recipient, batchPlacement)
			rthm.springEnsurePlaced(tx.Recipient, tx.Sender, batchPlacement)
		}
		return batchPlacement
	}
}

func (rthm *RelayCommitteeModule) springFillTouchedPlacement(
	txlist []*core.Transaction,
	batchPlacement map[string]uint64,
) {
	for _, tx := range txlist {
		if sid, ok := rthm.springAddrShard[string(tx.Sender)]; ok {
			batchPlacement[string(tx.Sender)] = sid
		}

		if sid, ok := rthm.springAddrShard[string(tx.Recipient)]; ok {
			batchPlacement[string(tx.Recipient)] = sid
		}
	}
}

func (rthm *RelayCommitteeModule) springPreparePlacementPPOBatch(
	txlist []*core.Transaction,
	batchPlacement map[string]uint64,
) {
	// SPRING paper semantics: sender_pos is updated after each placement
	// action within the current A-Shard block.
	trainActions := make([]SpringTrainAction, 0)
	for _, tx := range txlist {
		if action, ok := rthm.springPlaceAddressPPOSequential(tx.Sender, tx.Recipient, batchPlacement); ok {
			trainActions = append(trainActions, action)
		}
		if action, ok := rthm.springPlaceAddressPPOSequential(tx.Recipient, tx.Sender, batchPlacement); ok {
			trainActions = append(trainActions, action)
		}
	}

	if params.SpringOnlineTrain == 1 && len(trainActions) > 0 {
		rthm.springEnqueueTrainActionsLocked(trainActions[0].BatchID, trainActions)
	}
}

func (rthm *RelayCommitteeModule) springPlaceAddressPPOSequential(
	addr utils.Address,
	related utils.Address,
	batchPlacement map[string]uint64,
) (SpringTrainAction, bool) {
	key := string(addr)
	if key == "" {
		return SpringTrainAction{}, false
	}

	// 已经放置过的地址，不再重复决策。
	if _, ok := rthm.springAddrShard[key]; ok {
		return SpringTrainAction{}, false
	}

	relatedKey := string(related)

	// related_in_current_batch：
	// 表示 related 地址是否是在当前 TxBatch 内已经被前面的 sequential 决策放置过。
	relatedInCurrentBatch := false
	if relatedKey != "" {
		if _, ok := batchPlacement[relatedKey]; ok {
			relatedInCurrentBatch = true
		}
	}

	// 构造 PPO 状态。
	// 当前代码里的 springBuildState 已经会把最近 5 个块的 totalTx/crossTx 归一化到 0~1。
	state := rthm.springBuildState(related)

	// 从 state 的 sender_pos 区域反推 related 是否已知、在哪个 shard。
	// 这样可以保证诊断字段和 PPO 实际看到的 state 一致。
	relatedKnown, relatedShard := springExtractRelatedShardFromState(state)

	item := SpringBatchInferItem{
		Address: key,
		Related: relatedKey,
		State:   state,
	}

	results, inferCostUs, ok := rthm.springCallPythonBatch([]SpringBatchInferItem{item})

	result := SpringBatchInferResult{}
	if ok && len(results) == 1 {
		result = results[0]
	}

	var sid uint64
	source := ""
	confidence := 0.0

	if ok && result.Shard >= 0 && result.Shard < params.ShardNum {
		sid = uint64(result.Shard)
		source = result.Source
		confidence = result.Confidence
		if source == "" {
			source = "python_ppo"
		}
	} else {
		sid = rthm.springChooseShard(addr, related)
		source = "go_heuristic_sequential_fallback"
	}

	// sequential 语义的关键：
	// 当前地址决策后立即落表，后续地址构造 state 时能看到它。
	rthm.springAddrShard[key] = sid
	rthm.springShardLoad[sid]++
	batchPlacement[key] = sid

	chosenShard := int(sid)
	sameAsRelated := relatedKnown && chosenShard == relatedShard

	rthm.springAppendDecisionRecord(
		result.BatchID,
		key,
		item.Related,
		sid,
		source,
		confidence,
		result.LogProb,
		result.Value,
		state,
		inferCostUs,
		1,
	)

	rthm.sl.Slog.Printf(
		"[SPRING PPO SEQ PLACE] mode=%d addr=%s shard=%d related=%s source=%s confidence=%.6f related_known=%v related_shard=%d chosen_shard=%d same_as_related=%v related_in_current_batch=%v totalPlaced=%d\n",
		params.SpringMode,
		key,
		sid,
		item.Related,
		source,
		confidence,
		relatedKnown,
		relatedShard,
		chosenShard,
		sameAsRelated,
		relatedInCurrentBatch,
		len(rthm.springAddrShard),
	)

	// 只有真正由 PPO 网络产生的动作，才进入在线训练。
	// heuristic fallback 只是保证系统能跑，不作为 PPO 经验。
	if source != "python_ppo" || result.BatchID == 0 {
		return SpringTrainAction{}, false
	}

	return SpringTrainAction{
		BatchID:               result.BatchID,
		Address:               key,
		Related:               item.Related,
		State:                 state,
		Action:                chosenShard,
		LogProb:               result.LogProb,
		Value:                 result.Value,
		RelatedKnown:          relatedKnown,
		RelatedShard:          relatedShard,
		ChosenShard:           chosenShard,
		SameAsRelated:         sameAsRelated,
		RelatedInCurrentBatch: relatedInCurrentBatch,
	}, true
}

func (rthm *RelayCommitteeModule) txSending(txlist []*core.Transaction) {
	useSpringPlacement := params.SpringMode != 0

	batchPlacement := make(map[string]uint64)

	// SpringMode = 1 或 2 时，才进行 SPRING 新地址放置
	// SpringMode = 0 时，直接走原始 Hash Relay
	if useSpringPlacement {
		batchPlacement = rthm.springPreparePlacement(txlist)
	}

	// the txs will be sent
	sendToShard := make(map[uint64][]*core.Transaction)

	for idx := 0; idx <= len(txlist); idx++ {
		if idx > 0 && (idx%params.InjectSpeed == 0 || idx == len(txlist)) {
			if useSpringPlacement {
				counts := make([]int, params.ShardNum)
				for sid := uint64(0); sid < uint64(params.ShardNum); sid++ {
					counts[sid] = len(sendToShard[sid])
				}

				rthm.sl.Slog.Printf(
					"[SPRING SEND] nowData=%d txlistLen=%d idx=%d counts=%v placementSize=%d\n",
					rthm.nowDataNum,
					len(txlist),
					idx,
					counts,
					len(batchPlacement),
				)
			}
			// send to shard
			for sid := uint64(0); sid < uint64(params.ShardNum); sid++ {
				it := message.InjectTxs{
					Txs:       sendToShard[sid],
					ToShardID: sid,
				}

				// 只有 SPRING-Heuristic / SPRING-PPO 才需要同步放置表
				// 原始 Hash Relay 不需要 PlacementMap
				if useSpringPlacement {
					it.PlacementMap = batchPlacement
				}

				itByte, err := json.Marshal(it)
				if err != nil {
					log.Panic(err)
				}

				sendMsg := message.MergeMessage(message.CInject, itByte)
				for _, ip := range rthm.IpNodeTable[sid] {
					go networks.TcpDial(sendMsg, ip)
				}
			}

			sendToShard = make(map[uint64][]*core.Transaction)
			time.Sleep(time.Second)
		}

		if idx == len(txlist) {
			break
		}

		tx := txlist[idx]

		var sendersid uint64

		if useSpringPlacement {
			// SPRING-Heuristic / SPRING-PPO：
			// 交易发送到 sender 在 SPRING 放置表中的分片
			sendersid = rthm.springAddrShard[string(tx.Sender)]
		} else {
			// 原始 Hash Relay：
			// 交易发送到 sender 哈希映射得到的分片
			sendersid = uint64(utils.Addr2Shard(tx.Sender))
		}

		sendToShard[sendersid] = append(sendToShard[sendersid], tx)
	}
}

// read transactions, the Number of the transactions is - batchDataNum
func (rthm *RelayCommitteeModule) MsgSendingControl() {
	txfile, err := os.Open(rthm.csvPath)
	if err != nil {
		log.Panic(err)
	}
	defer txfile.Close()
	reader := csv.NewReader(txfile)
	txlist := make([]*core.Transaction, 0) // save the txs in this epoch (round)

	for {
		data, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Panic(err)
		}
		if tx, ok := data2tx(data, uint64(rthm.nowDataNum)); ok {
			txlist = append(txlist, tx)
			rthm.nowDataNum++
		}

		// re-shard condition, enough edges
		if len(txlist) == int(rthm.batchDataNum) || rthm.nowDataNum == rthm.dataTotalNum {
			rthm.txSending(txlist)
			// reset the variants about tx sending
			txlist = make([]*core.Transaction, 0)
			rthm.Ss.StopGap_Reset()
		}

		if rthm.nowDataNum == rthm.dataTotalNum {
			break
		}
	}
}

func (rthm *RelayCommitteeModule) HandleBlockInfo(b *message.BlockInfoMsg) {
	rthm.sl.Slog.Printf(
		"[BLOCK INFO] shard=%d epoch=%d body=%d inner=%d relay1=%d relay2=%d\n",
		b.SenderShardID,
		b.Epoch,
		b.BlockBodyLength,
		len(b.InnerShardTxs),
		len(b.Relay1Txs),
		len(b.Relay2Txs),
	)

	rthm.springLock.Lock()
	defer rthm.springLock.Unlock()

	// 注意：
	// 1. CrossTx 第一版只用 Relay1Tx。
	// 2. Relay2Tx 是跨片交易第二阶段，不要和 Relay1 一起重复计入 crossRate。
	stat := SpringBlockStat{
		NumTx:    b.BlockBodyLength,
		InnerTx:  len(b.InnerShardTxs),
		Relay1Tx: len(b.Relay1Txs),
		Relay2Tx: len(b.Relay2Txs),
		CrossTx:  len(b.Relay1Txs),
	}

	// 更新最近 5 个区块窗口，供 springBuildState() 继续使用。
	// 这里不再跳过 body=0 的空块，因为空块也代表该 shard 当前负载为 0。
	arr := rthm.springStats[b.SenderShardID]
	arr = append(arr, stat)
	if len(arr) > 5 {
		arr = arr[len(arr)-5:]
	}
	rthm.springStats[b.SenderShardID] = arr

	// 按 epoch 收集所有 shard 的真实反馈。
	if _, ok := rthm.springEpochFeedback[b.Epoch]; !ok {
		rthm.springEpochFeedback[b.Epoch] = make(map[uint64]SpringBlockStat)
	}
	rthm.springEpochFeedback[b.Epoch][b.SenderShardID] = stat

	// 收齐一个 epoch 的所有 shard 反馈后，计算一次 reward。
	if len(rthm.springEpochFeedback[b.Epoch]) == params.ShardNum && !rthm.springRewardedEpoch[b.Epoch] {
		record, ok := rthm.springBuildFeedbackRewardRecord(b.Epoch, rthm.springEpochFeedback[b.Epoch])
		if ok {
			rthm.springAppendFeedbackRecord(record)

			rthm.sl.Slog.Printf(
				"[SPRING ONLINE REWARD] epoch=%d total=%d inner=%d relay1=%d relay2=%d crossRate=%.6f rCSTR=%.6f rWLB=%.6f absDiff=%.6f normVar=%.6f reward=%.6f lambda=%.3f beta=%.3f loads=%v\n",
				record.Epoch,
				record.TotalTx,
				record.TotalInner,
				record.TotalRelay1,
				record.TotalRelay2,
				record.CrossRate,
				record.RCSTR,
				record.RWLB,
				record.AbsLoadDiff,
				record.NormalizedLoadVariance,
				record.Reward,
				record.Lambda,
				record.Beta,
				record.Loads,
			)

			if params.SpringMode == 2 && params.SpringOnlineTrain == 1 {
				onlineInput, updateOk := rthm.springBuildOnlineUpdateInputLocked(record)
				if updateOk {
					rthm.springWriteOnlineUpdateInput(onlineInput)

					// 在线训练阶段：生成 online_update 文件后，调用 Python 更新 PPO。
					// 不做显式负载保护，不改 PPO 动作，只用 reward 训练模型。
					rthm.springCallPythonOnlineUpdate(onlineInput)
				}
			} else if params.SpringMode == 2 && params.SpringOnlineTrain == 0 {
				rthm.sl.Slog.Printf(
					"[SPRING ONLINE UPDATE SKIP] epoch=%d reason=SpringOnlineTrain_disabled reward=%.6f crossRate=%.6f normVar=%.6f\n",
					record.Epoch,
					record.Reward,
					record.CrossRate,
					record.NormalizedLoadVariance,
				)
			}
		} else {
			rthm.sl.Slog.Printf(
				"[SPRING ONLINE REWARD SKIP] epoch=%d reason=empty_or_relay2_only\n",
				b.Epoch,
			)
		}
		rthm.springRewardedEpoch[b.Epoch] = true

		// 简单清理旧 epoch，避免长时间运行 map 越来越大。
		for oldEpoch := range rthm.springEpochFeedback {
			if oldEpoch < b.Epoch-10 {
				delete(rthm.springEpochFeedback, oldEpoch)
			}
		}
		for oldEpoch := range rthm.springRewardedEpoch {
			if oldEpoch < b.Epoch-10 {
				delete(rthm.springRewardedEpoch, oldEpoch)
			}
		}
	}
}

func (rthm *RelayCommitteeModule) springGetStat(sid uint64, back int) SpringBlockStat {
	arr := rthm.springStats[sid]
	idx := len(arr) - 1 - back

	if idx < 0 {
		return SpringBlockStat{}
	}

	return arr[idx]
}

func springNormalizeStateCount(v int) float64 {
	denom := float64(params.TxBatchSize)
	if denom <= 0 {
		denom = 100.0
	}

	x := float64(v) / denom
	if x < 0 {
		return 0
	}
	if x > 1 {
		return 1
	}
	return x
}

func springExtractRelatedShardFromState(state []float64) (bool, int) {
	base := len(state) - 1 - params.ShardNum
	if base < 0 {
		return false, -1
	}

	for sid := 0; sid < params.ShardNum; sid++ {
		if state[base+sid] > 0.5 {
			return true, sid
		}
	}
	return false, -1
}

func (rthm *RelayCommitteeModule) springBuildState(related utils.Address) []float64 {
	state := make([]float64, 0, 11*params.ShardNum+1)

	// 最近 5 个块的总交易数
	for back := 4; back >= 0; back-- {
		for sid := uint64(0); sid < uint64(params.ShardNum); sid++ {
			st := rthm.springGetStat(sid, back)
			state = append(state, springNormalizeStateCount(st.NumTx))
		}
	}

	// 最近 5 个块的跨片交易数
	for back := 4; back >= 0; back-- {
		for sid := uint64(0); sid < uint64(params.ShardNum); sid++ {
			st := rthm.springGetStat(sid, back)
			state = append(state, springNormalizeStateCount(st.CrossTx))
		}
	}

	// sender_pos：相关地址所在分片
	relatedSid := -1
	if sid, ok := rthm.springAddrShard[string(related)]; ok {
		relatedSid = int(sid)
	}

	for sid := 0; sid < params.ShardNum; sid++ {
		if sid == relatedSid {
			state = append(state, 1.0)
		} else {
			state = append(state, 0.0)
		}
	}

	// flag F：当前数据集里没有合约/普通账户类型，先统一设为 0
	state = append(state, 0.0)

	return state
}
