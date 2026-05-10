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
	NumTx   int
	CrossTx int
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

		springAddrShard: make(map[string]uint64),
		springShardLoad: make([]int, params.ShardNum),
		springStats:     springStats,
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

	for _, tx := range txlist {
		// 如果 sender 是第一次出现，优先参考 recipient 的已有分片
		rthm.springEnsurePlaced(tx.Sender, tx.Recipient, batchPlacement)

		// 如果 recipient 是第一次出现，优先参考 sender 的已有分片
		rthm.springEnsurePlaced(tx.Recipient, tx.Sender, batchPlacement)
	}

	return batchPlacement
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
				go networks.TcpDial(sendMsg, rthm.IpNodeTable[sid][0])
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
	rthm.sl.Slog.Printf("received from shard %d in epoch %d.\n", b.SenderShardID, b.Epoch)

	if b.BlockBodyLength == 0 {
		return
	}

	rthm.springLock.Lock()
	defer rthm.springLock.Unlock()

	stat := SpringBlockStat{
		NumTx:   b.BlockBodyLength,
		CrossTx: len(b.Relay1Txs) + len(b.Relay2Txs),
	}

	arr := rthm.springStats[b.SenderShardID]
	arr = append(arr, stat)

	if len(arr) > 5 {
		arr = arr[len(arr)-5:]
	}

	rthm.springStats[b.SenderShardID] = arr
}

func (rthm *RelayCommitteeModule) springGetStat(sid uint64, back int) SpringBlockStat {
	arr := rthm.springStats[sid]
	idx := len(arr) - 1 - back

	if idx < 0 {
		return SpringBlockStat{}
	}

	return arr[idx]
}

func (rthm *RelayCommitteeModule) springBuildState(related utils.Address) []float64 {
	state := make([]float64, 0, 11*params.ShardNum+1)

	// 最近 5 个块的总交易数
	for back := 4; back >= 0; back-- {
		for sid := uint64(0); sid < uint64(params.ShardNum); sid++ {
			st := rthm.springGetStat(sid, back)
			state = append(state, float64(st.NumTx))
		}
	}

	// 最近 5 个块的跨片交易数
	for back := 4; back >= 0; back-- {
		for sid := uint64(0); sid < uint64(params.ShardNum); sid++ {
			st := rthm.springGetStat(sid, back)
			state = append(state, float64(st.CrossTx))
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
