package measure

import (
	"blockEmulator/message"
	"blockEmulator/params"
	"math"
	"strconv"
)

type TestShardLoadVariance_Relay struct {
	epochID   int
	shardLoad [][]float64
}

func NewTestShardLoadVariance_Relay() *TestShardLoadVariance_Relay {
	return &TestShardLoadVariance_Relay{
		epochID:   -1,
		shardLoad: make([][]float64, 0),
	}
}

func (tslv *TestShardLoadVariance_Relay) OutputMetricName() string {
	return "Shard_Load_Variance"
}

func (tslv *TestShardLoadVariance_Relay) ensureEpoch(epochid int) {
	for tslv.epochID < epochid {
		load := make([]float64, params.ShardNum)
		tslv.shardLoad = append(tslv.shardLoad, load)
		tslv.epochID++
	}
}

func (tslv *TestShardLoadVariance_Relay) UpdateMeasureRecord(b *message.BlockInfoMsg) {
	if b.BlockBodyLength == 0 {
		return
	}

	epochid := b.Epoch
	if epochid < 0 {
		return
	}

	tslv.ensureEpoch(epochid)

	sid := int(b.SenderShardID)
	if sid < 0 || sid >= params.ShardNum {
		return
	}

	normalTxNum := len(b.InnerShardTxs)
	relay1TxNum := len(b.Relay1Txs)
	relay2TxNum := len(b.Relay2Txs)

	// 和原有 Tx_number / CrossTransaction_ratio 统计保持一致：
	// Relay1 + Relay2 共同代表一笔跨片交易，因此这里除以 2。
	load := float64(normalTxNum) + float64(relay1TxNum+relay2TxNum)/2.0

	tslv.shardLoad[epochid][sid] += load
}

func (tslv *TestShardLoadVariance_Relay) HandleExtraMessage([]byte) {}

func (tslv *TestShardLoadVariance_Relay) OutputRecord() ([]float64, float64) {
	tslv.writeToCSV()

	perEpochVariance := make([]float64, 0)
	totalVariance := 0.0

	for eid := 0; eid <= tslv.epochID; eid++ {
		variance := calcShardLoadVariance(tslv.shardLoad[eid])
		perEpochVariance = append(perEpochVariance, variance)
		totalVariance += variance
	}

	if len(perEpochVariance) == 0 {
		return perEpochVariance, 0
	}

	return perEpochVariance, totalVariance / float64(len(perEpochVariance))
}

func calcShardLoadVariance(loads []float64) float64 {
	if len(loads) == 0 {
		return 0
	}

	sum := 0.0
	for _, v := range loads {
		sum += v
	}

	avg := sum / float64(len(loads))

	variance := 0.0
	for _, v := range loads {
		variance += math.Pow(v-avg, 2)
	}

	return variance / float64(len(loads))
}

func (tslv *TestShardLoadVariance_Relay) writeToCSV() {
	fileName := tslv.OutputMetricName()

	measureName := []string{
		"EpochID",
		"ShardLoadVariance",
	}

	for sid := 0; sid < params.ShardNum; sid++ {
		measureName = append(measureName, "Shard_"+strconv.Itoa(sid)+"_Load")
	}

	measureVals := make([][]string, 0)

	for eid := 0; eid <= tslv.epochID; eid++ {
		variance := calcShardLoadVariance(tslv.shardLoad[eid])

		csvLine := []string{
			strconv.Itoa(eid),
			strconv.FormatFloat(variance, 'f', 8, 64),
		}

		for sid := 0; sid < params.ShardNum; sid++ {
			csvLine = append(csvLine, strconv.FormatFloat(tslv.shardLoad[eid][sid], 'f', 8, 64))
		}

		measureVals = append(measureVals, csvLine)
	}

	WriteMetricsToCSV(fileName, measureName, measureVals)
}
