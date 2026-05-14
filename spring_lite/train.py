import argparse
import csv
import math
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (
    BATCH_SIZE,
    BETA,
    CHECKPOINT_DIR,
    CLIP_EPS,
    DEFAULT_CSV_PATH,
    DEFAULT_SHARD_NUM,
    GAMMA,
    HIDDEN_DIM,
    LAMBDA_WEIGHT,
    LEARNING_RATE,
    MODEL_PATH,
    PPO_EPOCHS,
    state_dim,
)
from heuristic import addr2shard
from ppo import PPOAgent, RolloutBuffer


def parse_tx_row(row: List[str]) -> Optional[Tuple[str, str]]:
    """
    当前 selectedTxs_300K.csv 的有效交易字段沿用 Go 侧 data2tx：
    data[3] = from_address
    data[4] = to_address
    data[6] == "0"
    data[7] == "0"
    """
    if len(row) < 9:
        return None

    sender = row[3].strip()
    recipient = row[4].strip()

    if row[6] != "0" or row[7] != "0":
        return None

    if len(sender) <= 16 or len(recipient) <= 16:
        return None

    if sender == recipient:
        return None

    if sender.startswith("0x") or sender.startswith("0X"):
        sender = sender[2:]
    if recipient.startswith("0x") or recipient.startswith("0X"):
        recipient = recipient[2:]

    return sender, recipient


def calc_reward(num_tx: List[float], cross_tx: List[float], lam: float, beta: float) -> float:
    """
    SPRING 奖励思想：
    - 降低跨片交易比例
    - 保持负载均衡

    Paper formula:
    r_cstr = total / (cross + eps)
    r_wlb = exp(-beta * abs_diff)
    """
    total = float(sum(num_tx))
    cross = float(sum(cross_tx))

    if total <= 0:
        return 0.0

    r_cstr = total / (cross + 1e-6)

    avg = total / len(num_tx)
    abs_diff = sum(abs(x - avg) for x in num_tx)

    r_wlb = math.exp(-beta * abs_diff)

    return lam * r_cstr + (1.0 - lam) * r_wlb


class SpringPlacementSim:
    def __init__(self, shards: int, txs_per_block: int, lam: float, beta: float):
        self.shards = shards
        self.txs_per_block = txs_per_block
        self.lam = lam
        self.beta = beta

        self.addr_shard: Dict[str, int] = {}

        self.recent_stats = deque(maxlen=5)
        for _ in range(5):
            self.recent_stats.append(
                {
                    "num": [0.0 for _ in range(shards)],
                    "cross": [0.0 for _ in range(shards)],
                }
            )

        self.cur_num = [0.0 for _ in range(shards)]
        self.cur_cross = [0.0 for _ in range(shards)]
        self.cur_block_tx_count = 0

    def build_state(self, related_addr: str) -> List[float]:
        state: List[float] = []

        # 最近 5 个块总交易数
        for stat in self.recent_stats:
            state.extend(stat["num"])

        # 最近 5 个块跨片交易数
        for stat in self.recent_stats:
            state.extend(stat["cross"])

        # sender_pos：相关地址所在分片
        related_pos = [0.0 for _ in range(self.shards)]
        if related_addr in self.addr_shard:
            related_pos[self.addr_shard[related_addr]] = 1.0
        state.extend(related_pos)

        # flag F：当前数据无法区分合约账户和普通账户，先统一为 0
        state.append(0.0)

        expected_dim = state_dim(self.shards)
        if len(state) != expected_dim:
            raise RuntimeError(f"state dim mismatch: got {len(state)}, expected {expected_dim}")

        return state

    def place_addr(self, addr: str, shard: int):
        self.addr_shard[addr] = int(shard)

    def has_addr(self, addr: str) -> bool:
        return addr in self.addr_shard

    def get_addr_shard(self, addr: str) -> int:
        return self.addr_shard[addr]

    def finish_tx(self, sender: str, recipient: str):
        ssid = self.addr_shard[sender]
        rsid = self.addr_shard[recipient]

        self.cur_num[ssid] += 1.0
        if ssid != rsid:
            self.cur_cross[ssid] += 1.0

        self.cur_block_tx_count += 1

        done = False
        if self.cur_block_tx_count >= self.txs_per_block:
            self.recent_stats.append(
                {
                    "num": list(self.cur_num),
                    "cross": list(self.cur_cross),
                }
            )
            self.cur_num = [0.0 for _ in range(self.shards)]
            self.cur_cross = [0.0 for _ in range(self.shards)]
            self.cur_block_tx_count = 0
            done = True

        return done

    def current_reward(self) -> float:
        return calc_reward(self.cur_num, self.cur_cross, self.lam, self.beta)


def train(args):
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    sim = SpringPlacementSim(
        shards=args.shards,
        txs_per_block=args.txs_per_block,
        lam=args.lambda_weight,
        beta=args.beta,
    )

    agent = PPOAgent(
        state_dim=state_dim(args.shards),
        action_dim=args.shards,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        clip_eps=args.clip,
        ppo_epochs=args.ppo_epochs,
        entropy_coef=args.entropy_coef,
        device=args.device,
    )

    buffer = RolloutBuffer()

    total_valid_txs = 0
    total_new_addresses = 0
    update_count = 0

    for epoch in range(args.epochs):
        print(f"\n[TRAIN] epoch {epoch + 1}/{args.epochs}")

        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)

            for row in reader:
                parsed = parse_tx_row(row)
                if parsed is None:
                    continue

                sender, recipient = parsed
                total_valid_txs += 1

                pending_records = []

                if not sim.has_addr(sender):
                    state = sim.build_state(recipient)
                    action, log_prob, value = agent.select_action(state)
                    sim.place_addr(sender, action)
                    pending_records.append((state, action, log_prob, value))
                    total_new_addresses += 1

                if not sim.has_addr(recipient):
                    state = sim.build_state(sender)
                    action, log_prob, value = agent.select_action(state)
                    sim.place_addr(recipient, action)
                    pending_records.append((state, action, log_prob, value))
                    total_new_addresses += 1

                done = sim.finish_tx(sender, recipient)
                reward = sim.current_reward()

                for state, action, log_prob, value in pending_records:
                    buffer.add(
                        state=state,
                        action=action,
                        log_prob=log_prob,
                        reward=reward,
                        done=done,
                        value=value,
                    )

                if len(buffer) >= args.batch_size:
                    loss_info = agent.update(buffer)
                    buffer.clear()
                    update_count += 1

                    print(
                        f"[UPDATE] updates={update_count} "
                        f"valid_txs={total_valid_txs} "
                        f"new_addr={total_new_addresses} "
                        f"reward={reward:.4f} "
                        f"loss={loss_info}"
                    )

                if args.max_txs > 0 and total_valid_txs >= args.max_txs:
                    break

        if args.max_txs > 0 and total_valid_txs >= args.max_txs:
            break

    if len(buffer) > 0:
        loss_info = agent.update(buffer)
        update_count += 1
        print(f"[FINAL UPDATE] updates={update_count} loss={loss_info}")

    extra = {
        "shards": args.shards,
        "state_dim": state_dim(args.shards),
        "txs_per_block": args.txs_per_block,
        "lambda_weight": args.lambda_weight,
        "beta": args.beta,
        "total_valid_txs": total_valid_txs,
        "total_new_addresses": total_new_addresses,
    }

    agent.save(args.model, extra=extra)

    print("\n[DONE]")
    print(f"model saved to: {args.model}")
    print(f"valid txs: {total_valid_txs}")
    print(f"new addresses: {total_new_addresses}")
    print(f"updates: {update_count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=str(DEFAULT_CSV_PATH))
    parser.add_argument("--model", type=str, default=str(MODEL_PATH))
    parser.add_argument("--shards", type=int, default=DEFAULT_SHARD_NUM)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_txs", type=int, default=100000)
    parser.add_argument("--txs_per_block", type=int, default=500)

    parser.add_argument("--hidden_dim", type=int, default=HIDDEN_DIM)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--clip", type=float, default=CLIP_EPS)
    parser.add_argument("--ppo_epochs", type=int, default=PPO_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lambda_weight", type=float, default=LAMBDA_WEIGHT)
    parser.add_argument("--beta", type=float, default=BETA)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
