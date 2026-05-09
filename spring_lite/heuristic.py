from typing import List


def addr2shard(addr: str, shards: int) -> int:
    addr = addr.strip()
    if addr.startswith("0x") or addr.startswith("0X"):
        addr = addr[2:]

    if len(addr) > 8:
        addr = addr[-8:]

    try:
        return int(addr, 16) % shards
    except ValueError:
        return abs(hash(addr)) % shards


def heuristic_from_state(state: List[float], shards: int) -> int:
    """
    Python 推理兜底策略：
    1. 如果 sender_pos 显示交易相关地址已经在某个分片，优先靠近该分片；
    2. 同时使用最近 5 个窗口交易量作为负载惩罚；
    3. 如果没有相关地址，选近期负载最小分片。
    """
    if len(state) < 11 * shards + 1:
        return 0

    recent_num = state[: 5 * shards]
    sender_pos = state[10 * shards : 11 * shards]

    loads = [0.0 for _ in range(shards)]
    for w in range(5):
        for sid in range(shards):
            loads[sid] += float(recent_num[w * shards + sid])

    best_sid = 0
    best_score = -10**18

    has_related = any(v > 0.5 for v in sender_pos)

    for sid in range(shards):
        score = -loads[sid]

        if has_related and sender_pos[sid] > 0.5:
            score += 1000.0

        if score > best_score:
            best_score = score
            best_sid = sid

    return int(best_sid)