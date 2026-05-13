# 一次读取一个 batch_state_x.json，一次加载模型，然后一次输出这一批所有新地址的分片结果。
# 第一阶段增强版：
# 1. 保持原有批量推理功能；
# 2. 输出 log_prob 和 value，为后续在线 PPO 训练做准备；
# 3. 暂时不改变 Go 侧调用方式；
# 4. 如果模型不存在或维度不匹配，仍然安全回退到 heuristic，保证实验能跑通。

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.distributions import Categorical

from config import HIDDEN_DIM, MODEL_PATH, state_dim
from heuristic import heuristic_from_state
from ppo import PPOAgent


def load_input(path: str) -> Tuple[List[Dict[str, Any]], int]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    shards = int(data.get("shards", 4))
    items = data.get("items", [])

    if not isinstance(items, list):
        raise ValueError("items must be a list")

    normalized_items: List[Dict[str, Any]] = []

    for item in items:
        state = item.get("state", [])
        if not isinstance(state, list):
            state = []

        normalized_items.append(
            {
                "address": str(item.get("address", "")),
                "related": str(item.get("related", "")),
                "state": [float(x) for x in state],
            }
        )

    return normalized_items, shards


def safe_heuristic(item: Dict[str, Any], shards: int, reason: str) -> Dict[str, Any]:
    state = item.get("state", [])
    shard = heuristic_from_state(state, shards)

    if shard < 0 or shard >= shards:
        shard = 0
        reason = "safe_default"

    return {
        "address": item.get("address", ""),
        "related": item.get("related", ""),
        "shard": int(shard),
        "source": f"python_heuristic_{reason}",
        "confidence": 0.0,
        # 为了让 Go 侧结构统一，启发式兜底也输出这两个字段。
        # 但后续真正训练时，只把 source == python_ppo 的动作写入 PPO 经验。
        "log_prob": 0.0,
        "value": 0.0,
    }


def load_agent(shards: int, model_path: Path) -> Tuple[Optional[PPOAgent], str]:
    expected_dim = state_dim(shards)

    if not model_path.exists():
        return None, "no_model"

    agent = PPOAgent(
        state_dim=expected_dim,
        action_dim=shards,
        hidden_dim=HIDDEN_DIM,
        device="cpu",
    )

    try:
        payload = agent.load(model_path)
    except Exception as e:
        return None, f"load_failed_{type(e).__name__}"

    ckpt_state_dim = int(payload.get("state_dim", -1))
    ckpt_action_dim = int(payload.get("action_dim", -1))

    if ckpt_state_dim != expected_dim or ckpt_action_dim != shards:
        return None, "model_dim_mismatch"

    agent.net.eval()
    return agent, "python_ppo"


def infer_batch(
    items: List[Dict[str, Any]],
    shards: int,
    model_path: Path,
    sample: bool = False,
) -> List[Dict[str, Any]]:
    expected_dim = state_dim(shards)
    agent, model_status = load_agent(shards, model_path)

    if agent is None:
        return [safe_heuristic(item, shards, model_status) for item in items]

    outputs: List[Optional[Dict[str, Any]]] = [None for _ in items]
    valid_indices: List[int] = []
    valid_states: List[List[float]] = []

    for idx, item in enumerate(items):
        state = item.get("state", [])
        if len(state) != expected_dim:
            outputs[idx] = safe_heuristic(item, shards, "dim_mismatch")
            continue

        valid_indices.append(idx)
        valid_states.append(state)

    if valid_states:
        with torch.no_grad():
            states_t = torch.tensor(
                valid_states,
                dtype=torch.float32,
                device=agent.device,
            )

            logits, values = agent.net(states_t)
            dist = Categorical(logits=logits)
            probs = torch.softmax(logits, dim=-1)

            if sample:
                # 训练模式：按概率采样动作，保留探索。
                actions = dist.sample()
            else:
                # 评估模式：仍然取概率最大的 shard，保持你之前实验逻辑不变。
                actions = torch.argmax(probs, dim=-1)

            log_probs = dist.log_prob(actions)
            confidences = probs.gather(1, actions.unsqueeze(1)).squeeze(1)

            for local_idx, item_idx in enumerate(valid_indices):
                item = items[item_idx]

                shard = int(actions[local_idx].item())
                confidence = float(confidences[local_idx].item())
                log_prob = float(log_probs[local_idx].item())
                value = float(values[local_idx].item())

                if shard < 0 or shard >= shards:
                    shard = 0
                    source = "safe_default"
                    confidence = 0.0
                    log_prob = 0.0
                    value = 0.0
                else:
                    source = "python_ppo"

                outputs[item_idx] = {
                    "address": item.get("address", ""),
                    "related": item.get("related", ""),
                    "shard": shard,
                    "source": source,
                    "confidence": confidence,
                    "log_prob": log_prob,
                    "value": value,
                }

    final_outputs: List[Dict[str, Any]] = []

    for idx, item in enumerate(items):
        if outputs[idx] is None:
            outputs[idx] = safe_heuristic(item, shards, "unknown")
        final_outputs.append(outputs[idx])

    return final_outputs


def write_output(path: str, outputs: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"items": outputs}, f, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=str(MODEL_PATH))

    # 现在 Go 侧先不传这个参数，所以默认仍然是确定性推理。
    # 后面做在线训练时，再让 Go 调用时加 --sample。
    parser.add_argument("--sample", action="store_true")

    args = parser.parse_args()

    items, shards = load_input(args.state)
    outputs = infer_batch(
        items=items,
        shards=shards,
        model_path=Path(args.model),
        sample=bool(args.sample),
    )
    write_output(args.out, outputs)


if __name__ == "__main__":
    main()