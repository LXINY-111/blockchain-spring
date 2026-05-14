import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from config import (
    CHECKPOINT_DIR,
    CLIP_EPS,
    GAMMA,
    HIDDEN_DIM,
    LEARNING_RATE,
    MODEL_PATH,
    PPO_EPOCHS,
    state_dim,
)
from ppo import PPOAgent, RolloutBuffer


def now_ns() -> int:
    return time.time_ns()


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def load_update_input(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("online update input must be a JSON object")

    return data


def build_agent(shards: int, model_path: Path) -> Tuple[PPOAgent, Dict[str, Any], str]:
    expected_state_dim = state_dim(shards)

    agent = PPOAgent(
        state_dim=expected_state_dim,
        action_dim=shards,
        hidden_dim=HIDDEN_DIM,
        lr=LEARNING_RATE,
        gamma=GAMMA,
        clip_eps=CLIP_EPS,
        ppo_epochs=PPO_EPOCHS,
        device="cpu",
    )

    old_payload: Dict[str, Any] = {}
    source = "new_model"

    if model_path.exists():
        try:
            payload = agent.load(model_path)
            ckpt_state_dim = int(payload.get("state_dim", -1))
            ckpt_action_dim = int(payload.get("action_dim", -1))

            if ckpt_state_dim == expected_state_dim and ckpt_action_dim == shards:
                old_payload = payload
                source = "loaded_model"
            else:
                old_payload = {}
                source = "reset_dim_mismatch"
        except Exception as e:
            old_payload = {}
            source = f"reset_load_failed_{type(e).__name__}"

    return agent, old_payload, source


def build_buffer(data: Dict[str, Any]) -> Tuple[RolloutBuffer, Dict[str, Any]]:
    shards = safe_int(data.get("shards", 4), 4)
    expected_state_dim = state_dim(shards)

    reward = safe_float(data.get("reward", 0.0), 0.0)
    done = bool(data.get("done", True))

    actions = data.get("actions", [])
    if not isinstance(actions, list):
        raise ValueError("actions must be a list")

    buffer = RolloutBuffer()

    skipped = 0
    action_hist = [0 for _ in range(shards)]

    for item in actions:
        if not isinstance(item, dict):
            skipped += 1
            continue

        state = item.get("state", [])
        if not isinstance(state, list) or len(state) != expected_state_dim:
            skipped += 1
            continue

        action = safe_int(item.get("action", -1), -1)
        if action < 0 or action >= shards:
            skipped += 1
            continue

        log_prob = safe_float(item.get("log_prob", None), None)
        value = safe_float(item.get("value", None), None)

        # log_prob 和 value 是 PPO 更新必须依赖的旧策略信息。
        # 如果没有这两个值，就跳过，避免污染训练。
        if log_prob is None or value is None:
            skipped += 1
            continue

        clean_state = [safe_float(x, 0.0) for x in state]
        item_reward = safe_float(item.get("reward", reward), reward)
        item_done = bool(item.get("done", done))

        buffer.add(
            state=clean_state,
            action=action,
            log_prob=log_prob,
            reward=item_reward,
            done=item_done,
            value=value,
        )

        action_hist[action] += 1

    meta = {
        "shards": shards,
        "reward": reward,
        "done": done,
        "skipped": skipped,
        "action_hist": action_hist,
    }

    return buffer, meta


def append_train_log(log_path: Path, record: Dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_update(input_path: Path, model_path: Path, log_path: Path) -> Dict[str, Any]:
    data = load_update_input(input_path)

    batch_id = safe_int(data.get("batch_id", 0), 0)
    feedback_epoch = safe_int(data.get("feedback_epoch", -1), -1)
    shards = safe_int(data.get("shards", 4), 4)

    buffer, meta = build_buffer(data)

    result: Dict[str, Any] = {
        "ok": False,
        "time_unix_nano": now_ns(),
        "batch_id": batch_id,
        "feedback_epoch": feedback_epoch,
        "shards": shards,
        "num_actions": len(buffer),
        "skipped": meta["skipped"],
        "reward": meta["reward"],
        "done": meta["done"],
        "action_hist": meta["action_hist"],
        "cross_rate": safe_float(data.get("cross_rate", 0.0), 0.0),
        "normalized_load_variance": safe_float(
            data.get("normalized_load_variance", 0.0),
            0.0,
        ),
        "total_tx": safe_int(data.get("total_tx", 0), 0),
        "total_inner": safe_int(data.get("total_inner", 0), 0),
        "total_relay1": safe_int(data.get("total_relay1", 0), 0),
        "total_relay2": safe_int(data.get("total_relay2", 0), 0),
        "input_path": str(input_path),
        "model_path": str(model_path),
        "log_path": str(log_path),
        "loss_info": {},
        "model_source": "",
        "message": "",
    }

    # PPO 的优势归一化至少需要 2 条样本更稳。
    # 你的正常 batch 是 100+，这里只是防御异常输入。
    if len(buffer) < 2:
        result["message"] = "skip_update_too_few_samples"
        append_train_log(log_path, result)
        return result

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    agent, old_payload, model_source = build_agent(shards, model_path)
    result["model_source"] = model_source

    loss_info = agent.update(buffer)

    old_extra = old_payload.get("extra", {}) if isinstance(old_payload, dict) else {}
    if not isinstance(old_extra, dict):
        old_extra = {}

    old_update_count = safe_int(old_extra.get("online_update_count", 0), 0)
    new_update_count = old_update_count + 1

    extra = {
        "online_update_count": new_update_count,
        "last_batch_id": batch_id,
        "last_feedback_epoch": feedback_epoch,
        "last_reward": meta["reward"],
        "last_cross_rate": result["cross_rate"],
        "last_normalized_load_variance": result["normalized_load_variance"],
        "last_num_actions": len(buffer),
        "last_action_hist": meta["action_hist"],
        "last_update_time_unix_nano": result["time_unix_nano"],
    }

    agent.save(model_path, extra=extra)

    result["ok"] = True
    result["loss_info"] = loss_info
    result["online_update_count"] = new_update_count
    result["message"] = "updated"

    append_train_log(log_path, result)

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument(
        "--log",
        default=str(Path("spring_io") / "online_train_log.jsonl"),
    )

    args = parser.parse_args()

    try:
        result = run_update(
            input_path=Path(args.input),
            model_path=Path(args.model),
            log_path=Path(args.log),
        )
    except Exception as e:
        result = {
            "ok": False,
            "time_unix_nano": now_ns(),
            "message": f"exception_{type(e).__name__}: {e}",
            "input_path": args.input,
            "model_path": args.model,
            "log_path": args.log,
            "loss_info": {},
        }

        try:
            append_train_log(Path(args.log), result)
        except Exception:
            pass

    # 只向 stdout 输出 JSON，方便 Go 侧解析。
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
