import argparse
import json
from pathlib import Path

import torch

from config import HIDDEN_DIM, MODEL_PATH, state_dim
from heuristic import heuristic_from_state
from ppo import PPOAgent


def load_input(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    state = data.get("state", [])
    shards = int(data.get("shards", 4))

    if not isinstance(state, list):
        raise ValueError("state must be a list")

    return data, [float(x) for x in state], shards


def infer_with_model(state, shards: int, model_path: Path):
    expected_dim = state_dim(shards)

    if len(state) != expected_dim:
        return None, 0.0, "dim_mismatch"

    if not model_path.exists():
        return None, 0.0, "no_model"

    agent = PPOAgent(
        state_dim=expected_dim,
        action_dim=shards,
        hidden_dim=HIDDEN_DIM,
        device="cpu",
    )

    payload = agent.load(model_path)

    ckpt_state_dim = int(payload.get("state_dim", -1))
    ckpt_action_dim = int(payload.get("action_dim", -1))

    if ckpt_state_dim != expected_dim or ckpt_action_dim != shards:
        return None, 0.0, "model_dim_mismatch"

    action, confidence = agent.deterministic_action(state)
    return int(action), float(confidence), "python_ppo"


def write_output(path: str, shard: int, source: str, confidence: float):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "shard": int(shard),
                "source": source,
                "confidence": float(confidence),
            },
            f,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=str(MODEL_PATH))
    args = parser.parse_args()

    _, state, shards = load_input(args.state)
    model_path = Path(args.model)

    shard, confidence, source = infer_with_model(state, shards, model_path)

    if shard is None:
        shard = heuristic_from_state(state, shards)
        confidence = 0.0
        source = f"python_heuristic_{source}"

    if shard < 0 or shard >= shards:
        shard = 0
        confidence = 0.0
        source = "safe_default"

    write_output(args.out, shard, source, confidence)


if __name__ == "__main__":
    main()