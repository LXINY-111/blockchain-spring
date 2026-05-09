from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_CSV_PATH = ROOT_DIR / "selectedTxs_300K.csv"
CHECKPOINT_DIR = ROOT_DIR / "spring_lite" / "checkpoints"
MODEL_PATH = CHECKPOINT_DIR / "spring_ppo.pt"

DEFAULT_SHARD_NUM = 4

# SPRING 论文状态维度：11k + 1
# 5k 个最近总交易数 + 5k 个最近跨片交易数 + k 个 sender_pos + 1 个地址类型标记
def state_dim(shards: int) -> int:
    return 11 * shards + 1


HIDDEN_DIM = 64
LEARNING_RATE = 3e-4
GAMMA = 0.99
CLIP_EPS = 0.2
PPO_EPOCHS = 4
BATCH_SIZE = 2048

# 对齐 SPRING 表 1 的默认设置：lambda=0.5, beta=0.1
LAMBDA_WEIGHT = 0.5
BETA = 0.1

EPS = 1e-8