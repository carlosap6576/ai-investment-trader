"""Configuration for model training."""

from dataclasses import dataclass
from strategies.finbert_sentiment.constants import DATASETS_DIR

# =============================================================================
# TRAINING DEFAULTS
# =============================================================================

DEFAULT_BUY_THRESHOLD = 1.0
DEFAULT_SELL_THRESHOLD = -1.0
DEFAULT_EPOCHS = 20
DEFAULT_LEARNING_RATE = 0.005
DEFAULT_BATCH_SIZE = 1
DEFAULT_TRAIN_SPLIT = 0.8
DEFAULT_OPTIMIZER = "SGD"
DEFAULT_FRESH_START = False

# Smart Training Guard
DEFAULT_MIN_NEW_SAMPLES = 5
DEFAULT_COOLDOWN_MINUTES = 5

# =============================================================================
# MODEL ARCHITECTURE DEFAULTS
# =============================================================================

DEFAULT_HIDDEN_DIM = 256
DEFAULT_NUM_LAYERS = 2
DEFAULT_SEQ_LENGTH = 20

# =============================================================================
# VALID OPTIONS
# =============================================================================

VALID_OPTIMIZERS = ["SGD", "AdamW"]


# =============================================================================
# CONFIGURATION CLASS
# =============================================================================

@dataclass
class TrainConfig:
    """Configuration for model training."""
    symbol: str
    buy_threshold: float = DEFAULT_BUY_THRESHOLD
    sell_threshold: float = DEFAULT_SELL_THRESHOLD
    epochs: int = DEFAULT_EPOCHS
    learning_rate: float = DEFAULT_LEARNING_RATE
    batch_size: int = DEFAULT_BATCH_SIZE
    split: float = DEFAULT_TRAIN_SPLIT
    optimizer: str = DEFAULT_OPTIMIZER
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    num_layers: int = DEFAULT_NUM_LAYERS
    seq_length: int = DEFAULT_SEQ_LENGTH
    fresh: bool = DEFAULT_FRESH_START
    force: bool = False
    min_new_samples: int = DEFAULT_MIN_NEW_SAMPLES
    cooldown: int = DEFAULT_COOLDOWN_MINUTES

    def __post_init__(self):
        self.symbol = self.symbol.upper()
        self.symbol_dir = f"{DATASETS_DIR}/{self.symbol}"
        self.data_file = f"{self.symbol_dir}/news_with_price.json"
        self.model_file = f"{self.symbol_dir}/{self.symbol}.pth"
        self.metadata_file = f"{self.symbol_dir}/{self.symbol}_metadata.json"
        self.training_meta_file = f"{self.symbol_dir}/.training_meta.json"
