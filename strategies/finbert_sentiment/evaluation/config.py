"""Configuration for model evaluation."""

from dataclasses import dataclass
from strategies.finbert_sentiment.constants import DATASETS_DIR

# Import shared defaults from training (ensures test matches train)
from strategies.finbert_sentiment.training.config import (
    DEFAULT_BUY_THRESHOLD,
    DEFAULT_SELL_THRESHOLD,
    DEFAULT_TRAIN_SPLIT,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_NUM_LAYERS,
    DEFAULT_SEQ_LENGTH,
)

# =============================================================================
# EVALUATION DEFAULTS
# =============================================================================

DEFAULT_NUM_SAMPLES = 3
DEFAULT_SUMMARY_MODEL = "google/flan-t5-xl"


# =============================================================================
# CONFIGURATION CLASS
# =============================================================================

@dataclass
class TestConfig:
    """Configuration for model testing."""
    symbol: str
    buy_threshold: float = DEFAULT_BUY_THRESHOLD
    sell_threshold: float = DEFAULT_SELL_THRESHOLD
    split: float = DEFAULT_TRAIN_SPLIT
    samples: int = DEFAULT_NUM_SAMPLES
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    num_layers: int = DEFAULT_NUM_LAYERS
    seq_length: int = DEFAULT_SEQ_LENGTH
    summary: bool = False
    summary_model: str = DEFAULT_SUMMARY_MODEL

    def __post_init__(self):
        self.symbol = self.symbol.upper()
        self.symbol_dir = f"{DATASETS_DIR}/{self.symbol}"
        self.data_file = f"{self.symbol_dir}/news_with_price.json"
        self.model_file = f"{self.symbol_dir}/{self.symbol}.pth"
        self.metadata_file = f"{self.symbol_dir}/{self.symbol}_metadata.json"
