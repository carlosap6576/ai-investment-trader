"""
Centralized constants for the AI Investment Trader.

All default values, valid options, and configuration constants live here.
This is the SINGLE SOURCE OF TRUTH for all strategies.
"""

# =============================================================================
# PATHS
# =============================================================================

DATASETS_DIR = "datasets"

# =============================================================================
# DATA DOWNLOAD DEFAULTS
# =============================================================================

DEFAULT_PERIOD = "1mo"
DEFAULT_INTERVAL = "5m"
DEFAULT_NEWS_COUNT = 100
DEFAULT_NO_FILTER = False
DEFAULT_NO_SENTIMENT = False

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

# Continuous learning
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
DEFAULT_NUM_HEADS = 4
DEFAULT_DROPOUT = 0.1
NUM_CLASSES = 3  # SELL, HOLD, BUY

# =============================================================================
# TESTING DEFAULTS
# =============================================================================

DEFAULT_NUM_SAMPLES = 3
DEFAULT_SUMMARY_MODEL = "google/flan-t5-xl"

# =============================================================================
# VALID OPTIONS
# =============================================================================

VALID_OPTIMIZERS = ["SGD", "AdamW"]

VALID_PERIODS = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]

VALID_INTERVALS = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]

# Interval to seconds mapping
INTERVAL_SECONDS = {
    "1m": 60,
    "2m": 120,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "60m": 3600,
    "90m": 5400,
    "1h": 3600,
    "1d": 86400,
    "5d": 432000,
    "1wk": 604800,
    "1mo": 2592000,
    "3mo": 7776000,
}

# Period to approximate days mapping
PERIOD_TO_DAYS = {
    "1d": 1,
    "5d": 5,
    "1mo": 30,
    "3mo": 90,
    "6mo": 180,
    "1y": 365,
    "2y": 730,
    "5y": 1825,
    "10y": 3650,
    "ytd": 365,
    "max": 99999,
}

# Intraday data limits
INTERVAL_MAX_DAYS = {
    "1m": 7,
    "2m": 60,
    "5m": 60,
    "15m": 60,
    "30m": 60,
    "60m": 730,
    "90m": 730,
    "1h": 730,
    "1d": 99999,
    "5d": 99999,
    "1wk": 99999,
    "1mo": 99999,
    "3mo": 99999,
}

# =============================================================================
# LABEL NAMES
# =============================================================================

LABEL_NAMES = ["SELL", "HOLD", "BUY"]

# =============================================================================
# HIERARCHICAL FEATURE DIMENSIONS
# =============================================================================

TICKER_DIM = 12
SECTOR_DIM = 10
MARKET_DIM = 10
CROSS_DIM = 8
TOTAL_FEATURES = TICKER_DIM + SECTOR_DIM + MARKET_DIM + CROSS_DIM  # 40
