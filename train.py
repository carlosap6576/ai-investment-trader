"""
Train the trading signal classifier.
Loads prepared data, trains the model, and saves weights.
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
import numpy as np

# =============================================================================
# DEFAULTS - Used when CLI arguments are not provided
# =============================================================================

DEFAULT_BUY_THRESHOLD = 1.0      # +1% price increase triggers BUY
DEFAULT_SELL_THRESHOLD = -1.0    # -1% price decrease triggers SELL
DEFAULT_EPOCHS = 20              # Training iterations
DEFAULT_LEARNING_RATE = 0.005    # Optimizer step size
DEFAULT_BATCH_SIZE = 1           # Samples per gradient update
DEFAULT_TRAIN_SPLIT = 0.8        # 80% train, 20% test
DEFAULT_OPTIMIZER = "SGD"        # SGD or AdamW

# Model architecture (affects model size and capacity)
DEFAULT_HIDDEN_DIM = 256         # Internal dimension (256=small ~2.6MB, 512=large ~10MB)
DEFAULT_NUM_LAYERS = 2           # Transformer layers (2=fast, 4=thorough)

# Continuous learning (default: continue from existing model if available)
DEFAULT_FRESH_START = False      # False = continue learning, True = start from scratch

# Smart Training Guard (prevents overfitting on same data)
# NOTE: Defaults tuned for 5-minute trading intervals (PRICE_INTERVAL = "5m")
DEFAULT_MIN_NEW_SAMPLES = 5      # Require at least 5 new samples to train
DEFAULT_COOLDOWN_MINUTES = 5     # Don't train again within 5 minutes (matches data interval)

DATASETS_DIR = "datasets"
VALID_OPTIMIZERS = ["SGD", "AdamW"]

# Label encodings (one-hot)
LABEL_SELL = [1., 0., 0.]
LABEL_HOLD = [0., 1., 0.]
LABEL_BUY = [0., 0., 1.]

# =============================================================================
# HELP TEXT AND CLI INTERFACE
# =============================================================================

HELP_TEXT = """
================================================================================
                      TRADING SIGNAL CLASSIFIER
                           Model Training
================================================================================

DESCRIPTION:
    Trains the trading signal classifier on prepared market data.
    Loads news + price data, trains a transformer model, and saves weights.

    Before training, you must download data using:
        python download.py -s SYMBOL

--------------------------------------------------------------------------------
USAGE:
--------------------------------------------------------------------------------

    python train.py -s SYMBOL [OPTIONS]
    python train.py --symbol SYMBOL [OPTIONS]

--------------------------------------------------------------------------------
REQUIRED ARGUMENT:
--------------------------------------------------------------------------------

    -s, --symbol SYMBOL
        Trading symbol to train on. Must have data in datasets/SYMBOL/

        Examples: BTC-USD, AAPL, TSLA, ETH-USD

--------------------------------------------------------------------------------
SIGNAL THRESHOLDS:
--------------------------------------------------------------------------------

    -b, --buy-threshold PERCENT
        Price increase (%) to trigger a BUY signal.
        Default: 1.0

        Example: -b 0.5 means +0.5% price increase = BUY

    --sell-threshold PERCENT
        Price decrease (%) to trigger a SELL signal.
        Default: -1.0

        Example: --sell-threshold -0.5 means -0.5% price decrease = SELL

    How labels are assigned:
        - Price change > buy_threshold  → BUY
        - Price change < sell_threshold → SELL
        - Otherwise                     → HOLD

    Threshold Guidelines:
        +------------------+---------------------------+
        | Asset Type       | Recommended Thresholds    |
        +------------------+---------------------------+
        | Crypto (BTC)     | -b 1.0 --sell-threshold -1.0   |
        | Volatile (TSLA)  | -b 0.5 --sell-threshold -0.5   |
        | Large Cap (AAPL) | -b 0.3 --sell-threshold -0.3   |
        | Index (SPY)      | -b 0.2 --sell-threshold -0.2   |
        +------------------+---------------------------+

--------------------------------------------------------------------------------
TRAINING PARAMETERS:
--------------------------------------------------------------------------------

    -e, --epochs COUNT
        Number of training epochs (full passes through data).
        Default: 20

    -l, --learning-rate RATE
        Learning rate for optimizer.
        Default: 0.005

    -o, --optimizer TYPE
        Optimizer algorithm: SGD or AdamW
        Default: SGD

        SGD (Stochastic Gradient Descent):
          - "Steady Walker" - same step size every time
          - Simple, fast, less memory
          - Good for: prototyping, general use

        AdamW (Adam with Weight Decay):
          - "Smart Runner" - adapts step size automatically
          - Big steps when far, tiny steps when close to goal
          - Good for: precision-critical tasks (use with -l 0.001)

    --batch-size SIZE
        Number of samples per gradient update.
        Default: 1

    --split RATIO
        Train/test split ratio (0.0 to 1.0).
        Divides data: training portion (to learn) vs test portion (to validate).
        Default: 0.8 (80% train, 20% test)

        NOTE: This is NOT an accuracy score! It's how you divide your data.
        The 20% test set is hidden during training for honest evaluation.

        +-------------+---------+----------+----------------------------+
        | Split       | Train % | Test %   | Use Case                   |
        +-------------+---------+----------+----------------------------+
        | 0.9         | 90%     | 10%      | Large datasets (10k+)      |
        | 0.8         | 80%     | 20%      | Default - good balance     |
        | 0.7         | 70%     | 30%      | Need more validation       |
        +-------------+---------+----------+----------------------------+

--------------------------------------------------------------------------------
MODEL ARCHITECTURE (affects model size and capacity):
--------------------------------------------------------------------------------

    --hidden-dim SIZE
        Size of the model's internal "notebook" for learning patterns.
        Larger = more detailed pattern recognition, but bigger model file.
        Default: 256

        +------------+-------------+------------------------------------------+
        | Value      | Model Size  | Use Case                                 |
        +------------+-------------+------------------------------------------+
        | 256        | ~2.6 MB     | Fast training, general use (default)     |
        | 512        | ~10 MB      | Precision-critical (financial, medical)  |
        +------------+-------------+------------------------------------------+

    --num-layers COUNT
        Number of "expert friends" that review the prediction.
        More layers = more thorough checking, but slower training.
        Default: 2

        +------------+------------------------------------------+
        | Value      | Use Case                                 |
        +------------+------------------------------------------+
        | 2          | Fast training, general use (default)     |
        | 4          | Precision-critical (catches subtle patterns) |
        +------------+------------------------------------------+

    NOTE: These parameters affect MODEL SIZE, not training time estimates:
        - epochs, batch_size, learning_rate → affect training behavior
        - hidden_dim, num_layers → affect model capacity and file size

--------------------------------------------------------------------------------
CONTINUOUS LEARNING (The Student's Notebook):
--------------------------------------------------------------------------------

    By default, training CONTINUES from existing knowledge if a model exists.
    Think of it like a student who already has notes from previous classes:

        CONTINUE (default):
        ┌─────────────────────────────────────────────────────────────────┐
        │  Day 1: Student learns patterns A, B, C → saves notes           │
        │  Day 2: Student reads old notes, then learns D, E, F            │
        │  Day 3: Student reads notes (A-F), then learns G, H, I          │
        │                                                                 │
        │  Result: Student knows A through I! Knowledge accumulates.      │
        └─────────────────────────────────────────────────────────────────┘

        FRESH START (--fresh):
        ┌─────────────────────────────────────────────────────────────────┐
        │  Day 1: Student learns A, B, C → saves notes                    │
        │  Day 2: Student THROWS AWAY notes, starts over, learns D, E, F  │
        │  Day 3: Student THROWS AWAY notes, starts over, learns G, H, I  │
        │                                                                 │
        │  Result: Student only knows G, H, I. Previous learning lost!    │
        └─────────────────────────────────────────────────────────────────┘

    --fresh
        Start training from scratch (ignore existing model).
        Default: False (continue from existing model)

        Use --fresh when:
        - You want to completely retrain with new architecture
        - The existing model is corrupted or from wrong asset
        - You're experimenting with different hyperparameters

        Example:
          python train.py -s BTC-USD --fresh

    NOTE: Continuous learning is SAFE by default! Your trained model won't
    be accidentally overwritten. If you truly want a fresh start, either:
    - Use --fresh flag explicitly
    - Delete the .pth file manually

--------------------------------------------------------------------------------
SMART TRAINING GUARD (Daemon-Safe Automation):
--------------------------------------------------------------------------------

    The Smart Training Guard prevents overfitting when running automated/daemon
    processes. It detects if training is actually needed before proceeding.

    Think of it like a teacher who checks if there's new material to teach:

        ┌─────────────────────────────────────────────────────────────────┐
        │  Teacher arrives at classroom:                                  │
        │                                                                 │
        │  "Is there new material since last class?"                      │
        │    → YES: "Let's learn the new content!"                        │
        │    → NO:  "Nothing new? Class dismissed, come back tomorrow."   │
        │                                                                 │
        │  This prevents students from re-reading the same chapter        │
        │  100 times (overfitting).                                       │
        └─────────────────────────────────────────────────────────────────┘

    GUARD CHECKS (all must pass to train):
        1. Data Hash:       Has the training data content changed? (PRIMARY)
        2. New Samples:     Are there enough new samples? (default: 5+)
        3. Cooldown:        Has enough time passed? (default: 5 minutes)

    NOTE: Defaults are tuned for 5-minute trading intervals (PRICE_INTERVAL="5m").
    The Data Hash check is the PRIMARY guard - if data hasn't changed, training
    is always skipped regardless of other settings.

    --force
        Bypass ALL Smart Guard checks and train anyway.
        Use when you REALLY want to train regardless of checks.

        Example:
          python train.py -s BTC-USD --force

    --min-new-samples COUNT
        Minimum number of new samples required before training.
        Default: 5 (tuned for 5-minute data intervals)

        Higher values = more conservative (train less often)
        Lower values = more aggressive (train more often)

        Example:
          python train.py -s BTC-USD --min-new-samples 10

    --cooldown MINUTES
        Minimum time (in minutes) between training sessions.
        Default: 5 (matches PRICE_INTERVAL="5m" in download.py)

        Should generally match your data interval:
          • 5-minute data  → --cooldown 5
          • 15-minute data → --cooldown 15
          • 1-hour data    → --cooldown 60

        Example:
          python train.py -s BTC-USD --cooldown 15

    DAEMON USAGE:
        The Smart Guard makes train.py safe for cron jobs and daemons:

        # crontab entry - runs every hour, trains only when needed
        0 * * * * cd /path/to/project && python download.py -s BTC-USD && python train.py -s BTC-USD

        The script will:
        - Train if new data is available and cooldown passed
        - Skip gracefully if no changes detected
        - Exit with code 0 either way (daemon-friendly)

--------------------------------------------------------------------------------
OUTPUT:
--------------------------------------------------------------------------------

    --model-file PATH
        Output file for trained model weights.
        Default: datasets/{SYMBOL}/{SYMBOL}.pth

        The model is saved in the same folder as its training data.
        This keeps each symbol's data and model together:

            datasets/
            ├── BTC-USD/
            │   ├── news_with_price.json  (training data)
            │   └── BTC-USD.pth           (trained model)
            └── AAPL/
                ├── news_with_price.json
                └── AAPL.pth

    -h, --help
        Show this help message and exit.

--------------------------------------------------------------------------------
EXAMPLES:
--------------------------------------------------------------------------------

    # Basic training with defaults (continues from existing model if available)
    python train.py -s BTC-USD

    # Train with custom thresholds for less volatile stock
    python train.py -s AAPL -b 0.3 --sell-threshold -0.3

    # Train with more epochs and AdamW optimizer
    python train.py -s TSLA -e 50 -o AdamW

    # PRECISION MODE: For critical financial/medical decisions
    # (bigger model, more thorough, slower but more accurate)
    python train.py -s BTC-USD -e 200 --batch-size 1 -l 0.001 -o AdamW \
        --hidden-dim 512 --num-layers 4

    # Full customization
    python train.py -s ETH-USD -b 1.5 --sell-threshold -1.5 -e 30 -l 0.001 -o AdamW

    # Custom model output file (override default location)
    python train.py -s BTC-USD --model-file custom/my_btc_model.pth

    # Custom train/test split (more validation with 70/30 split)
    python train.py -s BTC-USD --split 0.7

    # CONTINUOUS LEARNING: Download new data, then continue training
    # (model gets smarter with each training session!)
    python download.py -s BTC-USD     # Get fresh news/prices
    python train.py -s BTC-USD        # Continue from existing knowledge

    # FRESH START: Completely retrain from scratch (ignores existing model)
    python train.py -s BTC-USD --fresh

--------------------------------------------------------------------------------
WORKFLOW:
--------------------------------------------------------------------------------

    1. Download data:    python download.py -s BTC-USD
    2. Train model:      python train.py -s BTC-USD
    3. Test model:       python test.py -s BTC-USD

================================================================================
"""


def print_help():
    """Print the full help text."""
    print(HELP_TEXT)


class FriendlyArgumentParser(argparse.ArgumentParser):
    """Custom ArgumentParser that shows friendly help on errors."""

    def error(self, message):
        """Override error to show helpful information."""
        sys.stderr.write(f"\n{'='*70}\n")
        sys.stderr.write(f"ERROR: {message}\n")
        sys.stderr.write(f"{'='*70}\n\n")

        sys.stderr.write("QUICK USAGE:\n")
        sys.stderr.write("  python train.py -s SYMBOL [OPTIONS]\n\n")

        sys.stderr.write("EXAMPLES:\n")
        sys.stderr.write("  python train.py -s BTC-USD\n")
        sys.stderr.write("  python train.py -s AAPL -b 0.3 --sell-threshold -0.3\n")
        sys.stderr.write("  python train.py -s TSLA -e 50 -o AdamW\n\n")

        sys.stderr.write("VALID OPTIMIZERS:\n")
        sys.stderr.write(f"  {', '.join(VALID_OPTIMIZERS)}\n\n")

        sys.stderr.write("For full help, run: python train.py --help\n\n")
        sys.exit(2)


def parse_args():
    """Parse command-line arguments."""
    # Check for help flag first
    if '-h' in sys.argv or '--help' in sys.argv or len(sys.argv) == 1:
        print_help()
        sys.exit(0)

    parser = FriendlyArgumentParser(add_help=False)

    # Required
    parser.add_argument("-s", "--symbol", required=True)

    # Signal thresholds
    parser.add_argument("-b", "--buy-threshold", type=float, default=DEFAULT_BUY_THRESHOLD)
    parser.add_argument("--sell-threshold", type=float, default=DEFAULT_SELL_THRESHOLD)

    # Training parameters
    parser.add_argument("-e", "--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("-l", "--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("-o", "--optimizer", default=DEFAULT_OPTIMIZER, choices=VALID_OPTIMIZERS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--split", type=float, default=DEFAULT_TRAIN_SPLIT)

    # Model architecture (affects model size)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--num-layers", type=int, default=DEFAULT_NUM_LAYERS)

    # Continuous learning (default: continue from existing model)
    parser.add_argument("--fresh", action="store_true", default=DEFAULT_FRESH_START,
                        help="Start training from scratch (ignore existing model)")

    # Smart Training Guard (daemon-safe automation)
    parser.add_argument("--force", action="store_true", default=False,
                        help="Bypass Smart Guard checks and train anyway")
    parser.add_argument("--min-new-samples", type=int, default=DEFAULT_MIN_NEW_SAMPLES,
                        help=f"Minimum new samples required to train (default: {DEFAULT_MIN_NEW_SAMPLES})")
    parser.add_argument("--cooldown", type=int, default=DEFAULT_COOLDOWN_MINUTES,
                        help=f"Minimum minutes between training sessions (default: {DEFAULT_COOLDOWN_MINUTES})")

    # Output (None = use default: datasets/{SYMBOL}/{SYMBOL}.pth)
    parser.add_argument("--model-file", default=None)
    parser.add_argument("-h", "--help", action="store_true")

    args = parser.parse_args()

    # Validate thresholds
    if args.buy_threshold <= 0:
        parser.error(f"Buy threshold must be positive (got {args.buy_threshold})")
    if args.sell_threshold >= 0:
        parser.error(f"Sell threshold must be negative (got {args.sell_threshold})")
    if args.split <= 0 or args.split >= 1:
        parser.error(f"Split must be between 0 and 1 (got {args.split})")
    if args.epochs < 1:
        parser.error(f"Epochs must be at least 1 (got {args.epochs})")
    if args.learning_rate <= 0:
        parser.error(f"Learning rate must be positive (got {args.learning_rate})")
    if args.batch_size < 1:
        parser.error(f"Batch size must be at least 1 (got {args.batch_size})")
    if args.hidden_dim < 64:
        parser.error(f"Hidden dimension must be at least 64 (got {args.hidden_dim})")
    if args.num_layers < 1:
        parser.error(f"Number of layers must be at least 1 (got {args.num_layers})")

    return args


class Config:
    """Configuration container for training parameters."""

    def __init__(self, symbol, buy_threshold, sell_threshold, epochs,
                 learning_rate, optimizer, batch_size, split, model_file,
                 hidden_dim, num_layers, fresh_start, force, min_new_samples, cooldown):
        self.symbol = symbol.upper()
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.split = split
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.fresh_start = fresh_start

        # Smart Training Guard settings
        self.force = force
        self.min_new_samples = min_new_samples
        self.cooldown = cooldown

        # Derived paths
        self.symbol_dir = f"{DATASETS_DIR}/{self.symbol}"
        self.training_data_file = f"{self.symbol_dir}/news_with_price.json"
        self.metadata_file = f"{self.symbol_dir}/.training_meta.json"

        # Model file: use provided path or default to datasets/{SYMBOL}/{SYMBOL}.pth
        if model_file:
            self.model_file = model_file
        else:
            self.model_file = f"{self.symbol_dir}/{self.symbol}.pth"

    def __str__(self):
        # Determine training mode description
        if self.fresh_start:
            training_mode = "Fresh start (--fresh)"
        else:
            training_mode = "Continue from existing (default)"

        # Smart Guard status
        if self.force:
            guard_status = "BYPASSED (--force)"
        else:
            guard_status = f"Active (min {self.min_new_samples} new samples, {self.cooldown}min cooldown)"

        return (
            f"\n"
            f"  [Data]\n"
            f"  Symbol:           {self.symbol}\n"
            f"  Data file:        {self.training_data_file}\n"
            f"\n"
            f"  [Signal Thresholds]\n"
            f"  Buy threshold:    > {self.buy_threshold}% price increase\n"
            f"  Sell threshold:   < {self.sell_threshold}% price decrease\n"
            f"\n"
            f"  [Training Parameters]\n"
            f"  Epochs:           {self.epochs}\n"
            f"  Batch size:       {self.batch_size}\n"
            f"  Learning rate:    {self.learning_rate}\n"
            f"  Optimizer:        {self.optimizer}\n"
            f"  Train/Test split: {self.split*100:.0f}% / {(1-self.split)*100:.0f}%\n"
            f"  Training mode:    {training_mode}\n"
            f"\n"
            f"  [Smart Training Guard]\n"
            f"  Guard status:     {guard_status}\n"
            f"\n"
            f"  [Model Architecture]\n"
            f"  Hidden dimension: {self.hidden_dim}\n"
            f"  Transformer layers: {self.num_layers}\n"
            f"\n"
            f"  [Output]\n"
            f"  Model file:       {self.model_file}"
        )


# =============================================================================
# SMART TRAINING GUARD FUNCTIONS
# =============================================================================

def calculate_data_hash(config):
    """
    Calculate SHA256 hash of training data content.
    This fingerprints the data to detect changes.
    """
    if not os.path.exists(config.training_data_file):
        return None

    with open(config.training_data_file, 'r') as f:
        data = json.load(f)

    # Sort by a consistent key to ensure same data = same hash
    # regardless of order in the file
    sorted_data = sorted(data, key=lambda x: (x.get('pubDate', ''), x.get('title', '')))

    # Create hash of the sorted content
    content_str = json.dumps(sorted_data, sort_keys=True)
    return hashlib.sha256(content_str.encode()).hexdigest()


def load_training_metadata(config):
    """
    Load training metadata from .training_meta.json file.
    Returns None if file doesn't exist.
    """
    if not os.path.exists(config.metadata_file):
        return None

    try:
        with open(config.metadata_file, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def save_training_metadata(config, data_hash, sample_count, accuracy, f1):
    """
    Save training metadata after successful training.
    """
    metadata = load_training_metadata(config) or {"training_history": []}

    # Current training info
    current = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_hash": data_hash,
        "sample_count": sample_count,
        "epochs": config.epochs,
        "accuracy": accuracy,
        "f1_score": f1
    }

    # Update last_training
    metadata["last_training"] = current

    # Add to history (keep last 10 entries)
    metadata["training_history"].insert(0, {
        "timestamp": current["timestamp"],
        "samples": sample_count,
        "accuracy": accuracy
    })
    metadata["training_history"] = metadata["training_history"][:10]

    # Save
    with open(config.metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)


def should_train(config):
    """
    Smart Training Guard: Determine if training should proceed.

    Returns:
        tuple: (should_train: bool, reason: str, details: dict)
    """
    # If --force flag is set, always train
    if config.force:
        return True, "Force flag set (--force)", {}

    # Load previous training metadata
    metadata = load_training_metadata(config)

    # If no previous training, always train
    if metadata is None or "last_training" not in metadata:
        return True, "No previous training found (first training session)", {}

    last = metadata["last_training"]
    details = {}

    # CHECK 1: Data hash - has the data changed?
    current_hash = calculate_data_hash(config)
    last_hash = last.get("data_hash")
    details["data_hash_changed"] = current_hash != last_hash

    if current_hash == last_hash:
        return False, "Data unchanged since last training", {
            "last_trained": last.get("timestamp"),
            "sample_count": last.get("sample_count"),
            "data_hash": "unchanged"
        }

    # CHECK 2: Cooldown - has enough time passed?
    last_timestamp = last.get("timestamp")
    if last_timestamp:
        try:
            last_time = datetime.fromisoformat(last_timestamp.replace('Z', '+00:00'))
            now = datetime.now(timezone.utc)
            minutes_since = (now - last_time).total_seconds() / 60
            details["minutes_since_last"] = round(minutes_since, 1)

            if minutes_since < config.cooldown:
                return False, f"Cooldown not met ({minutes_since:.0f}min < {config.cooldown}min)", {
                    "last_trained": last_timestamp,
                    "minutes_since": round(minutes_since, 1),
                    "cooldown_required": config.cooldown
                }
        except (ValueError, TypeError):
            pass  # If timestamp parsing fails, skip cooldown check

    # CHECK 3: New samples - are there enough new samples?
    with open(config.training_data_file, 'r') as f:
        current_count = len(json.load(f))
    last_count = last.get("sample_count", 0)
    new_samples = current_count - last_count
    details["new_samples"] = new_samples

    if new_samples < config.min_new_samples:
        return False, f"Not enough new samples ({new_samples} < {config.min_new_samples})", {
            "current_samples": current_count,
            "last_samples": last_count,
            "new_samples": new_samples,
            "required": config.min_new_samples
        }

    # All checks passed!
    return True, "New data detected - training approved", {
        "new_samples": new_samples,
        "data_hash": "changed",
        "cooldown": "passed"
    }


def print_guard_decision(should, reason, details):
    """Print the Smart Guard decision in a user-friendly format."""
    print("\n" + "-" * 60)
    print("SMART TRAINING GUARD")
    print("-" * 60)

    if should:
        print(f"  ✓ TRAINING APPROVED")
        print(f"    Reason: {reason}")
        if details:
            for key, value in details.items():
                print(f"    {key}: {value}")
    else:
        print(f"  ⏭️  TRAINING SKIPPED")
        print(f"    Reason: {reason}")
        if details:
            for key, value in details.items():
                print(f"    {key}: {value}")
        print()
        print(f"  To force training anyway: python train.py -s SYMBOL --force")

    print("-" * 60)


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def check_data_file(config):
    """Check if training data exists and has content."""
    if not os.path.exists(config.training_data_file):
        print(f"\nERROR: Training data file not found: {config.training_data_file}")
        print(f"\nTo create it, run:")
        print(f"  python download.py -s {config.symbol}")
        sys.exit(1)

    with open(config.training_data_file, 'r') as f:
        data = json.load(f)

    if len(data) == 0:
        print(f"\nERROR: Training data file is empty: {config.training_data_file}")
        print(f"\nTo populate it, run:")
        print(f"  python download.py -s {config.symbol}")
        sys.exit(1)

    return data


def load_data(config):
    """Load and prepare training data with labels."""
    print(f"  Source:           {config.training_data_file}")

    data = check_data_file(config)

    features = []
    labels = []
    label_counts = {"sell": 0, "hold": 0, "buy": 0}

    for item in data:
        # Create text feature from price and news
        feature_text = "\n".join([
            f"Price: {item['price']}",
            f"Headline: {item['title']}",
            f"Summary: {item['summary']}"
        ])
        features.append(feature_text)

        # Generate label based on percentage change
        pct = item['percentage']
        if pct < config.sell_threshold:
            labels.append(LABEL_SELL)
            label_counts["sell"] += 1
        elif pct > config.buy_threshold:
            labels.append(LABEL_BUY)
            label_counts["buy"] += 1
        else:
            labels.append(LABEL_HOLD)
            label_counts["hold"] += 1

    print(f"  Total samples:    {len(features)}")
    print(f"  Label distribution:")
    print(f"    SELL:  {label_counts['sell']:>4}  ({label_counts['sell']/len(features)*100:>5.1f}%)")
    print(f"    HOLD:  {label_counts['hold']:>4}  ({label_counts['hold']/len(features)*100:>5.1f}%)")
    print(f"    BUY:   {label_counts['buy']:>4}  ({label_counts['buy']/len(features)*100:>5.1f}%)")

    # Warn if distribution is heavily skewed
    total = len(features)
    if total > 0:
        hold_pct = label_counts["hold"] / total * 100
        if hold_pct > 90:
            print(f"\n  ⚠️  WARNING: {hold_pct:.0f}% of samples are HOLD!")
            print(f"      Consider adjusting thresholds (currently >{config.buy_threshold}% / <{config.sell_threshold}%)")
            print(f"      Try: -b {config.buy_threshold/2} --sell-threshold {config.sell_threshold/2}")

    return features, labels


def split_data(config, features, labels):
    """Split data into training and test sets."""
    split_index = int(config.split * len(features))

    train_features = features[:split_index]
    train_labels = labels[:split_index]
    test_features = features[split_index:]
    test_labels = labels[split_index:]

    print(f"  Train/Test split:")
    print(f"    Train set:      {len(train_features):>4} samples ({config.split*100:.0f}%)")
    print(f"    Test set:       {len(test_features):>4} samples ({(1-config.split)*100:.0f}%)")

    return train_features, train_labels, test_features, test_labels


def create_model(config):
    """Initialize model and optimizer, optionally loading existing weights."""
    # Lazy import to allow --help without dependencies
    try:
        import torch
        from torch import nn
        from models.gemma_transformer_classifier import SimpleGemmaTransformerClassifier
    except ImportError as e:
        sys.stderr.write(f"\nERROR: Required dependencies are not installed.\n")
        sys.stderr.write(f"\nTo install them, run:\n")
        sys.stderr.write(f"  pip install -r requirements.txt\n\n")
        sys.exit(1)

    # Create model with custom architecture if specified
    model = SimpleGemmaTransformerClassifier(
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers
    )

    # Continuous learning: Load existing weights if available and not fresh start
    loaded_existing = False
    if not config.fresh_start and os.path.exists(config.model_file):
        try:
            model.load_state_dict(torch.load(config.model_file, weights_only=True))
            loaded_existing = True
        except Exception as e:
            print(f"\n  WARNING: Could not load existing model: {e}")
            print(f"           Starting with fresh weights instead.")

    if config.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)

    criterion = nn.CrossEntropyLoss()

    print(f"\n" + "-" * 60)
    print("MODEL INITIALIZED")
    print("-" * 60)
    print(f"  Device:           {model.device}")
    print(f"  Hidden dimension: {config.hidden_dim}")
    print(f"  Transformer layers: {config.num_layers}")
    print(f"  Optimizer:        {config.optimizer}")
    print(f"  Learning rate:    {config.learning_rate}")

    # Show continuous learning status
    if loaded_existing:
        print(f"\n  ✓ CONTINUING from existing model!")
        print(f"    Loaded weights from: {config.model_file}")
        print(f"    (The student is reading their old notes before learning more)")
    elif config.fresh_start:
        print(f"\n  ✗ FRESH START (--fresh flag)")
        print(f"    Training from scratch with random weights")
        print(f"    (The student is starting with a blank notebook)")
    else:
        print(f"\n  ○ NEW MODEL (no existing model found)")
        print(f"    Will save to: {config.model_file}")
        print(f"    (First day of class - student has no previous notes)")

    print("-" * 60)

    return model, optimizer, criterion


def train_model(config, model, optimizer, criterion, train_features, train_labels):
    """Train the model."""
    import torch

    print(f"  Epochs:           {config.epochs}")
    print(f"  Batch size:       {config.batch_size}")
    print(f"  Samples:          {len(train_features)}")
    print()

    train_features = np.array(train_features)
    train_labels = np.array(train_labels)
    item_losses = []

    model.train()
    for epoch in range(config.epochs):
        # Shuffle data each epoch
        indices = np.random.permutation(len(train_features))
        inputs = train_features[indices]
        targets = train_labels[indices]

        epoch_loss = 0
        num_batches = len(inputs) // config.batch_size

        for i in range(num_batches):
            batch_input = inputs[i * config.batch_size : i * config.batch_size + config.batch_size]
            batch_target = torch.from_numpy(
                targets[i * config.batch_size : i * config.batch_size + config.batch_size]
            )

            optimizer.zero_grad()
            logits = model(batch_input)

            loss = criterion(
                logits,
                batch_target.float().to(model.device)
            )
            loss.backward()
            optimizer.step()

            item_losses.append(loss.item())
            epoch_loss += loss.item()

        # Rolling average loss (last 250 samples)
        avg_loss = sum(item_losses[-250:]) / len(item_losses[-250:])
        print(f"  Epoch {epoch + 1}/{config.epochs}: avg_loss={avg_loss:.4f}")


def evaluate(config, model, test_features, test_labels):
    """Evaluate model on test set."""
    import torch
    from sklearn.metrics import f1_score

    print(f"  Test samples:     {len(test_features)}")

    correct = 0
    total = 0
    all_predictions = []
    all_actuals = []

    model.eval()
    with torch.no_grad():
        for i in range(len(test_features)):
            input_text = [test_features[i]]
            target = torch.tensor(test_labels[i])

            logits = model(input_text)
            probs = logits.softmax(dim=-1).cpu()
            predicted = torch.argmax(probs, dim=-1)
            actual = torch.argmax(target.float().to(model.device))

            all_predictions.append(predicted.item())
            all_actuals.append(actual.item())

            if predicted.item() == actual.item():
                correct += 1
            total += 1

    accuracy = correct / total
    f1 = f1_score(all_actuals, all_predictions, average='weighted')

    print(f"  Results:")
    print(f"    Accuracy:       {correct}/{total} = {accuracy:.2%}")
    print(f"    F1 Score:       {f1:.4f}")

    return accuracy, f1


def save_model(config, model):
    """Save model weights to disk."""
    import torch

    # Create directory if needed
    model_dir = os.path.dirname(config.model_file)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    torch.save(model.state_dict(), config.model_file)

    print("\n" + "-" * 60)
    print("MODEL SAVED")
    print("-" * 60)
    print(f"  File:             {config.model_file}")
    print("-" * 60)


def main():
    """Main training pipeline."""
    args = parse_args()

    # Create configuration from arguments
    config = Config(
        symbol=args.symbol,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        batch_size=args.batch_size,
        split=args.split,
        model_file=args.model_file,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        fresh_start=args.fresh,
        force=args.force,
        min_new_samples=args.min_new_samples,
        cooldown=args.cooldown
    )

    print("\n" + "=" * 60)
    print("        TRADING SIGNAL CLASSIFIER - TRAINING")
    print("=" * 60)
    print(f"\nCONFIGURATION:{config}")
    print("\n" + "=" * 60)

    # Smart Training Guard: Check if training should proceed
    should, reason, details = should_train(config)
    print_guard_decision(should, reason, details)

    if not should:
        print("\nExiting (no training needed).")
        sys.exit(0)  # Exit cleanly for daemon compatibility

    # Calculate data hash before training (for metadata)
    data_hash = calculate_data_hash(config)

    # Load and prepare data
    print("\nLOADING DATA...")
    print("-" * 60)
    features, labels = load_data(config)
    train_features, train_labels, test_features, test_labels = split_data(config, features, labels)

    # Create model
    print("\nINITIALIZING MODEL...")
    model, optimizer, criterion = create_model(config)

    # Train
    print("\nTRAINING...")
    print("-" * 60)
    train_model(config, model, optimizer, criterion, train_features, train_labels)

    # Evaluate
    print("\n" + "-" * 60)
    print("EVALUATION")
    print("-" * 60)
    accuracy, f1 = evaluate(config, model, test_features, test_labels)

    # Save model
    save_model(config, model)

    # Save training metadata (for Smart Guard)
    save_training_metadata(config, data_hash, len(features), accuracy, f1)
    print(f"\n  Training metadata saved to: {config.metadata_file}")

    print("\nDone!")


if __name__ == "__main__":
    main()
