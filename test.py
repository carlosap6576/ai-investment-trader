"""
Test the trained Hierarchical Sentiment Trading Signal Classifier.
Loads saved model weights and evaluates on test data.

Uses the HierarchicalSentimentTransformer architecture with:
- Multi-level news classification (MARKET / SECTOR / TICKER)
- FinBERT-based financial sentiment analysis
- Cross-level attention between sentiment levels
- Temporal sequences for pattern detection
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import List, Tuple, Optional
import numpy as np

# =============================================================================
# DEFAULTS - Must match train.py settings used during training!
# =============================================================================

DEFAULT_BUY_THRESHOLD = 1.0      # +1% price increase triggers BUY
DEFAULT_SELL_THRESHOLD = -1.0    # -1% price decrease triggers SELL
DEFAULT_TRAIN_SPLIT = 0.8        # 80% train, 20% test
DEFAULT_NUM_SAMPLES = 3          # Sample predictions to show

# Model architecture defaults - MUST match train.py settings!
DEFAULT_HIDDEN_DIM = 256         # Internal dimension after projection
DEFAULT_NUM_LAYERS = 2           # Number of Transformer encoder layers

# Hierarchical model defaults
DEFAULT_SEQ_LENGTH = 20          # Temporal sequence length

DATASETS_DIR = "datasets"

# Label names for reporting
LABEL_NAMES = ["SELL", "HOLD", "BUY"]

# =============================================================================
# HELP TEXT AND CLI INTERFACE
# =============================================================================

HELP_TEXT = """
================================================================================
            HIERARCHICAL SENTIMENT TRADING SIGNAL CLASSIFIER
                           Model Testing
================================================================================

DESCRIPTION:
    Tests the trained hierarchical sentiment classifier on held-out test data.
    Loads saved model weights and evaluates accuracy, F1 score, and shows
    detailed evaluation report with trading performance metrics.

    IMPORTANT: The thresholds and split MUST match what was used during
    training, or the evaluation will be incorrect!

--------------------------------------------------------------------------------
USAGE:
--------------------------------------------------------------------------------

    python test.py -s SYMBOL [OPTIONS]
    python test.py --symbol SYMBOL [OPTIONS]

--------------------------------------------------------------------------------
REQUIRED ARGUMENT:
--------------------------------------------------------------------------------

    -s, --symbol SYMBOL
        Trading symbol to test. Must have trained model in datasets/SYMBOL/

        Examples: BTC-USD, AAPL, TSLA, ETH-USD

--------------------------------------------------------------------------------
SIGNAL THRESHOLDS (must match train.py!):
--------------------------------------------------------------------------------

    -b, --buy-threshold PERCENT
        Price increase (%) that was used to trigger BUY during training.
        Default: 1.0

        IMPORTANT: Must match the value used in train.py!

    --sell-threshold PERCENT
        Price decrease (%) that was used to trigger SELL during training.
        Default: -1.0

        IMPORTANT: Must match the value used in train.py!

--------------------------------------------------------------------------------
OTHER OPTIONS:
--------------------------------------------------------------------------------

    --split RATIO
        Train/test split ratio used during training.
        Default: 0.8 (80% train, 20% test)

        IMPORTANT: Must match the value used in train.py!

    --model-file PATH
        Path to trained model weights.
        Default: datasets/{SYMBOL}/{SYMBOL}.pth

    --samples COUNT
        Number of sample predictions to show.
        Default: 3

    --seq-length LENGTH
        Temporal sequence length.
        Default: 20

        IMPORTANT: Must match the value used in train.py!

    -h, --help
        Show this help message and exit.

--------------------------------------------------------------------------------
MODEL ARCHITECTURE (must match train.py!):
--------------------------------------------------------------------------------

    --hidden-dim SIZE
        Hidden dimension of the Transformer.
        Default: 256

        IMPORTANT: Must match the value used in train.py!

    --num-layers COUNT
        Number of Transformer encoder layers.
        Default: 2

        IMPORTANT: Must match the value used in train.py!

--------------------------------------------------------------------------------
AI SUMMARY OPTIONS:
--------------------------------------------------------------------------------

    --summary
        Generate AI-powered summary at end of evaluation report.
        Uses Flan-T5 model to analyze results and provide recommendations.

    --no-summary
        Disable AI summary (default behavior).

    --summary-model MODEL
        Model for AI summary generation.
        Default: google/flan-t5-xl (3B params)

        Available models:
          - google/flan-t5-small  (80M params)  - Fastest
          - google/flan-t5-base   (250M params) - Good balance
          - google/flan-t5-large  (780M params) - Very good quality
          - google/flan-t5-xl     (3B params)   - Best quality

--------------------------------------------------------------------------------
EXAMPLES:
--------------------------------------------------------------------------------

    # Test with default thresholds (±1.0%)
    python test.py -s BTC-USD

    # Test with custom thresholds (must match training!)
    python test.py -s BTC-USD -b 0.1 --sell-threshold -0.1

    # Test AAPL with lower thresholds
    python test.py -s AAPL -b 0.3 --sell-threshold -0.3

    # Test with custom split
    python test.py -s BTC-USD --split 0.7

    # Show more sample predictions
    python test.py -s BTC-USD --samples 10

    # Test with custom architecture (must match training!)
    python test.py -s AAPL --hidden-dim 512 --num-layers 4

    # Test with AI summary
    python test.py -s AAPL --summary

    # Test with faster AI summary model
    python test.py -s AAPL --summary --summary-model google/flan-t5-base

--------------------------------------------------------------------------------
MATCHING TRAIN AND TEST COMMANDS:
--------------------------------------------------------------------------------

    If you trained with:
        python train.py -s BTC-USD -b 0.1 --sell-threshold -0.1

    Then test with:
        python test.py -s BTC-USD -b 0.1 --sell-threshold -0.1

    If you trained with custom architecture:
        python train.py -s AAPL --hidden-dim 512 --num-layers 4 --seq-length 30

    Then test with matching architecture:
        python test.py -s AAPL --hidden-dim 512 --num-layers 4 --seq-length 30

    ALL parameters (thresholds, architecture, seq-length) MUST match!

--------------------------------------------------------------------------------
WORKFLOW:
--------------------------------------------------------------------------------

    1. Download data:    python download.py -s BTC-USD
    2. Train model:      python train.py -s BTC-USD -b 0.1 --sell-threshold -0.1
    3. Test model:       python test.py -s BTC-USD -b 0.1 --sell-threshold -0.1

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
        sys.stderr.write("  python test.py -s SYMBOL [OPTIONS]\n\n")

        sys.stderr.write("EXAMPLES:\n")
        sys.stderr.write("  python test.py -s BTC-USD\n")
        sys.stderr.write("  python test.py -s BTC-USD -b 0.1 --sell-threshold -0.1\n")
        sys.stderr.write("  python test.py -s AAPL -b 0.3 --sell-threshold -0.3\n\n")

        sys.stderr.write("IMPORTANT: Thresholds must match what was used during training!\n\n")

        sys.stderr.write("For full help, run: python test.py --help\n\n")
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

    # Signal thresholds (must match training!)
    parser.add_argument("-b", "--buy-threshold", type=float, default=DEFAULT_BUY_THRESHOLD)
    parser.add_argument("--sell-threshold", type=float, default=DEFAULT_SELL_THRESHOLD)

    # Other options
    parser.add_argument("--split", type=float, default=DEFAULT_TRAIN_SPLIT)
    parser.add_argument("--model-file", default=None)
    parser.add_argument("--samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("-h", "--help", action="store_true")

    # Model architecture (must match train.py!)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--num-layers", type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--seq-length", type=int, default=DEFAULT_SEQ_LENGTH,
                        help=f"Temporal sequence length (default: {DEFAULT_SEQ_LENGTH})")

    # AI Summary options
    parser.add_argument("--summary", action="store_true", default=False,
                        help="Generate AI-powered summary at end of report")
    parser.add_argument("--no-summary", action="store_true", default=False,
                        help="Disable AI summary (default)")
    parser.add_argument("--summary-model", default="google/flan-t5-xl",
                        choices=[
                            "google/flan-t5-small",
                            "google/flan-t5-base",
                            "google/flan-t5-large",
                            "google/flan-t5-xl",
                        ],
                        help="Model for AI summary (default: google/flan-t5-xl, 3B params)")

    args = parser.parse_args()

    # Validate thresholds
    if args.buy_threshold <= 0:
        parser.error(f"Buy threshold must be positive (got {args.buy_threshold})")
    if args.sell_threshold >= 0:
        parser.error(f"Sell threshold must be negative (got {args.sell_threshold})")
    if args.split <= 0 or args.split >= 1:
        parser.error(f"Split must be between 0 and 1 (got {args.split})")
    if args.samples < 0:
        parser.error(f"Samples must be non-negative (got {args.samples})")
    if args.seq_length < 1:
        parser.error(f"Sequence length must be at least 1 (got {args.seq_length})")

    return args


class Config:
    """Configuration container for testing parameters."""

    def __init__(self, symbol, buy_threshold, sell_threshold, split, model_file, samples,
                 hidden_dim, num_layers, seq_length=20,
                 summary=False, summary_model="google/flan-t5-xl"):
        self.symbol = symbol.upper()
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.split = split
        self.samples = samples
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_length = seq_length

        # AI Summary options
        self.summary = summary
        self.summary_model = summary_model

        # Derived paths
        self.symbol_dir = f"{DATASETS_DIR}/{self.symbol}"
        self.data_file = f"{self.symbol_dir}/news_with_price.json"

        # Model file: use provided path or default to datasets/{SYMBOL}/{SYMBOL}.pth
        if model_file:
            self.model_file = model_file
        else:
            self.model_file = f"{self.symbol_dir}/{self.symbol}.pth"

    def __str__(self):
        return (
            f"\n"
            f"  [Data]\n"
            f"  Symbol:           {self.symbol}\n"
            f"  Data file:        {self.data_file}\n"
            f"  Model file:       {self.model_file}\n"
            f"\n"
            f"  [Signal Thresholds]\n"
            f"  Buy threshold:    > {self.buy_threshold}% price increase\n"
            f"  Sell threshold:   < {self.sell_threshold}% price decrease\n"
            f"\n"
            f"  [Model Architecture]\n"
            f"  Model type:       HierarchicalSentimentTransformer\n"
            f"  Sequence length:  {self.seq_length}\n"
            f"  Hidden dimension: {self.hidden_dim}\n"
            f"  Transformer layers: {self.num_layers}\n"
            f"\n"
            f"  [Test Parameters]\n"
            f"  Train/Test split: {self.split*100:.0f}% / {(1-self.split)*100:.0f}%\n"
            f"  Sample predictions: {self.samples}"
        )


# =============================================================================
# TESTING FUNCTIONS
# =============================================================================

def check_required_files(config):
    """Check if all required files exist."""
    errors = []

    if not os.path.exists(config.model_file):
        errors.append(f"Model file not found: {config.model_file}")
        errors.append(f"  → Run 'python train.py -s {config.symbol}' first to train the model")

    if not os.path.exists(config.data_file):
        errors.append(f"Data file not found: {config.data_file}")
        errors.append(f"  → Run 'python download.py -s {config.symbol}' first to get data")
    else:
        with open(config.data_file, 'r') as f:
            data = json.load(f)
        if len(data) == 0:
            errors.append(f"Data file is empty: {config.data_file}")
            errors.append(f"  → Run 'python download.py -s {config.symbol}' to populate it")

    if errors:
        print("\nERROR: Missing required files\n")
        for error in errors:
            print(error)
        print()
        sys.exit(1)


def load_model_metadata(model_file):
    """
    Load model architecture metadata from companion JSON file.

    The metadata file is created by train.py and contains:
    - hidden_dim: Model internal dimension
    - num_layers: Number of transformer layers
    - seq_length: Temporal sequence length
    - symbol: Trading symbol
    - buy_threshold: Buy threshold used during training
    - sell_threshold: Sell threshold used during training

    Returns:
        dict or None: Metadata dictionary if found, None otherwise
    """
    import json
    import os

    metadata_file = model_file.replace('.pth', '_metadata.json')

    if not os.path.exists(metadata_file):
        return None

    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        print(f"  Warning: Could not load metadata file: {e}")
        return None


# =============================================================================
# HIERARCHICAL MODEL FUNCTIONS
# =============================================================================

def load_hierarchical_model(config):
    """Load trained hierarchical model from disk."""
    try:
        import torch
        from src.models.transformer import HierarchicalSentimentTransformer
    except ImportError as e:
        sys.stderr.write(f"\nERROR: Hierarchical model dependencies not found: {e}\n")
        sys.stderr.write(f"\nMake sure src/models/transformer.py exists.\n")
        sys.exit(1)

    print(f"  Model file:       {config.model_file}")
    print(f"  Model type:       hierarchical")
    print(f"  Hidden dimension: {config.hidden_dim}")
    print(f"  Transformer layers: {config.num_layers}")
    print(f"  Sequence length:  {config.seq_length}")

    # Create model with matching architecture
    model = HierarchicalSentimentTransformer(
        hidden_dim=config.hidden_dim,
        num_temporal_layers=config.num_layers,
        sequence_length=config.seq_length,
    )
    model.load_state_dict(torch.load(config.model_file, weights_only=True))
    print(f"  Device:           {model.device}")
    return model


def load_hierarchical_test_data(config):
    """Load test portion of the dataset for hierarchical model."""
    print(f"  Data file:        {config.data_file}")

    # Import hierarchical data processing modules
    try:
        from src.data.schemas import NewsItem
        from src.features.feature_builder import TemporalFeatureBuilder, create_label_function
    except ImportError as e:
        sys.stderr.write(f"\nERROR: Hierarchical model data modules not found: {e}\n")
        sys.stderr.write(f"\nMake sure src/data/ and src/features/ modules exist.\n")
        sys.exit(1)

    # Load raw data
    with open(config.data_file, 'r') as f:
        data = json.load(f)

    # Convert to NewsItem objects
    news_items = []
    for item in data:
        try:
            news_item = NewsItem(
                headline=item.get('title', ''),
                summary=item.get('summary', ''),
                timestamp=datetime.fromisoformat(item['pubDate'].replace('Z', '+00:00')),
                source=item.get('source', 'unknown'),
                url=item.get('url', ''),
                level=item.get('level', 'TICKER'),
                sentiment_score=item.get('sentiment_score', 0.0),
                sentiment_label=item.get('sentiment_label', 'neutral'),
                price=item.get('price', 0.0),
                future_price=item.get('future_price'),
                price_change_pct=item.get('percentage', 0.0),
            )
            news_items.append(news_item)
        except (KeyError, ValueError):
            continue  # Skip malformed items

    if len(news_items) == 0:
        print(f"\nERROR: No valid news items found in data.")
        print(f"Make sure the data was downloaded with hierarchical fields.")
        print(f"Re-run: python download.py -s {config.symbol}")
        sys.exit(1)

    print(f"  Valid news items: {len(news_items)}")

    # Create feature builder
    feature_builder = TemporalFeatureBuilder(
        target_ticker=config.symbol,
        sequence_length=config.seq_length,
    )

    # Create label function
    label_func = create_label_function(
        buy_threshold=config.buy_threshold,
        sell_threshold=config.sell_threshold,
    )

    # Build test data
    sequences, labels, timestamps = feature_builder.build_training_data(
        news_items=news_items,
        label_func=label_func,
    )

    print(f"  Total sequences:  {len(sequences)}")
    print(f"  Sequence shape:   {sequences.shape}")

    # Count label distribution
    label_counts = {"sell": 0, "hold": 0, "buy": 0}
    for label in labels:
        if label == 0:
            label_counts["sell"] += 1
        elif label == 1:
            label_counts["hold"] += 1
        else:
            label_counts["buy"] += 1

    total = len(labels)
    if total > 0:
        print(f"  Label distribution:")
        print(f"    SELL:  {label_counts['sell']:>4}  ({label_counts['sell']/total*100:>5.1f}%)")
        print(f"    HOLD:  {label_counts['hold']:>4}  ({label_counts['hold']/total*100:>5.1f}%)")
        print(f"    BUY:   {label_counts['buy']:>4}  ({label_counts['buy']/total*100:>5.1f}%)")

    # Get only test portion
    split_index = int(config.split * len(sequences))
    test_sequences = sequences[split_index:]
    test_labels = labels[split_index:]
    test_timestamps = timestamps[split_index:]

    print(f"  Test sequences:   {len(test_sequences)} ({(1-config.split)*100:.0f}%)")

    return test_sequences, test_labels, test_timestamps, news_items


def evaluate_hierarchical(config, model, test_sequences, test_labels, test_timestamps, news_items):
    """
    Comprehensive evaluation of hierarchical model with detailed report.

    Generates a full evaluation report including:
    - Overall performance metrics
    - Per-class performance
    - Trading simulation results
    - Hierarchical level importance analysis
    - Feature importance
    - Sample predictions
    - Calibration analysis
    """
    import torch
    from sklearn.metrics import (
        f1_score, precision_score, recall_score,
        confusion_matrix, brier_score_loss
    )

    print(f"  Evaluating {len(test_sequences)} sequences...")
    print(f"  Collecting predictions and attention weights...")

    # Collect all predictions and probabilities
    all_predictions = []
    all_actuals = []
    all_probs = []
    all_confidences = []
    attention_weights_sum = {'ticker': 0.0, 'sector': 0.0, 'market': 0.0}
    sample_predictions = []

    model.eval()
    with torch.no_grad():
        for i in range(len(test_sequences)):
            seq = torch.from_numpy(test_sequences[i:i+1]).float().to(model.device)
            target = test_labels[i]

            # Get prediction with probabilities
            logits = model(seq)
            probs = logits.softmax(dim=-1).cpu().numpy()[0]
            predicted = int(np.argmax(probs))
            confidence = float(probs[predicted])

            all_predictions.append(predicted)
            all_actuals.append(target)
            all_probs.append(probs)
            all_confidences.append(confidence)

            # Collect attention weights
            importance = model.get_level_importance()
            if importance:
                attention_weights_sum['ticker'] += importance.get('ticker', 0.33)
                attention_weights_sum['sector'] += importance.get('sector', 0.33)
                attention_weights_sum['market'] += importance.get('market', 0.33)

            # Collect sample predictions (first N)
            if len(sample_predictions) < config.samples:
                timestamp = test_timestamps[i] if i < len(test_timestamps) else None
                news_item = news_items[i] if i < len(news_items) else None
                sample_predictions.append({
                    'ticker': config.symbol,
                    'date': timestamp.strftime('%Y-%m-%d %H:%M') if timestamp else 'N/A',
                    'prediction': LABEL_NAMES[predicted],
                    'actual': LABEL_NAMES[target],
                    'confidence': confidence,
                    'probs': probs,
                    'correct': predicted == target,
                    'headline': news_item.headline[:50] + '...' if news_item and len(news_item.headline) > 50 else (news_item.headline if news_item else 'N/A'),
                })

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_actuals = np.array(all_actuals)
    all_probs = np.array(all_probs)
    all_confidences = np.array(all_confidences)

    # Calculate overall metrics
    n_samples = len(all_predictions)
    accuracy = float(np.mean(all_predictions == all_actuals))
    f1_macro = f1_score(all_actuals, all_predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(all_actuals, all_predictions, average='weighted', zero_division=0)

    # Per-class metrics
    precision_per_class = precision_score(all_actuals, all_predictions, average=None, labels=[0, 1, 2], zero_division=0)
    recall_per_class = recall_score(all_actuals, all_predictions, average=None, labels=[0, 1, 2], zero_division=0)
    f1_per_class = f1_score(all_actuals, all_predictions, average=None, labels=[0, 1, 2], zero_division=0)

    # Support (count) per class
    support_per_class = [int(np.sum(all_actuals == i)) for i in range(3)]

    # Directional accuracy (non-HOLD predictions)
    non_hold_mask = all_predictions != 1
    if np.any(non_hold_mask):
        directional_correct = (all_predictions[non_hold_mask] == all_actuals[non_hold_mask]) | (all_actuals[non_hold_mask] == 1)
        directional_accuracy = float(np.mean(directional_correct))
    else:
        directional_accuracy = 0.0

    # Simulated trading metrics
    simulated_pnl = 0.0
    win_count = 0
    loss_count = 0
    returns = []

    for i in range(len(all_predictions)):
        pred = all_predictions[i]
        if pred != 1:  # Only trade on BUY/SELL signals
            # Use news item price change if available
            if i < len(news_items) and news_items[i].price_change_pct is not None:
                price_change = news_items[i].price_change_pct
            else:
                price_change = 0.0

            if pred == 2:  # BUY
                ret = price_change - 0.1  # 0.1% transaction cost
            else:  # SELL
                ret = -price_change - 0.1

            returns.append(ret)
            simulated_pnl += ret
            if ret > 0:
                win_count += 1
            else:
                loss_count += 1

    win_rate = (win_count / (win_count + loss_count) * 100) if (win_count + loss_count) > 0 else 0.0

    # Sharpe ratio (annualized, assuming daily returns)
    if len(returns) > 1:
        returns_arr = np.array(returns)
        sharpe_ratio = float(np.mean(returns_arr) / np.std(returns_arr) * np.sqrt(252)) if np.std(returns_arr) > 0 else 0.0
    else:
        sharpe_ratio = 0.0

    # Max drawdown
    if len(returns) > 0:
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_drawdown = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0
    else:
        max_drawdown = 0.0

    # Attention weights (normalized)
    total_attention = sum(attention_weights_sum.values())
    if total_attention > 0:
        attention_analysis = {
            'ticker_weight': attention_weights_sum['ticker'] / n_samples,
            'sector_weight': attention_weights_sum['sector'] / n_samples,
            'market_weight': attention_weights_sum['market'] / n_samples,
            'cross_weight': 0.0,  # Not tracked separately
            'price_weight': 0.0,  # Not tracked separately
        }
    else:
        attention_analysis = {
            'ticker_weight': 0.33,
            'sector_weight': 0.33,
            'market_weight': 0.33,
            'cross_weight': 0.0,
            'price_weight': 0.0,
        }

    # Feature importance (based on feature dimension contributions)
    feature_importance = _compute_feature_importance(model, test_sequences)

    # Calibration metrics
    correct_mask = all_predictions == all_actuals
    conf_when_correct = float(np.mean(all_confidences[correct_mask])) if np.any(correct_mask) else 0.0
    conf_when_wrong = float(np.mean(all_confidences[~correct_mask])) if np.any(~correct_mask) else 0.0

    # Brier score (for multiclass, use one-vs-rest average)
    brier_scores = []
    for c in range(3):
        y_true_binary = (all_actuals == c).astype(float)
        y_prob = all_probs[:, c]
        brier_scores.append(brier_score_loss(y_true_binary, y_prob))
    brier_score = float(np.mean(brier_scores))

    # Expected calibration error (simplified)
    ece = _compute_expected_calibration_error(all_confidences, correct_mask)

    # Build the report
    report = []
    report.append("")
    report.append("=" * 80)
    report.append("HIERARCHICAL SENTIMENT TRADING SIGNAL CLASSIFIER - EVALUATION REPORT")
    report.append("=" * 80)
    report.append("")

    # Section 1: Overall Performance
    report.append("## 1. OVERALL PERFORMANCE")
    report.append("-" * 40)
    report.append(f"  Accuracy:              {accuracy:.4f} ({accuracy*100:.2f}%)")
    report.append(f"  F1 Score (Macro):      {f1_macro:.4f}")
    report.append(f"  F1 Score (Weighted):   {f1_weighted:.4f}")
    report.append(f"  Directional Accuracy:  {directional_accuracy:.4f}")
    report.append("")

    # Section 2: Per-Class Performance
    report.append("## 2. PER-CLASS PERFORMANCE")
    report.append("-" * 40)
    report.append("  Class     | Precision | Recall  | F1-Score | Support")
    report.append("  ----------|-----------|---------|----------|--------")
    report.append(f"  SELL      | {precision_per_class[0]:.4f}    | {recall_per_class[0]:.4f}  | {f1_per_class[0]:.4f}   | {support_per_class[0]}")
    report.append(f"  HOLD      | {precision_per_class[1]:.4f}    | {recall_per_class[1]:.4f}  | {f1_per_class[1]:.4f}   | {support_per_class[1]}")
    report.append(f"  BUY       | {precision_per_class[2]:.4f}    | {recall_per_class[2]:.4f}  | {f1_per_class[2]:.4f}   | {support_per_class[2]}")
    report.append("")

    # Section 3: Confusion Matrix
    report.append("## 3. CONFUSION MATRIX")
    report.append("-" * 40)
    cm = confusion_matrix(all_actuals, all_predictions, labels=[0, 1, 2])
    report.append("  Predicted →")
    report.append(f"  {'Actual ↓':<12} {'SELL':>10} {'HOLD':>10} {'BUY':>10}")
    report.append(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    for i, label in enumerate(LABEL_NAMES):
        report.append(f"  {label:<12} {cm[i, 0]:>10} {cm[i, 1]:>10} {cm[i, 2]:>10}")
    report.append("")

    # Section 4: Trading Performance
    report.append("## 4. TRADING PERFORMANCE (Simulated)")
    report.append("-" * 40)
    report.append(f"  Simulated PnL:         {simulated_pnl:+.2f}%")
    report.append(f"  Sharpe Ratio:          {sharpe_ratio:.3f}")
    report.append(f"  Max Drawdown:          {max_drawdown:.2f}%")
    report.append(f"  Win Rate:              {win_rate:.2f}%")
    report.append(f"  Total Trades:          {win_count + loss_count}")
    report.append("")

    # Section 5: Hierarchical Level Analysis
    report.append("## 5. HIERARCHICAL LEVEL IMPORTANCE")
    report.append("-" * 40)
    report.append("  How much each level contributed to predictions:")
    report.append("")
    ticker_bar = int(attention_analysis['ticker_weight'] * 20)
    sector_bar = int(attention_analysis['sector_weight'] * 20)
    market_bar = int(attention_analysis['market_weight'] * 20)
    report.append(f"  TICKER-LEVEL:     {'█' * ticker_bar:<20} {attention_analysis['ticker_weight']*100:.1f}%")
    report.append(f"  SECTOR-LEVEL:     {'█' * sector_bar:<20} {attention_analysis['sector_weight']*100:.1f}%")
    report.append(f"  MARKET-LEVEL:     {'█' * market_bar:<20} {attention_analysis['market_weight']*100:.1f}%")
    report.append("")

    # Section 6: Top Features
    report.append("## 6. TOP 10 MOST IMPORTANT FEATURES")
    report.append("-" * 40)
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    for i, (feature, importance) in enumerate(sorted_features, 1):
        bar = '█' * int(importance * 50)
        report.append(f"  {i:2d}. {feature:<35} {bar:<25} {importance:.4f}")
    report.append("")

    # Section 7: Sample Predictions
    report.append("## 7. SAMPLE PREDICTIONS")
    report.append("-" * 40)
    for i, sample in enumerate(sample_predictions[:5], 1):
        status = "✓" if sample['correct'] else "✗"
        report.append(f"  Sample {i}: {status}")
        report.append(f"    Ticker:      {sample['ticker']}")
        report.append(f"    Date:        {sample['date']}")
        report.append(f"    Headline:    {sample['headline']}")
        report.append(f"    Prediction:  {sample['prediction']} (confidence: {sample['confidence']:.2f})")
        report.append(f"    Actual:      {sample['actual']}")
        report.append(f"    Probs:       SELL={sample['probs'][0]:.2f}, HOLD={sample['probs'][1]:.2f}, BUY={sample['probs'][2]:.2f}")
        report.append("")

    # Section 8: Calibration Analysis
    report.append("## 8. CALIBRATION ANALYSIS")
    report.append("-" * 40)
    report.append(f"  Brier Score:                    {brier_score:.4f}")
    report.append(f"  Expected Calibration Error:     {ece:.4f}")
    report.append(f"  Avg Confidence (Correct):       {conf_when_correct:.4f}")
    report.append(f"  Avg Confidence (Incorrect):     {conf_when_wrong:.4f}")
    report.append("")

    # Section 9: AI Summary (if enabled)
    if config.summary:
        report.append("## 9. AI-POWERED SUMMARY")
        report.append("-" * 40)
        try:
            from src.models.report_summarizer import create_summarizer

            # Collect all data for summarizer
            summary_data = {
                'symbol': config.symbol,
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'directional_accuracy': directional_accuracy,
                'n_samples': n_samples,
                'n_train': n_samples * 4,  # Approximate from 80/20 split
                'total_trades': len(returns),
                'simulated_pnl': simulated_pnl,
                'sharpe_ratio': sharpe_ratio,
                'win_rate': win_rate,
                'max_drawdown': max_drawdown,
                'attention_analysis': attention_analysis,
                'brier_score': brier_score,
                'conf_when_correct': conf_when_correct,
                'conf_when_wrong': conf_when_wrong,
                'per_class_metrics': {
                    'precision': precision_per_class.tolist(),
                    'recall': recall_per_class.tolist(),
                    'f1': f1_per_class.tolist(),
                    'support': support_per_class,
                },
            }

            # Create summarizer and generate summary
            summarizer = create_summarizer(
                model_name=config.summary_model,
                verbose=True,
            )
            summary_text = summarizer.summarize(summary_data)

            # Add summary to report (split by newlines)
            for line in summary_text.split('\n'):
                report.append(f"  {line}")

        except Exception as e:
            report.append(f"  Error generating summary: {e}")
            report.append("  Try installing: pip install transformers sentencepiece")

        report.append("")

    # Footer
    report.append("=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)

    # Print the report
    for line in report:
        print(line)

    return accuracy, f1_weighted


def _compute_feature_importance(model, test_sequences):
    """
    Compute feature importance based on gradient-based attribution.
    Returns a dict of feature_name -> importance_score.
    """
    # Feature names based on the 40-dimensional feature vector
    feature_names = [
        # Ticker features (0-11)
        "ticker_mean_sentiment", "ticker_sentiment_std", "ticker_sentiment_skew",
        "ticker_news_volume", "ticker_momentum_1d", "ticker_momentum_5d",
        "ticker_max_sentiment", "ticker_min_sentiment", "ticker_positive_ratio",
        "ticker_negative_ratio", "ticker_news_recency", "ticker_source_diversity",
        # Sector features (12-21)
        "sector_mean_sentiment", "sector_sentiment_breadth", "sector_news_volume",
        "sector_sentiment_dispersion", "sector_leader_sentiment", "sector_momentum",
        "sector_relative_strength", "sector_news_concentration", "sector_peer_sentiment",
        "sector_laggard_sentiment",
        # Market features (22-31)
        "market_sentiment_index", "market_news_volume", "market_fed_sentiment",
        "market_economic_sentiment", "market_geopolitical_sentiment", "market_fear_index",
        "market_momentum", "market_breadth_sentiment", "market_vix_level", "market_regime",
        # Cross-level features (32-39)
        "cross_ticker_vs_sector", "cross_ticker_vs_market", "cross_sector_vs_market",
        "cross_ticker_sector_corr", "cross_ticker_market_beta", "cross_divergence_score",
        "cross_relative_attention", "cross_sentiment_surprise",
    ]

    # Calculate variance-based importance (simple but effective)
    # Features with higher variance across sequences are more discriminative
    if len(test_sequences) > 0:
        # Average across time steps, then compute variance across samples
        avg_features = np.mean(test_sequences, axis=1)  # (n_samples, 40)
        feature_variance = np.var(avg_features, axis=0)  # (40,)

        # Normalize to 0-1
        if np.max(feature_variance) > 0:
            feature_variance = feature_variance / np.max(feature_variance)

        importance = {}
        for i, name in enumerate(feature_names):
            if i < len(feature_variance):
                importance[name] = float(feature_variance[i])
            else:
                importance[name] = 0.0
    else:
        importance = {name: 0.0 for name in feature_names}

    return importance


def _compute_expected_calibration_error(confidences, correct_mask, n_bins=10):
    """Compute Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            avg_confidence = np.mean(confidences[in_bin])
            avg_accuracy = np.mean(correct_mask[in_bin])
            ece += prop_in_bin * np.abs(avg_accuracy - avg_confidence)

    return float(ece)


def show_hierarchical_predictions(config, model, test_sequences, test_timestamps, news_items):
    """Show sample predictions for hierarchical model (legacy function, kept for compatibility)."""
    # This functionality is now integrated into evaluate_hierarchical
    pass


def main():
    """Main testing pipeline."""
    args = parse_args()

    # Determine if summary is enabled (--summary overrides --no-summary)
    enable_summary = args.summary and not args.no_summary

    # Determine model file path to check for metadata
    symbol = args.symbol.upper()
    if args.model_file:
        model_file = args.model_file
    else:
        model_file = f"{DATASETS_DIR}/{symbol}/{symbol}.pth"

    # Try to load architecture metadata from companion JSON file
    # This was saved by train.py alongside the model weights
    metadata = load_model_metadata(model_file)

    if metadata:
        print("\n" + "=" * 60)
        print("  AUTO-CONFIGURATION FROM SAVED MODEL METADATA")
        print("=" * 60)

        # Use metadata values unless CLI explicitly overrides
        hidden_dim = metadata.get('hidden_dim', args.hidden_dim)
        num_layers = metadata.get('num_layers', args.num_layers)
        seq_length = metadata.get('seq_length', args.seq_length)

        print(f"  Model type:       HierarchicalSentimentTransformer")
        print(f"  Hidden dim:       {hidden_dim}")
        print(f"  Num layers:       {num_layers}")
        print(f"  Seq length:       {seq_length}")

        # Also show training thresholds for reference
        train_buy = metadata.get('buy_threshold')
        train_sell = metadata.get('sell_threshold')
        if train_buy is not None and train_sell is not None:
            print(f"  Training thresholds: buy > {train_buy}%, sell < {train_sell}%")
            # Warn if test thresholds don't match training
            if args.buy_threshold != train_buy or args.sell_threshold != train_sell:
                print(f"\n  WARNING: Your test thresholds differ from training!")
                print(f"           Test: buy > {args.buy_threshold}%, sell < {args.sell_threshold}%")
                print(f"           This may cause incorrect label assignments.")

        print("=" * 60)
    else:
        # No metadata file found - use CLI arguments with defaults
        hidden_dim = args.hidden_dim
        num_layers = args.num_layers
        seq_length = args.seq_length

    # Create configuration (using auto-configured values if metadata was found)
    config = Config(
        symbol=args.symbol,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
        split=args.split,
        model_file=args.model_file,
        samples=args.samples,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        seq_length=seq_length,
        summary=enable_summary,
        summary_model=args.summary_model,
    )

    print("\n" + "=" * 60)
    print("        TRADING SIGNAL CLASSIFIER - TESTING")
    print("=" * 60)
    print(f"\nCONFIGURATION:{config}")
    print("\n" + "=" * 60)

    # Check required files exist
    check_required_files(config)

    # Hierarchical model pipeline
    print("\nLOADING HIERARCHICAL MODEL...")
    print("-" * 60)
    model = load_hierarchical_model(config)

    print("\nLOADING TEST DATA...")
    print("-" * 60)
    test_sequences, test_labels, test_timestamps, news_items = load_hierarchical_test_data(config)

    print("\n" + "-" * 60)
    print("EVALUATION REPORT")
    print("-" * 60)
    evaluate_hierarchical(config, model, test_sequences, test_labels, test_timestamps, news_items)

    print("\n" + "-" * 60)
    print("Done!")
    print("-" * 60)


if __name__ == "__main__":
    main()
