"""
Test the trained trading signal classifier.
Loads saved model weights and evaluates on test data.
"""

import argparse
import json
import os
import sys

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

DATASETS_DIR = "datasets"

# Label encodings (one-hot)
LABEL_SELL = [1., 0., 0.]
LABEL_HOLD = [0., 1., 0.]
LABEL_BUY = [0., 0., 1.]

# Label names for reporting
LABEL_NAMES = ["SELL", "HOLD", "BUY"]

# =============================================================================
# HELP TEXT AND CLI INTERFACE
# =============================================================================

HELP_TEXT = """
================================================================================
                      TRADING SIGNAL CLASSIFIER
                           Model Testing
================================================================================

DESCRIPTION:
    Tests a trained trading signal classifier on held-out test data.
    Loads saved model weights and evaluates accuracy, F1 score, and shows
    sample predictions.

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

--------------------------------------------------------------------------------
MATCHING TRAIN AND TEST COMMANDS:
--------------------------------------------------------------------------------

    If you trained with:
        python train.py -s BTC-USD -b 0.1 --sell-threshold -0.1

    Then test with:
        python test.py -s BTC-USD -b 0.1 --sell-threshold -0.1

    If you trained with custom architecture:
        python train.py -s AAPL --hidden-dim 512 --num-layers 4

    Then test with matching architecture:
        python test.py -s AAPL --hidden-dim 512 --num-layers 4

    ALL parameters (thresholds AND architecture) MUST match!

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

    return args


class Config:
    """Configuration container for testing parameters."""

    def __init__(self, symbol, buy_threshold, sell_threshold, split, model_file, samples,
                 hidden_dim, num_layers):
        self.symbol = symbol.upper()
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.split = split
        self.samples = samples
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

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


def load_model(config):
    """Load trained model from disk."""
    # Lazy import to allow --help without dependencies
    try:
        import torch
        from models.gemma_transformer_classifier import SimpleGemmaTransformerClassifier
    except ImportError as e:
        sys.stderr.write(f"\nERROR: Required dependencies are not installed.\n")
        sys.stderr.write(f"\nTo install them, run:\n")
        sys.stderr.write(f"  pip install -r requirements.txt\n\n")
        sys.exit(1)

    print(f"  Model file:       {config.model_file}")
    print(f"  Hidden dimension: {config.hidden_dim}")
    print(f"  Transformer layers: {config.num_layers}")

    # Create model with matching architecture
    model = SimpleGemmaTransformerClassifier(
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers
    )
    model.load_state_dict(torch.load(config.model_file, weights_only=True))
    print(f"  Device:           {model.device}")
    return model


def load_test_data(config):
    """Load test portion of the dataset."""
    print(f"  Data file:        {config.data_file}")

    with open(config.data_file, 'r') as f:
        data = json.load(f)

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

    # Get only test portion
    split_index = int(config.split * len(features))
    test_features = features[split_index:]
    test_labels = labels[split_index:]

    print(f"  Total samples:    {len(features)}")
    print(f"  Test samples:     {len(test_features)} ({(1-config.split)*100:.0f}%)")
    print(f"  Label distribution:")
    print(f"    SELL:  {label_counts['sell']:>4}  ({label_counts['sell']/len(features)*100:>5.1f}%)")
    print(f"    HOLD:  {label_counts['hold']:>4}  ({label_counts['hold']/len(features)*100:>5.1f}%)")
    print(f"    BUY:   {label_counts['buy']:>4}  ({label_counts['buy']/len(features)*100:>5.1f}%)")

    return test_features, test_labels


def evaluate(config, model, test_features, test_labels):
    """Evaluate model on test set with detailed metrics."""
    import torch
    from sklearn.metrics import f1_score, classification_report

    print(f"  Evaluating {len(test_features)} samples...")

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

    # Calculate metrics
    accuracy = correct / total if total > 0 else 0
    f1 = f1_score(all_actuals, all_predictions, average='weighted', zero_division=0)

    print(f"\n  Results:")
    print(f"    Accuracy:       {correct}/{total} = {accuracy:.2%}")
    print(f"    F1 Score:       {f1:.4f}")

    # Detailed classification report
    print(f"\n  Classification Report:")
    report = classification_report(all_actuals, all_predictions,
                                   target_names=LABEL_NAMES,
                                   labels=[0, 1, 2],
                                   zero_division=0)
    # Indent the report
    for line in report.split('\n'):
        print(f"    {line}")

    return accuracy, f1


def show_sample_predictions(config, model, test_features):
    """Show sample predictions with friendly recommendations."""
    if config.samples <= 0:
        return

    print(f"  Showing {min(config.samples, len(test_features))} sample predictions:")

    # Take first few samples
    for i in range(min(config.samples, len(test_features))):
        # Extract just the headline from the feature text
        lines = test_features[i].split('\n')
        headline = lines[1].replace('Headline: ', '') if len(lines) > 1 else test_features[i][:50]

        # Get prediction using the friendly predict() method
        result = model.predict(test_features[i])

        print(f"\n    Sample {i+1}:")
        print(f"      Headline: {headline[:55]}{'...' if len(headline) > 55 else ''}")
        print(f"      → {result['recommendation']}")


def main():
    """Main testing pipeline."""
    args = parse_args()

    # Create configuration from arguments
    config = Config(
        symbol=args.symbol,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
        split=args.split,
        model_file=args.model_file,
        samples=args.samples,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    )

    print("\n" + "=" * 60)
    print("        TRADING SIGNAL CLASSIFIER - TESTING")
    print("=" * 60)
    print(f"\nCONFIGURATION:{config}")
    print("\n" + "=" * 60)

    # Check required files exist
    check_required_files(config)

    # Load model
    print("\nLOADING MODEL...")
    print("-" * 60)
    model = load_model(config)

    # Load test data
    print("\nLOADING TEST DATA...")
    print("-" * 60)
    test_features, test_labels = load_test_data(config)

    # Evaluate
    print("\n" + "-" * 60)
    print("EVALUATION")
    print("-" * 60)
    evaluate(config, model, test_features, test_labels)

    # Show sample predictions with friendly recommendations
    print("\n" + "-" * 60)
    print("SAMPLE PREDICTIONS")
    print("-" * 60)
    show_sample_predictions(config, model, test_features)

    print("\n" + "-" * 60)
    print("Done!")
    print("-" * 60)


if __name__ == "__main__":
    main()
