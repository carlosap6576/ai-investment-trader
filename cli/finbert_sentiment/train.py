#!/usr/bin/env python3
"""
CLI wrapper for training the Hierarchical Sentiment Trading Signal Classifier.

This is a thin wrapper that delegates to the ModelTrainer strategy class.
All business logic lives in strategies/finbert_sentiment/training/trainer.py.

Usage:
    python -m cli.finbert_sentiment.train -s SYMBOL [OPTIONS]

Examples:
    python -m cli.finbert_sentiment.train -s AAPL -e 50 -l 0.001
    python -m cli.finbert_sentiment.train -s BTC-USD -b 0.5 --sell-threshold -0.5 -o AdamW
"""

import argparse
import sys


def create_parser() -> argparse.ArgumentParser:
    """Create and return the argument parser."""
    # Lazy import config to keep --help fast
    from strategies.finbert_sentiment.training.config import (
        DEFAULT_BUY_THRESHOLD,
        DEFAULT_SELL_THRESHOLD,
        DEFAULT_EPOCHS,
        DEFAULT_LEARNING_RATE,
        DEFAULT_BATCH_SIZE,
        DEFAULT_TRAIN_SPLIT,
        DEFAULT_OPTIMIZER,
        DEFAULT_HIDDEN_DIM,
        DEFAULT_NUM_LAYERS,
        DEFAULT_SEQ_LENGTH,
        DEFAULT_FRESH_START,
        DEFAULT_MIN_NEW_SAMPLES,
        DEFAULT_COOLDOWN_MINUTES,
        VALID_OPTIMIZERS,
    )

    parser = argparse.ArgumentParser(
        prog="python -m cli.finbert_sentiment.train",
        description="Train the Hierarchical Sentiment Trading Signal Classifier.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m cli.finbert_sentiment.train -s AAPL -e 50 -l 0.001
  python -m cli.finbert_sentiment.train -s BTC-USD -b 0.5 --sell-threshold -0.5 -o AdamW
  python -m cli.finbert_sentiment.train -s TSLA --fresh --force

Training Profiles:
  Speed:     --batch-size 32 -e 20 -l 0.01
  Balanced:  --batch-size 8 -e 50 -l 0.005
  Precision: --batch-size 1 -e 200 -l 0.001 -o AdamW
""",
    )

    # Required
    parser.add_argument(
        "-s", "--symbol",
        required=True,
        help="Trading symbol (e.g., AAPL, BTC-USD)"
    )

    # Threshold parameters
    parser.add_argument(
        "-b", "--buy-threshold",
        type=float,
        default=DEFAULT_BUY_THRESHOLD,
        help=f"Price increase %% for BUY label (default: {DEFAULT_BUY_THRESHOLD})"
    )
    parser.add_argument(
        "--sell-threshold",
        type=float,
        default=DEFAULT_SELL_THRESHOLD,
        help=f"Price decrease %% for SELL label (default: {DEFAULT_SELL_THRESHOLD})"
    )

    # Training parameters
    parser.add_argument(
        "-e", "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Training epochs (default: {DEFAULT_EPOCHS})"
    )
    parser.add_argument(
        "-l", "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f"Optimizer learning rate (default: {DEFAULT_LEARNING_RATE})"
    )
    parser.add_argument(
        "-o", "--optimizer",
        default=DEFAULT_OPTIMIZER,
        choices=VALID_OPTIMIZERS,
        help=f"Optimizer type (default: {DEFAULT_OPTIMIZER})"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Samples per gradient update (default: {DEFAULT_BATCH_SIZE})"
    )
    parser.add_argument(
        "--split",
        type=float,
        default=DEFAULT_TRAIN_SPLIT,
        help=f"Train/test split ratio (default: {DEFAULT_TRAIN_SPLIT})"
    )

    # Model architecture
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=DEFAULT_HIDDEN_DIM,
        help=f"Model internal dimension (default: {DEFAULT_HIDDEN_DIM})"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=DEFAULT_NUM_LAYERS,
        help=f"Transformer encoder layers (default: {DEFAULT_NUM_LAYERS})"
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=DEFAULT_SEQ_LENGTH,
        help=f"Temporal sequence length (default: {DEFAULT_SEQ_LENGTH})"
    )

    # Training control
    parser.add_argument(
        "--fresh",
        action="store_true",
        default=DEFAULT_FRESH_START,
        help="Start from scratch (ignore existing model)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Bypass Smart Training Guard"
    )
    parser.add_argument(
        "--min-new-samples",
        type=int,
        default=DEFAULT_MIN_NEW_SAMPLES,
        help=f"Minimum new samples to train (default: {DEFAULT_MIN_NEW_SAMPLES})"
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        default=DEFAULT_COOLDOWN_MINUTES,
        help=f"Minutes between training sessions (default: {DEFAULT_COOLDOWN_MINUTES})"
    )

    return parser


def main() -> None:
    """Main entry point for the train command."""
    parser = create_parser()
    args = parser.parse_args()

    # Lazy import strategy modules after arg parsing (keeps --help fast)
    from strategies.finbert_sentiment.training.config import TrainConfig
    from strategies.finbert_sentiment.training.trainer import ModelTrainer

    # Create configuration
    config = TrainConfig(
        symbol=args.symbol,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        split=args.split,
        optimizer=args.optimizer,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        seq_length=args.seq_length,
        fresh=args.fresh,
        force=args.force,
        min_new_samples=args.min_new_samples,
        cooldown=args.cooldown,
    )

    # Run the trainer
    trainer = ModelTrainer(config)
    try:
        result = trainer.run()

        # Handle skip case (Smart Guard)
        if result.skipped:
            print(f"\nTraining skipped: {result.skip_reason}")
            sys.exit(0)

        # Print final results
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"  Accuracy: {result.accuracy:.2%}")
        print(f"  F1 Score: {result.f1_score:.2%}")
        print(f"  Total samples: {result.total_samples}")
        print(f"  Epochs completed: {result.epochs_completed}")
        print(f"  Model saved: {result.model_file}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\nTraining cancelled by user.")
        sys.exit(1)
    except FileNotFoundError as e:
        sys.stderr.write(f"Error: Data file not found. Run download first.\n{e}\n")
        sys.exit(1)
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
