#!/usr/bin/env python3
"""
CLI wrapper for evaluating the Hierarchical Sentiment Trading Signal Classifier.

This is a thin wrapper that delegates to the ModelEvaluator strategy class.
All business logic lives in strategies/finbert_sentiment/evaluation/evaluator.py.

Usage:
    python -m cli.finbert_sentiment.test -s SYMBOL [OPTIONS]

Examples:
    python -m cli.finbert_sentiment.test -s AAPL --samples 10
    python -m cli.finbert_sentiment.test -s BTC-USD -b 0.5 --sell-threshold -0.5 --summary
"""

import argparse
import sys


def create_parser() -> argparse.ArgumentParser:
    """Create and return the argument parser."""
    # Lazy import config to keep --help fast
    from strategies.finbert_sentiment.evaluation.config import (
        DEFAULT_BUY_THRESHOLD,
        DEFAULT_SELL_THRESHOLD,
        DEFAULT_TRAIN_SPLIT,
        DEFAULT_NUM_SAMPLES,
        DEFAULT_HIDDEN_DIM,
        DEFAULT_NUM_LAYERS,
        DEFAULT_SEQ_LENGTH,
        DEFAULT_SUMMARY_MODEL,
    )

    parser = argparse.ArgumentParser(
        prog="python -m cli.finbert_sentiment.test",
        description="Evaluate the Hierarchical Sentiment Trading Signal Classifier.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m cli.finbert_sentiment.test -s AAPL --samples 10
  python -m cli.finbert_sentiment.test -s BTC-USD -b 0.5 --sell-threshold -0.5
  python -m cli.finbert_sentiment.test -s TSLA --summary

Note: Thresholds must match training values for correct evaluation.
""",
    )

    # Required
    parser.add_argument(
        "-s", "--symbol",
        required=True,
        help="Trading symbol (must match training)"
    )

    # Threshold parameters (must match training)
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

    # Evaluation parameters
    parser.add_argument(
        "--split",
        type=float,
        default=DEFAULT_TRAIN_SPLIT,
        help=f"Train/test split ratio (default: {DEFAULT_TRAIN_SPLIT})"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help=f"Sample predictions to display (default: {DEFAULT_NUM_SAMPLES})"
    )

    # Model architecture (must match training)
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

    # Summary generation
    parser.add_argument(
        "--summary",
        action="store_true",
        default=False,
        help="Generate AI summary of results"
    )
    parser.add_argument(
        "--summary-model",
        default=DEFAULT_SUMMARY_MODEL,
        help=f"Model for AI summary (default: {DEFAULT_SUMMARY_MODEL})"
    )

    return parser


def main() -> None:
    """Main entry point for the test command."""
    parser = create_parser()
    args = parser.parse_args()

    # Lazy import strategy modules after arg parsing (keeps --help fast)
    from strategies.finbert_sentiment.evaluation.config import TestConfig
    from strategies.finbert_sentiment.evaluation.evaluator import ModelEvaluator

    # Create configuration
    config = TestConfig(
        symbol=args.symbol,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
        split=args.split,
        samples=args.samples,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        seq_length=args.seq_length,
        summary=args.summary,
        summary_model=args.summary_model,
    )

    # Run the evaluator
    evaluator = ModelEvaluator(config)
    try:
        result = evaluator.run()

        # Print final summary
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)
        print(f"  Accuracy:    {result['accuracy']:.2%}")
        print(f"  F1 Macro:    {result['f1_macro']:.2%}")
        print(f"  F1 Weighted: {result['f1_weighted']:.2%}")
        print(f"  Directional: {result['directional_accuracy']:.2%}")
        print(f"  Samples:     {result['n_samples']}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\nEvaluation cancelled by user.")
        sys.exit(1)
    except FileNotFoundError as e:
        sys.stderr.write(f"Error: Model or data file not found. Train first.\n{e}\n")
        sys.exit(1)
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
