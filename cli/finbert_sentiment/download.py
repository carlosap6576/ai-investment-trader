#!/usr/bin/env python3
"""
CLI wrapper for downloading and preparing market data.

This is a thin wrapper that delegates to the DataDownloader strategy class.
All business logic lives in strategies/finbert_sentiment/data/downloader.py.

Usage:
    python -m cli.finbert_sentiment.download -s SYMBOL [OPTIONS]

Examples:
    python -m cli.finbert_sentiment.download -s AAPL -p 1mo -i 5m
    python -m cli.finbert_sentiment.download -s BTC-USD -p 3mo -i 1h -n 500
"""

import argparse
import sys


def create_parser() -> argparse.ArgumentParser:
    """Create and return the argument parser."""
    # Lazy import config to keep --help fast
    from strategies.finbert_sentiment.data.config import (
        DEFAULT_PERIOD,
        DEFAULT_INTERVAL,
        DEFAULT_NEWS_COUNT,
        DEFAULT_NO_FILTER,
        DEFAULT_NO_SENTIMENT,
        VALID_PERIODS,
        VALID_INTERVALS,
    )

    parser = argparse.ArgumentParser(
        prog="python -m cli.finbert_sentiment.download",
        description="Download and prepare market data for training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m cli.finbert_sentiment.download -s AAPL -p 1mo -i 5m
  python -m cli.finbert_sentiment.download -s BTC-USD -p 3mo -i 1h -n 500
  python -m cli.finbert_sentiment.download -s TSLA --no-filter --no-sentiment

Intraday Data Limits (yfinance):
  1m  -> 7 days max
  5m  -> 60 days max
  1h  -> 730 days max
  1d+ -> unlimited
""",
    )

    parser.add_argument(
        "-s", "--symbol",
        required=True,
        help="Trading symbol (e.g., AAPL, BTC-USD, TSLA)"
    )
    parser.add_argument(
        "-p", "--period",
        default=DEFAULT_PERIOD,
        choices=VALID_PERIODS,
        help=f"How far back to fetch (default: {DEFAULT_PERIOD})"
    )
    parser.add_argument(
        "-i", "--interval",
        default=DEFAULT_INTERVAL,
        choices=VALID_INTERVALS,
        help=f"Candle interval (default: {DEFAULT_INTERVAL})"
    )
    parser.add_argument(
        "-n", "--news-count",
        type=int,
        default=DEFAULT_NEWS_COUNT,
        help=f"Number of news articles to fetch (default: {DEFAULT_NEWS_COUNT})"
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        default=DEFAULT_NO_FILTER,
        help="Disable news relevance filtering"
    )
    parser.add_argument(
        "--no-sentiment",
        action="store_true",
        default=DEFAULT_NO_SENTIMENT,
        help="Disable FinBERT sentiment analysis"
    )

    return parser


def main() -> None:
    """Main entry point for the download command."""
    parser = create_parser()
    args = parser.parse_args()

    # Lazy import strategy modules after arg parsing (keeps --help fast)
    from strategies.finbert_sentiment.data.config import (
        DownloadConfig,
        validate_period_interval,
    )
    from strategies.finbert_sentiment.data.preparer import DataPreparer

    # Validate period/interval combination
    is_valid, error_msg = validate_period_interval(args.period, args.interval)
    if not is_valid:
        sys.stderr.write(f"Error: {error_msg}\n")
        sys.exit(1)

    # Create configuration
    config = DownloadConfig(
        symbol=args.symbol,
        period=args.period,
        interval=args.interval,
        news_count=args.news_count,
        no_filter=args.no_filter,
        no_sentiment=args.no_sentiment,
    )

    # Run the preparer (downloads + prepares training data)
    preparer = DataPreparer(config)
    try:
        preparer.run()
    except KeyboardInterrupt:
        print("\nDownload cancelled by user.")
        sys.exit(1)
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
