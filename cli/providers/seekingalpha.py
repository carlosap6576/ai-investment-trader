#!/usr/bin/env python3
"""
CLI for Seeking Alpha data provider.

Usage:
    python -m cli.providers.seekingalpha -s AAPL
    python -m cli.providers.seekingalpha -s AAPL --news-only
    python -m cli.providers.seekingalpha -s AAPL --dividends-only
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m cli.providers.seekingalpha",
        description="Download data from Seeking Alpha.",
    )
    parser.add_argument("-s", "--symbol", required=True, help="Trading symbol")
    parser.add_argument("--news-only", action="store_true", help="Only download news")
    parser.add_argument("--dividends-only", action="store_true", help="Only download dividends")

    args = parser.parse_args()

    # Lazy imports
    from providers.seekingalpha import SeekingAlphaProvider, SeekingAlphaNewsProvider, SeekingAlphaDividendsProvider

    print(f"\n{'='*60}")
    print(f"  SEEKING ALPHA PROVIDER")
    print(f"  Symbol: {args.symbol.upper()}")
    print(f"{'='*60}\n")

    if args.news_only:
        provider = SeekingAlphaNewsProvider(args.symbol)
        result = provider.run()
        print(f"\nResult: {result.message}")
    elif args.dividends_only:
        provider = SeekingAlphaDividendsProvider(args.symbol)
        result = provider.run()
        print(f"\nResult: {result.message}")
    else:
        provider = SeekingAlphaProvider(args.symbol)
        results = provider.run()
        print(f"\nResults:")
        for key, result in results.items():
            print(f"  {key}: {result.message}")

    print()


if __name__ == "__main__":
    main()
