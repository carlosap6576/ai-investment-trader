#!/usr/bin/env python3
"""
CLI for Alpha Vantage data provider.

Usage:
    python -m cli.providers.alphavantage -s AAPL
    python -m cli.providers.alphavantage -s AAPL --endpoint quote
    python -m cli.providers.alphavantage -s AAPL --endpoint news
    python -m cli.providers.alphavantage -s AAPL --all
"""

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m cli.providers.alphavantage",
        description="Download data from Alpha Vantage.",
    )
    parser.add_argument("-s", "--symbol", required=True, help="Trading symbol")
    parser.add_argument(
        "-e", "--endpoint",
        choices=["quote", "news", "news_market", "overview", "income", "balance", "cashflow", "shares"],
        default="quote",
        help="Endpoint to fetch (default: quote)"
    )
    parser.add_argument("--all", action="store_true", help="Download all endpoints")

    args = parser.parse_args()

    # Lazy imports
    from providers.alphavantage import AlphaVantageProvider, AlphaVantageAllProvider

    print(f"\n{'='*60}")
    print(f"  ALPHA VANTAGE PROVIDER")
    print(f"  Symbol: {args.symbol.upper()}")
    print(f"{'='*60}\n")

    if args.all:
        provider = AlphaVantageAllProvider(args.symbol)
        results = provider.run()
        print(f"\nResults:")
        for endpoint, result in results.items():
            status = "OK" if result.success else "FAILED"
            print(f"  {endpoint}: [{status}] {result.message}")
    else:
        provider = AlphaVantageProvider(args.symbol, args.endpoint)
        result = provider.run()
        print(f"\nResult: {result.message}")

    print()


if __name__ == "__main__":
    main()
