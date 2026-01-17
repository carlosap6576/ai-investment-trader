#!/usr/bin/env python3
"""
CLI for Yahoo Finance data provider.

Usage:
    python -m cli.providers.yahoo -s AAPL
    python -m cli.providers.yahoo -s BTC-USD -p 3mo -i 1h
    python -m cli.providers.yahoo -s TSLA --prices-only
    python -m cli.providers.yahoo -s TSLA --news-only
"""

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m cli.providers.yahoo",
        description="Download data from Yahoo Finance.",
    )
    parser.add_argument("-s", "--symbol", required=True, help="Trading symbol")
    parser.add_argument("-p", "--period", default="1mo", help="Data period (default: 1mo)")
    parser.add_argument("-i", "--interval", default="5m", help="Data interval (default: 5m)")
    parser.add_argument("-n", "--news-count", type=int, default=100, help="News articles to fetch")
    parser.add_argument("--prices-only", action="store_true", help="Only download prices")
    parser.add_argument("--news-only", action="store_true", help="Only download news")

    args = parser.parse_args()

    # Lazy imports
    from providers.yahoo import YahooProvider, YahooPriceProvider, YahooNewsProvider

    print(f"\n{'='*60}")
    print(f"  YAHOO FINANCE PROVIDER")
    print(f"  Symbol: {args.symbol.upper()}")
    print(f"{'='*60}\n")

    if args.prices_only:
        provider = YahooPriceProvider(args.symbol, args.period, args.interval)
        result = provider.run()
        print(f"\nResult: {result.message}")
    elif args.news_only:
        provider = YahooNewsProvider(args.symbol, args.news_count)
        result = provider.run()
        print(f"\nResult: {result.message}")
    else:
        provider = YahooProvider(args.symbol, args.period, args.interval, args.news_count)
        results = provider.run()
        print(f"\nResults:")
        for key, result in results.items():
            print(f"  {key}: {result.message}")

    print()


if __name__ == "__main__":
    main()
