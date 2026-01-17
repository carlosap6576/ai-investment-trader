#!/usr/bin/env python3
"""
CLI for Polygon (Massive) data provider.

Usage:
    python -m cli.providers.polygon -s AAPL
    python -m cli.providers.polygon -s AAPL --days 30
    python -m cli.providers.polygon -s AAPL --timespan hour
    python -m cli.providers.polygon -s AAPL --aggregates-only
    python -m cli.providers.polygon -s AAPL --news-only
    python -m cli.providers.polygon -s AAPL --ticker-only
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m cli.providers.polygon",
        description="Download data from Polygon (Massive) API.",
    )
    parser.add_argument("-s", "--symbol", required=True, help="Trading symbol (e.g., AAPL)")
    parser.add_argument("-d", "--days", type=int, default=7, help="Days of history for aggregates (default: 7)")
    parser.add_argument("-t", "--timespan", default="day",
                        choices=["minute", "hour", "day", "week", "month"],
                        help="Bar timespan (default: day)")
    parser.add_argument("-n", "--news-limit", type=int, default=100, help="Max news articles (default: 100)")
    parser.add_argument("--aggregates-only", action="store_true", help="Only download aggregates")
    parser.add_argument("--previous-only", action="store_true", help="Only download previous day bar")
    parser.add_argument("--ticker-only", action="store_true", help="Only download ticker details")
    parser.add_argument("--news-only", action="store_true", help="Only download news")

    args = parser.parse_args()

    # Lazy imports
    from providers.polygon import (
        PolygonProvider,
        PolygonAggregatesProvider,
        PolygonPreviousDayProvider,
        PolygonTickerProvider,
        PolygonNewsProvider,
    )
    from datetime import datetime, timedelta

    print(f"\n{'='*60}")
    print(f"  POLYGON (MASSIVE) PROVIDER")
    print(f"  Symbol: {args.symbol.upper()}")
    print(f"{'='*60}\n")

    if args.aggregates_only:
        today = datetime.now()
        from_date = (today - timedelta(days=args.days)).strftime("%Y-%m-%d")
        to_date = today.strftime("%Y-%m-%d")
        provider = PolygonAggregatesProvider(
            args.symbol, timespan=args.timespan,
            from_date=from_date, to_date=to_date
        )
        result = provider.run()
        print(f"\nResult: {result.message}")
    elif args.previous_only:
        provider = PolygonPreviousDayProvider(args.symbol)
        result = provider.run()
        print(f"\nResult: {result.message}")
    elif args.ticker_only:
        provider = PolygonTickerProvider(args.symbol)
        result = provider.run()
        print(f"\nResult: {result.message}")
    elif args.news_only:
        provider = PolygonNewsProvider(args.symbol, args.news_limit)
        result = provider.run()
        print(f"\nResult: {result.message}")
    else:
        provider = PolygonProvider(
            args.symbol,
            days=args.days,
            timespan=args.timespan,
            news_limit=args.news_limit,
        )
        results = provider.run()
        print(f"\nResults:")
        for key, result in results.items():
            status = "OK" if result.success else "FAILED"
            print(f"  {key}: [{status}] {result.record_count} records - {result.message}")

    print()


if __name__ == "__main__":
    main()
