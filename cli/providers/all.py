#!/usr/bin/env python3
"""
CLI to download from all data providers.

Note: This application supports STOCKS ONLY (NASDAQ, NYSE, etc.)
      Cryptocurrency symbols are not supported.

Usage:
    python -m cli.providers.all -s AAPL
    python -m cli.providers.all -s MSFT -p 3mo -i 1h
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m cli.providers.all",
        description="Download stock data from all providers (stocks only, no crypto).",
    )
    parser.add_argument("-s", "--symbol", required=True, help="Stock ticker symbol (e.g., AAPL, MSFT)")
    parser.add_argument("-p", "--period", default="1mo", help="Data period for Yahoo (default: 1mo)")
    parser.add_argument("-i", "--interval", default="5m", help="Data interval for Yahoo (default: 5m)")
    parser.add_argument("-n", "--news-count", type=int, default=100, help="News articles from Yahoo")

    args = parser.parse_args()

    # Lazy imports
    from providers.yahoo import YahooProvider
    from providers.alphavantage import AlphaVantageAllProvider
    from providers.seekingalpha import SeekingAlphaProvider
    from providers.financialdatasets import FinancialDatasetsProvider
    from providers.nasdaq import NasdaqProvider
    from providers.polygon import PolygonProvider

    print(f"\n{'='*60}")
    print(f"  ALL PROVIDERS (Stocks Only)")
    print(f"  Symbol: {args.symbol.upper()}")
    print(f"{'='*60}\n")

    all_results = {}

    # Yahoo Finance
    print("\n[1/6] YAHOO FINANCE")
    print("-" * 40)
    yahoo = YahooProvider(args.symbol, args.period, args.interval, args.news_count)
    all_results["yahoo"] = yahoo.run()

    # Alpha Vantage
    print("\n[2/6] ALPHA VANTAGE")
    print("-" * 40)
    alphavantage = AlphaVantageAllProvider(args.symbol)
    all_results["alphavantage"] = alphavantage.run()

    # Seeking Alpha
    print("\n[3/6] SEEKING ALPHA")
    print("-" * 40)
    seekingalpha = SeekingAlphaProvider(args.symbol)
    all_results["seekingalpha"] = seekingalpha.run()

    # FinancialDatasets
    print("\n[4/6] FINANCIALDATASETS.AI")
    print("-" * 40)
    financialdatasets = FinancialDatasetsProvider(args.symbol)
    all_results["financialdatasets"] = financialdatasets.run()

    # Nasdaq Data Link
    print("\n[5/6] NASDAQ DATA LINK")
    print("-" * 40)
    nasdaq = NasdaqProvider(args.symbol)
    all_results["nasdaq"] = nasdaq.run()

    # Polygon (Massive)
    print("\n[6/6] POLYGON")
    print("-" * 40)
    polygon = PolygonProvider(args.symbol)
    all_results["polygon"] = polygon.run()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    for provider_name, results in all_results.items():
        print(f"\n{provider_name.upper()}:")
        if isinstance(results, dict):
            for key, result in results.items():
                status = "OK" if result.success else "FAILED"
                print(f"  {key}: [{status}] {result.record_count} records")
        else:
            status = "OK" if results.success else "FAILED"
            print(f"  [{status}] {results.record_count} records")

    print()


if __name__ == "__main__":
    main()
