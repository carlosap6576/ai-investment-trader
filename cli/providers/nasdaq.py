#!/usr/bin/env python3
"""
CLI for Nasdaq Data Link provider.

Usage:
    # Stock prices (uses WIKI/PRICES table)
    python -m cli.providers.nasdaq -s AAPL
    python -m cli.providers.nasdaq -s MSFT --start-date 2020-01-01 --end-date 2021-12-31

    # Economic data (uses timeseries get())
    python -m cli.providers.nasdaq -s GDP --database FRED --timeseries
    python -m cli.providers.nasdaq -s UNRATE --database FRED --timeseries

    # Custom table
    python -m cli.providers.nasdaq -s AAPL --table ZACKS/FC
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m cli.providers.nasdaq",
        description="Download data from Nasdaq Data Link.",
    )
    parser.add_argument("-s", "--symbol", required=True, help="Trading symbol or dataset code")
    parser.add_argument("-d", "--database", default="WIKI", help="Database code (default: WIKI)")
    parser.add_argument("-t", "--table", help="Table name for get_table() (e.g., WIKI/PRICES, ZACKS/FC)")
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    parser.add_argument("--timeseries", action="store_true", help="Use get() for time-series data (FRED, etc.)")

    args = parser.parse_args()

    # Lazy imports
    from providers.nasdaq import (
        NasdaqProvider,
        NasdaqTimeseriesProvider,
        NasdaqTableProvider,
        NasdaqPricesProvider,
    )

    print(f"\n{'='*60}")
    print(f"  NASDAQ DATA LINK PROVIDER")
    print(f"  Symbol: {args.symbol.upper()}")
    if args.timeseries:
        print(f"  Database: {args.database.upper()}")
        print(f"  Method: get() (timeseries)")
    elif args.table:
        print(f"  Table: {args.table}")
        print(f"  Method: get_table()")
    else:
        print(f"  Table: WIKI/PRICES")
        print(f"  Method: get_table()")
    print(f"{'='*60}\n")

    if args.table:
        # Custom table
        provider = NasdaqTableProvider(
            args.symbol,
            args.table,
            args.start_date,
            args.end_date,
        )
        result = provider.run()
        print(f"\nResult: {result.message}")
    elif args.timeseries:
        # Time-series data (FRED, etc.)
        provider = NasdaqTimeseriesProvider(
            args.symbol,
            args.database,
            args.start_date,
            args.end_date,
        )
        result = provider.run()
        print(f"\nResult: {result.message}")
    else:
        # Default: stock prices from WIKI/PRICES
        provider = NasdaqProvider(
            args.symbol,
            args.database,
            args.start_date,
            args.end_date,
            use_timeseries=False,
        )
        results = provider.run()
        print(f"\nResults:")
        for key, result in results.items():
            status = "OK" if result.success else "FAILED"
            print(f"  {key}: [{status}] {result.record_count} records - {result.message}")

    print()


if __name__ == "__main__":
    main()
