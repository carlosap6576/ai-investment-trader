#!/usr/bin/env python3
"""
CLI for FinancialDatasets.ai data provider.

Usage:
    python -m cli.providers.financialdatasets -s AAPL
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m cli.providers.financialdatasets",
        description="Download data from FinancialDatasets.ai.",
    )
    parser.add_argument("-s", "--symbol", required=True, help="Trading symbol")

    args = parser.parse_args()

    # Lazy imports
    from providers.financialdatasets import FinancialDatasetsProvider

    print(f"\n{'='*60}")
    print(f"  FINANCIALDATASETS.AI PROVIDER")
    print(f"  Symbol: {args.symbol.upper()}")
    print(f"{'='*60}\n")

    provider = FinancialDatasetsProvider(args.symbol)
    result = provider.run()

    print(f"\nResult: {result.message}")
    print()


if __name__ == "__main__":
    main()
