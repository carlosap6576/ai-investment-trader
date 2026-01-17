"""
Data Preparer for the FinBERT Sentiment Strategy.

Downloads data using the providers module, then prepares training data
by merging news with prices and adding sentiment analysis.

Usage:
    from strategies.finbert_sentiment.data.config import DownloadConfig
    from strategies.finbert_sentiment.data.preparer import DataPreparer

    config = DownloadConfig(symbol="AAPL", period="1mo", interval="5m")
    preparer = DataPreparer(config)
    preparer.run()
"""

from __future__ import annotations

import datetime
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from strategies.finbert_sentiment.constants import DATASETS_DIR
from strategies.finbert_sentiment.data.config import DownloadConfig


class DataPreparer:
    """
    Downloads and prepares training data for the FinBERT Sentiment Strategy.

    Uses providers for data acquisition, then merges and enriches the data
    with sentiment analysis for model training.
    """

    def __init__(self, config: DownloadConfig) -> None:
        self.config = config

        # Lazy-loaded modules
        self._yf = None
        self._NewsLevelClassifier = None
        self._FinBERTSentimentAnalyzer = None
        self._get_sector = None

    # =========================================================================
    # LAZY LOADERS
    # =========================================================================

    @property
    def yf(self):
        """Lazily load yfinance module."""
        if self._yf is None:
            import yfinance
            self._yf = yfinance
        return self._yf

    @property
    def NewsLevelClassifier(self):
        """Lazily load NewsLevelClassifier if available."""
        if self._NewsLevelClassifier is None:
            try:
                from src.data.news_classifier import NewsLevelClassifier as NLC
                self._NewsLevelClassifier = NLC
            except ImportError:
                self._NewsLevelClassifier = False
        return self._NewsLevelClassifier if self._NewsLevelClassifier is not False else None

    @property
    def FinBERTSentimentAnalyzer(self):
        """Lazily load FinBERTSentimentAnalyzer if available."""
        if self._FinBERTSentimentAnalyzer is None:
            if self.config.no_sentiment:
                self._FinBERTSentimentAnalyzer = False
            else:
                try:
                    from src.models.finbert_sentiment import FinBERTSentimentAnalyzer as FSA
                    self._FinBERTSentimentAnalyzer = FSA
                except ImportError:
                    self._FinBERTSentimentAnalyzer = False
        return self._FinBERTSentimentAnalyzer if self._FinBERTSentimentAnalyzer is not False else None

    @property
    def get_sector(self):
        """Lazily load sector mapping function."""
        if self._get_sector is None:
            try:
                from src.data.sector_mapping import get_sector
                self._get_sector = get_sector
            except ImportError:
                self._get_sector = False
        return self._get_sector if self._get_sector is not False else None

    # =========================================================================
    # VALIDATION & SETUP
    # =========================================================================

    def validate_ticker(self) -> Tuple[bool, Optional[str]]:
        """Validate the ticker symbol exists."""
        config = self.config
        print(f"  Validating ticker: {config.symbol}")

        try:
            ticker = self.yf.Ticker(config.symbol)
            info = ticker.info

            if not info or info.get('regularMarketPrice') is None:
                if 'symbol' not in info:
                    return False, f"  ERROR: '{config.symbol}' is not a valid ticker symbol.\n"

            print(f"  Ticker validated: {info.get('shortName', config.symbol)}")
            return True, None

        except Exception as e:
            return False, f"  ERROR: Could not validate ticker: {e}\n"

    def ensure_directory(self) -> None:
        """Create symbol directory if needed."""
        os.makedirs(self.config.symbol_dir, exist_ok=True)

    def init_data_files(self) -> None:
        """Initialize empty data files."""
        self.ensure_directory()

        # Create empty files if they don't exist
        for filename in ["historical_data.json", "news.json"]:
            filepath = f"{self.config.symbol_dir}/{filename}"
            if not os.path.exists(filepath):
                with open(filepath, 'w') as f:
                    if filename == "historical_data.json":
                        json.dump({}, f)
                    else:
                        json.dump([], f)

        print(f"  Initialized data files in {self.config.symbol_dir}/")

    # =========================================================================
    # DOWNLOAD USING PROVIDERS
    # =========================================================================

    def download_all(self) -> None:
        """Download data from all providers."""
        from providers.yahoo import YahooProvider
        from providers.alphavantage import AlphaVantageAllProvider
        from providers.seekingalpha import SeekingAlphaProvider
        from providers.financialdatasets import FinancialDatasetsProvider

        config = self.config

        # Yahoo Finance (prices + news)
        print("\n  [Yahoo Finance]")
        yahoo = YahooProvider(
            config.symbol,
            config.period,
            config.interval,
            config.news_count
        )
        yahoo.run()

        # Alpha Vantage (all endpoints)
        print("\n  [Alpha Vantage]")
        alphavantage = AlphaVantageAllProvider(config.symbol)
        alphavantage.run()

        # Seeking Alpha (news + dividends)
        print("\n  [Seeking Alpha]")
        seekingalpha = SeekingAlphaProvider(config.symbol)
        seekingalpha.run()

        # FinancialDatasets (news)
        print("\n  [FinancialDatasets.ai]")
        financialdatasets = FinancialDatasetsProvider(config.symbol)
        financialdatasets.run()

    # =========================================================================
    # DATA PREPARATION (Strategy-specific)
    # =========================================================================

    def prepare_data(self) -> None:
        """
        Merge news with price data to create training dataset.

        For each news article:
        1. Filter irrelevant news (unless no_filter is True)
        2. Classify into MARKET/SECTOR/TICKER levels
        3. Find the price at publication time
        4. Find the price one interval later
        5. Calculate percentage change
        6. Run FinBERT sentiment analysis (unless no_sentiment is True)
        """
        config = self.config
        print("  Preparing training data...")
        output = []

        # Load data
        try:
            with open(config.historical_data_file, 'r') as f:
                ticker_prices = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            ticker_prices = {}

        try:
            with open(config.news_file, 'r') as f:
                news = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            news = []

        if not ticker_prices:
            print("  WARNING: No price data available. Skipping data preparation.")
            return

        if not news:
            print("  WARNING: No news data available. Skipping data preparation.")
            return

        # Initialize news classifier
        classifier = None
        if self.NewsLevelClassifier is not None:
            strict_filtering = not config.no_filter
            classifier = self.NewsLevelClassifier(config.symbol, strict_filtering=strict_filtering)
            print(f"  News filtering: {'disabled' if config.no_filter else 'enabled'}")

        # Get sector for the symbol
        sector_gics = None
        if self.get_sector is not None:
            sector_gics = self.get_sector(config.symbol)
            if sector_gics:
                print(f"  Sector: {sector_gics}")

        # Initialize FinBERT sentiment analyzer
        sentiment_analyzer = None
        if self.FinBERTSentimentAnalyzer is not None:
            print("  Loading FinBERT for sentiment analysis...")
            try:
                sentiment_analyzer = self.FinBERTSentimentAnalyzer()
            except Exception as e:
                print(f"  WARNING: Could not load FinBERT: {e}")
                sentiment_analyzer = None

        matched = 0
        skipped = 0
        filtered = 0
        level_counts = {"MARKET": 0, "SECTOR": 0, "TICKER": 0}

        # Collect texts for batch sentiment analysis
        texts_for_sentiment = []
        items_for_sentiment = []

        for item in news:
            # Handle different news formats
            try:
                if 'content' in item:
                    title = item['content']['title']
                    summary = item['content'].get('summary', '')
                    pubDate = item['content']['pubDate']
                else:
                    title = item.get('title', '')
                    summary = item.get('summary', '')
                    pubDate = item.get('pubDate', item.get('providerPublishTime', ''))
            except (KeyError, TypeError):
                skipped += 1
                continue

            if not pubDate or not title:
                skipped += 1
                continue

            # Classify news level
            level = None
            if classifier is not None:
                level = classifier.classify({'headline': title, 'summary': summary})
                if level is None:
                    filtered += 1
                    continue
                level_counts[level] += 1
            else:
                level = "TICKER"
                level_counts["TICKER"] += 1

            # Convert pubDate to unix timestamp
            try:
                if isinstance(pubDate, str):
                    pubDate_ts = int(datetime.datetime.strptime(pubDate, '%Y-%m-%dT%H:%M:%SZ').timestamp())
                else:
                    pubDate_ts = int(pubDate)
            except (ValueError, TypeError):
                skipped += 1
                continue

            # Round down to nearest interval boundary
            index = pubDate_ts - (pubDate_ts % config.interval_seconds)

            # Look up prices (keys are milliseconds)
            price = ticker_prices.get(f"{index}000")
            future_price = ticker_prices.get(f"{index + config.interval_seconds}000")

            if price is None or future_price is None:
                skipped += 1
                continue

            # Calculate price change
            difference = future_price - price
            percentage = (difference / price) * 100

            # Build training record
            record = {
                'title': title,
                'index': index,
                'price': price,
                'future_price': future_price,
                'difference': difference,
                'percentage': percentage,
                'summary': summary,
                'pubDate': pubDate if isinstance(pubDate, str) else datetime.datetime.fromtimestamp(pubDate).isoformat(),
                'pubDate_ts': pubDate_ts,
                'level': level,
                'sector_gics': sector_gics,
                'sentiment_score': None,
                'sentiment_label': None,
                'sentiment_confidence': None,
            }

            output.append(record)

            # Prepare text for sentiment analysis
            if sentiment_analyzer is not None:
                text = f"{title} {summary}".strip()
                texts_for_sentiment.append(text)
                items_for_sentiment.append(record)

            matched += 1

        # Run batch sentiment analysis
        if sentiment_analyzer is not None and texts_for_sentiment:
            print(f"  Running FinBERT sentiment analysis on {len(texts_for_sentiment)} articles...")
            try:
                results = sentiment_analyzer.analyze_batch(texts_for_sentiment)
                for record, result in zip(items_for_sentiment, results):
                    record['sentiment_score'] = result.score
                    record['sentiment_label'] = result.label
                    record['sentiment_confidence'] = result.confidence
                print(f"  Sentiment analysis complete.")
            except Exception as e:
                print(f"  WARNING: Sentiment analysis failed: {e}")

        # Save training data
        with open(config.training_data_file, 'w') as f:
            json.dump(output, f, indent=2)

        # Print summary
        print(f"  Matched {matched} articles with price data")
        if filtered > 0:
            print(f"  Filtered out {filtered} irrelevant articles")
        if skipped > 0:
            print(f"  Skipped {skipped} articles (missing data)")
        print(f"  Level distribution: MARKET={level_counts['MARKET']}, "
              f"SECTOR={level_counts['SECTOR']}, TICKER={level_counts['TICKER']}")
        print(f"  Saved training data to {config.training_data_file}")

    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================

    def run(self) -> bool:
        """
        Execute the full download and preparation process.

        Returns:
            True if successful, False if validation failed.
        """
        config = self.config

        print("=" * 60)
        print("FINBERT SENTIMENT DATA PREPARER")
        print("=" * 60)

        # Step 1: Validate ticker
        print("\n[1/4] VALIDATING TICKER")
        print("-" * 60)

        is_valid, error_msg = self.validate_ticker()
        if not is_valid:
            sys.stderr.write(error_msg)
            return False

        # Step 2: Initialize
        print("\n[2/4] INITIALIZING")
        print("-" * 60)

        print(f"  Configuration:")
        print(f"    Symbol:     {config.symbol}")
        print(f"    Period:     {config.period}")
        print(f"    Interval:   {config.interval} ({config.interval_seconds}s)")
        print(f"    News count: {config.news_count}")
        print(f"    Filtering:  {'disabled' if config.no_filter else 'enabled'}")
        print(f"    Sentiment:  {'disabled' if config.no_sentiment else 'enabled'}")
        print(f"    Output dir: {config.symbol_dir}/")

        self.init_data_files()

        # Step 3: Download from all providers
        print("\n[3/4] DOWNLOADING DATA")
        print("-" * 60)

        self.download_all()

        # Step 4: Prepare training data
        print("\n[4/4] PREPARING TRAINING DATA")
        print("-" * 60)

        self.prepare_data()

        # Summary
        print("\n" + "=" * 60)
        print("COMPLETE!")
        print("=" * 60)
        print(f"\nFiles created in {config.symbol_dir}/")
        print(f"\nNext steps:")
        print(f"  python -m cli.finbert_sentiment.train -s {config.symbol}")
        print()

        return True
