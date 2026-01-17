"""Yahoo Finance data providers."""

import json
from typing import Any, Dict, List

from providers.base import BaseProvider
from providers.config import (
    YAHOO_ENDPOINTS,
    DEFAULT_PERIOD,
    DEFAULT_INTERVAL,
)


class YahooPriceProvider(BaseProvider):
    """
    Downloads historical price data from Yahoo Finance via yfinance.

    Output: datasets/{SYMBOL}/historical_data.json
    Format: {timestamp: price, ...}
    """

    def __init__(
        self,
        symbol: str,
        period: str = DEFAULT_PERIOD,
        interval: str = DEFAULT_INTERVAL,
    ) -> None:
        super().__init__(symbol)
        self.period = period
        self.interval = interval
        self._yf = None

    @property
    def yf(self):
        """Lazy load yfinance."""
        if self._yf is None:
            import yfinance
            self._yf = yfinance
        return self._yf

    @property
    def name(self) -> str:
        return "Yahoo Prices"

    @property
    def output_filename(self) -> str:
        return YAHOO_ENDPOINTS["prices"]["output_file"]

    def download(self) -> Dict[str, float]:
        """Download price data. Returns dict of timestamp -> price."""
        print(f"  Fetching {self.interval} price data for last {self.period}...")

        data = self.yf.download(
            tickers=self.symbol,
            period=self.period,
            interval=self.interval,
            progress=False,
        )

        if data.empty:
            print(f"  WARNING: No price data returned for {self.symbol}")
            return {}

        # Convert to JSON format
        encoded = data.to_json()
        decoded = json.loads(encoded)

        # Extract open prices (handle different yfinance formats)
        price_key = f"('Open', '{self.symbol}')"
        prices = decoded.get(price_key, {})

        # Fallback for single-ticker format
        if not prices and 'Open' in decoded:
            prices = decoded['Open']

        return prices

    def load_existing(self) -> Dict[str, float]:
        """Load existing price data."""
        try:
            with open(self.output_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def merge_with_existing(self, new_data: Dict[str, float]) -> Dict[str, float]:
        """Merge new prices with existing, new takes precedence."""
        existing = self.load_existing()
        # Keep existing timestamps not in new data
        for timestamp, price in existing.items():
            if timestamp not in new_data:
                new_data[timestamp] = price
        return new_data

    def save(self, data: Dict[str, float]) -> str:
        """Save price data."""
        self.ensure_directory()
        with open(self.output_path, 'w') as f:
            json.dump(data, f, indent=2)
        return self.output_path


class YahooNewsProvider(BaseProvider):
    """
    Downloads news articles from Yahoo Finance via yfinance.

    Output: datasets/{SYMBOL}/news.json
    Format: [{title, link, pubDate, ...}, ...]
    """

    def __init__(self, symbol: str, count: int = 100) -> None:
        super().__init__(symbol)
        self.count = count
        self._yf = None

    @property
    def yf(self):
        """Lazy load yfinance."""
        if self._yf is None:
            import yfinance
            self._yf = yfinance
        return self._yf

    @property
    def name(self) -> str:
        return "Yahoo News"

    @property
    def output_filename(self) -> str:
        return YAHOO_ENDPOINTS["news"]["output_file"]

    def download(self) -> List[Dict[str, Any]]:
        """Download news articles."""
        print(f"  Fetching up to {self.count} news articles...")

        ticker = self.yf.Ticker(self.symbol)

        try:
            # Try new API first (yfinance >= 0.2.40)
            news = ticker.get_news(count=self.count)
        except TypeError:
            # Fall back to old API
            try:
                news = ticker.get_news()
            except Exception as e:
                print(f"  WARNING: Could not fetch news: {e}")
                news = []
        except Exception as e:
            print(f"  WARNING: Could not fetch news: {e}")
            news = []

        return news if news else []


class YahooProvider:
    """
    Convenience wrapper to download all Yahoo Finance data.

    Usage:
        provider = YahooProvider("AAPL")
        provider.run()  # Downloads both prices and news
    """

    def __init__(
        self,
        symbol: str,
        period: str = DEFAULT_PERIOD,
        interval: str = DEFAULT_INTERVAL,
        news_count: int = 100,
    ) -> None:
        self.symbol = symbol.upper()
        self.price_provider = YahooPriceProvider(symbol, period, interval)
        self.news_provider = YahooNewsProvider(symbol, news_count)

    def run(self) -> Dict[str, Any]:
        """Run all Yahoo downloads."""
        return {
            "prices": self.price_provider.run(),
            "news": self.news_provider.run(),
        }
