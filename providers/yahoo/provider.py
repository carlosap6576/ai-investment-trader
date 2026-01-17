"""Yahoo Finance data providers."""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List

from ..base import BaseProvider
from .config import (
    ENDPOINTS,
    DEFAULT_PERIOD,
    DEFAULT_INTERVAL,
)


class YahooPriceProvider(BaseProvider):
    """
    Downloads historical price data from Yahoo Finance via yfinance.

    Output: datasets/{SYMBOL}/yahoo_historical_data.json
    Format: [{timestamp, datetime, open, high, low, close, volume}, ...]
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
        return ENDPOINTS["prices"]["output_file"]

    def download(self) -> List[Dict[str, Any]]:
        """Download price data. Returns list of OHLCV records."""
        print(f"  Fetching {self.interval} price data for last {self.period}...")

        data = self.yf.download(
            tickers=self.symbol,
            period=self.period,
            interval=self.interval,
            progress=False,
        )

        if data.empty:
            print(f"  WARNING: No price data returned for {self.symbol}")
            return []

        # Convert DataFrame to list of OHLCV records
        records = []
        for idx, row in data.iterrows():
            # Handle both single-ticker and multi-ticker column formats
            try:
                # Multi-ticker format: ('Open', 'AAPL')
                open_price = row[('Open', self.symbol)]
                high_price = row[('High', self.symbol)]
                low_price = row[('Low', self.symbol)]
                close_price = row[('Close', self.symbol)]
                volume = row[('Volume', self.symbol)]
            except KeyError:
                # Single-ticker format: 'Open'
                open_price = row['Open']
                high_price = row['High']
                low_price = row['Low']
                close_price = row['Close']
                volume = row['Volume']

            # Convert timestamp to milliseconds (UTC)
            timestamp_ms = int(idx.timestamp() * 1000)

            # Convert to UTC datetime string with Z suffix for consistency
            utc_dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)

            record = {
                "timestamp": timestamp_ms,
                "datetime": utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "open": round(float(open_price), 4) if open_price == open_price else None,
                "high": round(float(high_price), 4) if high_price == high_price else None,
                "low": round(float(low_price), 4) if low_price == low_price else None,
                "close": round(float(close_price), 4) if close_price == close_price else None,
                "volume": int(volume) if volume == volume else None,
            }
            records.append(record)

        return records

    def load_existing(self) -> List[Dict[str, Any]]:
        """Load existing price data."""
        try:
            with open(self.output_path, 'r') as f:
                data = json.load(f)
                # Handle legacy dict format for backwards compatibility
                if isinstance(data, dict):
                    return self._convert_legacy_format(data)
                return data
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _convert_legacy_format(self, legacy_data: Dict[str, float]) -> List[Dict[str, Any]]:
        """Convert old {timestamp: price} format to new array format."""
        records = []
        for ts_str, price in legacy_data.items():
            timestamp_ms = int(ts_str)
            utc_dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
            records.append({
                "timestamp": timestamp_ms,
                "datetime": utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "open": round(float(price), 4) if price is not None else None,
                "high": None,
                "low": None,
                "close": None,
                "volume": None,
            })
        return sorted(records, key=lambda x: x["timestamp"])

    def merge_with_existing(self, new_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge new prices with existing, new takes precedence."""
        existing = self.load_existing()

        # Build lookup of new timestamps
        new_timestamps = {r["timestamp"] for r in new_data}

        # Keep existing records not in new data
        merged = list(new_data)
        for record in existing:
            if record["timestamp"] not in new_timestamps:
                merged.append(record)

        # Sort by timestamp
        merged.sort(key=lambda x: x["timestamp"])
        return merged

    def save(self, data: List[Dict[str, Any]]) -> str:
        """Save price data."""
        self.ensure_directory()
        with open(self.output_path, 'w') as f:
            json.dump(data, f, indent=2)
        return self.output_path


class YahooNewsProvider(BaseProvider):
    """
    Downloads news articles from Yahoo Finance via yfinance.

    Output: datasets/{SYMBOL}/yahoo_news.json
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
        return ENDPOINTS["news"]["output_file"]

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
