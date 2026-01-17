"""Polygon (Massive) data providers."""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
import requests

from ..base import BaseProvider
from .config import (
    API_KEY,
    BASE_URL,
    ENDPOINTS,
)


class PolygonAggregatesProvider(BaseProvider):
    """
    Downloads aggregate bars (OHLCV) from Polygon.

    Usage:
        provider = PolygonAggregatesProvider("AAPL")
        provider.run()  # Gets last 7 days of daily bars

    Output: datasets/{SYMBOL}/polygon_aggregates.json
    """

    def __init__(
        self,
        symbol: str,
        multiplier: int = 1,
        timespan: str = "day",
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        limit: int = 5000,
    ) -> None:
        super().__init__(symbol)
        self.multiplier = multiplier
        self.timespan = timespan
        self.limit = limit

        # Default to last 7 days (free tier friendly)
        today = datetime.now()
        self.to_date = to_date or today.strftime("%Y-%m-%d")
        self.from_date = from_date or (today - timedelta(days=7)).strftime("%Y-%m-%d")

    @property
    def name(self) -> str:
        return "Polygon Aggregates"

    @property
    def output_filename(self) -> str:
        return ENDPOINTS["aggregates"]["output_file"]

    def download(self) -> List[Dict[str, Any]]:
        """Download aggregate bars."""
        print(f"  Fetching {self.multiplier}{self.timespan} bars for {self.symbol} ({self.from_date} to {self.to_date})...")

        url = f"{BASE_URL}/v2/aggs/ticker/{self.symbol}/range/{self.multiplier}/{self.timespan}/{self.from_date}/{self.to_date}"

        try:
            response = requests.get(
                url,
                params={
                    "apiKey": API_KEY,
                    "adjusted": "true",
                    "sort": "asc",
                    "limit": self.limit,
                },
                timeout=30,
            )

            if response.status_code != 200:
                print(f"  WARNING: API returned {response.status_code}: {response.text[:100]}")
                return []

            data = response.json()

            status = data.get("status")
            if status not in ("OK", "DELAYED"):
                print(f"  WARNING: {status}: {data.get('message', 'Unknown error')}")
                return []

            if status == "DELAYED":
                print(f"  Note: Data is delayed (free tier)")


            results = data.get("results", [])
            if not results:
                print(f"  WARNING: No data returned for {self.symbol}")
                return []

            # Convert to readable format (UTC with Z suffix)
            records = []
            for bar in results:
                ts = bar.get("t")
                utc_dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc) if ts else None
                records.append({
                    "timestamp": ts,
                    "datetime": utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ") if utc_dt else None,
                    "open": bar.get("o"),
                    "high": bar.get("h"),
                    "low": bar.get("l"),
                    "close": bar.get("c"),
                    "volume": bar.get("v"),
                    "vwap": bar.get("vw"),
                    "transactions": bar.get("n"),
                })

            return records

        except Exception as e:
            print(f"  WARNING: Could not fetch aggregates: {e}")
            return []


class PolygonPreviousDayProvider(BaseProvider):
    """
    Downloads previous day's bar from Polygon.

    Usage:
        provider = PolygonPreviousDayProvider("AAPL")
        provider.run()

    Output: datasets/{SYMBOL}/polygon_previous.json
    """

    def __init__(self, symbol: str) -> None:
        super().__init__(symbol)

    @property
    def name(self) -> str:
        return "Polygon Previous Day"

    @property
    def output_filename(self) -> str:
        return ENDPOINTS["previous"]["output_file"]

    def download(self) -> List[Dict[str, Any]]:
        """Download previous day's bar."""
        print(f"  Fetching previous day bar for {self.symbol}...")

        url = f"{BASE_URL}/v2/aggs/ticker/{self.symbol}/prev"

        try:
            response = requests.get(
                url,
                params={"apiKey": API_KEY, "adjusted": "true"},
                timeout=30,
            )

            if response.status_code != 200:
                print(f"  WARNING: API returned {response.status_code}: {response.text[:100]}")
                return []

            data = response.json()
            results = data.get("results", [])

            if not results:
                print(f"  WARNING: No previous day data for {self.symbol}")
                return []

            # Convert to readable format (UTC with Z suffix)
            records = []
            for bar in results:
                ts = bar.get("t")
                utc_dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc) if ts else None
                records.append({
                    "ticker": bar.get("T"),
                    "timestamp": ts,
                    "datetime": utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ") if utc_dt else None,
                    "open": bar.get("o"),
                    "high": bar.get("h"),
                    "low": bar.get("l"),
                    "close": bar.get("c"),
                    "volume": bar.get("v"),
                    "vwap": bar.get("vw"),
                    "transactions": bar.get("n"),
                })

            return records

        except Exception as e:
            print(f"  WARNING: Could not fetch previous day: {e}")
            return []


class PolygonTickerProvider(BaseProvider):
    """
    Downloads ticker details from Polygon.

    Usage:
        provider = PolygonTickerProvider("AAPL")
        provider.run()

    Output: datasets/{SYMBOL}/polygon_ticker.json
    """

    def __init__(self, symbol: str) -> None:
        super().__init__(symbol)

    @property
    def name(self) -> str:
        return "Polygon Ticker"

    @property
    def output_filename(self) -> str:
        return ENDPOINTS["ticker"]["output_file"]

    def download(self) -> List[Dict[str, Any]]:
        """Download ticker details."""
        print(f"  Fetching ticker details for {self.symbol}...")

        url = f"{BASE_URL}/v3/reference/tickers/{self.symbol}"

        try:
            response = requests.get(
                url,
                params={"apiKey": API_KEY},
                timeout=30,
            )

            if response.status_code != 200:
                print(f"  WARNING: API returned {response.status_code}: {response.text[:100]}")
                return []

            data = response.json()
            result = data.get("results", {})

            if not result:
                print(f"  WARNING: No ticker details for {self.symbol}")
                return []

            # Return as list for consistency
            return [result]

        except Exception as e:
            print(f"  WARNING: Could not fetch ticker details: {e}")
            return []


class PolygonNewsProvider(BaseProvider):
    """
    Downloads news articles from Polygon.

    Usage:
        provider = PolygonNewsProvider("AAPL")
        provider.run()

    Output: datasets/{SYMBOL}/polygon_news.json
    """

    def __init__(self, symbol: str, limit: int = 100) -> None:
        super().__init__(symbol)
        self.limit = min(limit, 1000)  # API max is 1000

    @property
    def name(self) -> str:
        return "Polygon News"

    @property
    def output_filename(self) -> str:
        return ENDPOINTS["news"]["output_file"]

    def download(self) -> List[Dict[str, Any]]:
        """Download news articles."""
        print(f"  Fetching up to {self.limit} news articles for {self.symbol}...")

        url = f"{BASE_URL}/v2/reference/news"

        try:
            response = requests.get(
                url,
                params={
                    "apiKey": API_KEY,
                    "ticker": self.symbol,
                    "limit": self.limit,
                    "order": "desc",
                    "sort": "published_utc",
                },
                timeout=30,
            )

            if response.status_code != 200:
                print(f"  WARNING: API returned {response.status_code}: {response.text[:100]}")
                return []

            data = response.json()
            results = data.get("results", [])

            if not results:
                print(f"  WARNING: No news found for {self.symbol}")
                return []

            # Simplify the news format
            records = []
            for article in results:
                records.append({
                    "id": article.get("id"),
                    "title": article.get("title"),
                    "author": article.get("author"),
                    "published_utc": article.get("published_utc"),
                    "article_url": article.get("article_url"),
                    "description": article.get("description"),
                    "tickers": article.get("tickers", []),
                    "keywords": article.get("keywords", []),
                    "publisher": article.get("publisher", {}).get("name"),
                })

            return records

        except Exception as e:
            print(f"  WARNING: Could not fetch news: {e}")
            return []


class PolygonProvider:
    """
    Convenience wrapper to download all Polygon data.

    Usage:
        provider = PolygonProvider("AAPL")
        provider.run()  # Downloads aggregates, previous day, ticker info, and news
    """

    def __init__(
        self,
        symbol: str,
        days: int = 7,
        timespan: str = "day",
        news_limit: int = 100,
    ) -> None:
        self.symbol = symbol.upper()

        today = datetime.now()
        from_date = (today - timedelta(days=days)).strftime("%Y-%m-%d")
        to_date = today.strftime("%Y-%m-%d")

        self.aggregates_provider = PolygonAggregatesProvider(
            symbol, multiplier=1, timespan=timespan,
            from_date=from_date, to_date=to_date
        )
        self.previous_provider = PolygonPreviousDayProvider(symbol)
        self.ticker_provider = PolygonTickerProvider(symbol)
        self.news_provider = PolygonNewsProvider(symbol, news_limit)

    def run(self) -> Dict[str, Any]:
        """Run all Polygon downloads."""
        return {
            "aggregates": self.aggregates_provider.run(),
            "previous": self.previous_provider.run(),
            "ticker": self.ticker_provider.run(),
            "news": self.news_provider.run(),
        }
