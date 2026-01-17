"""Seeking Alpha data provider."""

import json
import urllib.error
import urllib.request
from typing import Any, Dict, List

from providers.base import BaseProvider
from providers.config import (
    SEEKINGALPHA_BASE_URL,
    SEEKINGALPHA_ENDPOINTS,
)


class SeekingAlphaNewsProvider(BaseProvider):
    """
    Downloads news articles from Seeking Alpha API.

    Output: datasets/{SYMBOL}/seekingalpha_news.json
    """

    def __init__(self, symbol: str) -> None:
        super().__init__(symbol)
        self.endpoint_config = SEEKINGALPHA_ENDPOINTS["news"]

    @property
    def name(self) -> str:
        return "SeekingAlpha News"

    @property
    def output_filename(self) -> str:
        return self.endpoint_config["output_file"]

    def download(self) -> List[Dict[str, Any]]:
        """Download news from Seeking Alpha."""
        path = self.endpoint_config["path"].replace("{symbol}", self.symbol.lower())
        params = self.endpoint_config["params"]
        url = f"{SEEKINGALPHA_BASE_URL}{path}?id={self.symbol.lower()}&{params}"

        try:
            req = urllib.request.Request(
                url,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'application/json',
                }
            )

            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode('utf-8'))

            news = data.get('data', [])

            if news:
                print(f"  Fetched {len(news)} articles")
            else:
                print(f"  WARNING: No news returned")

            return news

        except urllib.error.HTTPError as e:
            print(f"  WARNING: HTTP error: {e.code} {e.reason}")
            return []
        except urllib.error.URLError as e:
            print(f"  WARNING: Could not connect: {e.reason}")
            return []
        except json.JSONDecodeError as e:
            print(f"  WARNING: Invalid JSON: {e}")
            return []
        except Exception as e:
            print(f"  WARNING: Error: {e}")
            return []


class SeekingAlphaDividendsProvider(BaseProvider):
    """
    Downloads dividend history from Seeking Alpha API.

    Output: datasets/{SYMBOL}/seekingalpha_dividends.json
    """

    def __init__(self, symbol: str) -> None:
        super().__init__(symbol)
        self.endpoint_config = SEEKINGALPHA_ENDPOINTS["dividends"]

    @property
    def name(self) -> str:
        return "SeekingAlpha Dividends"

    @property
    def output_filename(self) -> str:
        return self.endpoint_config["output_file"]

    def download(self) -> List[Dict[str, Any]]:
        """Download dividend history from Seeking Alpha."""
        path = self.endpoint_config["path"].replace("{symbol}", self.symbol.lower())
        params = self.endpoint_config["params"]
        url = f"{SEEKINGALPHA_BASE_URL}{path}?{params}"

        try:
            req = urllib.request.Request(
                url,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'application/json',
                }
            )

            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode('utf-8'))

            dividends = data.get('data', [])

            if dividends:
                print(f"  Fetched {len(dividends)} dividend records")
            else:
                print(f"  WARNING: No dividends returned")

            return dividends

        except urllib.error.HTTPError as e:
            print(f"  WARNING: HTTP error: {e.code} {e.reason}")
            return []
        except urllib.error.URLError as e:
            print(f"  WARNING: Could not connect: {e.reason}")
            return []
        except json.JSONDecodeError as e:
            print(f"  WARNING: Invalid JSON: {e}")
            return []
        except Exception as e:
            print(f"  WARNING: Error: {e}")
            return []


class SeekingAlphaProvider:
    """
    Convenience wrapper to download all Seeking Alpha data.

    Usage:
        provider = SeekingAlphaProvider("AAPL")
        results = provider.run()
    """

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol.upper()
        self.news_provider = SeekingAlphaNewsProvider(symbol)
        self.dividends_provider = SeekingAlphaDividendsProvider(symbol)

    def run(self) -> Dict[str, Any]:
        """Run all SeekingAlpha downloads."""
        return {
            "news": self.news_provider.run(),
            "dividends": self.dividends_provider.run(),
        }
