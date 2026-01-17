"""Alpha Vantage data provider."""

import json
import urllib.error
import urllib.request
from typing import Any, Dict, List

from ..base import BaseProvider
from .config import (
    API_KEY,
    BASE_URL,
    ENDPOINTS,
)


class AlphaVantageProvider(BaseProvider):
    """
    Downloads data from Alpha Vantage API.

    Supports multiple endpoints:
        - quote: Real-time quote
        - news: Ticker-specific news sentiment
        - news_market: Market-wide news sentiment
        - overview: Company overview
        - income: Income statement
        - balance: Balance sheet
        - cashflow: Cash flow statement
        - shares: Shares outstanding

    Usage:
        provider = AlphaVantageProvider("AAPL", endpoint="quote")
        result = provider.run()
    """

    def __init__(self, symbol: str, endpoint: str = "quote") -> None:
        """
        Initialize AlphaVantage provider.

        Args:
            symbol: Trading symbol
            endpoint: One of: quote, news, news_market, overview, income, balance, cashflow, shares
        """
        super().__init__(symbol)
        if endpoint not in ENDPOINTS:
            raise ValueError(f"Unknown endpoint: {endpoint}. Valid: {list(ENDPOINTS.keys())}")
        self.endpoint = endpoint
        self.endpoint_config = ENDPOINTS[endpoint]

    @property
    def name(self) -> str:
        return f"AlphaVantage {self.endpoint}"

    @property
    def output_filename(self) -> str:
        filename = self.endpoint_config["output_file"]
        return filename.replace("{symbol}", self.symbol)

    def _build_url(self) -> str:
        """Build the API URL for the endpoint."""
        function = self.endpoint_config["function"]
        url = f"{BASE_URL}/query?function={function}&apikey={API_KEY}"

        # Add symbol for non-market endpoints
        if self.endpoint not in ["news_market"]:
            if self.endpoint == "news":
                url += f"&tickers={self.symbol}"
            else:
                url += f"&symbol={self.symbol}"

        # Add limit for news endpoints
        if "limit" in self.endpoint_config:
            url += f"&limit={self.endpoint_config['limit']}"

        return url

    def download(self) -> List[Dict[str, Any]]:
        """Download data from AlphaVantage. Returns list with single result dict."""
        url = self._build_url()

        try:
            req = urllib.request.Request(
                url,
                headers={
                    'User-Agent': 'ai-investment-trader/1.0',
                    'Accept': 'application/json',
                }
            )

            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode('utf-8'))

            # Log what we got
            if self.endpoint == "quote" and 'Global Quote' in data:
                print(f"  Fetched quote data")
            elif self.endpoint in ["news", "news_market"] and 'feed' in data:
                print(f"  Fetched {len(data.get('feed', []))} news articles")
            elif self.endpoint == "overview" and 'Symbol' in data:
                print(f"  Fetched company overview")
            elif self.endpoint in ["income", "balance", "cashflow"]:
                reports = data.get('annualReports', []) + data.get('quarterlyReports', [])
                print(f"  Fetched {len(reports)} financial reports")
            elif self.endpoint == "shares":
                print(f"  Fetched shares data")
            else:
                print(f"  WARNING: Unexpected response format")

            return [data]

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

    def save(self, data: List[Dict[str, Any]]) -> str:
        """Save data - unwrap the single-item list."""
        self.ensure_directory()
        # AlphaVantage returns single objects, not arrays
        save_data = data[0] if data else {}
        with open(self.output_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        return self.output_path

    def merge_with_existing(self, new_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """AlphaVantage data is replaced, not merged."""
        return new_data


class AlphaVantageAllProvider:
    """
    Convenience wrapper to download Alpha Vantage data for a symbol.

    Note: Only quote endpoint is enabled to avoid rate limit errors.
    Other endpoints (news, overview, financials) require premium API.

    Usage:
        provider = AlphaVantageAllProvider("AAPL")
        results = provider.run()
    """

    # Only quote endpoint - others hit rate limits on free tier
    ENDPOINTS = ["quote"]

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol.upper()

    def run(self) -> Dict[str, Any]:
        """Run AlphaVantage downloads."""
        results = {}
        for endpoint in self.ENDPOINTS:
            provider = AlphaVantageProvider(self.symbol, endpoint)
            results[endpoint] = provider.run()
        return results
