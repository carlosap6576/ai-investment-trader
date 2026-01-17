"""FinancialDatasets.ai data provider."""

import json
import urllib.error
import urllib.request
from typing import Any, Dict, List

from ..base import BaseProvider
from .config import (
    BASE_URL,
    ENDPOINTS,
)


class FinancialDatasetsProvider(BaseProvider):
    """
    Downloads news articles from FinancialDatasets.ai API.

    Output: datasets/{SYMBOL}/financialdataset_news.json

    The API returns news with pre-computed sentiment scores.
    Data is normalized to a consistent format.
    """

    def __init__(self, symbol: str) -> None:
        super().__init__(symbol)
        self.endpoint_config = ENDPOINTS["news"]

    @property
    def name(self) -> str:
        return "FinancialDatasets"

    @property
    def output_filename(self) -> str:
        return self.endpoint_config["output_file"]

    def download(self) -> List[Dict[str, Any]]:
        """Download and normalize news from FinancialDatasets.ai."""
        path = self.endpoint_config["path"]
        url = f"{BASE_URL}{path}?ticker={self.symbol}"

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

            raw_news = data.get('news', [])

            if not raw_news:
                print(f"  WARNING: No news returned")
                return []

            # Normalize to consistent format
            normalized = []
            for item in raw_news:
                normalized.append({
                    'title': item.get('title', ''),
                    'summary': '',
                    'pubDate': item.get('date', ''),
                    'source': item.get('source', ''),
                    'author': item.get('author', ''),
                    'url': item.get('url', ''),
                    'image_url': item.get('image_url', ''),
                    'api_sentiment': item.get('sentiment', ''),
                    'ticker': item.get('ticker', self.symbol),
                    '_source_api': 'financialdatasets',
                })

            print(f"  Fetched {len(normalized)} articles")
            return normalized

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
