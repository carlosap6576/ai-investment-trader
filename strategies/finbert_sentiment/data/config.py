"""Configuration for data downloading and preparation."""

from dataclasses import dataclass
from strategies.finbert_sentiment.constants import DATASETS_DIR

# =============================================================================
# DOWNLOAD DEFAULTS
# =============================================================================

DEFAULT_PERIOD = "1mo"
DEFAULT_INTERVAL = "5m"
DEFAULT_NEWS_COUNT = 100
DEFAULT_NO_FILTER = False
DEFAULT_NO_SENTIMENT = False

# =============================================================================
# VALID OPTIONS
# =============================================================================

VALID_PERIODS = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]

INTERVAL_SECONDS = {
    "1m": 60,
    "2m": 120,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "60m": 3600,
    "90m": 5400,
    "1h": 3600,
    "1d": 86400,
    "5d": 432000,
    "1wk": 604800,
    "1mo": 2592000,
    "3mo": 7776000,
}

VALID_INTERVALS = list(INTERVAL_SECONDS.keys())

PERIOD_TO_DAYS = {
    "1d": 1,
    "5d": 5,
    "1mo": 30,
    "3mo": 90,
    "6mo": 180,
    "1y": 365,
    "2y": 730,
    "5y": 1825,
    "10y": 3650,
    "ytd": 365,
    "max": 99999,
}

INTERVAL_MAX_DAYS = {
    "1m": 7,
    "2m": 60,
    "5m": 60,
    "15m": 60,
    "30m": 60,
    "60m": 730,
    "90m": 730,
    "1h": 730,
    "1d": 99999,
    "5d": 99999,
    "1wk": 99999,
    "1mo": 99999,
    "3mo": 99999,
}

# =============================================================================
# EXTERNAL API CONFIGURATIONS
# =============================================================================

API_SOURCES = {
    "financialdatasets": {
        "base_url": "https://api.financialdatasets.ai",
        "news_endpoint": "/news/",
        "output_file": "financialdataset_news.json",
    },
    "seekingalpha": {
        "base_url": "https://seekingalpha.com",
        "news_endpoint": "/api/v3/symbols/{symbol}/news",
        "output_file": "seekingalpha_news.json",
        "params": "filter[category]=news_card&filter[until]=0&include=author,primaryTickers,secondaryTickers,sentiments,otherTags&page[number]=1&page[size]=1000",
    },
    "seekingalpha_dividends": {
        "base_url": "https://seekingalpha.com",
        "endpoint": "/api/v3/symbols/{symbol}/dividend_history",
        "output_file": "seekingalpha_dividends.json",
        "params": "years=5",
    },
    "alphavantage": {
        "base_url": "https://www.alphavantage.co",
        "endpoint": "/query",
        "output_file": "alphavantage_quote.json",
        "api_key": "TQ2FAYKUZX90VN81",
        "function": "GLOBAL_QUOTE",
    },
    "alphavantage_news": {
        "base_url": "https://www.alphavantage.co",
        "endpoint": "/query",
        "output_file": "alphavantage_news_{symbol}.json",
        "api_key": "TQ2FAYKUZX90VN81",
        "function": "NEWS_SENTIMENT",
        "limit": 1000,
    },
    "alphavantage_news_market": {
        "base_url": "https://www.alphavantage.co",
        "endpoint": "/query",
        "output_file": "alphavantage_news_market.json",
        "api_key": "TQ2FAYKUZX90VN81",
        "function": "NEWS_SENTIMENT",
        "limit": 1000,
    },
    "alphavantage_overview": {
        "base_url": "https://www.alphavantage.co",
        "endpoint": "/query",
        "output_file": "alphavantage_overview.json",
        "api_key": "TQ2FAYKUZX90VN81",
        "function": "OVERVIEW",
    },
    "alphavantage_income": {
        "base_url": "https://www.alphavantage.co",
        "endpoint": "/query",
        "output_file": "alphavantage_income.json",
        "api_key": "TQ2FAYKUZX90VN81",
        "function": "INCOME_STATEMENT",
    },
    "alphavantage_balance": {
        "base_url": "https://www.alphavantage.co",
        "endpoint": "/query",
        "output_file": "alphavantage_balance.json",
        "api_key": "TQ2FAYKUZX90VN81",
        "function": "BALANCE_SHEET",
    },
    "alphavantage_cashflow": {
        "base_url": "https://www.alphavantage.co",
        "endpoint": "/query",
        "output_file": "alphavantage_cashflow.json",
        "api_key": "TQ2FAYKUZX90VN81",
        "function": "CASH_FLOW",
    },
    "alphavantage_shares": {
        "base_url": "https://www.alphavantage.co",
        "endpoint": "/query",
        "output_file": "alphavantage_shares.json",
        "api_key": "TQ2FAYKUZX90VN81",
        "function": "SHARES_OUTSTANDING",
    },
}


# =============================================================================
# CONFIGURATION CLASS
# =============================================================================

@dataclass
class DownloadConfig:
    """Configuration for data download."""
    symbol: str
    period: str = DEFAULT_PERIOD
    interval: str = DEFAULT_INTERVAL
    news_count: int = DEFAULT_NEWS_COUNT
    no_filter: bool = DEFAULT_NO_FILTER
    no_sentiment: bool = DEFAULT_NO_SENTIMENT

    def __post_init__(self):
        self.symbol = self.symbol.upper()
        self.interval_seconds = INTERVAL_SECONDS[self.interval]
        self.symbol_dir = f"{DATASETS_DIR}/{self.symbol}"
        self.historical_data_file = f"{self.symbol_dir}/historical_data.json"
        self.news_file = f"{self.symbol_dir}/news.json"
        self.training_data_file = f"{self.symbol_dir}/news_with_price.json"


# =============================================================================
# VALIDATION
# =============================================================================

def validate_period_interval(period: str, interval: str) -> tuple:
    """
    Validate that the period is compatible with the interval.

    Returns: (is_valid, error_message)
    """
    period_days = PERIOD_TO_DAYS.get(period, 0)
    max_days = INTERVAL_MAX_DAYS.get(interval, 99999)

    if period_days > max_days:
        suggestions = [p for p, days in PERIOD_TO_DAYS.items()
                       if days <= max_days and p in VALID_PERIODS][:5]
        suggestion_str = ", ".join(suggestions)

        return False, (
            f"Invalid combination: {interval} data is only available for "
            f"the last {max_days} days, but {period} is ~{period_days} days. "
            f"Valid periods for {interval}: {suggestion_str}"
        )

    return True, None
