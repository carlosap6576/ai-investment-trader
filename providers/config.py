"""Configuration for data providers."""

DATASETS_DIR = "datasets"

# =============================================================================
# ALPHAVANTAGE
# =============================================================================

ALPHAVANTAGE_API_KEY = "TQ2FAYKUZX90VN81"
ALPHAVANTAGE_BASE_URL = "https://www.alphavantage.co"

ALPHAVANTAGE_ENDPOINTS = {
    "quote": {
        "function": "GLOBAL_QUOTE",
        "output_file": "alphavantage_quote.json",
    },
    "news": {
        "function": "NEWS_SENTIMENT",
        "output_file": "alphavantage_news_{symbol}.json",
        "limit": 1000,
    },
    "news_market": {
        "function": "NEWS_SENTIMENT",
        "output_file": "alphavantage_news_market.json",
        "limit": 1000,
    },
    "overview": {
        "function": "OVERVIEW",
        "output_file": "alphavantage_overview.json",
    },
    "income": {
        "function": "INCOME_STATEMENT",
        "output_file": "alphavantage_income.json",
    },
    "balance": {
        "function": "BALANCE_SHEET",
        "output_file": "alphavantage_balance.json",
    },
    "cashflow": {
        "function": "CASH_FLOW",
        "output_file": "alphavantage_cashflow.json",
    },
    "shares": {
        "function": "SHARES_OUTSTANDING",
        "output_file": "alphavantage_shares.json",
    },
}

# =============================================================================
# SEEKINGALPHA
# =============================================================================

SEEKINGALPHA_BASE_URL = "https://seekingalpha.com"

SEEKINGALPHA_ENDPOINTS = {
    "news": {
        "path": "/api/v3/symbols/{symbol}/news",
        "params": "filter[category]=news_card&filter[until]=0&include=author,primaryTickers,secondaryTickers,sentiments,otherTags&page[number]=1&page[size]=1000",
        "output_file": "seekingalpha_news.json",
    },
    "dividends": {
        "path": "/api/v3/symbols/{symbol}/dividend_history",
        "params": "years=5",
        "output_file": "seekingalpha_dividends.json",
    },
}

# =============================================================================
# FINANCIALDATASETS
# =============================================================================

FINANCIALDATASETS_BASE_URL = "https://api.financialdatasets.ai"

FINANCIALDATASETS_ENDPOINTS = {
    "news": {
        "path": "/news/",
        "output_file": "financialdataset_news.json",
    },
}

# =============================================================================
# YAHOO FINANCE
# =============================================================================

YAHOO_ENDPOINTS = {
    "prices": {
        "output_file": "historical_data.json",
    },
    "news": {
        "output_file": "news.json",
    },
}

# Valid intervals for Yahoo Finance
VALID_INTERVALS = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
VALID_PERIODS = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]

DEFAULT_PERIOD = "1mo"
DEFAULT_INTERVAL = "5m"
