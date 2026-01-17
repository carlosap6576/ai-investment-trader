"""Alpha Vantage provider configuration."""

# API credentials
API_KEY = "TQ2FAYKUZX90VN81"
BASE_URL = "https://www.alphavantage.co"

# API endpoints
ENDPOINTS = {
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
