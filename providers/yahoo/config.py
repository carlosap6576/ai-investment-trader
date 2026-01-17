"""Yahoo Finance provider configuration."""

# API endpoints and output files
ENDPOINTS = {
    "prices": {
        "output_file": "yahoo_historical_data.json",
    },
    "news": {
        "output_file": "yahoo_news.json",
    },
}

# Valid intervals for Yahoo Finance
VALID_INTERVALS = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
VALID_PERIODS = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]

# Defaults
DEFAULT_PERIOD = "1mo"
DEFAULT_INTERVAL = "5m"
