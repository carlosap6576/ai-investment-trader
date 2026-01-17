"""Nasdaq Data Link provider configuration."""

# API credentials
API_KEY = "kZwt_6FeNfJHKKayu8w2"

# Available databases
DATABASES = {
    "wiki": "WIKI",      # EOD stock prices (free, data through 2018)
    "zacks": "ZACKS",    # Fundamentals
    "fred": "FRED",      # Economic indicators
}

# API endpoints
ENDPOINTS = {
    "timeseries": {"output_file": "nasdaq_{database}_{symbol}.json"},
    "table": {"output_file": "nasdaq_table_{table}.json"},
    "prices": {"output_file": "nasdaq_prices.json"},
}
