"""Polygon (Massive) provider configuration."""

# API credentials
API_KEY = "jdJglwDYWJqx7pZeiLN8_PIuKy8o_mT8"
BASE_URL = "https://api.polygon.io"

# API endpoints
ENDPOINTS = {
    "aggregates": {"output_file": "polygon_aggregates.json"},
    "previous": {"output_file": "polygon_previous.json"},
    "ticker": {"output_file": "polygon_ticker.json"},
    "news": {"output_file": "polygon_news.json"},
}

# Valid timespans for aggregates
TIMESPANS = ["minute", "hour", "day", "week", "month", "quarter", "year"]
