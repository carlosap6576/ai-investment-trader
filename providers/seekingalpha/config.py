"""Seeking Alpha provider configuration."""

# API base URL
BASE_URL = "https://seekingalpha.com"

# API endpoints
ENDPOINTS = {
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
