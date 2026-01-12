"""
News Level Classifier for hierarchical sentiment analysis.

This module classifies news articles into three levels:
- MARKET: Fed announcements, GDP, unemployment, geopolitical events
- SECTOR: Industry-wide news, regulatory changes affecting sectors
- TICKER: Company-specific earnings, management, products, lawsuits
"""

from typing import Dict, List, Literal, Optional, Set, Tuple, Union

from .sector_mapping import (
    SECTOR_KEYWORDS,
    get_sector,
    get_ticker_keywords,
)


# Market-level keywords (macroeconomic and geopolitical)
MARKET_KEYWORDS: Set[str] = {
    # Federal Reserve / Monetary Policy
    "fed", "fomc", "federal reserve", "interest rate", "rate hike", "rate cut",
    "monetary policy", "quantitative easing", "quantitative tightening",
    "jerome powell", "powell", "central bank", "fed chair", "fed meeting",
    "federal funds", "basis points", "bps",

    # Economic Indicators
    "gdp", "gross domestic product", "inflation", "cpi", "ppi",
    "unemployment", "jobless claims", "nonfarm payroll", "payrolls",
    "retail sales", "consumer confidence", "consumer sentiment",
    "pmi", "manufacturing index", "services index", "ism",
    "housing starts", "existing home sales", "new home sales",
    "durable goods", "industrial production", "capacity utilization",
    "trade balance", "trade deficit", "current account",

    # Recession / Growth
    "recession", "economic growth", "economic slowdown", "soft landing",
    "hard landing", "stagflation", "deflation", "hyperinflation",
    "yield curve", "inverted yield curve", "treasury yield",

    # Government / Fiscal Policy
    "treasury", "treasury secretary", "debt ceiling", "government shutdown",
    "fiscal policy", "stimulus", "infrastructure bill", "tax reform",
    "government spending", "budget deficit", "national debt",
    "janet yellen", "yellen",

    # Geopolitical
    "tariff", "trade war", "sanctions", "geopolitical", "war",
    "conflict", "military", "nato", "opec", "opec+",
    "china trade", "us-china", "russia", "ukraine",
    "middle east", "israel", "iran", "north korea",
    "election", "presidential", "congress", "senate", "house",

    # Market-wide events
    "stock market", "market crash", "bear market", "bull market",
    "market correction", "volatility", "vix", "fear index",
    "circuit breaker", "flash crash", "black monday", "black swan",
    "market rally", "risk-off", "risk-on", "flight to safety",
    "all-time high", "record high", "market selloff",

    # Global
    "global economy", "world bank", "imf", "international monetary fund",
    "eurozone", "ecb", "european central bank", "boe", "bank of england",
    "boj", "bank of japan", "pboc", "peoples bank of china",
    "emerging markets", "developed markets",
}

# Sector-level keywords (industry-wide, not company-specific)
SECTOR_LEVEL_KEYWORDS: Set[str] = {
    # Sector-wide terms
    "sector", "industry", "industry-wide", "across the sector",
    "sector stocks", "sector etf", "sector rotation",

    # Regulatory (affects whole sector)
    "regulation", "regulatory", "sec", "ftc", "antitrust",
    "legislation", "bill", "law", "compliance", "ruling",
    "court ruling", "supreme court", "appeals court",
    "class action", "settlement",

    # Sector-specific
    "tech sector", "technology sector", "tech stocks",
    "energy sector", "oil stocks", "energy stocks",
    "financial sector", "bank stocks", "banking sector",
    "healthcare sector", "pharma stocks", "biotech sector",
    "consumer sector", "retail sector", "retail stocks",
    "industrial sector", "manufacturing sector",
    "utilities sector", "utility stocks",
    "real estate sector", "reit", "reits",
    "materials sector", "basic materials",
    "communication sector", "telecom sector", "media stocks",

    # Industry trends
    "industry trends", "industry outlook", "sector outlook",
    "industry forecast", "sector forecast",
    "supply chain", "global supply", "chip shortage", "semiconductor shortage",
    "labor shortage", "wage pressure", "input costs",
    "commodity prices", "raw materials",
}


class NewsLevelClassifier:
    """
    Classifies news articles into hierarchical levels:
    - MARKET: Macroeconomic and geopolitical news
    - SECTOR: Industry-wide news affecting multiple companies
    - TICKER: Company-specific news

    Also filters irrelevant news that doesn't relate to the target ticker.
    """

    def __init__(self, target_ticker: str, strict_filtering: bool = True):
        """
        Initialize the classifier.

        Args:
            target_ticker: The ticker symbol we're analyzing (e.g., "AAPL")
            strict_filtering: If True, filter out news that doesn't mention
                            the ticker or related keywords
        """
        self.target_ticker = target_ticker.upper()
        self.strict_filtering = strict_filtering
        self.target_sector = get_sector(target_ticker)
        self.ticker_keywords = get_ticker_keywords(target_ticker)

    def classify(self, item: Dict) -> Optional[Literal["MARKET", "SECTOR", "TICKER"]]:
        """
        Classify a news article into a hierarchical level.

        Args:
            item: News article dict with 'headline' and optionally 'summary'

        Returns:
            - "MARKET" if macroeconomic/geopolitical news
            - "SECTOR" if industry-wide news
            - "TICKER" if company-specific news
            - None if the news is irrelevant (should be filtered out)
        """
        # Extract text content
        text = self._extract_text(item)
        if not text:
            return None

        text_lower = text.lower()

        # Step 1: Check relevance (for strict filtering)
        if self.strict_filtering and not self._is_relevant(text_lower):
            return None

        # Step 2: Classify into level
        return self._classify_level(text_lower)

    def classify_with_details(self, item: Dict) -> Tuple[
        Optional[Literal["MARKET", "SECTOR", "TICKER"]],
        Dict
    ]:
        """
        Classify with additional details about why the classification was made.

        Args:
            item: News article dict

        Returns:
            Tuple of (level, details_dict)
        """
        text = self._extract_text(item)
        if not text:
            return None, {"reason": "empty_text"}

        text_lower = text.lower()

        # Check relevance
        is_relevant, relevance_reason = self._check_relevance_with_reason(text_lower)
        if self.strict_filtering and not is_relevant:
            return None, {
                "reason": "irrelevant",
                "relevance_check": relevance_reason,
            }

        # Classify level
        level, level_reason = self._classify_level_with_reason(text_lower)

        return level, {
            "reason": level_reason,
            "is_relevant": is_relevant,
            "relevance_check": relevance_reason,
            "market_keyword_count": self._count_market_keywords(text_lower),
            "sector_keyword_count": self._count_sector_keywords(text_lower),
            "ticker_keyword_count": self._count_ticker_keywords(text_lower),
        }

    def _extract_text(self, item: Dict) -> str:
        """Extract headline and summary text from news item."""
        headline = ""
        summary = ""

        # Handle different news item formats
        if 'content' in item:
            content = item['content']
            headline = content.get('title', '') or content.get('headline', '')
            summary = content.get('summary', '') or content.get('description', '')
        else:
            headline = item.get('title', '') or item.get('headline', '')
            summary = item.get('summary', '') or item.get('description', '')

        return f"{headline} {summary}".strip()

    def _is_relevant(self, text_lower: str) -> bool:
        """Check if news is relevant to the target ticker."""
        # Check 1: Ticker symbol appears in text
        if self.target_ticker.lower() in text_lower:
            return True

        # Check 2: Any of the ticker's keywords appear
        for keyword in self.ticker_keywords:
            if keyword.lower() in text_lower:
                return True

        # Check 3: Market-level news is always relevant
        if self._count_market_keywords(text_lower) >= 2:
            return True

        # Check 4: Sector-level news for the same sector is relevant
        if self.target_sector:
            sector_keywords = SECTOR_KEYWORDS.get(self.target_sector, set())
            sector_match_count = sum(1 for kw in sector_keywords if kw in text_lower)
            if sector_match_count >= 2:
                return True

        return False

    def _check_relevance_with_reason(self, text_lower: str) -> Tuple[bool, str]:
        """Check relevance and return the reason."""
        # Check 1: Ticker symbol appears in text
        if self.target_ticker.lower() in text_lower:
            return True, f"ticker_symbol_match: {self.target_ticker}"

        # Check 2: Any of the ticker's keywords appear
        for keyword in self.ticker_keywords:
            if keyword.lower() in text_lower:
                return True, f"ticker_keyword_match: {keyword}"

        # Check 3: Market-level news is always relevant
        market_count = self._count_market_keywords(text_lower)
        if market_count >= 2:
            return True, f"market_keywords: {market_count}"

        # Check 4: Sector-level news for the same sector
        if self.target_sector:
            sector_keywords = SECTOR_KEYWORDS.get(self.target_sector, set())
            sector_match_count = sum(1 for kw in sector_keywords if kw in text_lower)
            if sector_match_count >= 2:
                return True, f"sector_keywords: {sector_match_count}"

        return False, "no_match"

    def _classify_level(self, text_lower: str) -> Literal["MARKET", "SECTOR", "TICKER"]:
        """Classify the news level based on keyword analysis."""
        market_count = self._count_market_keywords(text_lower)
        sector_count = self._count_sector_keywords(text_lower)
        ticker_count = self._count_ticker_keywords(text_lower)

        # If strong market signal, classify as MARKET
        if market_count >= 2 and market_count > sector_count and market_count > ticker_count:
            return "MARKET"

        # If strong sector signal (and not just one company), classify as SECTOR
        if sector_count >= 2 and sector_count > ticker_count:
            return "SECTOR"

        # Default to TICKER (company-specific)
        return "TICKER"

    def _classify_level_with_reason(
        self, text_lower: str
    ) -> Tuple[Literal["MARKET", "SECTOR", "TICKER"], str]:
        """Classify with reason."""
        market_count = self._count_market_keywords(text_lower)
        sector_count = self._count_sector_keywords(text_lower)
        ticker_count = self._count_ticker_keywords(text_lower)

        if market_count >= 2 and market_count > sector_count and market_count > ticker_count:
            return "MARKET", f"market_keywords_dominant: {market_count}"

        if sector_count >= 2 and sector_count > ticker_count:
            return "SECTOR", f"sector_keywords_dominant: {sector_count}"

        return "TICKER", f"ticker_default: ticker={ticker_count}, sector={sector_count}, market={market_count}"

    def _count_market_keywords(self, text_lower: str) -> int:
        """Count market-level keywords in text."""
        return sum(1 for kw in MARKET_KEYWORDS if kw in text_lower)

    def _count_sector_keywords(self, text_lower: str) -> int:
        """Count sector-level keywords in text."""
        count = 0
        for kw in SECTOR_LEVEL_KEYWORDS:
            if kw in text_lower:
                count += 1

        # Also check sector-specific keywords
        if self.target_sector and self.target_sector in SECTOR_KEYWORDS:
            for kw in SECTOR_KEYWORDS[self.target_sector]:
                if kw in text_lower:
                    count += 1

        return count

    def _count_ticker_keywords(self, text_lower: str) -> int:
        """Count ticker-level keywords in text."""
        count = 0

        # Check for ticker symbol
        if self.target_ticker.lower() in text_lower:
            count += 2  # Weight ticker symbol higher

        # Check for ticker keywords
        for keyword in self.ticker_keywords:
            if keyword.lower() in text_lower:
                count += 1

        return count

    def filter_and_classify_batch(
        self, items: List[Dict]
    ) -> Tuple[List[Dict], Dict[str, Union[int, float]]]:
        """
        Filter and classify a batch of news items.

        Args:
            items: List of news article dicts

        Returns:
            Tuple of (classified_items, stats)
            - classified_items: List of items with 'level' field added
            - stats: Dictionary with filtering/classification statistics
        """
        stats: Dict[str, Union[int, float]] = {
            "total": len(items),
            "filtered_out": 0,
            "market": 0,
            "sector": 0,
            "ticker": 0,
        }

        classified = []
        for item in items:
            level = self.classify(item)

            if level is None:
                stats["filtered_out"] += 1
                continue

            # Add level to item
            item_copy = item.copy()
            item_copy['level'] = level
            classified.append(item_copy)

            # Update stats
            level_key = level.lower()
            if level_key in stats:
                current = stats[level_key]
                if isinstance(current, int):
                    stats[level_key] = current + 1

        stats["relevant"] = len(classified)
        relevance = round((len(classified) / len(items) * 100), 1) if items else 0.0
        stats["relevance_rate"] = relevance

        return classified, stats


def create_classifier(target_ticker: str, strict: bool = True) -> NewsLevelClassifier:
    """
    Factory function to create a news classifier.

    Args:
        target_ticker: The ticker symbol to analyze
        strict: Whether to use strict filtering

    Returns:
        Configured NewsLevelClassifier instance
    """
    return NewsLevelClassifier(target_ticker, strict_filtering=strict)
