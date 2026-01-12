"""
Data schemas for hierarchical sentiment analysis.

This module defines the core data structures used throughout the pipeline.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Literal, Optional


@dataclass
class SentimentResult:
    """Result from FinBERT sentiment analysis."""
    score: float           # -1 (negative) to +1 (positive)
    confidence: float      # 0 to 1
    label: Literal["positive", "negative", "neutral"]


@dataclass
class TimeWindow:
    """Trading-hour aligned time window for aggregation."""
    start: datetime
    end: datetime

    @classmethod
    def from_timestamp(cls, timestamp: datetime) -> 'TimeWindow':
        """
        Create a trading-hour aligned window containing the given timestamp.
        Trading windows are 9:30 AM to 9:30 AM next day (for US markets).
        """
        # Get date components
        hour = timestamp.hour
        minute = timestamp.minute

        # If before 9:30 AM, window started previous day at 9:30 AM
        if hour < 9 or (hour == 9 and minute < 30):
            start_date = timestamp.date()
            # Go back one day
            from datetime import timedelta
            start_date = start_date - timedelta(days=1)
        else:
            start_date = timestamp.date()

        # Window is 9:30 AM to 9:30 AM next day
        start = datetime(start_date.year, start_date.month, start_date.day, 9, 30)
        from datetime import timedelta
        end = start + timedelta(days=1)

        return cls(start=start, end=end)


@dataclass
class NewsItem:
    """
    A news article with hierarchical classification and sentiment.

    This is the enriched schema that includes all fields needed for
    hierarchical sentiment analysis.
    """
    # === Core fields (from yfinance) ===
    headline: str
    summary: Optional[str] = None
    timestamp: Optional[datetime] = None
    source: Optional[str] = None
    url: Optional[str] = None

    # === Hierarchical classification ===
    level: Optional[Literal["MARKET", "SECTOR", "TICKER"]] = None

    # === Entity associations ===
    primary_ticker: Optional[str] = None        # For TICKER-level news
    sector_gics: Optional[str] = None           # GICS sector code
    related_tickers: List[str] = field(default_factory=list)  # For spillover analysis

    # === Sentiment scores (populated by FinBERT) ===
    sentiment_score: Optional[float] = None     # -1 to +1
    sentiment_confidence: Optional[float] = None  # 0 to 1
    sentiment_label: Optional[Literal["positive", "negative", "neutral"]] = None

    # === Price data (for training labels) ===
    price: Optional[float] = None               # Price at news time
    future_price: Optional[float] = None        # Price after interval
    price_change_pct: Optional[float] = None    # Percentage change

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'headline': self.headline,
            'summary': self.summary,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'source': self.source,
            'url': self.url,
            'level': self.level,
            'primary_ticker': self.primary_ticker,
            'sector_gics': self.sector_gics,
            'related_tickers': self.related_tickers,
            'sentiment_score': self.sentiment_score,
            'sentiment_confidence': self.sentiment_confidence,
            'sentiment_label': self.sentiment_label,
            'price': self.price,
            'future_price': self.future_price,
            'price_change_pct': self.price_change_pct,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'NewsItem':
        """Create from dictionary (JSON deserialization)."""
        timestamp = data.get('timestamp')
        if timestamp and isinstance(timestamp, str):
            # Parse ISO format timestamp
            try:
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except ValueError:
                timestamp = None

        return cls(
            headline=data.get('headline', ''),
            summary=data.get('summary'),
            timestamp=timestamp,
            source=data.get('source'),
            url=data.get('url'),
            level=data.get('level'),
            primary_ticker=data.get('primary_ticker'),
            sector_gics=data.get('sector_gics'),
            related_tickers=data.get('related_tickers', []),
            sentiment_score=data.get('sentiment_score'),
            sentiment_confidence=data.get('sentiment_confidence'),
            sentiment_label=data.get('sentiment_label'),
            price=data.get('price'),
            future_price=data.get('future_price'),
            price_change_pct=data.get('price_change_pct'),
        )


@dataclass
class TickerSentimentFeatures:
    """Aggregated ticker-level sentiment features."""
    mean_sentiment: float = 0.0
    sentiment_std: float = 0.0
    sentiment_skew: float = 0.0
    news_volume: int = 0
    sentiment_momentum_1d: float = 0.0
    sentiment_momentum_5d: float = 0.0
    max_sentiment: float = 0.0
    min_sentiment: float = 0.0
    positive_ratio: float = 0.0
    negative_ratio: float = 0.0
    news_recency_hours: float = 0.0
    source_diversity: float = 0.0


@dataclass
class SectorSentimentFeatures:
    """Aggregated sector-level sentiment features."""
    mean_sentiment: float = 0.0
    sentiment_breadth: float = 0.0  # % of sector tickers positive
    news_volume: int = 0
    sentiment_dispersion: float = 0.0
    leader_sentiment: float = 0.0   # Sentiment of sector leader by market cap
    sentiment_momentum: float = 0.0
    relative_strength: float = 0.0  # Sector vs market sentiment
    news_concentration: float = 0.0  # Herfindahl index of news distribution
    peer_sentiment: float = 0.0      # Average of top 5 peers by correlation
    laggard_sentiment: float = 0.0   # Sentiment of worst-performing peer


@dataclass
class MarketSentimentFeatures:
    """Aggregated market-level sentiment features."""
    sentiment_index: float = 0.0    # Overall market sentiment
    news_volume: int = 0
    fed_sentiment: float = 0.0      # Federal Reserve news sentiment
    economic_sentiment: float = 0.0  # Economic indicator news sentiment
    geopolitical_sentiment: float = 0.0  # Geopolitical news sentiment
    fear_index: float = 0.0         # Negative news concentration
    sentiment_momentum: float = 0.0
    breadth_sentiment: float = 0.0  # % of all tickers positive
    vix_level: Optional[float] = None  # Actual VIX if available
    market_regime: int = 0          # 0=bear, 1=neutral, 2=bull


@dataclass
class CrossLevelFeatures:
    """Cross-level deviation and correlation features."""
    ticker_vs_sector_deviation: float = 0.0
    ticker_vs_market_deviation: float = 0.0
    sector_vs_market_deviation: float = 0.0
    ticker_sector_correlation: float = 0.0
    ticker_market_beta: float = 0.0
    sentiment_divergence_score: float = 0.0
    relative_news_attention: float = 0.0
    sentiment_surprise: float = 0.0


@dataclass
class HierarchicalFeatureVector:
    """
    Complete feature vector for one ticker at one timestamp.
    Total dimensions: ~45-60 features depending on configuration.
    """
    ticker_features: TickerSentimentFeatures = field(default_factory=TickerSentimentFeatures)
    sector_features: SectorSentimentFeatures = field(default_factory=SectorSentimentFeatures)
    market_features: MarketSentimentFeatures = field(default_factory=MarketSentimentFeatures)
    cross_level_features: CrossLevelFeatures = field(default_factory=CrossLevelFeatures)

    def to_tensor_dict(self) -> dict:
        """Convert to dictionary of tensors for model input."""
        import torch

        ticker = [
            self.ticker_features.mean_sentiment,
            self.ticker_features.sentiment_std,
            self.ticker_features.sentiment_skew,
            self.ticker_features.news_volume,
            self.ticker_features.sentiment_momentum_1d,
            self.ticker_features.sentiment_momentum_5d,
            self.ticker_features.max_sentiment,
            self.ticker_features.min_sentiment,
            self.ticker_features.positive_ratio,
            self.ticker_features.negative_ratio,
            self.ticker_features.news_recency_hours,
            self.ticker_features.source_diversity,
        ]

        sector = [
            self.sector_features.mean_sentiment,
            self.sector_features.sentiment_breadth,
            self.sector_features.news_volume,
            self.sector_features.sentiment_dispersion,
            self.sector_features.leader_sentiment,
            self.sector_features.sentiment_momentum,
            self.sector_features.relative_strength,
            self.sector_features.news_concentration,
            self.sector_features.peer_sentiment,
            self.sector_features.laggard_sentiment,
        ]

        market = [
            self.market_features.sentiment_index,
            self.market_features.news_volume,
            self.market_features.fed_sentiment,
            self.market_features.economic_sentiment,
            self.market_features.geopolitical_sentiment,
            self.market_features.fear_index,
            self.market_features.sentiment_momentum,
            self.market_features.breadth_sentiment,
            self.market_features.vix_level or 0.0,
            self.market_features.market_regime,
        ]

        cross = [
            self.cross_level_features.ticker_vs_sector_deviation,
            self.cross_level_features.ticker_vs_market_deviation,
            self.cross_level_features.sector_vs_market_deviation,
            self.cross_level_features.ticker_sector_correlation,
            self.cross_level_features.ticker_market_beta,
            self.cross_level_features.sentiment_divergence_score,
            self.cross_level_features.relative_news_attention,
            self.cross_level_features.sentiment_surprise,
        ]

        return {
            'ticker': torch.tensor(ticker, dtype=torch.float32),
            'sector': torch.tensor(sector, dtype=torch.float32),
            'market': torch.tensor(market, dtype=torch.float32),
            'cross': torch.tensor(cross, dtype=torch.float32),
        }
