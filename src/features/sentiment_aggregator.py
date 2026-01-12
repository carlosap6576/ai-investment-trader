"""
Sentiment Aggregator for hierarchical sentiment analysis.

Aggregates sentiment scores across time windows and hierarchical levels
(MARKET, SECTOR, TICKER) to create feature vectors for the model.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..data.schemas import (
    CrossLevelFeatures,
    HierarchicalFeatureVector,
    MarketSentimentFeatures,
    NewsItem,
    SectorSentimentFeatures,
    TickerSentimentFeatures,
    TimeWindow,
)


@dataclass
class SentimentStats:
    """Statistical summary of sentiment scores."""
    mean: float = 0.0
    std: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    count: int = 0
    positive_ratio: float = 0.0
    negative_ratio: float = 0.0
    skew: float = 0.0


def calculate_stats(scores: List[float]) -> SentimentStats:
    """Calculate statistical summary of sentiment scores."""
    if not scores:
        return SentimentStats()

    arr = np.array(scores)
    n = len(arr)

    mean = float(np.mean(arr))
    std = float(np.std(arr)) if n > 1 else 0.0
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))

    positive_ratio = float(np.sum(arr > 0) / n) if n > 0 else 0.0
    negative_ratio = float(np.sum(arr < 0) / n) if n > 0 else 0.0

    # Calculate skewness
    if n > 2 and std > 0:
        skew = float(np.mean(((arr - mean) / std) ** 3))
    else:
        skew = 0.0

    return SentimentStats(
        mean=mean,
        std=std,
        min_val=min_val,
        max_val=max_val,
        count=n,
        positive_ratio=positive_ratio,
        negative_ratio=negative_ratio,
        skew=skew,
    )


def calculate_momentum(current_mean: float, previous_mean: float) -> float:
    """Calculate sentiment momentum (change from previous period)."""
    return current_mean - previous_mean


class SentimentAggregator:
    """
    Aggregates sentiment by level and time window.

    Supports trading-hour aligned windows (9:30 AM to 9:30 AM next day)
    and calculates various sentiment features for each level.
    """

    def __init__(
        self,
        target_ticker: str,
        target_sector: Optional[str] = None,
        window_hours: int = 24,
    ):
        """
        Initialize the aggregator.

        Args:
            target_ticker: The main ticker symbol we're analyzing
            target_sector: The GICS sector code for the ticker
            window_hours: Size of the aggregation window in hours
        """
        self.target_ticker = target_ticker.upper()
        self.target_sector = target_sector
        self.window_hours = window_hours

        # Storage for news items by level
        self._ticker_news: List[NewsItem] = []
        self._sector_news: List[NewsItem] = []
        self._market_news: List[NewsItem] = []

    def add_news(self, news_items: List[NewsItem]) -> None:
        """
        Add news items to the aggregator.

        Items are sorted into MARKET, SECTOR, or TICKER buckets based on their level.
        """
        for item in news_items:
            if item.level == "MARKET":
                self._market_news.append(item)
            elif item.level == "SECTOR":
                self._sector_news.append(item)
            elif item.level == "TICKER":
                self._ticker_news.append(item)

    def clear(self) -> None:
        """Clear all stored news."""
        self._ticker_news.clear()
        self._sector_news.clear()
        self._market_news.clear()

    def get_news_in_window(
        self,
        news_list: List[NewsItem],
        window: TimeWindow,
    ) -> List[NewsItem]:
        """Filter news to only those within the time window."""
        return [
            item for item in news_list
            if item.timestamp and window.start <= item.timestamp < window.end
        ]

    def aggregate_ticker_features(
        self,
        window: TimeWindow,
        previous_window: Optional[TimeWindow] = None,
    ) -> TickerSentimentFeatures:
        """
        Aggregate ticker-level sentiment features.

        Args:
            window: Current time window
            previous_window: Previous window for momentum calculation

        Returns:
            TickerSentimentFeatures with 12 features
        """
        current_news = self.get_news_in_window(self._ticker_news, window)
        scores = [n.sentiment_score for n in current_news if n.sentiment_score is not None]
        stats = calculate_stats(scores)

        # Calculate momentum if previous window provided
        momentum_1d = 0.0
        momentum_5d = 0.0
        if previous_window:
            prev_news = self.get_news_in_window(self._ticker_news, previous_window)
            prev_scores = [n.sentiment_score for n in prev_news if n.sentiment_score is not None]
            prev_stats = calculate_stats(prev_scores)
            momentum_1d = calculate_momentum(stats.mean, prev_stats.mean)

        # Calculate news recency (hours since most recent news)
        news_recency_hours = 0.0
        if current_news:
            timestamps = [n.timestamp for n in current_news if n.timestamp]
            if timestamps:
                most_recent = max(timestamps)
                news_recency_hours = (window.end - most_recent).total_seconds() / 3600

        # Calculate source diversity (unique sources / total)
        sources = [n.source for n in current_news if n.source]
        source_diversity = len(set(sources)) / len(sources) if sources else 0.0

        return TickerSentimentFeatures(
            mean_sentiment=stats.mean,
            sentiment_std=stats.std,
            sentiment_skew=stats.skew,
            news_volume=stats.count,
            sentiment_momentum_1d=momentum_1d,
            sentiment_momentum_5d=momentum_5d,  # Would need 5-day history
            max_sentiment=stats.max_val,
            min_sentiment=stats.min_val,
            positive_ratio=stats.positive_ratio,
            negative_ratio=stats.negative_ratio,
            news_recency_hours=news_recency_hours,
            source_diversity=source_diversity,
        )

    def aggregate_sector_features(
        self,
        window: TimeWindow,
        peer_sentiments: Optional[Dict[str, float]] = None,
    ) -> SectorSentimentFeatures:
        """
        Aggregate sector-level sentiment features.

        Args:
            window: Current time window
            peer_sentiments: Optional dict of peer ticker -> sentiment

        Returns:
            SectorSentimentFeatures with 10 features
        """
        current_news = self.get_news_in_window(self._sector_news, window)
        scores = [n.sentiment_score for n in current_news if n.sentiment_score is not None]
        stats = calculate_stats(scores)

        # Sentiment breadth (% of sector tickers with positive sentiment)
        sentiment_breadth = 0.0
        if peer_sentiments:
            positive_peers = sum(1 for s in peer_sentiments.values() if s > 0)
            sentiment_breadth = positive_peers / len(peer_sentiments)

        # Sentiment dispersion (std of peer sentiments)
        sentiment_dispersion = 0.0
        if peer_sentiments:
            peer_values = list(peer_sentiments.values())
            sentiment_dispersion = float(np.std(peer_values)) if len(peer_values) > 1 else 0.0

        # Leader sentiment (would need market cap data)
        leader_sentiment = stats.mean  # Placeholder

        # Peer sentiment (average of top 5 peers by correlation)
        peer_sentiment = 0.0
        if peer_sentiments:
            sorted_peers = sorted(peer_sentiments.values(), reverse=True)[:5]
            peer_sentiment = float(np.mean(sorted_peers)) if sorted_peers else 0.0

        # Laggard sentiment (worst-performing peer)
        laggard_sentiment = 0.0
        if peer_sentiments:
            laggard_sentiment = min(peer_sentiments.values())

        # News concentration (Herfindahl index of news distribution)
        news_concentration = 0.0
        if current_news:
            source_counts = defaultdict(int)
            for n in current_news:
                source_counts[n.source or "unknown"] += 1
            total = sum(source_counts.values())
            if total > 0:
                shares = [(c / total) ** 2 for c in source_counts.values()]
                news_concentration = sum(shares)

        return SectorSentimentFeatures(
            mean_sentiment=stats.mean,
            sentiment_breadth=sentiment_breadth,
            news_volume=stats.count,
            sentiment_dispersion=sentiment_dispersion,
            leader_sentiment=leader_sentiment,
            sentiment_momentum=0.0,  # Would need historical data
            relative_strength=0.0,  # Would need market comparison
            news_concentration=news_concentration,
            peer_sentiment=peer_sentiment,
            laggard_sentiment=laggard_sentiment,
        )

    def aggregate_market_features(
        self,
        window: TimeWindow,
        vix_level: Optional[float] = None,
    ) -> MarketSentimentFeatures:
        """
        Aggregate market-level sentiment features.

        Args:
            window: Current time window
            vix_level: Optional VIX level if available

        Returns:
            MarketSentimentFeatures with 10 features
        """
        current_news = self.get_news_in_window(self._market_news, window)
        scores = [n.sentiment_score for n in current_news if n.sentiment_score is not None]
        stats = calculate_stats(scores)

        # Categorize market news by topic
        fed_scores = []
        econ_scores = []
        geopolitical_scores = []

        fed_keywords = {"fed", "fomc", "powell", "interest rate", "monetary"}
        econ_keywords = {"gdp", "inflation", "unemployment", "jobs", "cpi", "ppi"}
        geo_keywords = {"tariff", "war", "sanctions", "geopolitical", "election"}

        for item in current_news:
            if item.sentiment_score is None:
                continue
            text = f"{item.headline or ''} {item.summary or ''}".lower()

            if any(kw in text for kw in fed_keywords):
                fed_scores.append(item.sentiment_score)
            elif any(kw in text for kw in econ_keywords):
                econ_scores.append(item.sentiment_score)
            elif any(kw in text for kw in geo_keywords):
                geopolitical_scores.append(item.sentiment_score)

        fed_sentiment = float(np.mean(fed_scores)) if fed_scores else 0.0
        economic_sentiment = float(np.mean(econ_scores)) if econ_scores else 0.0
        geopolitical_sentiment = float(np.mean(geopolitical_scores)) if geopolitical_scores else 0.0

        # Fear index (concentration of negative news)
        fear_index = stats.negative_ratio

        # Market regime (0=bear, 1=neutral, 2=bull)
        if stats.mean < -0.2:
            market_regime = 0
        elif stats.mean > 0.2:
            market_regime = 2
        else:
            market_regime = 1

        return MarketSentimentFeatures(
            sentiment_index=stats.mean,
            news_volume=stats.count,
            fed_sentiment=fed_sentiment,
            economic_sentiment=economic_sentiment,
            geopolitical_sentiment=geopolitical_sentiment,
            fear_index=fear_index,
            sentiment_momentum=0.0,  # Would need historical data
            breadth_sentiment=stats.positive_ratio - stats.negative_ratio,
            vix_level=vix_level,
            market_regime=market_regime,
        )

    def aggregate_cross_level_features(
        self,
        ticker_features: TickerSentimentFeatures,
        sector_features: SectorSentimentFeatures,
        market_features: MarketSentimentFeatures,
    ) -> CrossLevelFeatures:
        """
        Calculate cross-level deviation and correlation features.

        These features capture how the ticker sentiment differs from
        sector and market sentiment, which can be predictive signals.

        Args:
            ticker_features: Ticker-level features
            sector_features: Sector-level features
            market_features: Market-level features

        Returns:
            CrossLevelFeatures with 8 features
        """
        # Deviations
        ticker_vs_sector = ticker_features.mean_sentiment - sector_features.mean_sentiment
        ticker_vs_market = ticker_features.mean_sentiment - market_features.sentiment_index
        sector_vs_market = sector_features.mean_sentiment - market_features.sentiment_index

        # Sentiment divergence score (how much ticker differs from both)
        divergence = abs(ticker_vs_sector) + abs(ticker_vs_market)

        # Relative news attention (ticker volume vs sector volume)
        relative_attention = 0.0
        if sector_features.news_volume > 0:
            relative_attention = ticker_features.news_volume / sector_features.news_volume

        # Sentiment surprise (ticker momentum vs market momentum)
        sentiment_surprise = ticker_features.sentiment_momentum_1d - market_features.sentiment_momentum

        return CrossLevelFeatures(
            ticker_vs_sector_deviation=ticker_vs_sector,
            ticker_vs_market_deviation=ticker_vs_market,
            sector_vs_market_deviation=sector_vs_market,
            ticker_sector_correlation=0.0,  # Would need time series
            ticker_market_beta=0.0,  # Would need historical calculation
            sentiment_divergence_score=divergence,
            relative_news_attention=relative_attention,
            sentiment_surprise=sentiment_surprise,
        )

    def build_feature_vector(
        self,
        window: TimeWindow,
        previous_window: Optional[TimeWindow] = None,
        peer_sentiments: Optional[Dict[str, float]] = None,
        vix_level: Optional[float] = None,
    ) -> HierarchicalFeatureVector:
        """
        Build a complete hierarchical feature vector.

        Args:
            window: Current time window for aggregation
            previous_window: Previous window for momentum features
            peer_sentiments: Dict of peer ticker -> sentiment for sector features
            vix_level: Optional VIX level for market features

        Returns:
            HierarchicalFeatureVector with all 40 features
        """
        ticker_features = self.aggregate_ticker_features(window, previous_window)
        sector_features = self.aggregate_sector_features(window, peer_sentiments)
        market_features = self.aggregate_market_features(window, vix_level)
        cross_features = self.aggregate_cross_level_features(
            ticker_features, sector_features, market_features
        )

        return HierarchicalFeatureVector(
            ticker_features=ticker_features,
            sector_features=sector_features,
            market_features=market_features,
            cross_level_features=cross_features,
        )


def create_aggregator(
    target_ticker: str,
    target_sector: Optional[str] = None,
) -> SentimentAggregator:
    """
    Factory function to create a sentiment aggregator.

    Args:
        target_ticker: The ticker symbol to analyze
        target_sector: Optional GICS sector code

    Returns:
        Configured SentimentAggregator instance
    """
    return SentimentAggregator(
        target_ticker=target_ticker,
        target_sector=target_sector,
    )
