"""
Feature Builder for hierarchical sentiment analysis.

Builds temporal sequences of HierarchicalFeatureVector for the transformer model.
Handles normalization and sequence padding.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..data.schemas import (
    HierarchicalFeatureVector,
    NewsItem,
    TimeWindow,
)
from .sentiment_aggregator import SentimentAggregator


@dataclass
class NormalizationStats:
    """Statistics for feature normalization."""
    mean: np.ndarray
    std: np.ndarray
    min_vals: np.ndarray
    max_vals: np.ndarray


class FeatureNormalizer:
    """
    Normalizes sentiment features using z-score or min-max normalization.

    Tracks running statistics to normalize new data consistently.
    """

    def __init__(self, method: str = "zscore"):
        """
        Initialize normalizer.

        Args:
            method: Normalization method ("zscore" or "minmax")
        """
        if method not in ("zscore", "minmax"):
            raise ValueError(f"Unknown normalization method: {method}")

        self.method = method
        self._fitted = False
        self._stats: Optional[NormalizationStats] = None

    def fit(self, features: np.ndarray) -> None:
        """
        Compute normalization statistics from training data.

        Args:
            features: Array of shape (n_samples, n_features)
        """
        if features.ndim != 2:
            raise ValueError(f"Expected 2D array, got {features.ndim}D")

        self._stats = NormalizationStats(
            mean=np.mean(features, axis=0),
            std=np.std(features, axis=0),
            min_vals=np.min(features, axis=0),
            max_vals=np.max(features, axis=0),
        )

        # Avoid division by zero
        self._stats.std = np.where(
            self._stats.std < 1e-8,
            1.0,
            self._stats.std
        )
        range_vals = self._stats.max_vals - self._stats.min_vals
        range_vals = np.where(range_vals < 1e-8, 1.0, range_vals)

        self._fitted = True

    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features using fitted statistics.

        Args:
            features: Array of shape (n_samples, n_features) or (n_features,)

        Returns:
            Normalized features
        """
        if not self._fitted or self._stats is None:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")

        if self.method == "zscore":
            return (features - self._stats.mean) / self._stats.std
        else:  # minmax
            range_vals = self._stats.max_vals - self._stats.min_vals
            range_vals = np.where(range_vals < 1e-8, 1.0, range_vals)
            return (features - self._stats.min_vals) / range_vals

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(features)
        return self.transform(features)

    def inverse_transform(self, normalized: np.ndarray) -> np.ndarray:
        """Convert normalized features back to original scale."""
        if not self._fitted or self._stats is None:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")

        if self.method == "zscore":
            return normalized * self._stats.std + self._stats.mean
        else:  # minmax
            range_vals = self._stats.max_vals - self._stats.min_vals
            return normalized * range_vals + self._stats.min_vals

    def get_stats(self) -> Optional[Dict]:
        """Get normalization statistics as dictionary."""
        if self._stats is None:
            return None
        return {
            "mean": self._stats.mean.tolist(),
            "std": self._stats.std.tolist(),
            "min": self._stats.min_vals.tolist(),
            "max": self._stats.max_vals.tolist(),
        }


class TemporalFeatureBuilder:
    """
    Builds temporal sequences of features for the transformer model.

    Creates sliding windows of feature vectors, handling padding for
    sequences shorter than the target length.
    """

    def __init__(
        self,
        target_ticker: str,
        target_sector: Optional[str] = None,
        sequence_length: int = 20,
        window_hours: int = 24,
    ):
        """
        Initialize the feature builder.

        Args:
            target_ticker: Ticker symbol to build features for
            target_sector: GICS sector code
            sequence_length: Number of time steps in each sequence
            window_hours: Size of each aggregation window in hours
        """
        self.target_ticker = target_ticker.upper()
        self.target_sector = target_sector
        self.sequence_length = sequence_length
        self.window_hours = window_hours

        # Create aggregator
        self.aggregator = SentimentAggregator(
            target_ticker=target_ticker,
            target_sector=target_sector,
            window_hours=window_hours,
        )

        # Normalizers for each feature group
        self.normalizers: Dict[str, FeatureNormalizer] = {
            "ticker": FeatureNormalizer(method="zscore"),
            "sector": FeatureNormalizer(method="zscore"),
            "market": FeatureNormalizer(method="zscore"),
            "cross": FeatureNormalizer(method="zscore"),
        }

    def build_features_for_window(
        self,
        news_items: List[NewsItem],
        window: TimeWindow,
        previous_window: Optional[TimeWindow] = None,
        peer_sentiments: Optional[Dict[str, float]] = None,
        vix_level: Optional[float] = None,
    ) -> HierarchicalFeatureVector:
        """
        Build a single feature vector for one time window.

        Args:
            news_items: All news items (will be filtered by window)
            window: The time window to aggregate
            previous_window: Previous window for momentum features
            peer_sentiments: Dict of peer ticker -> sentiment
            vix_level: Optional VIX level

        Returns:
            HierarchicalFeatureVector for this window
        """
        # Clear and add news
        self.aggregator.clear()
        self.aggregator.add_news(news_items)

        # Build feature vector
        return self.aggregator.build_feature_vector(
            window=window,
            previous_window=previous_window,
            peer_sentiments=peer_sentiments,
            vix_level=vix_level,
        )

    def build_temporal_sequence(
        self,
        news_items: List[NewsItem],
        end_time: datetime,
        peer_sentiments: Optional[Dict[str, float]] = None,
        vix_level: Optional[float] = None,
    ) -> List[HierarchicalFeatureVector]:
        """
        Build a temporal sequence of feature vectors.

        Args:
            news_items: All available news items
            end_time: End time of the sequence
            peer_sentiments: Dict of peer ticker -> sentiment
            vix_level: Optional VIX level

        Returns:
            List of sequence_length feature vectors
        """
        sequence: List[HierarchicalFeatureVector] = []
        window_delta = timedelta(hours=self.window_hours)

        # Build windows backwards from end_time
        for i in range(self.sequence_length):
            # Calculate window boundaries
            window_end = end_time - (i * window_delta)
            window_start = window_end - window_delta
            window = TimeWindow(start=window_start, end=window_end)

            # Previous window for momentum
            prev_window_end = window_start
            prev_window_start = prev_window_end - window_delta
            prev_window = TimeWindow(start=prev_window_start, end=prev_window_end)

            # Build features for this window
            features = self.build_features_for_window(
                news_items=news_items,
                window=window,
                previous_window=prev_window,
                peer_sentiments=peer_sentiments,
                vix_level=vix_level,
            )
            sequence.append(features)

        # Reverse so oldest is first, newest is last
        sequence.reverse()
        return sequence

    def sequence_to_array(
        self,
        sequence: List[HierarchicalFeatureVector],
    ) -> np.ndarray:
        """
        Convert a sequence of feature vectors to a numpy array.

        Args:
            sequence: List of HierarchicalFeatureVector

        Returns:
            Array of shape (sequence_length, total_features)
            where total_features = 12 + 10 + 10 + 8 = 40
        """
        arrays = []
        for fv in sequence:
            tensor_dict = fv.to_tensor_dict()
            # Concatenate all feature groups
            combined = np.concatenate([
                tensor_dict['ticker'].numpy(),
                tensor_dict['sector'].numpy(),
                tensor_dict['market'].numpy(),
                tensor_dict['cross'].numpy(),
            ])
            arrays.append(combined)

        return np.array(arrays)

    def build_training_data(
        self,
        news_items: List[NewsItem],
        label_func: Callable[[NewsItem], int],
    ) -> Tuple[np.ndarray, np.ndarray, List[datetime]]:
        """
        Build training data from news items.

        Args:
            news_items: All news items with price data
            label_func: Function that takes NewsItem and returns label (0, 1, or 2)

        Returns:
            Tuple of (features, labels, timestamps)
            - features: Array of shape (n_samples, sequence_length, n_features)
            - labels: Array of shape (n_samples,)
            - timestamps: List of end timestamps for each sample
        """
        # Sort news by timestamp
        items_with_time = [n for n in news_items if n.timestamp is not None]
        items_with_time.sort(key=lambda x: x.timestamp)  # type: ignore

        if not items_with_time:
            return np.array([]), np.array([]), []

        # Find unique time points (where we have price labels)
        label_items = [n for n in items_with_time if n.price_change_pct is not None]
        if not label_items:
            return np.array([]), np.array([]), []

        features_list = []
        labels_list = []
        timestamps_list = []

        for item in label_items:
            if item.timestamp is None:
                continue

            # Build sequence ending at this item's timestamp
            sequence = self.build_temporal_sequence(
                news_items=items_with_time,
                end_time=item.timestamp,
            )

            # Convert to array
            seq_array = self.sequence_to_array(sequence)
            features_list.append(seq_array)

            # Get label
            label = label_func(item)
            labels_list.append(label)
            timestamps_list.append(item.timestamp)

        if not features_list:
            return np.array([]), np.array([]), []

        features = np.array(features_list)
        labels = np.array(labels_list)

        return features, labels, timestamps_list

    def fit_normalizers(self, features: np.ndarray) -> None:
        """
        Fit normalizers on training features.

        Args:
            features: Array of shape (n_samples, sequence_length, n_features)
        """
        # Reshape to (n_samples * sequence_length, n_features)
        _, _, n_features = features.shape
        flat_features = features.reshape(-1, n_features)

        # Split by feature groups
        # ticker: 12, sector: 10, market: 10, cross: 8
        ticker_features = flat_features[:, :12]
        sector_features = flat_features[:, 12:22]
        market_features = flat_features[:, 22:32]
        cross_features = flat_features[:, 32:]

        # Fit each normalizer
        self.normalizers["ticker"].fit(ticker_features)
        self.normalizers["sector"].fit(sector_features)
        self.normalizers["market"].fit(market_features)
        self.normalizers["cross"].fit(cross_features)

    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features using fitted normalizers.

        Args:
            features: Array of shape (n_samples, sequence_length, n_features)
                     or (sequence_length, n_features)

        Returns:
            Normalized features with same shape
        """
        if features.ndim == 2:
            features = features[np.newaxis, :, :]
            squeeze = True
        else:
            squeeze = False

        n_samples, seq_len, _ = features.shape
        normalized = np.zeros_like(features)

        for i in range(n_samples):
            for j in range(seq_len):
                row = features[i, j]

                # Split and normalize each group
                ticker_norm = self.normalizers["ticker"].transform(row[:12])
                sector_norm = self.normalizers["sector"].transform(row[12:22])
                market_norm = self.normalizers["market"].transform(row[22:32])
                cross_norm = self.normalizers["cross"].transform(row[32:])

                # Recombine
                normalized[i, j] = np.concatenate([
                    ticker_norm, sector_norm, market_norm, cross_norm
                ])

        if squeeze:
            normalized = normalized[0]

        return normalized


def create_feature_builder(
    target_ticker: str,
    target_sector: Optional[str] = None,
    sequence_length: int = 20,
) -> TemporalFeatureBuilder:
    """
    Factory function to create a feature builder.

    Args:
        target_ticker: Ticker symbol
        target_sector: Optional GICS sector code
        sequence_length: Number of time steps per sequence

    Returns:
        Configured TemporalFeatureBuilder instance
    """
    return TemporalFeatureBuilder(
        target_ticker=target_ticker,
        target_sector=target_sector,
        sequence_length=sequence_length,
    )


def create_label_function(
    buy_threshold: float = 1.0,
    sell_threshold: float = -1.0,
) -> Callable[[NewsItem], int]:
    """
    Create a label function for training data.

    Args:
        buy_threshold: Price change % above which label is BUY (2)
        sell_threshold: Price change % below which label is SELL (0)

    Returns:
        Function that takes NewsItem and returns 0 (SELL), 1 (HOLD), or 2 (BUY)
    """
    def label_func(item: NewsItem) -> int:
        if item.price_change_pct is None:
            return 1  # Default to HOLD

        pct = item.price_change_pct
        if pct < sell_threshold:
            return 0  # SELL
        elif pct > buy_threshold:
            return 2  # BUY
        else:
            return 1  # HOLD

    return label_func
