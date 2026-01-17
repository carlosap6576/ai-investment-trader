"""
Base interface for sentiment analyzers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class SentimentResult:
    """Result from sentiment analysis."""
    score: float           # -1 (negative) to +1 (positive)
    confidence: float      # 0 to 1
    label: str             # "positive", "negative", or "neutral"
    probabilities: Dict[str, float]  # Full probability distribution


class BaseSentimentAnalyzer(ABC):
    """Base class for sentiment analyzers."""

    @abstractmethod
    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of a single text.

        Args:
            text: Text to analyze.

        Returns:
            SentimentResult with score, confidence, and label.
        """
        pass

    @abstractmethod
    def analyze_batch(self, texts: List[str], batch_size: int = 32) -> List[SentimentResult]:
        """
        Analyze sentiment of multiple texts efficiently.

        Args:
            texts: List of texts to analyze.
            batch_size: Number of texts to process at once.

        Returns:
            List of SentimentResult for each text.
        """
        pass
