"""
FinBERT Sentiment Analyzer for financial text.

FinBERT is a BERT model fine-tuned on financial text (Financial PhraseBank + Reuters TRC2).
It provides more accurate sentiment analysis on financial news compared to generic models
because it properly handles financial jargon (e.g., "liability", "exposure" are neutral in finance).
"""

import hashlib
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ai.sentiment.base import BaseSentimentAnalyzer, SentimentResult


class FinBERTSentimentAnalyzer(BaseSentimentAnalyzer):
    """
    FinBERT-based sentiment analyzer for financial news.

    Uses ProsusAI/finbert model which is trained specifically for financial text.
    Provides sentiment scores on a -1 to +1 scale.
    """

    MODEL_NAME = "ProsusAI/finbert"
    LABELS = ["positive", "negative", "neutral"]

    def __init__(
        self,
        device: Optional[str] = None,
        cache_enabled: bool = True,
        max_length: int = 512,
    ):
        """
        Initialize the FinBERT sentiment analyzer.

        Args:
            device: Device to run model on ("cuda", "mps", "cpu", or None for auto).
            cache_enabled: Whether to cache sentiment results.
            max_length: Maximum token length for inputs.
        """
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.max_length = max_length
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, SentimentResult] = {}

        print(f"  Loading FinBERT model ({self.MODEL_NAME})...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()
        print(f"  FinBERT loaded on {self.device}")

    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of a single text.

        Args:
            text: Text to analyze (headline or summary).

        Returns:
            SentimentResult with score, confidence, and label.
        """
        cache_key = self._get_cache_key(text)
        if self.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        probs = probs.cpu()

        # Calculate sentiment score: P(positive) - P(negative)
        score = float(probs[0].item()) - float(probs[1].item())

        pred_idx = int(probs.argmax().item())
        label = self.LABELS[pred_idx]
        confidence = float(probs[pred_idx].item())

        probabilities = {
            self.LABELS[i]: float(probs[i].item())
            for i in range(len(self.LABELS))
        }

        result = SentimentResult(
            score=score,
            confidence=confidence,
            label=label,
            probabilities=probabilities,
        )

        if self.cache_enabled:
            self._cache[cache_key] = result

        return result

    def analyze_batch(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[SentimentResult]:
        """
        Analyze sentiment of multiple texts efficiently.

        Args:
            texts: List of texts to analyze.
            batch_size: Number of texts to process at once.

        Returns:
            List of SentimentResult for each text.
        """
        uncached_texts = []
        uncached_indices = []
        cached_results: Dict[int, SentimentResult] = {}

        if self.cache_enabled:
            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text)
                if cache_key in self._cache:
                    cached_results[i] = self._cache[cache_key]
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))

        uncached_results = []
        for i in range(0, len(uncached_texts), batch_size):
            batch = uncached_texts[i:i + batch_size]
            batch_results = self._process_batch(batch)
            uncached_results.extend(batch_results)

            if self.cache_enabled:
                for text, result in zip(batch, batch_results):
                    cache_key = self._get_cache_key(text)
                    self._cache[cache_key] = result

        results_list: List[Optional[SentimentResult]] = [None] * len(texts)
        for i, result in cached_results.items():
            results_list[i] = result
        for i, result in zip(uncached_indices, uncached_results):
            results_list[i] = result

        return [r for r in results_list if r is not None]

    def _process_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Process a batch of texts."""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            all_probs = torch.softmax(outputs.logits, dim=-1).cpu()

        results: List[SentimentResult] = []
        for probs in all_probs:
            score = float((probs[0] - probs[1]).item())
            pred_idx = int(probs.argmax().item())
            label = self.LABELS[pred_idx]
            confidence = float(probs[pred_idx].item())

            probabilities = {
                self.LABELS[i]: float(probs[i].item())
                for i in range(len(self.LABELS))
            }

            results.append(SentimentResult(
                score=score,
                confidence=confidence,
                label=label,
                probabilities=probabilities,
            ))

        return results

    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()

    def clear_cache(self):
        """Clear the sentiment cache."""
        self._cache.clear()

    @property
    def cache_size(self) -> int:
        """Return the number of cached results."""
        return len(self._cache)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self._cache),
            "device": self.device,
        }


# Singleton instance for reuse
_analyzer_instance: Optional[FinBERTSentimentAnalyzer] = None


def get_sentiment_analyzer(
    device: Optional[str] = None,
    cache_enabled: bool = True,
) -> FinBERTSentimentAnalyzer:
    """
    Get or create the global sentiment analyzer instance.

    Uses a singleton pattern to avoid loading the model multiple times.

    Args:
        device: Device to run on (only used on first call).
        cache_enabled: Whether to enable caching (only used on first call).

    Returns:
        FinBERTSentimentAnalyzer instance.
    """
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = FinBERTSentimentAnalyzer(
            device=device,
            cache_enabled=cache_enabled,
        )
    return _analyzer_instance
