"""
AI/ML infrastructure for the trading platform.

This module provides reusable AI components that can be used by any strategy:
- embeddings: Text to vector models (Gemma)
- sentiment: Sentiment analysis (FinBERT)
- summarizers: Report summarization (Flan-T5)
- llm: Large language models (future: Ollama)
"""

from ai.embeddings.gemma import GemmaEmbedding, get_gemma_embedding
from ai.sentiment.finbert import FinBERTSentimentAnalyzer, get_sentiment_analyzer
from ai.sentiment.base import SentimentResult
from ai.summarizers.flan_t5 import FlanT5Summarizer, create_summarizer

__all__ = [
    # Embeddings
    "GemmaEmbedding",
    "get_gemma_embedding",
    # Sentiment
    "FinBERTSentimentAnalyzer",
    "get_sentiment_analyzer",
    "SentimentResult",
    # Summarizers
    "FlanT5Summarizer",
    "create_summarizer",
]
