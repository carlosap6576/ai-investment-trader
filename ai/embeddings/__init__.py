"""Text embedding models."""

from ai.embeddings.base import BaseEmbedding
from ai.embeddings.gemma import GemmaEmbedding, get_gemma_embedding

__all__ = [
    "BaseEmbedding",
    "GemmaEmbedding",
    "get_gemma_embedding",
]
