"""
Gemma Embedding Model.

Uses Google's embeddinggemma-300m model for text embeddings.
This is a lightweight embedding model suitable for financial text.
"""

import hashlib
from typing import Dict, List, Optional

import torch
from sentence_transformers import SentenceTransformer

from ai.embeddings.base import BaseEmbedding


# Default model configuration
EMBEDDING_MODEL = "google/embeddinggemma-300m"


class GemmaEmbedding(BaseEmbedding):
    """
    Gemma-based text embedding model.

    Uses Google's embeddinggemma-300m for converting text to dense vectors.
    Includes caching to avoid recomputing embeddings for the same text.
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        device: Optional[str] = None,
        cache_enabled: bool = True,
    ):
        """
        Initialize the Gemma embedding model.

        Args:
            model_name: HuggingFace model identifier.
            device: Device to run on ("cuda", "mps", "cpu", or None for auto).
            cache_enabled: Whether to cache embeddings.
        """
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print("Using Apple Silicon MPS device")
            else:
                self.device = torch.device("cpu")
                print("Using CPU")
        else:
            self.device = torch.device(device)

        self.model_name = model_name
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, torch.Tensor] = {}

        print(f"  Loading Gemma embedding model ({model_name})...")
        self.model = SentenceTransformer(model_name, device=str(self.device))
        print(f"  Gemma loaded on {self.device}")

    def embed(self, text: str) -> torch.Tensor:
        """
        Convert text to a vector embedding.

        Uses caching to avoid recomputing embeddings for the same text.

        Args:
            text: Input text string.

        Returns:
            Embedding tensor of shape (embedding_dim,).
        """
        if self.cache_enabled:
            key = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if key in self._cache:
                return self._cache[key]

        features = self.model.tokenize([text])
        features = {name: tensor.to(self.device) for name, tensor in features.items()}

        with torch.no_grad():
            outputs = self.model(features)

        token_embeddings = outputs["token_embeddings"]
        attention_mask = features["attention_mask"]

        # Mean pooling
        mask = attention_mask.unsqueeze(-1)
        pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        embedding = pooled.squeeze(0)

        if self.cache_enabled:
            self._cache[key] = embedding

        return embedding

    def embed_batch(self, texts: List[str]) -> torch.Tensor:
        """
        Convert multiple texts to embeddings.

        Args:
            texts: List of input text strings.

        Returns:
            Batch of embedding tensors of shape (batch_size, embedding_dim).
        """
        embeddings = [self.embed(text) for text in texts]
        return torch.stack(embeddings)

    def get_embedding_dimension(self) -> int:
        """Return the dimension of the embedding vectors."""
        return self.model.get_sentence_embedding_dimension()

    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()

    @property
    def cache_size(self) -> int:
        """Return the number of cached embeddings."""
        return len(self._cache)


# Singleton instance for reuse
_embedding_instance: Optional[GemmaEmbedding] = None


def get_gemma_embedding(
    device: Optional[str] = None,
    cache_enabled: bool = True,
) -> GemmaEmbedding:
    """
    Get or create the global Gemma embedding instance.

    Uses a singleton pattern to avoid loading the model multiple times.

    Args:
        device: Device to run on (only used on first call).
        cache_enabled: Whether to enable caching (only used on first call).

    Returns:
        GemmaEmbedding instance.
    """
    global _embedding_instance
    if _embedding_instance is None:
        _embedding_instance = GemmaEmbedding(
            device=device,
            cache_enabled=cache_enabled,
        )
    return _embedding_instance
