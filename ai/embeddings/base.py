"""
Base interface for text embedding models.
"""

from abc import ABC, abstractmethod
from typing import List

import torch


class BaseEmbedding(ABC):
    """Base class for text embedding models."""

    @abstractmethod
    def embed(self, text: str) -> torch.Tensor:
        """
        Convert a single text to embedding vector.

        Args:
            text: Input text string.

        Returns:
            Embedding tensor.
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> torch.Tensor:
        """
        Convert multiple texts to embedding vectors.

        Args:
            texts: List of input text strings.

        Returns:
            Batch of embedding tensors.
        """
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Return the dimension of the embedding vectors."""
        pass
