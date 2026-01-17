"""
Base interfaces for AI/ML components.

All AI models (embeddings, sentiment, summarizers, LLMs) inherit from these base classes.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch


class BaseAIModel(ABC):
    """Base class for all AI models."""

    def __init__(self, device: Optional[str] = None):
        """
        Initialize the AI model.

        Args:
            device: Device to run on ("cuda", "mps", "cpu", or None for auto).
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

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return information about the model."""
        pass
