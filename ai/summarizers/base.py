"""
Base interface for text summarizers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseSummarizer(ABC):
    """Base class for text summarizers."""

    @abstractmethod
    def summarize(self, report_data: Dict[str, Any]) -> str:
        """
        Generate a summary from data.

        Args:
            report_data: Dictionary containing data to summarize.

        Returns:
            Summary string.
        """
        pass
