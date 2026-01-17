"""Base class for all data providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json
import os


@dataclass
class ProviderResult:
    """Result from a provider download operation."""
    success: bool
    file_path: str
    record_count: int
    message: str


class BaseProvider(ABC):
    """
    Abstract base class for data providers.

    Each provider is responsible for downloading data from a single source
    and saving it to the datasets/{SYMBOL}/ directory.

    Subclasses must implement:
        - name: Provider identifier
        - download(): Fetch data from the source
        - output_filename: Name of the output file
    """

    DATASETS_DIR = "datasets"

    def __init__(self, symbol: str) -> None:
        """
        Initialize the provider.

        Args:
            symbol: Trading symbol (e.g., AAPL, BTC-USD)
        """
        self.symbol = symbol.upper()
        self.output_dir = f"{self.DATASETS_DIR}/{self.symbol}"

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging."""
        pass

    @property
    @abstractmethod
    def output_filename(self) -> str:
        """Output filename (without directory)."""
        pass

    @property
    def output_path(self) -> str:
        """Full path to output file."""
        return f"{self.output_dir}/{self.output_filename}"

    @abstractmethod
    def download(self) -> List[Dict[str, Any]]:
        """
        Download data from the provider.

        Returns:
            List of records (dicts) downloaded from the source.
        """
        pass

    def ensure_directory(self) -> None:
        """Create output directory if it doesn't exist."""
        os.makedirs(self.output_dir, exist_ok=True)

    def load_existing(self) -> List[Dict[str, Any]]:
        """Load existing data from output file."""
        try:
            with open(self.output_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def save(self, data: List[Dict[str, Any]]) -> str:
        """
        Save data to output file.

        Args:
            data: List of records to save

        Returns:
            Path to saved file
        """
        self.ensure_directory()
        with open(self.output_path, 'w') as f:
            json.dump(data, f, indent=2)
        return self.output_path

    def merge_with_existing(self, new_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge new data with existing, avoiding duplicates.

        Override this method for custom deduplication logic.
        """
        existing = self.load_existing()
        # Simple append - subclasses can override for smarter merging
        return existing + new_data

    def run(self, merge: bool = True) -> ProviderResult:
        """
        Execute the full download pipeline.

        Args:
            merge: If True, merge with existing data. If False, replace.

        Returns:
            ProviderResult with status and details
        """
        print(f"[{self.name}] Downloading for {self.symbol}...")

        try:
            new_data = self.download()

            if not new_data:
                return ProviderResult(
                    success=True,
                    file_path=self.output_path,
                    record_count=0,
                    message=f"No new data from {self.name}"
                )

            if merge:
                data = self.merge_with_existing(new_data)
            else:
                data = new_data

            self.save(data)

            print(f"[{self.name}] Saved {len(data)} records to {self.output_path}")

            return ProviderResult(
                success=True,
                file_path=self.output_path,
                record_count=len(data),
                message=f"Downloaded {len(new_data)} new records"
            )

        except Exception as e:
            print(f"[{self.name}] Error: {e}")
            return ProviderResult(
                success=False,
                file_path=self.output_path,
                record_count=0,
                message=str(e)
            )
