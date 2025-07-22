"""
Core interfaces and protocols for the modular MLX dataloader system.
Defines the contracts that all components must follow.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol

import mlx.core as mx


class Dataset(Protocol):
    """Protocol for dataset implementations."""

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        ...

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single item from the dataset."""
        ...

    def get_metadata(self) -> dict[str, Any]:
        """Return dataset metadata (schema, stats, etc.)."""
        ...


class DataLoader(Protocol):
    """Protocol for data loaders."""

    def __iter__(self) -> Iterator[dict[str, mx.array]]:
        """Iterate over batches."""
        ...

    def __len__(self) -> int:
        """Return number of batches."""
        ...

    @property
    def batch_size(self) -> int:
        """Get batch size."""
        ...

    @property
    def num_samples(self) -> int:
        """Get total number of samples."""
        ...


class DataFormat(ABC):
    """Abstract base class for data format handlers."""

    @abstractmethod
    def read(self, path: Path, **kwargs) -> Any:
        """Read data from file."""
        pass

    @abstractmethod
    def write(self, data: Any, path: Path, **kwargs) -> None:
        """Write data to file."""
        pass

    @abstractmethod
    def validate(self, path: Path) -> bool:
        """Validate if file is in correct format."""
        pass

    @abstractmethod
    def infer_schema(self, path: Path) -> dict[str, Any]:
        """Infer data schema from file."""
        pass


class BatchProcessor(Protocol):
    """Protocol for batch processing."""

    def process_batch(self, batch: list[dict[str, Any]]) -> dict[str, mx.array]:
        """Process a batch of samples into MLX arrays."""
        ...

    def collate(self, samples: list[dict[str, Any]]) -> dict[str, mx.array]:
        """Collate samples into a batch."""
        ...


class Augmenter(Protocol):
    """Protocol for data augmentation."""

    def augment(self, data: Any, **kwargs) -> Any:
        """Augment a single data sample."""
        ...

    def augment_batch(self, batch: list[Any], **kwargs) -> list[Any]:
        """Augment a batch of data samples."""
        ...

    def get_config(self) -> dict[str, Any]:
        """Get augmentation configuration."""
        ...

    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        ...


class FeatureAugmenter(Protocol):
    """Protocol for feature-specific augmentation."""

    def augment_feature(
        self, value: Any, feature_name: str, feature_type: str, **kwargs
    ) -> Any:
        """Augment a single feature value."""
        ...

    def can_augment(self, feature_type: str) -> bool:
        """Check if this augmenter can handle the feature type."""
        ...


__all__ = [
    "Dataset",
    "DataLoader",
    "DataFormat",
    "BatchProcessor",
    "Augmenter",
    "FeatureAugmenter",
]
