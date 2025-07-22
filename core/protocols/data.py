"""Data pipeline protocols for k-bert.

These protocols define the contracts for data loading, processing, and augmentation.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol

from core.ports.compute import Array


class Dataset(Protocol):
    """Protocol for dataset implementations."""

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        ...

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single item from the dataset.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Dictionary containing the data sample
        """
        ...

    def get_metadata(self) -> dict[str, Any]:
        """Return dataset metadata (schema, stats, etc.).
        
        Returns:
            Dictionary containing metadata about the dataset
        """
        ...


class DataLoader(Protocol):
    """Protocol for data loaders."""

    def __iter__(self) -> Iterator[dict[str, Array]]:
        """Iterate over batches.
        
        Returns:
            Iterator yielding batches as dictionaries of MLX arrays
        """
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


class DataFormat(Protocol):
    """Protocol for data format handlers."""

    def read(self, path: Path, **kwargs) -> Any:
        """Read data from file.
        
        Args:
            path: Path to read from
            **kwargs: Additional format-specific arguments
            
        Returns:
            Loaded data
        """
        ...

    def write(self, data: Any, path: Path, **kwargs) -> None:
        """Write data to file.
        
        Args:
            data: Data to write
            path: Path to write to
            **kwargs: Additional format-specific arguments
        """
        ...

    def validate(self, path: Path) -> bool:
        """Validate if file is in correct format.
        
        Args:
            path: Path to validate
            
        Returns:
            True if file is valid
        """
        ...

    def infer_schema(self, path: Path) -> dict[str, Any]:
        """Infer data schema from file.
        
        Args:
            path: Path to analyze
            
        Returns:
            Dictionary describing the data schema
        """
        ...


class BatchProcessor(Protocol):
    """Protocol for batch processing."""

    def process_batch(self, batch: list[dict[str, Any]]) -> dict[str, Array]:
        """Process a batch of samples into MLX arrays.
        
        Args:
            batch: List of sample dictionaries
            
        Returns:
            Dictionary of MLX arrays
        """
        ...

    def collate(self, samples: list[dict[str, Any]]) -> dict[str, Array]:
        """Collate samples into a batch.
        
        Args:
            samples: List of samples to collate
            
        Returns:
            Batch as dictionary of MLX arrays
        """
        ...


class Augmenter(Protocol):
    """Protocol for data augmentation."""

    def augment(self, data: Any, **kwargs) -> Any:
        """Augment a single data sample.
        
        Args:
            data: Data to augment
            **kwargs: Augmentation parameters
            
        Returns:
            Augmented data
        """
        ...

    def augment_batch(self, batch: list[Any], **kwargs) -> list[Any]:
        """Augment a batch of data samples.
        
        Args:
            batch: List of data samples
            **kwargs: Augmentation parameters
            
        Returns:
            List of augmented samples
        """
        ...

    def get_config(self) -> dict[str, Any]:
        """Get augmentation configuration.
        
        Returns:
            Dictionary of configuration parameters
        """
        ...

    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility.
        
        Args:
            seed: Random seed
        """
        ...


class FeatureAugmenter(Protocol):
    """Protocol for feature-specific augmentation."""

    def augment_feature(
        self, value: Any, feature_name: str, feature_type: str, **kwargs
    ) -> Any:
        """Augment a single feature value.
        
        Args:
            value: Feature value to augment
            feature_name: Name of the feature
            feature_type: Type of the feature
            **kwargs: Additional parameters
            
        Returns:
            Augmented feature value
        """
        ...

    def can_augment(self, feature_type: str) -> bool:
        """Check if this augmenter can handle the feature type.
        
        Args:
            feature_type: Type of feature to check
            
        Returns:
            True if this augmenter can handle the feature type
        """
        ...