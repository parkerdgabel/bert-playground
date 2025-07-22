"""Base classes for Kaggle dataset abstraction using composition.

This module defines the core interfaces and base classes for handling
Kaggle datasets with optimizations for BERT models and MLX framework.

This version uses composition internally while maintaining backward compatibility.
"""

from abc import ABC
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Union

import mlx.core as mx
import pandas as pd
from loguru import logger


class CompetitionType(Enum):
    """Types of Kaggle competitions."""

    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"
    REGRESSION = "regression"
    ORDINAL_REGRESSION = "ordinal_regression"
    TIME_SERIES = "time_series"
    RANKING = "ranking"
    STRUCTURED_PREDICTION = "structured_prediction"
    GENERATIVE = "generative"
    UNKNOWN = "unknown"


@dataclass
class DatasetSpec:
    """Specification for a Kaggle dataset.

    This class defines the metadata and configuration for a specific
    Kaggle competition dataset, including optimization hints for BERT
    and MLX processing.
    """

    # Basic identification
    competition_name: str
    dataset_path: Union[str, Path]
    competition_type: CompetitionType

    # Data characteristics
    num_samples: int
    num_features: int
    target_column: str | None = None
    text_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    numerical_columns: list[str] = field(default_factory=list)

    # Target characteristics
    num_classes: int | None = None
    class_distribution: dict[str, int] | None = None
    is_balanced: bool = True

    # Performance optimization hints
    recommended_batch_size: int = 32
    recommended_max_length: int = 512
    use_attention_mask: bool = True
    enable_caching: bool = True

    # MLX optimization settings
    use_unified_memory: bool = True
    prefetch_size: int = 4
    num_workers: int = 4

    # BERT-specific settings
    text_template: str | None = None
    tokenizer_backend: str = "transformers"  # "transformers" or "mlx"

    def __post_init__(self):
        """Validate and normalize the dataset specification."""
        self.dataset_path = Path(self.dataset_path)

        # Validate competition type and num_classes consistency
        if self.competition_type == CompetitionType.BINARY_CLASSIFICATION:
            if self.num_classes is None:
                self.num_classes = 2
            elif self.num_classes != 2:
                logger.warning(
                    f"Binary classification should have 2 classes, got {self.num_classes}"
                )

        elif self.competition_type == CompetitionType.REGRESSION:
            if self.num_classes is None:
                self.num_classes = 1

        # Set reasonable defaults based on dataset size
        if self.num_samples > 100000:
            self.recommended_batch_size = min(64, self.recommended_batch_size)
            self.prefetch_size = 8
        elif self.num_samples < 5000:
            self.recommended_batch_size = max(16, self.recommended_batch_size)
            self.prefetch_size = 2


class KaggleDataset(ABC):
    """Abstract base class for Kaggle datasets.

    This class provides a unified interface for all Kaggle competition datasets,
    with built-in optimizations for BERT models and MLX framework on Apple Silicon.

    The design follows these principles:
    1. Unified interface across all competition types
    2. MLX-optimized data loading with unified memory
    3. BERT-compatible tokenization and batching
    4. Automatic dataset analysis and optimization
    5. Efficient caching and streaming capabilities
    
    This version uses composition internally while maintaining the same API.
    """

    def __init__(
        self,
        spec: DatasetSpec,
        split: str = "train",
        transform: Optional[Callable] = None,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        """Initialize the Kaggle dataset.

        Args:
            spec: Dataset specification containing metadata and configuration
            split: Dataset split ("train", "validation", "test")
            transform: Optional transform function to apply to samples
            cache_dir: Directory for caching processed data
        """
        self.spec = spec
        self.split = split
        self.transform = transform
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Import here to avoid circular dependency
        from ..composed_dataset import ComposedDataset

        # Use composition internally
        self._composed_dataset = ComposedDataset(
            spec=spec,
            split=split,
            transform=transform,
            cache_dir=cache_dir,
        )

        # Expose components for backward compatibility
        self._data = self._composed_dataset._data
        self._device = self._composed_dataset._device
        self._stream = self._composed_dataset._stream

        # Legacy attributes (deprecated)
        self._cached_samples = None
        self._sample_cache = {}

        # Allow subclasses to override loading/validation
        if hasattr(self, '_custom_load_data'):
            self._custom_load_data()
        if hasattr(self, '_custom_validate_data'):
            self._custom_validate_data()

        logger.info(
            f"Initialized {self.__class__.__name__} for {spec.competition_name} "
            f"({split}) with {len(self)} samples"
        )

    def _load_data(self) -> None:
        """Load the raw data from files.

        This method can be overridden by subclasses for custom loading.
        Default implementation is handled by ComposedDataset.
        """
        # Update reference to composed dataset's data
        self._data = self._composed_dataset._data

    def _validate_data(self) -> None:
        """Validate the loaded data.

        This method can be overridden by subclasses for custom validation.
        Default implementation is handled by ComposedDataset.
        """
        # Validation is already done in ComposedDataset
        pass

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Get a single sample by index."""
        return self._composed_dataset[index]

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self._composed_dataset)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over dataset samples."""
        return iter(self._composed_dataset)

    def get_sample_by_id(self, sample_id: str) -> dict[str, Any] | None:
        """Get a sample by its unique ID."""
        return self._composed_dataset.get_sample_by_id(sample_id)

    def get_batch(self, indices: list[int]) -> dict[str, mx.array]:
        """Get a batch of samples as MLX arrays."""
        return self._composed_dataset.get_batch(indices)

    def _collate_batch(self, samples: list[dict[str, Any]]) -> dict[str, mx.array]:
        """Collate a list of samples into a batch."""
        return self._composed_dataset._collate_batch(samples)

    def get_competition_info(self) -> dict[str, Any]:
        """Get competition information and statistics."""
        return self._composed_dataset.metadata_manager.get_competition_info()

    def get_sample_text(self, index: int) -> str:
        """Get the text representation of a sample."""
        sample = self[index]
        return sample.get("text", "")

    def get_data_statistics(self) -> dict[str, Any]:
        """Get detailed statistics about the dataset."""
        return self._composed_dataset.get_statistics()

    def enable_caching(self, cache_dir: Optional[Union[str, Path]] = None) -> None:
        """Enable sample caching for faster access."""
        self._composed_dataset.enable_caching(cache_dir)
        if cache_dir:
            self.cache_dir = Path(cache_dir)

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._composed_dataset.clear_cache()

    def get_mlx_device_info(self) -> dict[str, Any]:
        """Get MLX device and memory information."""
        return {
            "device": str(self._device),
            "unified_memory": self.spec.use_unified_memory,
            "default_stream": str(self._stream),
            "mlx_available": True,
        }
    
    # Additional methods for backward compatibility
    def get_optimization_hints(self) -> dict[str, Any]:
        """Get optimization hints for training."""
        return self._composed_dataset.get_optimization_hints()

    def get_feature_summary(self) -> dict[str, Any]:
        """Get summary of dataset features."""
        return self._composed_dataset.get_feature_summary()

    def get_text_statistics(self) -> dict[str, Any]:
        """Get statistics about text representations."""
        return self._composed_dataset.get_text_statistics()

    def add_custom_transform(self, transform: Callable) -> None:
        """Add a custom transformation to the pipeline."""
        self._composed_dataset.add_custom_transform(transform)

    def export_info(self) -> dict[str, Any]:
        """Export comprehensive dataset information."""
        return self._composed_dataset.export_info()