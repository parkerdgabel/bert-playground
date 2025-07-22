"""Composed dataset implementation using component architecture.

This module provides a dataset implementation that uses composition
to combine focused components for clean, maintainable code.
"""

from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Union

import mlx.core as mx
import pandas as pd
from loguru import logger

from .components import (
    CacheManager,
    ColumnSelector,
    DataReader,
    DataTransformer,
    DataValidator,
    MetadataManager,
)
from .core.base import DatasetSpec


class ComposedDataset:
    """Dataset implementation using composition of focused components."""

    def __init__(
        self,
        spec: DatasetSpec,
        split: str = "train",
        transform: Optional[Callable] = None,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        """Initialize the composed dataset.

        Args:
            spec: Dataset specification
            split: Dataset split ("train", "validation", "test")
            transform: Optional transform function
            cache_dir: Directory for caching
        """
        self.spec = spec
        self.split = split
        self.transform = transform

        # Initialize components
        self.reader = DataReader()
        self.validator = DataValidator(spec)
        self.transformer = DataTransformer(spec)
        self.metadata_manager = MetadataManager(spec)
        self.cache_manager = CacheManager(cache_dir)
        self.column_selector = ColumnSelector()

        # Internal state
        self._data: Optional[pd.DataFrame] = None
        self._prepared_data: Optional[pd.DataFrame] = None

        # MLX settings
        self._device = mx.default_device()
        self._stream = mx.default_stream(self._device)

        # Load and prepare data
        self._load_and_prepare_data()

    def _load_and_prepare_data(self) -> None:
        """Load, validate, and prepare the dataset."""
        # Try to load from cache first
        cache_key = self.cache_manager.create_cache_key(
            self.spec.dataset_path,
            self.split,
            self.spec.competition_name,
        )

        cached_data = self.cache_manager.load_dataframe(cache_key)
        if cached_data is not None:
            logger.info(f"Loaded dataset from cache: {cache_key}")
            self._data = cached_data
        else:
            # Load from file
            file_path = self._get_data_file_path()
            self._data = self.reader.read(file_path)

            # Cache the raw data
            if self.spec.enable_caching:
                self.cache_manager.cache_dataframe(cache_key, self._data)

        # Validate data
        errors = self.validator.validate(self._data, self.split)
        if errors:
            logger.warning(f"Data validation found {len(errors)} issues")

        # Infer column types
        self.column_selector.infer_column_types(self._data)

        # Prepare for BERT
        self._prepare_data()

        logger.info(
            f"Initialized dataset for {self.spec.competition_name} "
            f"({self.split}) with {len(self)} samples"
        )

    def _get_data_file_path(self) -> Path:
        """Get the file path for the current split."""
        base_path = Path(self.spec.dataset_path)

        # Common naming patterns
        if self.split == "train":
            candidates = ["train.csv", "train.parquet", "training.csv"]
        elif self.split == "validation":
            candidates = ["val.csv", "valid.csv", "validation.csv", "dev.csv"]
        elif self.split == "test":
            candidates = ["test.csv", "test.parquet", "testing.csv"]
        else:
            candidates = [f"{self.split}.csv", f"{self.split}.parquet"]

        # Check each candidate
        for candidate in candidates:
            file_path = base_path / candidate
            if file_path.exists():
                return file_path

        # Default to split name
        return base_path / f"{self.split}.csv"

    def _prepare_data(self) -> None:
        """Prepare data for BERT processing."""
        # Check if already prepared
        cache_key = self.cache_manager.create_cache_key(
            self.spec.dataset_path,
            self.split,
            "prepared",
        )

        cached_prepared = self.cache_manager.load_dataframe(cache_key)
        if cached_prepared is not None:
            self._prepared_data = cached_prepared
            return

        # Prepare data
        self._prepared_data = self.transformer.prepare_for_bert(self._data)

        # Cache prepared data
        if self.spec.enable_caching:
            self.cache_manager.cache_dataframe(cache_key, self._prepared_data)

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self._data) if self._data is not None else 0

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Get a single sample by index.

        Args:
            index: Sample index

        Returns:
            Dictionary containing sample data
        """
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range [0, {len(self)})")

        # Check cache
        cache_key = f"sample_{self.split}_{index}"
        cached_sample = self.cache_manager.get(cache_key)
        if cached_sample is not None:
            return cached_sample

        # Get row from prepared data
        row = self._prepared_data.iloc[index]

        # Create sample dictionary
        sample = {
            "text": row.get("text", ""),
            "metadata": {"index": index, "split": self.split},
        }

        # Add labels for training data
        if self.spec.target_column and self.spec.target_column in row:
            sample["labels"] = row[self.spec.target_column]

        # Add ID if available
        id_columns = self.column_selector.get_columns_by_type(self._data, "id")
        if id_columns:
            sample["id"] = row[id_columns[0]]

        # Apply custom transform if provided
        if self.transform:
            sample = self.transform(sample)

        # Cache the sample
        if self.spec.enable_caching:
            self.cache_manager.set(cache_key, sample, persist=False)

        return sample

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over dataset samples."""
        for i in range(len(self)):
            yield self[i]

    def get_batch(self, indices: list[int]) -> dict[str, mx.array]:
        """Get a batch of samples as MLX arrays.

        Args:
            indices: List of sample indices

        Returns:
            Dictionary of batched MLX arrays
        """
        samples = [self[i] for i in indices]
        return self._collate_batch(samples)

    def _collate_batch(self, samples: list[dict[str, Any]]) -> dict[str, mx.array]:
        """Collate samples into a batch.

        Args:
            samples: List of sample dictionaries

        Returns:
            Dictionary of batched MLX arrays
        """
        if not samples:
            return {}

        batch = {}

        # Handle text data
        if "text" in samples[0]:
            batch["text"] = [s["text"] for s in samples]

        # Handle labels
        if "labels" in samples[0]:
            labels = [s["labels"] for s in samples]
            batch["labels"] = mx.array(labels, dtype=mx.float32)

        # Handle metadata
        batch["metadata"] = [s.get("metadata", {}) for s in samples]

        return batch

    def get_sample_by_id(self, sample_id: str) -> Optional[dict[str, Any]]:
        """Get a sample by its unique ID.

        Args:
            sample_id: Unique identifier

        Returns:
            Sample dictionary or None
        """
        # Try as index first
        try:
            idx = int(sample_id)
            if 0 <= idx < len(self):
                return self[idx]
        except ValueError:
            pass

        # Search by ID column
        id_columns = self.column_selector.get_columns_by_type(self._data, "id")
        for col in id_columns:
            mask = self._data[col] == sample_id
            if mask.any():
                idx = self._data[mask].index[0]
                return self[idx]

        return None

    def get_statistics(self) -> dict[str, Any]:
        """Get comprehensive dataset statistics.

        Returns:
            Dictionary containing statistics
        """
        cache_key = f"stats_{self.split}"
        return self.metadata_manager.compute_statistics(self._data, cache_key)

    def get_optimization_hints(self) -> dict[str, Any]:
        """Get optimization hints for training.

        Returns:
            Dictionary containing optimization recommendations
        """
        return self.metadata_manager.get_optimization_hints(self._prepared_data)

    def get_feature_summary(self) -> dict[str, Any]:
        """Get summary of dataset features.

        Returns:
            Dictionary containing feature information
        """
        return self.column_selector.create_feature_summary(self._data)

    def enable_caching(self, cache_dir: Optional[Union[str, Path]] = None) -> None:
        """Enable or update caching settings.

        Args:
            cache_dir: New cache directory (optional)
        """
        if cache_dir:
            self.cache_manager = CacheManager(cache_dir)
        self.spec.enable_caching = True
        logger.info("Caching enabled")

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self.cache_manager.clear()
        logger.info("Cache cleared")

    def get_text_statistics(self) -> dict[str, Any]:
        """Get statistics about text representations.

        Returns:
            Dictionary of text statistics
        """
        return self.transformer.get_text_statistics(self._prepared_data)

    def add_custom_transform(self, transform: Callable) -> None:
        """Add a custom transformation to the pipeline.

        Args:
            transform: Transformation function
        """
        self.transformer.add_custom_transform(transform)
        # Clear prepared data to force re-preparation
        self._prepared_data = None
        self._prepare_data()

    def export_info(self) -> dict[str, Any]:
        """Export comprehensive dataset information.

        Returns:
            Dictionary containing all dataset information
        """
        return {
            "specification": self.metadata_manager.export_metadata(),
            "statistics": self.get_statistics(),
            "features": self.get_feature_summary(),
            "optimization": self.get_optimization_hints(),
            "text_stats": self.get_text_statistics(),
            "cache_info": self.cache_manager.get_cache_info(),
            "validation": self.validator.get_validation_report(self._data),
        }