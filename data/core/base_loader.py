"""
Base loader implementation for creating MLX data streams.
Handles batching, shuffling, and stream optimization.
"""

import random
from collections.abc import Callable, Iterator

# Note: MLX used for all array operations
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.data as dx
from loguru import logger

from .interfaces import BatchProcessor, Dataset, OptimizationProfile


class BaseMLXLoader:
    """
    Base class for creating MLX data loaders.
    Handles stream creation, batching, and optimization.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        prefetch_size: int = 4,
        buffer_size: int = 1000,
        optimization_profile: str | OptimizationProfile | None = None,
        batch_processor: BatchProcessor | None = None,
        seed: int | None = None,
        **kwargs,
    ):
        """
        Initialize MLX loader.

        Args:
            dataset: Dataset to load from
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker threads
            prefetch_size: Number of batches to prefetch
            buffer_size: Size of shuffle buffer
            optimization_profile: Optimization profile or name
            batch_processor: Custom batch processor
            seed: Random seed for reproducibility
            **kwargs: Additional arguments
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        # Apply optimization profile if provided
        if optimization_profile:
            self._apply_optimization_profile(optimization_profile)
        else:
            self.num_workers = num_workers
            self.prefetch_size = prefetch_size
            self.buffer_size = buffer_size

        self.batch_processor = batch_processor or DefaultBatchProcessor()

        # Create stream
        self._stream = None
        self._create_stream()

    def _apply_optimization_profile(self, profile: str | OptimizationProfile) -> None:
        """Apply optimization profile settings."""
        from .interfaces import OPTIMIZATION_PROFILES

        if isinstance(profile, str):
            if profile not in OPTIMIZATION_PROFILES:
                raise ValueError(f"Unknown optimization profile: {profile}")
            profile = OPTIMIZATION_PROFILES[profile]

        self.num_workers = profile.num_workers
        self.prefetch_size = profile.prefetch_size
        self.buffer_size = profile.buffer_size

        logger.debug(f"Applied optimization profile: {profile}")

    def _create_stream(self) -> None:
        """Create MLX data stream."""
        # For small datasets, use buffer-based approach
        if len(self.dataset) < 10000:
            self._create_buffer_stream()
        else:
            self._create_generator_stream()

    def _create_buffer_stream(self) -> None:
        """Create stream from pre-loaded buffer (for small datasets)."""
        logger.debug(f"Creating buffer-based stream for {len(self.dataset)} samples")

        # Pre-process all data
        all_samples = []
        for i in range(len(self.dataset)):
            sample = self.dataset[i]
            all_samples.append(sample)

        # Group into batches
        batches = []
        for i in range(0, len(all_samples), self.batch_size):
            batch = all_samples[i : i + self.batch_size]
            if batch:  # Skip empty batches
                processed_batch = self.batch_processor.process_batch(batch)
                batches.append(processed_batch)

        # Shuffle batches if needed
        if self.shuffle:
            if self.seed is not None:
                random.seed(self.seed)
            random.shuffle(batches)

        # Create buffer and stream
        buffer = dx.buffer_from_vector(batches)
        self._stream = buffer.to_stream()

        # Apply prefetching
        if self.prefetch_size > 0:
            self._stream = self._stream.prefetch(self.prefetch_size, self.num_workers)

    def _create_generator_stream(self) -> None:
        """Create stream from generator (for large datasets)."""
        logger.debug(f"Creating generator-based stream for {len(self.dataset)} samples")

        def sample_generator():
            """Generate samples with optional shuffling."""
            indices = list(range(len(self.dataset)))

            if self.shuffle:
                if self.seed is not None:
                    mx.random.seed(self.seed)
                # Use MLX permutation
                perm = mx.random.permutation(len(indices))
                indices = [indices[i] for i in perm.tolist()]

            for idx in indices:
                yield self.dataset[idx]

        # Create base stream from generator
        stream = dx.stream(sample_generator)

        # Apply batching
        stream = stream.batch(self.batch_size)

        # Apply batch processing
        stream = stream.map(
            lambda batch: self.batch_processor.process_batch(batch),
            num_workers=self.num_workers,
        )

        # Apply shuffling buffer if needed
        if self.shuffle and self.buffer_size > 0:
            stream = stream.shuffle(self.buffer_size)

        # Apply prefetching
        if self.prefetch_size > 0:
            stream = stream.prefetch(self.prefetch_size, self.num_workers)

        self._stream = stream

    def __iter__(self) -> Iterator[dict[str, mx.array]]:
        """Iterate over batches."""
        if self._stream is None:
            raise RuntimeError("Stream not initialized")

        for batch in self._stream:
            yield batch

    def __len__(self) -> int:
        """Return number of batches."""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    @property
    def num_samples(self) -> int:
        """Get total number of samples."""
        return len(self.dataset)

    def reset(self) -> None:
        """Reset the stream (recreate it)."""
        self._create_stream()

    def set_batch_size(self, batch_size: int) -> None:
        """Change batch size and recreate stream."""
        self.batch_size = batch_size
        self._create_stream()

    def set_shuffle(self, shuffle: bool) -> None:
        """Change shuffle setting and recreate stream."""
        self.shuffle = shuffle
        self._create_stream()


class DefaultBatchProcessor:
    """Default batch processor for MLX arrays."""

    def __init__(
        self,
        input_key: str = "input_ids",
        mask_key: str = "attention_mask",
        label_key: str = "label",
        additional_keys: list[str] | None = None,
    ):
        """
        Initialize batch processor.

        Args:
            input_key: Key for input IDs
            mask_key: Key for attention mask
            label_key: Key for labels
            additional_keys: Additional keys to process
        """
        self.input_key = input_key
        self.mask_key = mask_key
        self.label_key = label_key
        self.additional_keys = additional_keys or []

    def process_batch(self, batch: list[dict[str, Any]]) -> dict[str, mx.array]:
        """
        Process a batch of samples into MLX arrays.

        Args:
            batch: List of sample dictionaries

        Returns:
            Dictionary of MLX arrays
        """
        if not batch:
            raise ValueError("Empty batch")

        result = {}

        # Process input IDs
        if self.input_key in batch[0]:
            input_ids = [sample[self.input_key] for sample in batch]
            result[self.input_key] = mx.array(input_ids, dtype=mx.int32)

        # Process attention mask
        if self.mask_key in batch[0]:
            attention_masks = [sample[self.mask_key] for sample in batch]
            result[self.mask_key] = mx.array(attention_masks, dtype=mx.float32)

        # Process labels
        if self.label_key in batch[0]:
            labels = [sample[self.label_key] for sample in batch]
            # Handle different label types
            if isinstance(labels[0], (list, mx.array)):
                result[self.label_key] = mx.array(labels, dtype=mx.float32)
            else:
                result[self.label_key] = mx.array(labels, dtype=mx.int32)

        # Process additional keys
        for key in self.additional_keys:
            if key in batch[0]:
                values = [sample[key] for sample in batch]
                # Infer dtype based on values
                if isinstance(values[0], int):
                    dtype = mx.int32
                elif isinstance(values[0], float):
                    dtype = mx.float32
                else:
                    # Default to float32 for unknown types
                    dtype = mx.float32
                result[key] = mx.array(values, dtype=dtype)

        return result

    def collate(self, samples: list[dict[str, Any]]) -> dict[str, mx.array]:
        """Alias for process_batch for compatibility."""
        return self.process_batch(samples)


class StreamedMLXLoader(BaseMLXLoader):
    """
    Optimized loader that streams directly from files.
    Useful for very large datasets that don't fit in memory.
    """

    def __init__(
        self,
        data_path: str | Path,
        dataset_factory: Callable[[str], Dataset],
        chunk_size: int = 1000,
        **kwargs,
    ):
        """
        Initialize streamed loader.

        Args:
            data_path: Path to data file
            dataset_factory: Factory function to create dataset chunks
            chunk_size: Size of chunks to process
            **kwargs: Arguments passed to BaseMLXLoader
        """
        self.data_path = Path(data_path)
        self.dataset_factory = dataset_factory
        self.chunk_size = chunk_size

        # Create a dummy dataset for metadata
        dummy_dataset = dataset_factory(str(data_path))
        super().__init__(dummy_dataset, **kwargs)

    def _create_stream(self) -> None:
        """Create chunked stream."""

        def chunk_generator():
            """Generate dataset chunks."""
            # This would be implemented based on the specific file format
            # For now, we'll use the dataset factory
            dataset = self.dataset_factory(str(self.data_path))

            for i in range(0, len(dataset), self.chunk_size):
                chunk_indices = list(range(i, min(i + self.chunk_size, len(dataset))))
                chunk = dataset.get_batch(chunk_indices)

                # Process chunk into batches
                for j in range(0, len(chunk), self.batch_size):
                    batch = chunk[j : j + self.batch_size]
                    if batch:
                        yield self.batch_processor.process_batch(batch)

        # Create stream from generator
        stream = dx.stream(chunk_generator)

        # Apply shuffling if needed
        if self.shuffle and self.buffer_size > 0:
            stream = stream.shuffle(self.buffer_size)

        # Apply prefetching
        if self.prefetch_size > 0:
            stream = stream.prefetch(self.prefetch_size, self.num_workers)

        self._stream = stream


def create_mlx_loader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    optimization: str = "training",
    **kwargs,
) -> BaseMLXLoader:
    """
    Factory function to create MLX loader.

    Args:
        dataset: Dataset to load from
        batch_size: Batch size
        shuffle: Whether to shuffle
        optimization: Optimization profile name
        **kwargs: Additional arguments

    Returns:
        MLX loader instance
    """
    return BaseMLXLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        optimization_profile=optimization,
        **kwargs,
    )
