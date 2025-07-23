"""MLX-optimized data loader for Apple Silicon.

This module provides a high-performance data loader optimized for MLX by:
- Using direct iteration without complex threading/multiprocessing
- Leveraging unified memory for zero-copy operations
- Supporting lazy evaluation
- Minimizing state management to avoid concurrency issues
"""

import queue
import random
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from loguru import logger

# Remove old import - KaggleDataset is no longer used
from ..preprocessing.tokenizer_cache import PreTokenizedDataset


@dataclass
class MLXLoaderConfig:
    """Configuration for MLX data loader."""

    # Batch configuration
    batch_size: int = 32
    shuffle: bool = True
    drop_last: bool = False

    # MLX-specific optimization
    use_unified_memory: bool = True
    lazy_evaluation: bool = True
    prefetch_size: int = 2  # Number of batches to prefetch
    prefetch_timeout: float = 10.0  # Timeout for prefetch queue

    # Tokenization
    max_length: int = 512
    padding: str = "max_length"  # "max_length" or "longest"
    truncation: bool = True
    tokenizer_chunk_size: int = 100  # Process texts in chunks for better perf

    # Pre-tokenization
    use_pretokenized: bool = False  # Use pre-tokenized data if available
    pretokenized_cache_dir: str = "data/cache/tokenized"  # Cache directory


class MLXDataLoader:
    """High-performance data loader optimized for MLX and Apple Silicon.

    Features:
    - Direct iteration without complex prefetching (avoids threading issues)
    - Leverages unified memory for zero-copy operations
    - Supports MLX-native tokenization when available
    - Efficient batching with minimal overhead
    """

    def __init__(
        self,
        dataset: Any,  # Generic dataset interface
        config: MLXLoaderConfig | None = None,
        tokenizer=None,
        pretokenized_data: PreTokenizedDataset | None = None,
    ):
        """Initialize MLX data loader.

        Args:
            dataset: Kaggle dataset instance
            config: Loader configuration
            tokenizer: Optional tokenizer for text processing
        """
        self.dataset = dataset
        self.config = config or MLXLoaderConfig()
        self.tokenizer = tokenizer
        self.pretokenized_data = pretokenized_data

        # Use pretokenized dataset if available
        if self.pretokenized_data is not None:
            self.dataset = self.pretokenized_data
            logger.debug("Using pre-tokenized dataset")

        # MLX device
        self.device = mx.default_device()

        # Initialize indices
        self.indices = list(range(len(self.dataset)))
        if self.config.shuffle:
            random.shuffle(self.indices)

        # Calculate number of batches
        if self.config.drop_last:
            self.num_batches = len(self.indices) // self.config.batch_size
        else:
            self.num_batches = (
                len(self.indices) + self.config.batch_size - 1
            ) // self.config.batch_size

        # Prefetching setup
        self.prefetch_queue = None
        self._stop_prefetch = None
        self._prefetch_thread = None

        logger.debug(
            f"MLXDataLoader initialized: {self.num_batches} batches, "
            f"batch_size={self.config.batch_size}"
        )

    def __len__(self) -> int:
        """Get number of batches."""
        return self.num_batches

    def __iter__(self) -> Iterator[dict[str, mx.array]]:
        """Iterate over batches with optional prefetching."""
        # Reshuffle if needed
        if self.config.shuffle:
            random.shuffle(self.indices)

        # Use prefetching if enabled
        if self.config.prefetch_size > 0:
            yield from self._iter_with_prefetch()
        else:
            yield from self._iter_no_prefetch()

    def _iter_no_prefetch(self) -> Iterator[dict[str, mx.array]]:
        """Simple iteration without prefetching."""
        for batch_idx in range(self.num_batches):
            # Get batch indices
            start_idx = batch_idx * self.config.batch_size
            end_idx = min(start_idx + self.config.batch_size, len(self.indices))
            batch_indices = self.indices[start_idx:end_idx]

            # Get samples
            samples = [self.dataset[idx] for idx in batch_indices]

            # Collate into batch
            batch = self._collate_samples(samples)

            # CRITICAL: Force evaluation of batch arrays to prevent lazy evaluation buildup
            # This prevents hanging when computing gradients on unevaluated arrays
            # See: https://github.com/ml-explore/mlx/issues/451
            if batch:
                # Evaluate all arrays in the batch
                arrays_to_eval = [v for v in batch.values() if isinstance(v, mx.array)]
                if arrays_to_eval:
                    mx.eval(*arrays_to_eval)

            yield batch

    def _iter_with_prefetch(self) -> Iterator[dict[str, mx.array]]:
        """Iteration with asynchronous prefetching."""
        # Initialize prefetch queue and thread
        self.prefetch_queue = queue.Queue(maxsize=self.config.prefetch_size)
        self._stop_prefetch = threading.Event()

        # Start prefetch thread
        self._prefetch_thread = threading.Thread(target=self._prefetch_worker)
        self._prefetch_thread.daemon = True
        self._prefetch_thread.start()

        try:
            for _ in range(self.num_batches):
                # Get batch from queue with timeout
                try:
                    batch = self.prefetch_queue.get(
                        timeout=self.config.prefetch_timeout
                    )
                    if batch is None:  # Sentinel value
                        break
                    yield batch
                except queue.Empty:
                    logger.warning("Data loading timeout - consider reducing batch size or disabling prefetch")
                    break
        finally:
            # Clean up prefetch thread
            self._stop_prefetch.set()
            if self._prefetch_thread is not None:
                self._prefetch_thread.join(timeout=1.0)

    def _prefetch_worker(self):
        """Worker thread for prefetching batches."""
        try:
            for batch_idx in range(self.num_batches):
                if self._stop_prefetch.is_set():
                    break

                # Get batch indices
                start_idx = batch_idx * self.config.batch_size
                end_idx = min(start_idx + self.config.batch_size, len(self.indices))
                batch_indices = self.indices[start_idx:end_idx]

                # Get samples
                samples = [self.dataset[idx] for idx in batch_indices]

                # Collate into batch
                batch = self._collate_samples(samples)

                # CRITICAL: Force evaluation of batch arrays to prevent lazy evaluation buildup
                # This prevents hanging when computing gradients on unevaluated arrays
                # See: https://github.com/ml-explore/mlx/issues/451
                if batch:
                    # Evaluate all arrays in the batch
                    arrays_to_eval = [
                        v for v in batch.values() if isinstance(v, mx.array)
                    ]
                    if arrays_to_eval:
                        mx.eval(*arrays_to_eval)

                # Put in queue (blocks if full)
                self.prefetch_queue.put(batch)

            # Add sentinel to indicate completion
            self.prefetch_queue.put(None)
        except Exception as e:
            logger.error(f"Error in prefetch worker: {e}")
            self.prefetch_queue.put(None)

    def _collate_samples(self, samples: list[dict[str, Any]]) -> dict[str, mx.array]:
        """Collate samples into a batch.

        Args:
            samples: List of sample dictionaries

        Returns:
            Dictionary of batched MLX arrays
        """
        if not samples:
            return {}

        batch = {}

        # Check if we have pre-tokenized data
        has_tokenized_data = (
            "input_ids" in samples[0] and samples[0]["input_ids"] is not None
        )

        # Handle text data
        if "text" in samples[0] and not has_tokenized_data:
            texts = [sample["text"] for sample in samples]

            if self.tokenizer:
                # Tokenize texts
                tokenized = self._tokenize_batch(texts)
                batch.update(tokenized)
            else:
                raise ValueError("Tokenizer is required for text data")

        # Handle pre-tokenized data
        if has_tokenized_data:
            batch["input_ids"] = self._pad_sequences(
                [sample["input_ids"] for sample in samples]
            )

        if "attention_mask" in samples[0] and samples[0]["attention_mask"] is not None:
            batch["attention_mask"] = self._pad_sequences(
                [sample["attention_mask"] for sample in samples]
            )

        if "token_type_ids" in samples[0] and samples[0]["token_type_ids"] is not None:
            batch["token_type_ids"] = self._pad_sequences(
                [sample["token_type_ids"] for sample in samples]
            )

        # Handle labels
        if "labels" in samples[0]:
            labels = [
                sample["labels"] for sample in samples if sample["labels"] is not None
            ]
            if labels:
                if isinstance(labels[0], (int, float)):
                    # Single label per sample
                    batch["labels"] = mx.array(labels, dtype=mx.float32)
                else:
                    # Multi-label case
                    batch["labels"] = mx.array(labels, dtype=mx.float32)

        # Handle metadata
        if "metadata" in samples[0]:
            batch["metadata"] = [sample["metadata"] for sample in samples]

        return batch

    def _tokenize_batch(self, texts: list[str]) -> dict[str, mx.array]:
        """Tokenize a batch of texts with optional chunking for better performance.

        Args:
            texts: List of text strings

        Returns:
            Dictionary with tokenized data as MLX arrays
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer required for tokenization")

        # Check if we're using MLX embeddings tokenizer
        if hasattr(self.tokenizer, "backend") and self.tokenizer.backend == "mlx":
            # Use MLX-native tokenization if available
            encodings = self.tokenizer(
                texts,
                padding=self.config.padding,
                truncation=self.config.truncation,
                max_length=self.config.max_length,
                return_tensors="mlx",  # Return MLX tensors directly
            )
            return encodings
        else:
            # Process in chunks for better performance with standard tokenizers
            chunk_size = self.config.tokenizer_chunk_size
            all_encodings = []

            for i in range(0, len(texts), chunk_size):
                chunk = texts[i : i + chunk_size]
                chunk_encodings = self.tokenizer(
                    chunk,
                    padding=self.config.padding,
                    truncation=self.config.truncation,
                    max_length=self.config.max_length,
                    return_tensors="np",
                )
                all_encodings.append(chunk_encodings)

            # Combine chunks
            import numpy as np

            combined = {}
            for key in ["input_ids", "attention_mask", "token_type_ids"]:
                if key in all_encodings[0]:
                    values = [enc[key] for enc in all_encodings]
                    if len(values) > 1:
                        combined_values = np.concatenate(values, axis=0)
                    else:
                        combined_values = values[0]

                    # Convert to MLX array
                    if hasattr(combined_values, "numpy"):
                        combined_values = combined_values.numpy()
                    combined[key] = mx.array(combined_values, dtype=mx.int32)

            return combined

    def _pad_sequences(self, sequences: list[list[int]]) -> mx.array:
        """Pad sequences to uniform length using MLX operations.

        Args:
            sequences: List of sequences

        Returns:
            Padded sequences as MLX array
        """
        if not sequences:
            return mx.zeros((0, 0), dtype=mx.int32)

        # Convert sequences to MLX arrays
        mlx_sequences = []
        max_len = 0

        for seq in sequences:
            # Convert to MLX array if needed
            if not isinstance(seq, mx.array):
                if hasattr(seq, "tolist"):
                    seq = seq.tolist()
                elif not isinstance(seq, list):
                    seq = list(seq)
                seq = mx.array(seq, dtype=mx.int32)
            mlx_sequences.append(seq)
            max_len = max(max_len, seq.shape[0])

        # Limit to configured max length
        if self.config.padding == "max_length":
            max_len = self.config.max_length
        else:
            max_len = min(max_len, self.config.max_length)

        # Pad using MLX operations
        padded = []
        for seq in mlx_sequences:
            if seq.shape[0] > max_len:
                # Truncate
                padded_seq = seq[:max_len]
            else:
                # Pad with zeros using MLX
                pad_width = max_len - seq.shape[0]
                padded_seq = mx.pad(seq, pad_width=[(0, pad_width)], constant_values=0)
            padded.append(padded_seq)

        # Stack into batch
        return mx.stack(padded)
