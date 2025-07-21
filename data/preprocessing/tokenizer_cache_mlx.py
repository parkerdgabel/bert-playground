"""Pre-tokenization cache for efficient data loading with MLX.

This module provides a caching system that pre-tokenizes entire datasets
and stores them as MLX arrays in unified memory for zero-copy operations.

Note: This is the MLX-optimized version. The transformers library still
returns numpy arrays, but we immediately convert them to MLX for storage
and processing.
"""

import hashlib
import json
from pathlib import Path

import mlx.core as mx
from loguru import logger
from transformers import PreTrainedTokenizer


class TokenizerCache:
    """Cache pre-tokenized datasets for efficient MLX training.

    This class handles:
    - Pre-tokenizing entire datasets into MLX arrays
    - Caching tokenized data to disk for reuse
    - Zero-copy loading from cache
    - Unified memory management for Apple Silicon

    All data is stored as MLX arrays for optimal performance.
    """

    def __init__(
        self,
        cache_dir: str | Path = "data/cache/tokenized",
        max_length: int = 256,
        tokenizer: PreTrainedTokenizer | None = None,
    ):
        """Initialize the tokenizer cache.

        Args:
            cache_dir: Directory to store cached tokenized data
            max_length: Maximum sequence length for tokenization
            tokenizer: Pre-trained tokenizer instance
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_length = max_length
        self.tokenizer = tokenizer

    def _get_cache_key(
        self,
        dataset_path: str | Path,
        tokenizer_name: str,
        max_length: int,
        split: str = "train",
    ) -> str:
        """Generate a unique cache key for the dataset configuration."""
        # Create a unique key based on dataset, tokenizer, and parameters
        key_parts = [str(dataset_path), tokenizer_name, str(max_length), split]
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the cache file path for a given cache key."""
        return self.cache_dir / f"{cache_key}.safetensors"

    def tokenize_and_cache(
        self,
        texts: list[str],
        labels: list[int] | None = None,
        dataset_path: str | Path | None = None,
        split: str = "train",
        force_rebuild: bool = False,
    ) -> dict[str, mx.array]:
        """Pre-tokenize texts and cache as MLX arrays.

        Args:
            texts: List of text samples
            labels: Optional list of labels
            dataset_path: Path to original dataset (for cache key)
            split: Dataset split name
            force_rebuild: Force rebuilding cache even if exists

        Returns:
            Dictionary of MLX arrays with tokenized data
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be provided to tokenize data")

        # Generate cache key
        cache_key = self._get_cache_key(
            dataset_path or "unknown",
            self.tokenizer.name_or_path,
            self.max_length,
            split,
        )
        cache_path = self._get_cache_path(cache_key)

        # Check if cache exists and load if available
        if not force_rebuild and cache_path.exists():
            logger.info(f"Loading pre-tokenized data from cache: {cache_path}")
            return self.load_from_cache(cache_path)

        logger.info(f"Pre-tokenizing {len(texts)} samples...")

        # Tokenize all texts at once for efficiency
        # Note: Transformers returns numpy arrays, but we'll convert to MLX
        encoded = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )

        # Convert to MLX arrays for unified memory and optimal performance
        # Use int32 dtype for MLX compilation compatibility
        tokenized_data = {
            "input_ids": mx.array(encoded["input_ids"], dtype=mx.int32),
            "attention_mask": mx.array(encoded["attention_mask"], dtype=mx.int32),
        }

        # Add token_type_ids if available
        if "token_type_ids" in encoded:
            tokenized_data["token_type_ids"] = mx.array(encoded["token_type_ids"], dtype=mx.int32)

        # Add labels if provided
        if labels is not None:
            tokenized_data["labels"] = mx.array(labels)

        # Save to cache
        self.save_to_cache(tokenized_data, cache_path)

        if cache_path.exists():
            logger.info(f"Cached pre-tokenized data to: {cache_path}")
            logger.info(f"Cache size: {cache_path.stat().st_size / 1024 / 1024:.2f} MB")
        else:
            logger.error(f"Failed to cache data to: {cache_path}")

        return tokenized_data

    def save_to_cache(self, data: dict[str, mx.array], cache_path: Path):
        """Save pre-tokenized data to cache.

        Uses safetensors format for efficient loading.
        All data is stored as MLX arrays.
        """
        # Save using MLX's native safetensors format
        mx.save_safetensors(str(cache_path), data)

        # Also save metadata
        metadata = {
            "num_samples": data["input_ids"].shape[0],
            "max_length": data["input_ids"].shape[1],
            "keys": list(data.keys()),
            "shapes": {k: list(v.shape) for k, v in data.items()},
            "dtypes": {k: str(v.dtype) for k, v in data.items()},
            "format": "mlx_safetensors",
        }

        metadata_path = cache_path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def load_from_cache(self, cache_path: Path) -> dict[str, mx.array]:
        """Load pre-tokenized data from cache.

        Returns MLX arrays in unified memory for zero-copy operations.
        """
        # Load metadata first
        metadata_path = cache_path.with_suffix(".json")
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            logger.debug(f"Loading cache with metadata: {metadata}")

        # Load safetensors file - returns dict of MLX arrays
        data = mx.load(str(cache_path))

        return data

    def get_cache_info(self) -> dict[str, dict]:
        """Get information about all cached datasets."""
        cache_info = {}

        for cache_file in self.cache_dir.glob("*.safetensors"):
            metadata_file = cache_file.with_suffix(".json")
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)

                cache_info[cache_file.stem] = {
                    "path": str(cache_file),
                    "size_mb": cache_file.stat().st_size / 1024 / 1024,
                    **metadata,
                }

        return cache_info

    def clear_cache(self):
        """Clear all cached tokenized data."""
        for cache_file in self.cache_dir.glob("*.safetensors"):
            cache_file.unlink()
            metadata_file = cache_file.with_suffix(".json")
            if metadata_file.exists():
                metadata_file.unlink()

        logger.info(f"Cleared tokenizer cache at: {self.cache_dir}")


class PreTokenizedDataset:
    """Dataset wrapper for pre-tokenized MLX arrays.

    This provides a simple interface for accessing pre-tokenized data
    without any tokenization overhead during training.

    All operations use MLX arrays for zero-copy slicing.
    """

    def __init__(self, tokenized_data: dict[str, mx.array]):
        """Initialize with pre-tokenized data.

        Args:
            tokenized_data: Dictionary of MLX arrays
        """
        self.data = tokenized_data
        self.num_samples = tokenized_data["input_ids"].shape[0]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, mx.array]:
        """Get a single sample (zero-copy slice)."""
        return {k: v[idx] for k, v in self.data.items()}

    def get_batch(self, indices: list[int] | mx.array) -> dict[str, mx.array]:
        """Get a batch of samples (zero-copy slice).

        Args:
            indices: List of indices or MLX array of indices

        Returns:
            Dictionary of MLX arrays for the batch
        """
        # Convert list to MLX array if needed
        if isinstance(indices, list):
            indices = mx.array(indices, dtype=mx.int32)

        return {k: v[indices] for k, v in self.data.items()}

    @property
    def shape_info(self) -> dict[str, tuple[int, ...]]:
        """Get shape information for all arrays."""
        return {k: tuple(v.shape) for k, v in self.data.items()}
