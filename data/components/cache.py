"""Cache manager component for efficient data access.

This component handles caching of processed data, samples, and metadata
for improved performance.
"""

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
from loguru import logger


class CacheManager:
    """Manages caching for datasets and processed data."""

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        """Initialize the cache manager.

        Args:
            cache_dir: Directory for storing cache files
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._memory_cache: dict[str, Any] = {}
        self._cache_info: dict[str, dict] = {}

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_cache_info()

    def get(self, key: str, default: Any = None) -> Any:
        """Get item from cache.

        Args:
            key: Cache key
            default: Default value if not found

        Returns:
            Cached value or default
        """
        # Check memory cache first
        if key in self._memory_cache:
            logger.debug(f"Cache hit (memory): {key}")
            return self._memory_cache[key]

        # Check disk cache
        if self.cache_dir:
            cache_path = self._get_cache_path(key)
            if cache_path.exists():
                logger.debug(f"Cache hit (disk): {key}")
                value = self._load_from_disk(cache_path)
                # Store in memory for faster access
                self._memory_cache[key] = value
                return value

        logger.debug(f"Cache miss: {key}")
        return default

    def set(self, key: str, value: Any, persist: bool = True) -> None:
        """Store item in cache.

        Args:
            key: Cache key
            value: Value to cache
            persist: Whether to persist to disk
        """
        # Store in memory
        self._memory_cache[key] = value

        # Store on disk if requested
        if persist and self.cache_dir:
            cache_path = self._get_cache_path(key)
            self._save_to_disk(cache_path, value)
            
            # Update cache info
            self._cache_info[key] = {
                "size": cache_path.stat().st_size,
                "type": type(value).__name__,
                "path": str(cache_path),
            }
            self._save_cache_info()

        logger.debug(f"Cached: {key} (persist={persist})")

    def has(self, key: str) -> bool:
        """Check if key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if key exists
        """
        return key in self._memory_cache or (
            self.cache_dir and self._get_cache_path(key).exists()
        )

    def remove(self, key: str) -> None:
        """Remove item from cache.

        Args:
            key: Cache key
        """
        # Remove from memory
        self._memory_cache.pop(key, None)

        # Remove from disk
        if self.cache_dir:
            cache_path = self._get_cache_path(key)
            if cache_path.exists():
                cache_path.unlink()
                logger.debug(f"Removed from disk cache: {key}")

        # Update cache info
        self._cache_info.pop(key, None)
        if self.cache_dir:
            self._save_cache_info()

    def clear(self) -> None:
        """Clear all cached data."""
        # Clear memory cache
        self._memory_cache.clear()

        # Clear disk cache
        if self.cache_dir and self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            for cache_file in self.cache_dir.glob("*.parquet"):
                cache_file.unlink()
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()

        # Clear cache info
        self._cache_info.clear()
        if self.cache_dir:
            self._save_cache_info()

        logger.info("Cleared all cache")

    def get_cache_info(self) -> dict[str, Any]:
        """Get information about cached items.

        Returns:
            Dictionary with cache statistics
        """
        memory_size = sum(
            self._estimate_memory_size(v) for v in self._memory_cache.values()
        )

        disk_size = 0
        if self.cache_dir and self.cache_dir.exists():
            disk_size = sum(
                f.stat().st_size
                for f in self.cache_dir.iterdir()
                if f.is_file()
            )

        return {
            "memory_items": len(self._memory_cache),
            "disk_items": len(self._cache_info),
            "memory_size_mb": memory_size / 1024 / 1024,
            "disk_size_mb": disk_size / 1024 / 1024,
            "cache_dir": str(self.cache_dir) if self.cache_dir else None,
            "items": self._cache_info,
        }

    def cache_dataframe(self, key: str, df: pd.DataFrame, format: str = "parquet") -> None:
        """Cache a pandas DataFrame efficiently.

        Args:
            key: Cache key
            df: DataFrame to cache
            format: Storage format ('parquet' or 'pickle')
        """
        if not self.cache_dir:
            # Only use memory cache
            self._memory_cache[key] = df
            return

        if format == "parquet":
            cache_path = self.cache_dir / f"{key}.parquet"
            df.to_parquet(cache_path)
        else:
            cache_path = self.cache_dir / f"{key}.pkl"
            df.to_pickle(cache_path)

        # Update cache info
        self._cache_info[key] = {
            "size": cache_path.stat().st_size,
            "type": "DataFrame",
            "format": format,
            "shape": df.shape,
            "path": str(cache_path),
        }
        self._save_cache_info()

        logger.debug(f"Cached DataFrame: {key} (format={format})")

    def load_dataframe(self, key: str) -> Optional[pd.DataFrame]:
        """Load a cached DataFrame.

        Args:
            key: Cache key

        Returns:
            Cached DataFrame or None
        """
        # Check memory cache
        if key in self._memory_cache:
            return self._memory_cache[key]

        # Check disk cache
        if not self.cache_dir:
            return None

        # Try different formats
        parquet_path = self.cache_dir / f"{key}.parquet"
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            self._memory_cache[key] = df
            return df

        pickle_path = self.cache_dir / f"{key}.pkl"
        if pickle_path.exists():
            df = pd.read_pickle(pickle_path)
            self._memory_cache[key] = df
            return df

        return None

    def create_cache_key(self, *args, **kwargs) -> str:
        """Create a cache key from arguments.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Unique cache key
        """
        # Create a string representation of all arguments
        key_parts = [str(arg) for arg in args]
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        key_string = "|".join(key_parts)

        # Create hash for consistent key length
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_cache_path(self, key: str) -> Path:
        """Get the file path for a cache key."""
        return self.cache_dir / f"{key}.pkl"

    def _save_to_disk(self, path: Path, value: Any) -> None:
        """Save value to disk."""
        with open(path, "wb") as f:
            pickle.dump(value, f)

    def _load_from_disk(self, path: Path) -> Any:
        """Load value from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)

    def _load_cache_info(self) -> None:
        """Load cache information from disk."""
        info_path = self.cache_dir / "cache_info.json"
        if info_path.exists():
            with open(info_path, "r") as f:
                self._cache_info = json.load(f)

    def _save_cache_info(self) -> None:
        """Save cache information to disk."""
        if not self.cache_dir:
            return

        info_path = self.cache_dir / "cache_info.json"
        with open(info_path, "w") as f:
            json.dump(self._cache_info, f, indent=2)

    def _estimate_memory_size(self, obj: Any) -> int:
        """Estimate memory size of an object in bytes."""
        if isinstance(obj, pd.DataFrame):
            return obj.memory_usage(deep=True).sum()
        elif isinstance(obj, dict):
            return sum(self._estimate_memory_size(v) for v in obj.values())
        elif isinstance(obj, (list, tuple)):
            return sum(self._estimate_memory_size(v) for v in obj)
        else:
            # Rough estimate
            return len(pickle.dumps(obj))

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure cache info is saved."""
        if self.cache_dir:
            self._save_cache_info()