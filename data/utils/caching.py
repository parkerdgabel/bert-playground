"""
Caching utilities for the MLX dataloader system.
Provides efficient caching mechanisms for tokenized data and processed features.
"""

import os
import json
import pickle
import hashlib
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Callable
import numpy as np
import mlx.core as mx
from loguru import logger
from datetime import datetime
import threading
from collections import OrderedDict


class BaseCache:
    """Base class for caching implementations."""
    
    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        """
        Initialize cache.
        
        Args:
            cache_dir: Directory for cache storage
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "mlx_dataloader"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        # Create a deterministic string representation
        key_data = {
            "args": args,
            "kwargs": kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def clear(self) -> None:
        """Clear all cached data."""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Cleared cache at {self.cache_dir}")


class DiskCache(BaseCache):
    """Disk-based cache implementation."""
    
    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        max_size_mb: Optional[int] = None,
        ttl_hours: Optional[int] = None,
    ):
        """
        Initialize disk cache.
        
        Args:
            cache_dir: Directory for cache storage
            max_size_mb: Maximum cache size in MB
            ttl_hours: Time to live in hours
        """
        super().__init__(cache_dir)
        self.max_size_mb = max_size_mb
        self.ttl_hours = ttl_hours
        
        # Create metadata file
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self) -> None:
        """Save cache metadata."""
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f)
    
    def _check_ttl(self, key: str) -> bool:
        """Check if cached item is still valid."""
        if self.ttl_hours is None:
            return True
        
        if key in self.metadata:
            created_at = datetime.fromisoformat(self.metadata[key]["created_at"])
            age_hours = (datetime.now() - created_at).total_seconds() / 3600
            return age_hours < self.ttl_hours
        
        return False
    
    def _check_size_limit(self) -> None:
        """Check and enforce size limit."""
        if self.max_size_mb is None:
            return
        
        # Calculate total size
        total_size = 0
        for file in self.cache_dir.glob("*.pkl"):
            total_size += file.stat().st_size
        
        total_size_mb = total_size / (1024 * 1024)
        
        # Remove old files if over limit
        if total_size_mb > self.max_size_mb:
            # Sort by access time
            files = [(f, f.stat().st_atime) for f in self.cache_dir.glob("*.pkl")]
            files.sort(key=lambda x: x[1])
            
            # Remove oldest files
            while total_size_mb > self.max_size_mb * 0.9 and files:  # Keep 90% after cleanup
                file, _ = files.pop(0)
                size_mb = file.stat().st_size / (1024 * 1024)
                file.unlink()
                total_size_mb -= size_mb
                
                # Remove from metadata
                key = file.stem
                if key in self.metadata:
                    del self.metadata[key]
            
            self._save_metadata()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        cache_file = self.cache_dir / f"{key}.pkl"
        
        if cache_file.exists() and self._check_ttl(key):
            try:
                with open(cache_file, "rb") as f:
                    data = pickle.load(f)
                logger.debug(f"Cache hit: {key}")
                return data
            except Exception as e:
                logger.warning(f"Failed to load cache {key}: {e}")
                cache_file.unlink()
                if key in self.metadata:
                    del self.metadata[key]
                    self._save_metadata()
        
        logger.debug(f"Cache miss: {key}")
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set item in cache."""
        self._check_size_limit()
        
        cache_file = self.cache_dir / f"{key}.pkl"
        
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(value, f)
            
            # Update metadata
            self.metadata[key] = {
                "created_at": datetime.now().isoformat(),
                "size": cache_file.stat().st_size,
            }
            self._save_metadata()
            
            logger.debug(f"Cached: {key}")
        except Exception as e:
            logger.warning(f"Failed to cache {key}: {e}")
            if cache_file.exists():
                cache_file.unlink()
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        cache_file = self.cache_dir / f"{key}.pkl"
        return cache_file.exists() and self._check_ttl(key)


class MemoryCache(BaseCache):
    """Memory-based cache with LRU eviction."""
    
    def __init__(
        self,
        max_items: int = 1000,
        max_memory_mb: Optional[int] = None,
    ):
        """
        Initialize memory cache.
        
        Args:
            max_items: Maximum number of items
            max_memory_mb: Maximum memory usage in MB
        """
        super().__init__()
        self.max_items = max_items
        self.max_memory_mb = max_memory_mb
        self.cache: OrderedDict = OrderedDict()
        self.lock = threading.Lock()
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        if isinstance(obj, (mx.array, np.ndarray)):
            return obj.nbytes
        elif isinstance(obj, dict):
            total = 0
            for k, v in obj.items():
                total += len(str(k).encode()) + self._estimate_size(v)
            return total
        elif isinstance(obj, (list, tuple)):
            return sum(self._estimate_size(item) for item in obj)
        else:
            return len(pickle.dumps(obj))
    
    def _check_memory_limit(self) -> None:
        """Check and enforce memory limit."""
        if self.max_memory_mb is None:
            return
        
        # Estimate total memory usage
        total_bytes = sum(self._estimate_size(v) for v in self.cache.values())
        total_mb = total_bytes / (1024 * 1024)
        
        # Evict if over limit
        while total_mb > self.max_memory_mb and self.cache:
            # Remove least recently used
            self.cache.popitem(last=False)
            total_bytes = sum(self._estimate_size(v) for v in self.cache.values())
            total_mb = total_bytes / (1024 * 1024)
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                return self.cache[key]
            return None
    
    def set(self, key: str, value: Any) -> None:
        """Set item in cache."""
        with self.lock:
            # Remove if exists to update position
            if key in self.cache:
                del self.cache[key]
            
            # Add to end
            self.cache[key] = value
            
            # Check item limit
            if len(self.cache) > self.max_items:
                self.cache.popitem(last=False)
            
            # Check memory limit
            self._check_memory_limit()
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        return key in self.cache
    
    def clear(self) -> None:
        """Clear cache."""
        with self.lock:
            self.cache.clear()


class TokenizationCache:
    """Specialized cache for tokenized data."""
    
    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        tokenizer_name: Optional[str] = None,
    ):
        """
        Initialize tokenization cache.
        
        Args:
            cache_dir: Cache directory
            tokenizer_name: Tokenizer identifier
        """
        self.base_cache = DiskCache(cache_dir)
        self.tokenizer_name = tokenizer_name
    
    def get_key(
        self,
        text: str,
        max_length: int,
        padding: Union[bool, str],
        truncation: bool,
    ) -> str:
        """Generate cache key for tokenization."""
        key_data = {
            "text": text,
            "max_length": max_length,
            "padding": padding,
            "truncation": truncation,
            "tokenizer": self.tokenizer_name,
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get_tokenized(
        self,
        text: str,
        max_length: int,
        padding: Union[bool, str],
        truncation: bool,
    ) -> Optional[Dict[str, List[int]]]:
        """Get tokenized data from cache."""
        key = self.get_key(text, max_length, padding, truncation)
        return self.base_cache.get(key)
    
    def set_tokenized(
        self,
        text: str,
        max_length: int,
        padding: Union[bool, str],
        truncation: bool,
        encoding: Dict[str, List[int]],
    ) -> None:
        """Cache tokenized data."""
        key = self.get_key(text, max_length, padding, truncation)
        self.base_cache.set(key, encoding)


class ComputedCache:
    """Cache for expensive computations."""
    
    def __init__(
        self,
        compute_func: Callable,
        cache: Optional[BaseCache] = None,
        cache_key_func: Optional[Callable] = None,
    ):
        """
        Initialize computed cache.
        
        Args:
            compute_func: Function to compute values
            cache: Cache backend
            cache_key_func: Function to generate cache keys
        """
        self.compute_func = compute_func
        self.cache = cache or MemoryCache()
        self.cache_key_func = cache_key_func or self._default_key_func
    
    def _default_key_func(self, *args, **kwargs) -> str:
        """Default cache key function."""
        key_data = {"args": args, "kwargs": kwargs}
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def __call__(self, *args, **kwargs) -> Any:
        """Get or compute value."""
        # Generate cache key
        key = self.cache_key_func(*args, **kwargs)
        
        # Check cache
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        
        # Compute value
        result = self.compute_func(*args, **kwargs)
        
        # Cache result
        self.cache.set(key, result)
        
        return result
    
    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()


class DatasetCache:
    """Cache for entire datasets or batches."""
    
    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        cache_format: str = "npz",  # "npz", "safetensors", "pickle"
    ):
        """
        Initialize dataset cache.
        
        Args:
            cache_dir: Cache directory
            cache_format: Format for saving arrays
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "mlx_datasets"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_format = cache_format
    
    def save_dataset(
        self,
        dataset_name: str,
        data: Dict[str, Union[mx.array, np.ndarray, List]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save dataset to cache."""
        dataset_dir = self.cache_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        
        if self.cache_format == "npz":
            # Convert MLX arrays to numpy
            np_data = {}
            for key, value in data.items():
                if isinstance(value, mx.array):
                    np_data[key] = np.array(value)
                elif isinstance(value, list):
                    np_data[key] = np.array(value)
                else:
                    np_data[key] = value
            
            np.savez_compressed(dataset_dir / "data.npz", **np_data)
            
        elif self.cache_format == "safetensors":
            # Save using MLX safetensors
            mx.save_safetensors(dataset_dir / "data.safetensors", data)
            
        elif self.cache_format == "pickle":
            with open(dataset_dir / "data.pkl", "wb") as f:
                pickle.dump(data, f)
        
        # Save metadata
        if metadata:
            with open(dataset_dir / "metadata.json", "w") as f:
                json.dump(metadata, f)
        
        logger.info(f"Cached dataset: {dataset_name}")
    
    def load_dataset(
        self,
        dataset_name: str
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Load dataset from cache."""
        dataset_dir = self.cache_dir / dataset_name
        
        if not dataset_dir.exists():
            return None
        
        data = None
        metadata = {}
        
        # Load data
        if self.cache_format == "npz":
            data_file = dataset_dir / "data.npz"
            if data_file.exists():
                np_data = np.load(data_file)
                data = {key: mx.array(np_data[key]) for key in np_data.files}
                
        elif self.cache_format == "safetensors":
            data_file = dataset_dir / "data.safetensors"
            if data_file.exists():
                data = mx.load(str(data_file))
                
        elif self.cache_format == "pickle":
            data_file = dataset_dir / "data.pkl"
            if data_file.exists():
                with open(data_file, "rb") as f:
                    data = pickle.load(f)
        
        # Load metadata
        metadata_file = dataset_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
        
        if data is not None:
            logger.info(f"Loaded cached dataset: {dataset_name}")
            return data, metadata
        
        return None
    
    def exists(self, dataset_name: str) -> bool:
        """Check if dataset exists in cache."""
        dataset_dir = self.cache_dir / dataset_name
        return dataset_dir.exists()
    
    def list_cached(self) -> List[str]:
        """List all cached datasets."""
        return [d.name for d in self.cache_dir.iterdir() if d.is_dir()]
    
    def clear_dataset(self, dataset_name: str) -> None:
        """Clear specific dataset from cache."""
        dataset_dir = self.cache_dir / dataset_name
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
            logger.info(f"Cleared cached dataset: {dataset_name}")