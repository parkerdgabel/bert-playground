"""
Cache configuration for dataloaders.
"""

from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger

from data.utils.caching import (
    BaseCache,
    DiskCache,
    MemoryCache,
    TokenizationCache,
    DatasetCache,
)


@dataclass
class CacheConfig:
    """Configuration for caching."""
    
    # Cache type
    cache_type: str = "disk"  # disk, memory, hybrid
    
    # Common settings
    enabled: bool = True
    cache_dir: Optional[Union[str, Path]] = None
    
    # Disk cache settings
    disk_max_size_mb: Optional[int] = None
    disk_ttl_hours: Optional[int] = None
    
    # Memory cache settings
    memory_max_items: int = 1000
    memory_max_size_mb: Optional[int] = None
    
    # Cache levels
    cache_tokenized: bool = True
    cache_converted_text: bool = True
    cache_transformed: bool = False
    cache_final_batches: bool = False
    
    # Dataset cache
    dataset_cache_enabled: bool = False
    dataset_cache_format: str = "npz"  # npz, safetensors, pickle
    
    # Advanced settings
    compression: bool = True
    async_writes: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cache_type": self.cache_type,
            "enabled": self.enabled,
            "cache_dir": str(self.cache_dir) if self.cache_dir else None,
            "disk_max_size_mb": self.disk_max_size_mb,
            "disk_ttl_hours": self.disk_ttl_hours,
            "memory_max_items": self.memory_max_items,
            "memory_max_size_mb": self.memory_max_size_mb,
            "cache_tokenized": self.cache_tokenized,
            "cache_converted_text": self.cache_converted_text,
            "cache_transformed": self.cache_transformed,
            "cache_final_batches": self.cache_final_batches,
            "dataset_cache_enabled": self.dataset_cache_enabled,
            "dataset_cache_format": self.dataset_cache_format,
            "compression": self.compression,
            "async_writes": self.async_writes,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "CacheConfig":
        """Create from dictionary."""
        # Handle Path conversion
        if "cache_dir" in config_dict and config_dict["cache_dir"]:
            config_dict["cache_dir"] = Path(config_dict["cache_dir"])
        return cls(**config_dict)
    
    def create_cache(self, cache_name: str = "main") -> BaseCache:
        """
        Create cache instance based on configuration.
        
        Args:
            cache_name: Name for the cache
            
        Returns:
            Cache instance
        """
        if not self.enabled:
            return None
        
        cache_dir = self.cache_dir
        if cache_dir:
            cache_dir = Path(cache_dir) / cache_name
        
        if self.cache_type == "disk":
            return DiskCache(
                cache_dir=cache_dir,
                max_size_mb=self.disk_max_size_mb,
                ttl_hours=self.disk_ttl_hours,
            )
        
        elif self.cache_type == "memory":
            return MemoryCache(
                max_items=self.memory_max_items,
                max_memory_mb=self.memory_max_size_mb,
            )
        
        elif self.cache_type == "hybrid":
            # Create hybrid cache with memory L1 and disk L2
            return HybridCache(
                memory_cache=MemoryCache(
                    max_items=min(self.memory_max_items, 100),
                    max_memory_mb=self.memory_max_size_mb,
                ),
                disk_cache=DiskCache(
                    cache_dir=cache_dir,
                    max_size_mb=self.disk_max_size_mb,
                    ttl_hours=self.disk_ttl_hours,
                )
            )
        
        else:
            raise ValueError(f"Unknown cache type: {self.cache_type}")
    
    def create_tokenization_cache(self, tokenizer_name: str) -> Optional[TokenizationCache]:
        """Create tokenization cache if enabled."""
        if not self.enabled or not self.cache_tokenized:
            return None
        
        cache_dir = self.cache_dir
        if cache_dir:
            cache_dir = Path(cache_dir) / "tokenization"
        
        return TokenizationCache(
            cache_dir=cache_dir,
            tokenizer_name=tokenizer_name,
        )
    
    def create_dataset_cache(self) -> Optional[DatasetCache]:
        """Create dataset cache if enabled."""
        if not self.enabled or not self.dataset_cache_enabled:
            return None
        
        cache_dir = self.cache_dir
        if cache_dir:
            cache_dir = Path(cache_dir) / "datasets"
        
        return DatasetCache(
            cache_dir=cache_dir,
            cache_format=self.dataset_cache_format,
        )


class HybridCache(BaseCache):
    """Hybrid cache with memory L1 and disk L2."""
    
    def __init__(
        self,
        memory_cache: MemoryCache,
        disk_cache: DiskCache,
    ):
        """
        Initialize hybrid cache.
        
        Args:
            memory_cache: Memory cache (L1)
            disk_cache: Disk cache (L2)
        """
        super().__init__(disk_cache.cache_dir if disk_cache else None)
        self.memory_cache = memory_cache
        self.disk_cache = disk_cache
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache (check L1 then L2)."""
        # Check memory cache first
        value = self.memory_cache.get(key)
        if value is not None:
            return value
        
        # Check disk cache
        value = self.disk_cache.get(key)
        if value is not None:
            # Promote to memory cache
            self.memory_cache.set(key, value)
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set item in cache (both L1 and L2)."""
        # Set in memory cache
        self.memory_cache.set(key, value)
        
        # Set in disk cache
        self.disk_cache.set(key, value)
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        return self.memory_cache.exists(key) or self.disk_cache.exists(key)
    
    def clear(self) -> None:
        """Clear both caches."""
        self.memory_cache.clear()
        self.disk_cache.clear()


def create_cache(
    config: Union[Dict[str, Any], CacheConfig],
    cache_name: str = "main"
) -> Optional[BaseCache]:
    """
    Create cache from configuration.
    
    Args:
        config: Cache configuration
        cache_name: Name for the cache
        
    Returns:
        Cache instance or None
    """
    if isinstance(config, dict):
        cache_config = CacheConfig.from_dict(config)
    else:
        cache_config = config
    
    return cache_config.create_cache(cache_name)


# Preset cache configurations
PRESET_CACHE_CONFIGS = {
    "disabled": CacheConfig(enabled=False),
    
    "minimal": CacheConfig(
        cache_type="memory",
        memory_max_items=100,
        cache_tokenized=True,
        cache_converted_text=False,
        cache_transformed=False,
    ),
    
    "standard": CacheConfig(
        cache_type="disk",
        disk_max_size_mb=1000,
        cache_tokenized=True,
        cache_converted_text=True,
        cache_transformed=False,
    ),
    
    "aggressive": CacheConfig(
        cache_type="hybrid",
        memory_max_items=1000,
        memory_max_size_mb=500,
        disk_max_size_mb=5000,
        cache_tokenized=True,
        cache_converted_text=True,
        cache_transformed=True,
        dataset_cache_enabled=True,
    ),
    
    "development": CacheConfig(
        cache_type="memory",
        memory_max_items=500,
        cache_tokenized=True,
        cache_converted_text=True,
        cache_transformed=False,
        disk_ttl_hours=1,  # Short TTL for development
    ),
}


def get_preset_cache_config(name: str) -> CacheConfig:
    """
    Get preset cache configuration.
    
    Args:
        name: Preset name
        
    Returns:
        Cache configuration
    """
    if name not in PRESET_CACHE_CONFIGS:
        raise ValueError(
            f"Unknown preset: {name}. "
            f"Available: {list(PRESET_CACHE_CONFIGS.keys())}"
        )
    
    # Return a copy
    preset = PRESET_CACHE_CONFIGS[name]
    return CacheConfig.from_dict(preset.to_dict())