"""Enhanced unified data loader with additional features from mlx_enhanced_loader.

This module extends the unified loader with:
- Persistent pickle-based caching
- Dynamic batch size updates
- Explicit test data handling
- Double buffering support
"""

import os
import pickle
from pathlib import Path
from typing import Optional, Union

import mlx.core as mx
from loguru import logger

from .unified_loader import UnifiedTitanicDataPipeline, OptimizationLevel


class EnhancedUnifiedDataPipeline(UnifiedTitanicDataPipeline):
    """Enhanced unified data pipeline with additional optimizations."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer_name: str = "answerdotai/ModernBERT-base", 
        max_length: int = 256,
        batch_size: int = 32,
        is_training: bool = True,
        augment: bool = True,
        is_test: bool = False,  # Explicit test data handling
        # Optimization parameters
        optimization_level: Union[str, OptimizationLevel] = OptimizationLevel.AUTO,
        prefetch_size: int = 4,
        num_threads: int = 4,
        cache_size: int = 1000,
        use_mlx_data: Optional[bool] = None,
        pre_tokenize: Optional[bool] = None,
        # Enhanced features
        persistent_cache: Optional[bool] = None,  # Auto-enabled for OPTIMIZED
        enable_dynamic_batch: bool = False,  # Allow batch size updates
        cache_dir: str = "cache/tokenized",  # Directory for persistent cache
        double_buffer: bool = False,  # Use double buffering
    ):
        """Initialize enhanced unified data pipeline.
        
        Args:
            All args from UnifiedTitanicDataPipeline plus:
            is_test: Whether this is test data (no labels)
            persistent_cache: Enable pickle-based caching
            enable_dynamic_batch: Allow batch size updates
            cache_dir: Directory for persistent cache files
            double_buffer: Use double buffering for prefetch
        """
        self.is_test = is_test
        self.enable_dynamic_batch = enable_dynamic_batch
        self.cache_dir = Path(cache_dir)
        self.double_buffer = double_buffer
        
        # Auto-enable persistent cache for OPTIMIZED level
        if persistent_cache is None and optimization_level == OptimizationLevel.OPTIMIZED:
            persistent_cache = True
        self.persistent_cache = persistent_cache or False
        
        # Create cache directory if needed
        if self.persistent_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize base class
        # For test data, we don't have labels
        super().__init__(
            data_path=data_path,
            tokenizer_name=tokenizer_name,
            max_length=max_length,
            batch_size=batch_size,
            is_training=is_training,
            augment=augment,
            optimization_level=optimization_level,
            prefetch_size=prefetch_size,
            num_threads=num_threads,
            cache_size=cache_size,
            use_mlx_data=use_mlx_data,
            pre_tokenize=pre_tokenize,
        )
        
        # Load persistent cache if available
        if self.persistent_cache:
            self._load_persistent_cache()
    
    def prepare_data(self):
        """Override to handle test data and persistent caching."""
        # Check persistent cache first
        if self.persistent_cache:
            cached_data = self._try_load_cache()
            if cached_data is not None:
                logger.info("Loaded data from persistent cache")
                return cached_data
        
        # Use parent's prepare_data but handle test data
        if self.is_test:
            # Temporarily disable label processing
            original_df = self.df.copy()
            if "Survived" not in self.df.columns:
                # Add dummy labels for processing
                self.df["Survived"] = -1
            
            data = super().prepare_data()
            
            # Remove labels from result
            data.pop("labels", None)
            
            # Restore original df
            self.df = original_df
        else:
            data = super().prepare_data()
        
        # Save to persistent cache if enabled
        if self.persistent_cache:
            self._save_cache(data)
        
        return data
    
    def update_batch_size(self, new_batch_size: int):
        """Dynamically update batch size."""
        if not self.enable_dynamic_batch:
            logger.warning("Dynamic batch size updates are disabled")
            return
        
        old_batch_size = self.batch_size
        self.batch_size = new_batch_size
        
        logger.info(f"Updated batch size: {old_batch_size} -> {new_batch_size}")
        
        # Recreate stream if using MLX-Data
        if self.use_mlx_data and self.stream is not None:
            logger.info("Recreating MLX-Data stream with new batch size")
            self.stream = self._create_mlx_stream()
    
    def _get_cache_key(self) -> str:
        """Generate cache key based on data characteristics."""
        import hashlib
        
        # Create a unique key based on file path, size, and settings
        key_parts = [
            str(self.data_path),
            str(self.df.shape),
            self.tokenizer.name_or_path,
            str(self.max_length),
            str(self.augment),
            str(self.is_test),
        ]
        
        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _try_load_cache(self) -> Optional[dict]:
        """Try to load data from persistent cache."""
        cache_key = self._get_cache_key()
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    data = pickle.load(f)
                logger.info(f"Loaded cached data from {cache_file}")
                return data
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        
        return None
    
    def _save_cache(self, data: dict):
        """Save data to persistent cache."""
        cache_key = self._get_cache_key()
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Saved data to cache: {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _load_persistent_cache(self):
        """Load all tokenized data from persistent cache if available."""
        if not self.pre_tokenize:
            return
        
        cached_data = self._try_load_cache()
        if cached_data is not None and "input_ids" in cached_data:
            # Replace tokenized data
            self.tokenized_cache = {}
            for i in range(len(cached_data["input_ids"])):
                self.tokenized_cache[i] = {
                    "input_ids": cached_data["input_ids"][i],
                    "attention_mask": cached_data["attention_mask"][i],
                }
            logger.info(f"Loaded {len(self.tokenized_cache)} tokenized samples from cache")
    
    def _create_mlx_stream(self):
        """Override to support double buffering."""
        stream = super()._create_mlx_stream()
        
        if self.double_buffer and stream is not None:
            # Add double buffering
            logger.info("Enabling double buffering for MLX stream")
            stream = stream.prefetch(2 * self.prefetch_size, self.num_threads)
        
        return stream


# Convenience function
def create_enhanced_dataloaders(
    train_path: str,
    val_path: Optional[str] = None,
    test_path: Optional[str] = None,
    tokenizer_name: str = "answerdotai/ModernBERT-base",
    batch_size: int = 32,
    max_length: int = 256,
    optimization_level: str = "auto",
    **kwargs
):
    """Create enhanced data loaders with all optimizations.
    
    Returns:
        tuple: (train_loader, val_loader, test_loader) - val/test can be None
    """
    # Create training loader
    train_loader = EnhancedUnifiedDataPipeline(
        data_path=train_path,
        tokenizer_name=tokenizer_name,
        batch_size=batch_size,
        max_length=max_length,
        is_training=True,
        optimization_level=optimization_level,
        **kwargs
    )
    
    # Create validation loader
    val_loader = None
    if val_path:
        val_loader = EnhancedUnifiedDataPipeline(
            data_path=val_path,
            tokenizer_name=tokenizer_name,
            batch_size=batch_size,
            max_length=max_length,
            is_training=False,
            augment=False,
            optimization_level=optimization_level,
            **kwargs
        )
    
    # Create test loader
    test_loader = None
    if test_path:
        test_loader = EnhancedUnifiedDataPipeline(
            data_path=test_path,
            tokenizer_name=tokenizer_name,
            batch_size=batch_size,
            max_length=max_length,
            is_training=False,
            is_test=True,
            augment=False,
            optimization_level=optimization_level,
            **kwargs
        )
    
    return train_loader, val_loader, test_loader