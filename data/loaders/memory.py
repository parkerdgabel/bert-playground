"""Unified memory manager for MLX on Apple Silicon.

This module provides advanced memory management capabilities that leverage
Apple Silicon's unified memory architecture for optimal performance.
"""

import gc
import threading
import time
import weakref
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import mlx.core as mx
from loguru import logger


@dataclass
class MemoryConfig:
    """Configuration for unified memory management."""
    
    # Memory limits
    max_unified_memory_mb: int = 8192  # 8GB default
    tensor_cache_limit_mb: int = 1024  # 1GB for tensor cache
    warning_threshold: float = 0.8     # Warn at 80% usage
    cleanup_threshold: float = 0.9     # Cleanup at 90% usage
    
    # Optimization settings
    enable_automatic_cleanup: bool = True
    enable_tensor_pooling: bool = True
    enable_lazy_allocation: bool = True
    
    # Monitoring
    monitoring_interval_seconds: float = 5.0
    enable_detailed_tracking: bool = False


class TensorPool:
    """Pool for reusing MLX tensors to reduce allocation overhead."""
    
    def __init__(self, max_size_mb: int = 512):
        """Initialize tensor pool.
        
        Args:
            max_size_mb: Maximum pool size in megabytes
        """
        self.max_size_mb = max_size_mb
        self._pools: Dict[Tuple[tuple, mx.Dtype], deque] = defaultdict(deque)
        self._current_size_mb = 0
        self._lock = threading.RLock()
        
    def get_tensor(self, shape: Tuple[int, ...], dtype: mx.Dtype) -> mx.array:
        """Get a tensor from the pool or create new one.
        
        Args:
            shape: Tensor shape
            dtype: Tensor data type
            
        Returns:
            MLX array
        """
        key = (shape, dtype)
        
        with self._lock:
            pool = self._pools[key]
            if pool:
                tensor = pool.popleft()
                # Zero out the tensor for reuse
                tensor = mx.zeros_like(tensor)
                return tensor
                
        # Create new tensor if pool is empty
        return mx.zeros(shape, dtype=dtype)
    
    def return_tensor(self, tensor: mx.array) -> None:
        """Return a tensor to the pool.
        
        Args:
            tensor: Tensor to return
        """
        if tensor.size == 0:
            return
            
        key = (tensor.shape, tensor.dtype)
        tensor_size_mb = tensor.nbytes / 1024 / 1024
        
        with self._lock:
            # Check if adding this tensor would exceed pool size
            if self._current_size_mb + tensor_size_mb <= self.max_size_mb:
                self._pools[key].append(tensor)
                self._current_size_mb += tensor_size_mb
            
    def clear(self) -> None:
        """Clear all pooled tensors."""
        with self._lock:
            self._pools.clear()
            self._current_size_mb = 0
            
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            return {
                'current_size_mb': self._current_size_mb,
                'max_size_mb': self.max_size_mb,
                'num_shapes': len(self._pools),
                'total_tensors': sum(len(pool) for pool in self._pools.values()),
            }


class UnifiedMemoryManager:
    """Advanced memory manager for MLX unified memory architecture.
    
    This manager optimizes memory usage by:
    1. Tracking tensor allocations across CPU/GPU
    2. Implementing smart cleanup strategies
    3. Providing tensor pooling for common shapes
    4. Monitoring memory pressure and responding automatically
    """
    
    _instance: Optional["UnifiedMemoryManager"] = None
    _lock = threading.Lock()
    
    def __new__(cls, config: Optional[MemoryConfig] = None) -> "UnifiedMemoryManager":
        """Singleton pattern for global memory management."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        """Initialize memory manager.
        
        Args:
            config: Memory management configuration
        """
        if getattr(self, '_initialized', False):
            return
            
        self.config = config or MemoryConfig()
        
        # Memory tracking
        self._allocated_tensors: Set[int] = set()
        self._tensor_sizes: Dict[int, int] = {}
        self._allocation_history: deque = deque(maxlen=1000)
        
        # Tensor pooling
        self.tensor_pool = TensorPool(self.config.tensor_cache_limit_mb)
        
        # Monitoring
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        
        # Performance tracking
        self._allocations_count = 0
        self._deallocations_count = 0
        self._pool_hits = 0
        self._pool_misses = 0
        
        self.logger = logger.bind(component="UnifiedMemoryManager")
        
        # Start monitoring if enabled
        if self.config.enable_automatic_cleanup:
            self.start_monitoring()
            
        self._initialized = True
        self.logger.info("Initialized UnifiedMemoryManager")
    
    def allocate_tensor(
        self,
        shape: Tuple[int, ...],
        dtype: mx.Dtype = mx.float32,
        use_pool: bool = True,
    ) -> mx.array:
        """Allocate a tensor with memory tracking.
        
        Args:
            shape: Tensor shape
            dtype: Data type
            use_pool: Whether to use tensor pool
            
        Returns:
            Allocated MLX array
        """
        if use_pool and self.config.enable_tensor_pooling:
            tensor = self.tensor_pool.get_tensor(shape, dtype)
            if id(tensor) in self._allocated_tensors:
                self._pool_hits += 1
            else:
                self._pool_misses += 1
        else:
            tensor = mx.zeros(shape, dtype=dtype)
            self._pool_misses += 1
            
        # Track allocation
        tensor_id = id(tensor)
        self._allocated_tensors.add(tensor_id)
        self._tensor_sizes[tensor_id] = tensor.nbytes
        self._allocations_count += 1
        
        # Record allocation history
        self._allocation_history.append({
            'timestamp': time.time(),
            'action': 'allocate',
            'shape': shape,
            'dtype': str(dtype),
            'size_bytes': tensor.nbytes,
        })
        
        return tensor
    
    def deallocate_tensor(self, tensor: mx.array, return_to_pool: bool = True) -> None:
        """Deallocate a tensor.
        
        Args:
            tensor: Tensor to deallocate
            return_to_pool: Whether to return to pool for reuse
        """
        tensor_id = id(tensor)
        
        if tensor_id in self._allocated_tensors:
            self._allocated_tensors.remove(tensor_id)
            size_bytes = self._tensor_sizes.pop(tensor_id, 0)
            self._deallocations_count += 1
            
            # Record deallocation
            self._allocation_history.append({
                'timestamp': time.time(),
                'action': 'deallocate',
                'shape': tensor.shape,
                'dtype': str(tensor.dtype),
                'size_bytes': size_bytes,
            })
            
            # Return to pool if requested
            if return_to_pool and self.config.enable_tensor_pooling:
                self.tensor_pool.return_tensor(tensor)
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics.
        
        Returns:
            Memory usage information
        """
        # Calculate tracked tensor memory
        tracked_memory_bytes = sum(self._tensor_sizes.values())
        tracked_memory_mb = tracked_memory_bytes / 1024 / 1024
        
        # Get MLX memory info if available
        try:
            # This would need to be implemented based on MLX's memory APIs
            # For now, use tracked memory as approximation
            unified_memory_mb = tracked_memory_mb
        except:
            unified_memory_mb = tracked_memory_mb
            
        return {
            'tracked_tensors': len(self._allocated_tensors),
            'tracked_memory_mb': tracked_memory_mb,
            'unified_memory_mb': unified_memory_mb,
            'max_memory_mb': self.config.max_unified_memory_mb,
            'memory_usage_percent': (unified_memory_mb / self.config.max_unified_memory_mb) * 100,
            'pool_stats': self.tensor_pool.get_stats(),
            'allocation_stats': {
                'allocations': self._allocations_count,
                'deallocations': self._deallocations_count,
                'pool_hits': self._pool_hits,
                'pool_misses': self._pool_misses,
                'pool_hit_rate': self._pool_hits / (self._pool_hits + self._pool_misses) if (self._pool_hits + self._pool_misses) > 0 else 0,
            }
        }
    
    def cleanup_memory(self, aggressive: bool = False) -> Dict[str, int]:
        """Perform memory cleanup.
        
        Args:
            aggressive: Whether to perform aggressive cleanup
            
        Returns:
            Cleanup statistics
        """
        self.logger.info(f"Starting memory cleanup (aggressive={aggressive})")
        
        initial_memory = self.get_memory_usage()
        
        # Clear tensor pool
        pool_tensors_cleared = len(self.tensor_pool._pools)
        self.tensor_pool.clear()
        
        # Force garbage collection
        gc.collect()
        
        # MLX-specific cleanup if available
        try:
            mx.eval([])  # Force evaluation of pending operations
        except:
            pass
            
        if aggressive:
            # More aggressive cleanup
            # Clear allocation history
            history_entries_cleared = len(self._allocation_history)
            self._allocation_history.clear()
            
            # Multiple GC passes
            for _ in range(3):
                gc.collect()
        else:
            history_entries_cleared = 0
            
        final_memory = self.get_memory_usage()
        memory_freed_mb = initial_memory['unified_memory_mb'] - final_memory['unified_memory_mb']
        
        cleanup_stats = {
            'memory_freed_mb': memory_freed_mb,
            'pool_tensors_cleared': pool_tensors_cleared,
            'history_entries_cleared': history_entries_cleared,
        }
        
        self.logger.info(f"Memory cleanup completed: freed {memory_freed_mb:.1f} MB")
        return cleanup_stats
    
    def start_monitoring(self) -> None:
        """Start automatic memory monitoring."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            return
            
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_worker,
            name="MemoryMonitor",
            daemon=True
        )
        self._monitoring_thread.start()
        
        self.logger.info("Started memory monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop automatic memory monitoring."""
        if not self._monitoring_thread:
            return
            
        self._stop_monitoring.set()
        self._monitoring_thread.join(timeout=5.0)
        self._monitoring_thread = None
        
        self.logger.info("Stopped memory monitoring")
    
    def _monitoring_worker(self) -> None:
        """Worker thread for memory monitoring."""
        while not self._stop_monitoring.is_set():
            try:
                memory_info = self.get_memory_usage()
                usage_percent = memory_info['memory_usage_percent']
                
                if usage_percent > self.config.cleanup_threshold * 100:
                    self.logger.warning(f"Memory usage high ({usage_percent:.1f}%), performing cleanup")
                    self.cleanup_memory(aggressive=True)
                elif usage_percent > self.config.warning_threshold * 100:
                    self.logger.warning(f"Memory usage warning ({usage_percent:.1f}%)")
                    
                # Log detailed stats if enabled
                if self.config.enable_detailed_tracking:
                    self.logger.debug(f"Memory stats: {memory_info}")
                    
            except Exception as e:
                self.logger.error(f"Memory monitoring error: {e}")
                
            # Wait for next monitoring cycle
            self._stop_monitoring.wait(self.config.monitoring_interval_seconds)
    
    def create_memory_context(self) -> "MemoryContext":
        """Create a memory context for automatic cleanup.
        
        Returns:
            Memory context manager
        """
        return MemoryContext(self)
    
    def get_optimization_hints(self) -> Dict[str, Any]:
        """Get optimization hints based on current memory usage.
        
        Returns:
            Optimization recommendations
        """
        memory_info = self.get_memory_usage()
        usage_percent = memory_info['memory_usage_percent']
        pool_hit_rate = memory_info['allocation_stats']['pool_hit_rate']
        
        hints = {
            'memory_pressure': 'low' if usage_percent < 50 else 'medium' if usage_percent < 80 else 'high',
            'recommendations': [],
        }
        
        if usage_percent > 80:
            hints['recommendations'].append('Consider reducing batch size')
            hints['recommendations'].append('Enable more aggressive cleanup')
            
        if pool_hit_rate < 0.5:
            hints['recommendations'].append('Review tensor shapes for better pooling')
            
        if memory_info['tracked_tensors'] > 1000:
            hints['recommendations'].append('Consider tensor lifetime management')
            
        return hints
    
    def __del__(self):
        """Cleanup when manager is destroyed."""
        self.stop_monitoring()


class MemoryContext:
    """Context manager for automatic memory cleanup."""
    
    def __init__(self, memory_manager: UnifiedMemoryManager):
        """Initialize memory context.
        
        Args:
            memory_manager: Memory manager instance
        """
        self.memory_manager = memory_manager
        self.initial_memory = None
        
    def __enter__(self) -> "MemoryContext":
        """Enter memory context."""
        self.initial_memory = self.memory_manager.get_memory_usage()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit memory context with cleanup."""
        # Perform cleanup
        self.memory_manager.cleanup_memory()
        
        final_memory = self.memory_manager.get_memory_usage()
        if self.initial_memory:
            memory_change = final_memory['unified_memory_mb'] - self.initial_memory['unified_memory_mb']
            logger.debug(f"Memory context: {memory_change:+.1f} MB change")