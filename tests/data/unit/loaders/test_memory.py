"""Tests for unified memory management."""

import pytest
from unittest.mock import Mock, patch
import gc
import weakref

import mlx.core as mx

from data.loaders.memory import UnifiedMemoryManager, MemoryConfig
from tests.data.fixtures.configs import create_memory_config
from tests.data.fixtures.utils import (
    check_memory_usage,
    create_large_tensor,
    measure_memory_allocation_time,
)


class TestMemoryConfig:
    """Test MemoryConfig dataclass."""
    
    def test_default_config(self):
        """Test default memory configuration."""
        config = MemoryConfig()
        
        assert config.pool_size_mb == 512
        assert config.max_cache_size_mb == 256
        assert config.enable_pooling == True
        assert config.auto_cleanup == True
        assert config.enable_unified_memory == True
        
    def test_custom_config(self):
        """Test custom memory configuration."""
        config = create_memory_config(
            pool_size_mb=1024,
            max_cache_size_mb=512,
            enable_pooling=False,
            auto_cleanup=False,
            zero_copy_threshold_mb=20,
        )
        
        assert config.pool_size_mb == 1024
        assert config.max_cache_size_mb == 512
        assert config.enable_pooling == False
        assert config.auto_cleanup == False
        assert config.zero_copy_threshold_mb == 20
        
    def test_validation(self):
        """Test configuration validation."""
        # Pool size must be positive
        with pytest.raises(ValueError):
            MemoryConfig(pool_size_mb=0)
            
        # Cache size must be positive
        with pytest.raises(ValueError):
            MemoryConfig(max_cache_size_mb=-1)
            
        # Cache size shouldn't exceed pool size
        with pytest.raises(ValueError):
            MemoryConfig(pool_size_mb=100, max_cache_size_mb=200)


class TestUnifiedMemoryManager:
    """Test UnifiedMemoryManager class."""
    
    @pytest.fixture
    def memory_config(self):
        """Create memory configuration."""
        return create_memory_config(
            pool_size_mb=256,
            max_cache_size_mb=128,
            enable_pooling=True,
            auto_cleanup=True,
        )
        
    def test_memory_manager_creation(self, memory_config):
        """Test memory manager creation."""
        manager = UnifiedMemoryManager(memory_config)
        
        assert manager.config == memory_config
        assert hasattr(manager, '_pool')
        assert hasattr(manager, '_cache')
        assert hasattr(manager, '_allocations')
        
    def test_tensor_allocation(self, memory_config):
        """Test tensor allocation from pool."""
        manager = UnifiedMemoryManager(memory_config)
        
        # Allocate tensor
        tensor = manager.allocate_tensor(shape=(32, 128), dtype=mx.float32)
        
        assert isinstance(tensor, mx.array)
        assert tensor.shape == (32, 128)
        assert tensor.dtype == mx.float32
        
    def test_tensor_allocation_different_types(self, memory_config):
        """Test allocation of different tensor types."""
        manager = UnifiedMemoryManager(memory_config)
        
        # Float32
        tensor_f32 = manager.allocate_tensor(shape=(10, 10), dtype=mx.float32)
        assert tensor_f32.dtype == mx.float32
        
        # Int32
        tensor_i32 = manager.allocate_tensor(shape=(10, 10), dtype=mx.int32)
        assert tensor_i32.dtype == mx.int32
        
        # Bool
        tensor_bool = manager.allocate_tensor(shape=(10, 10), dtype=mx.bool_)
        assert tensor_bool.dtype == mx.bool_
        
    def test_tensor_deallocation(self, memory_config):
        """Test tensor deallocation."""
        manager = UnifiedMemoryManager(memory_config)
        
        # Allocate and deallocate
        tensor = manager.allocate_tensor(shape=(16, 64), dtype=mx.float32)
        tensor_id = id(tensor)
        
        # Track allocation
        allocation_info = manager.get_allocation_info()
        initial_count = allocation_info['active_allocations']
        
        manager.deallocate_tensor(tensor)
        
        # Check deallocation
        allocation_info = manager.get_allocation_info()
        assert allocation_info['active_allocations'] == initial_count - 1
        
        # Tensor should be returned to pool
        pool_info = manager.get_pool_info()
        assert pool_info['available_tensors'] > 0
        
    def test_cache_operations(self, memory_config):
        """Test caching operations."""
        manager = UnifiedMemoryManager(memory_config)
        
        # Cache a tensor
        tensor = mx.array([1, 2, 3, 4])
        cache_key = "test_tensor"
        
        manager.cache_tensor(cache_key, tensor)
        
        # Retrieve from cache
        cached_tensor = manager.get_cached_tensor(cache_key)
        
        assert cached_tensor is not None
        assert mx.array_equal(cached_tensor, tensor)
        
        # Cache hit rate should improve
        initial_stats = manager.get_cache_statistics()
        cached_tensor = manager.get_cached_tensor(cache_key)
        final_stats = manager.get_cache_statistics()
        
        assert final_stats['cache_hits'] > initial_stats['cache_hits']
        
    def test_cache_eviction(self):
        """Test cache eviction when limit reached."""
        # Small cache for testing eviction
        config = create_memory_config(
            pool_size_mb=256,
            max_cache_size_mb=1,  # Very small cache
            enable_pooling=True,
        )
        
        manager = UnifiedMemoryManager(config)
        
        # Fill cache beyond limit
        large_tensor = create_large_tensor(shape=(1000, 1000))
        
        manager.cache_tensor("large1", large_tensor)
        manager.cache_tensor("large2", large_tensor)
        manager.cache_tensor("large3", large_tensor)
        
        # Some tensors should be evicted
        cache_info = manager.get_cache_info()
        assert cache_info['cached_items'] <= 2
        
        # Check eviction policy (LRU)
        # First tensor might be evicted
        first_tensor = manager.get_cached_tensor("large1")
        if first_tensor is None:
            # Was evicted
            assert cache_info['evictions'] > 0
            
    def test_memory_cleanup(self, memory_config):
        """Test automatic memory cleanup."""
        manager = UnifiedMemoryManager(memory_config)
        
        # Allocate many tensors
        tensors = []
        for i in range(10):
            tensor = manager.allocate_tensor(shape=(100, 100), dtype=mx.float32)
            tensors.append(tensor)
            
        initial_memory = manager.get_memory_info()
        
        # Clear references
        del tensors
        
        # Trigger cleanup
        manager.cleanup()
        
        # Memory should be cleaned up
        final_memory = manager.get_memory_info()
        assert final_memory['allocated_mb'] < initial_memory['allocated_mb']
        
    def test_automatic_cleanup(self):
        """Test automatic cleanup with auto_cleanup enabled."""
        config = create_memory_config(
            auto_cleanup=True,
            cleanup_threshold_mb=50,
        )
        
        manager = UnifiedMemoryManager(config)
        
        # Allocate tensors until cleanup triggers
        tensors = []
        for i in range(100):
            tensor = manager.allocate_tensor(shape=(100, 100), dtype=mx.float32)
            tensors.append(tensor)
            
            # Clear some references to allow cleanup
            if i % 10 == 0:
                tensors = tensors[-5:]  # Keep only last 5
                
        # Cleanup should have been triggered
        cleanup_stats = manager.get_cleanup_statistics()
        assert cleanup_stats['cleanup_count'] > 0
        
    def test_memory_statistics(self, memory_config):
        """Test memory statistics."""
        manager = UnifiedMemoryManager(memory_config)
        
        # Allocate some tensors
        tensor1 = manager.allocate_tensor(shape=(64, 64), dtype=mx.float32)
        tensor2 = manager.allocate_tensor(shape=(32, 32), dtype=mx.int32)
        
        stats = manager.get_memory_statistics()
        
        assert 'total_allocated_mb' in stats
        assert 'peak_allocated_mb' in stats
        assert 'cache_hit_rate' in stats
        assert 'pool_utilization' in stats
        assert 'fragmentation_ratio' in stats
        assert 'allocation_count' in stats
        assert 'deallocation_count' in stats
        
        assert stats['total_allocated_mb'] > 0
        assert stats['allocation_count'] >= 2
        
    def test_unified_memory_optimization(self):
        """Test unified memory optimizations."""
        config = create_memory_config(
            pool_size_mb=256,
            enable_unified_memory=True,
            zero_copy_threshold_mb=10,
        )
        
        manager = UnifiedMemoryManager(config)
        
        # Small tensor should use regular allocation
        small_tensor = manager.allocate_tensor(shape=(10, 10), dtype=mx.float32)
        
        # Large tensor should use zero-copy
        large_tensor = manager.allocate_tensor(shape=(1000, 1000), dtype=mx.float32)
        
        allocation_info = manager.get_allocation_info()
        assert 'zero_copy_allocations' in allocation_info
        assert allocation_info['zero_copy_allocations'] >= 1
        
    def test_memory_monitoring(self, memory_config):
        """Test memory monitoring and alerts."""
        manager = UnifiedMemoryManager(memory_config)
        
        # Enable monitoring
        manager.enable_monitoring(alert_threshold=0.8)
        
        # Allocate memory close to threshold
        large_tensors = []
        alert_triggered = False
        
        def alert_callback(info):
            nonlocal alert_triggered
            alert_triggered = True
            
        manager.set_alert_callback(alert_callback)
        
        # Allocate until alert
        for i in range(100):
            try:
                tensor = manager.allocate_tensor(shape=(100, 100), dtype=mx.float32)
                large_tensors.append(tensor)
            except:
                break
                
        # Alert should have been triggered
        assert alert_triggered or manager.get_pool_utilization() > 0.7
        
    def test_memory_defragmentation(self, memory_config):
        """Test memory defragmentation."""
        manager = UnifiedMemoryManager(memory_config)
        
        # Create fragmented memory
        tensors = []
        for i in range(20):
            tensor = manager.allocate_tensor(shape=(50, 50), dtype=mx.float32)
            tensors.append((i, tensor))
            
        # Free every other tensor to create fragmentation
        for i, tensor in tensors[::2]:
            manager.deallocate_tensor(tensor)
            
        initial_stats = manager.get_memory_statistics()
        initial_fragmentation = initial_stats['fragmentation_ratio']
        
        # Defragment
        manager.defragment()
        
        # Check fragmentation improved
        final_stats = manager.get_memory_statistics()
        final_fragmentation = final_stats['fragmentation_ratio']
        
        assert final_fragmentation <= initial_fragmentation
        
    def test_cross_device_memory(self, memory_config):
        """Test cross-device memory management."""
        manager = UnifiedMemoryManager(memory_config)
        
        # Test allocation on different devices (if available)
        tensor_cpu = manager.allocate_tensor(
            shape=(32, 32), 
            dtype=mx.float32, 
            device=mx.cpu
        )
        
        assert tensor_cpu.shape == (32, 32)
        
        # Check device tracking
        device_info = manager.get_device_memory_info()
        assert 'cpu' in device_info
        assert device_info['cpu']['allocated_mb'] > 0
        
    def test_memory_profiling(self, memory_config):
        """Test memory profiling capabilities."""
        manager = UnifiedMemoryManager(memory_config)
        
        # Enable profiling
        manager.enable_profiling()
        
        # Perform some operations
        tensor1 = manager.allocate_tensor(shape=(64, 64), dtype=mx.float32)
        manager.cache_tensor("profile_test", tensor1)
        cached = manager.get_cached_tensor("profile_test")
        manager.deallocate_tensor(tensor1)
        
        # Get profiling results
        profile = manager.get_profiling_results()
        
        assert 'allocation_count' in profile
        assert 'deallocation_count' in profile
        assert 'cache_operations' in profile
        assert 'total_memory_allocated_mb' in profile
        assert 'allocation_time_ms' in profile
        
        assert profile['allocation_count'] >= 1
        assert profile['cache_operations']['cache_puts'] >= 1
        
    def test_memory_pooling_efficiency(self, memory_config, measure_memory_allocation_time):
        """Test memory pooling efficiency."""
        # Without pooling
        config_no_pool = create_memory_config(enable_pooling=False)
        manager_no_pool = UnifiedMemoryManager(config_no_pool)
        
        time_no_pool = measure_memory_allocation_time(
            manager_no_pool,
            num_allocations=100,
            shape=(100, 100)
        )
        
        # With pooling
        config_pool = create_memory_config(enable_pooling=True)
        manager_pool = UnifiedMemoryManager(config_pool)
        
        # Pre-warm pool
        for _ in range(10):
            t = manager_pool.allocate_tensor(shape=(100, 100), dtype=mx.float32)
            manager_pool.deallocate_tensor(t)
            
        time_pool = measure_memory_allocation_time(
            manager_pool,
            num_allocations=100,
            shape=(100, 100)
        )
        
        # Pooling should be faster
        assert time_pool < time_no_pool * 0.8  # At least 20% faster
        
    def test_weak_references(self, memory_config):
        """Test weak reference support for cache."""
        manager = UnifiedMemoryManager(memory_config)
        
        # Create tensor and cache with weak reference
        tensor = manager.allocate_tensor(shape=(10, 10), dtype=mx.float32)
        weak_ref = weakref.ref(tensor)
        
        manager.cache_tensor("weak_test", tensor, use_weak_ref=True)
        
        # Tensor should be retrievable
        cached = manager.get_cached_tensor("weak_test")
        assert cached is not None
        
        # Delete strong reference
        del tensor
        del cached
        gc.collect()
        
        # Weak reference should be cleared
        assert weak_ref() is None
        
        # Cache should return None
        cached = manager.get_cached_tensor("weak_test")
        assert cached is None


@pytest.mark.integration  
class TestMemoryIntegration:
    """Integration tests for memory management."""
    
    def test_memory_manager_with_data_loader(self):
        """Test memory manager integration with data loader."""
        from data.loaders.mlx_loader import MLXDataLoader, MLXLoaderConfig
        from tests.data.fixtures.datasets import MockKaggleDataset
        
        # Create memory manager
        memory_config = create_memory_config(
            pool_size_mb=256,
            enable_unified_memory=True,
        )
        memory_manager = UnifiedMemoryManager(memory_config)
        
        # Create dataset and loader
        spec = create_dataset_spec(num_samples=100)
        dataset = MockKaggleDataset(spec, size=100)
        
        loader_config = MLXLoaderConfig(
            batch_size=32,
        )
        
        loader = MLXDataLoader(dataset, loader_config)
        
        # Load batches
        initial_memory = memory_manager.get_memory_info()
        
        batches = []
        for i, batch in enumerate(loader):
            batches.append(batch)
            if i >= 3:
                break
                
        # Memory should be managed efficiently
        final_memory = memory_manager.get_memory_info()
        memory_growth = final_memory['allocated_mb'] - initial_memory['allocated_mb']
        
        # Should reuse memory
        assert memory_growth < 50  # Less than 50MB growth
        
    def test_multi_manager_coordination(self):
        """Test multiple memory managers working together."""
        config1 = create_memory_config(pool_size_mb=128)
        config2 = create_memory_config(pool_size_mb=128)
        
        manager1 = UnifiedMemoryManager(config1)
        manager2 = UnifiedMemoryManager(config2)
        
        # Allocate from both
        tensors1 = []
        tensors2 = []
        
        for i in range(10):
            t1 = manager1.allocate_tensor(shape=(50, 50), dtype=mx.float32)
            t2 = manager2.allocate_tensor(shape=(50, 50), dtype=mx.float32)
            tensors1.append(t1)
            tensors2.append(t2)
            
        # Both should work independently
        assert manager1.get_allocation_info()['active_allocations'] == 10
        assert manager2.get_allocation_info()['active_allocations'] == 10
        
        # Global memory tracking
        global_memory = UnifiedMemoryManager.get_global_memory_info()
        assert global_memory['total_managers'] >= 2


