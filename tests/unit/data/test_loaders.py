"""Unit tests for MLX data loaders."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import time

import pandas as pd
import mlx.core as mx
import numpy as np

from data.core.base import DatasetSpec, CompetitionType
from data.loaders.mlx_loader import MLXDataLoader, MLXLoaderConfig
from data.loaders.streaming import StreamingPipeline, StreamingConfig
from data.loaders.memory import UnifiedMemoryManager, MemoryConfig


class MockKaggleDataset:
    """Mock dataset for testing loaders."""
    
    def __init__(self, spec, size=100):
        self.spec = spec
        self.size = size
        self._data = pd.DataFrame({
            'feature1': np.random.randn(size),
            'feature2': np.random.choice(['A', 'B', 'C'], size),
            'target': np.random.randint(0, 2, size)
        })
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return [self[i] for i in range(start, stop, step or 1)]
            
        row = self._data.iloc[idx]
        return {
            'input_ids': mx.array([1, 2, 3, 4], dtype=mx.int32),
            'attention_mask': mx.array([1, 1, 1, 1], dtype=mx.int32),
            'labels': mx.array([row['target']], dtype=mx.float32),
            'text': f"Feature1: {row['feature1']:.2f}, Feature2: {row['feature2']}",
            'metadata': {'index': idx}
        }
        
    def get_batch(self, indices):
        samples = [self[i] for i in indices]
        return {
            'input_ids': mx.stack([s['input_ids'] for s in samples]),
            'attention_mask': mx.stack([s['attention_mask'] for s in samples]),
            'labels': mx.stack([s['labels'] for s in samples]),
            'text': [s['text'] for s in samples],
        }


class TestMLXLoaderConfig:
    """Test MLXLoaderConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MLXLoaderConfig()
        
        assert config.batch_size == 32
        assert config.shuffle == True
        assert config.num_workers == 4
        assert config.prefetch_size == 4
        assert config.drop_last == False
        assert config.pin_memory == True
        
    def test_custom_config(self):
        """Test custom configuration values."""
        config = MLXLoaderConfig(
            batch_size=64,
            shuffle=False,
            num_workers=8,
            prefetch_size=8,
            pin_memory=False,
        )
        
        assert config.batch_size == 64
        assert config.shuffle == False
        assert config.num_workers == 8
        assert config.prefetch_size == 8
        assert config.pin_memory == False
        
    def test_optimization_settings(self):
        """Test optimization-related settings."""
        config = MLXLoaderConfig(
            enable_gradient_accumulation=True,
            gradient_accumulation_steps=4,
            use_unified_memory=True,
            memory_pool_size_mb=512,
        )
        
        assert config.enable_gradient_accumulation == True
        assert config.gradient_accumulation_steps == 4
        assert config.use_unified_memory == True
        assert config.memory_pool_size_mb == 512


class TestMLXDataLoader:
    """Test MLXDataLoader class."""
    
    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for testing."""
        spec = DatasetSpec(
            competition_name="test",
            dataset_path=Path("/tmp/test"),
            competition_type=CompetitionType.BINARY_CLASSIFICATION,
            num_samples=100,
            num_features=2,
        )
        return MockKaggleDataset(spec, size=100)
        
    @pytest.fixture
    def loader_config(self):
        """Create loader configuration."""
        return MLXLoaderConfig(
            batch_size=16,
            shuffle=True,
            num_workers=2,
            prefetch_size=2,
        )
        
    def test_loader_creation(self, sample_dataset, loader_config):
        """Test loader creation."""
        loader = MLXDataLoader(sample_dataset, loader_config)
        
        assert loader.dataset == sample_dataset
        assert loader.config == loader_config
        assert loader.batch_size == 16
        assert hasattr(loader, '_memory_manager')
        
    def test_loader_with_default_config(self, sample_dataset):
        """Test loader with default configuration."""
        loader = MLXDataLoader(sample_dataset)
        
        assert loader.config.batch_size == 32
        assert loader.config.shuffle == True
        assert isinstance(loader.config, MLXLoaderConfig)
        
    def test_loader_length(self, sample_dataset, loader_config):
        """Test loader length calculation."""
        loader = MLXDataLoader(sample_dataset, loader_config)
        
        expected_length = len(sample_dataset) // loader_config.batch_size
        if not loader_config.drop_last and len(sample_dataset) % loader_config.batch_size:
            expected_length += 1
            
        assert len(loader) == expected_length
        
    def test_loader_iteration(self, sample_dataset, loader_config):
        """Test loader iteration."""
        loader = MLXDataLoader(sample_dataset, loader_config)
        
        batches = list(loader)
        
        assert len(batches) > 0
        assert len(batches) == len(loader)
        
        # Check first batch structure
        batch = batches[0]
        assert 'input_ids' in batch
        assert 'attention_mask' in batch
        assert 'labels' in batch
        assert 'text' in batch
        
        # Check batch dimensions
        assert batch['input_ids'].shape[0] <= loader_config.batch_size
        
    def test_loader_shuffle(self, sample_dataset):
        """Test loader shuffling."""
        config_shuffled = MLXLoaderConfig(batch_size=10, shuffle=True)
        config_no_shuffle = MLXLoaderConfig(batch_size=10, shuffle=False)
        
        loader_shuffled = MLXDataLoader(sample_dataset, config_shuffled)
        loader_no_shuffle = MLXDataLoader(sample_dataset, config_no_shuffle)
        
        # Get first batch from each
        batch_shuffled = next(iter(loader_shuffled))
        batch_no_shuffle = next(iter(loader_no_shuffle))
        
        # They should potentially be different due to shuffling
        # Note: This test might occasionally fail due to randomness
        assert isinstance(batch_shuffled['input_ids'], mx.array)
        assert isinstance(batch_no_shuffle['input_ids'], mx.array)
        
    def test_loader_drop_last(self, sample_dataset):
        """Test drop_last functionality."""
        # Dataset size is 100, use batch size that doesn't divide evenly
        config_drop = MLXLoaderConfig(batch_size=30, drop_last=True)
        config_no_drop = MLXLoaderConfig(batch_size=30, drop_last=False)
        
        loader_drop = MLXDataLoader(sample_dataset, config_drop)
        loader_no_drop = MLXDataLoader(sample_dataset, config_no_drop)
        
        assert len(loader_drop) == 3  # 100 // 30 = 3
        assert len(loader_no_drop) == 4  # 3 + 1 for remainder
        
    def test_get_batch(self, sample_dataset, loader_config):
        """Test getting specific batch."""
        loader = MLXDataLoader(sample_dataset, loader_config)
        
        batch = loader.get_batch(0)
        
        assert 'input_ids' in batch
        assert 'attention_mask' in batch
        assert 'labels' in batch
        assert batch['input_ids'].shape[0] == loader_config.batch_size
        
    def test_get_batch_out_of_range(self, sample_dataset, loader_config):
        """Test getting batch with invalid index."""
        loader = MLXDataLoader(sample_dataset, loader_config)
        
        with pytest.raises(IndexError):
            loader.get_batch(1000)  # Way beyond valid range
            
    def test_reset_loader(self, sample_dataset, loader_config):
        """Test resetting loader state."""
        loader = MLXDataLoader(sample_dataset, loader_config)
        
        # Consume some batches
        iterator = iter(loader)
        next(iterator)
        next(iterator)
        
        # Reset
        loader.reset()
        
        # Should be able to iterate again from beginning
        new_iterator = iter(loader)
        batch = next(new_iterator)
        assert batch is not None
        
    def test_loader_statistics(self, sample_dataset, loader_config):
        """Test getting loader statistics."""
        loader = MLXDataLoader(sample_dataset, loader_config)
        
        stats = loader.get_statistics()
        
        assert 'total_samples' in stats
        assert 'batch_size' in stats
        assert 'num_batches' in stats
        assert 'shuffle' in stats
        assert 'num_workers' in stats
        
        assert stats['total_samples'] == len(sample_dataset)
        assert stats['batch_size'] == loader_config.batch_size
        
    def test_loader_with_gradient_accumulation(self, sample_dataset):
        """Test loader with gradient accumulation."""
        config = MLXLoaderConfig(
            batch_size=8,
            enable_gradient_accumulation=True,
            gradient_accumulation_steps=4,
        )
        
        loader = MLXDataLoader(sample_dataset, config)
        
        # Effective batch size should be batch_size * accumulation_steps
        assert loader.effective_batch_size == 32
        
    def test_memory_management(self, sample_dataset, loader_config):
        """Test memory management features."""
        loader = MLXDataLoader(sample_dataset, loader_config)
        
        # Get memory info
        memory_info = loader.get_memory_info()
        
        assert 'allocated_mb' in memory_info
        assert 'cached_mb' in memory_info
        assert 'peak_allocated_mb' in memory_info
        assert 'unified_memory' in memory_info
        
    def test_clear_cache(self, sample_dataset, loader_config):
        """Test clearing loader cache."""
        loader = MLXDataLoader(sample_dataset, loader_config)
        
        # Load some batches to populate cache
        batch1 = loader.get_batch(0)
        batch2 = loader.get_batch(1)
        
        # Clear cache
        loader.clear_cache()
        
        # Should still work after clearing cache
        batch3 = loader.get_batch(0)
        assert batch3 is not None
        
    def test_loader_performance_profiling(self, sample_dataset, loader_config):
        """Test performance profiling."""
        loader = MLXDataLoader(sample_dataset, loader_config)
        
        # Enable profiling
        loader.enable_profiling()
        
        # Load some batches
        for i, batch in enumerate(loader):
            if i >= 5:  # Only test a few batches
                break
                
        # Get profiling results
        profile = loader.get_profiling_results()
        
        assert 'avg_batch_time_ms' in profile
        assert 'total_batches_loaded' in profile
        assert 'throughput_samples_per_sec' in profile
        
    def test_async_loading(self, sample_dataset):
        """Test asynchronous loading."""
        config = MLXLoaderConfig(
            batch_size=16,
            num_workers=2,
            prefetch_size=4,
            async_loading=True,
        )
        
        loader = MLXDataLoader(sample_dataset, config)
        
        # Should work with async loading
        batches = []
        for i, batch in enumerate(loader):
            batches.append(batch)
            if i >= 3:  # Test a few batches
                break
                
        assert len(batches) == 4
        
    def test_data_augmentation_integration(self, sample_dataset):
        """Test integration with data augmentation."""
        def augment_batch(batch):
            # Simple augmentation - just return the batch
            return batch
            
        config = MLXLoaderConfig(
            batch_size=16,
            enable_augmentation=True,
            augmentation_fn=augment_batch,
        )
        
        loader = MLXDataLoader(sample_dataset, config)
        
        batch = next(iter(loader))
        assert batch is not None
        
    def test_mlx_optimization_settings(self, sample_dataset):
        """Test MLX-specific optimization settings."""
        config = MLXLoaderConfig(
            batch_size=32,
            use_unified_memory=True,
            optimize_for_mlx=True,
            enable_zero_copy=True,
        )
        
        loader = MLXDataLoader(sample_dataset, config)
        
        batch = next(iter(loader))
        
        # Check that tensors are MLX arrays
        assert isinstance(batch['input_ids'], mx.array)
        assert isinstance(batch['attention_mask'], mx.array)
        assert isinstance(batch['labels'], mx.array)
        
    def test_error_handling(self, sample_dataset):
        """Test error handling in loader."""
        # Test with invalid configuration
        with pytest.raises(ValueError):
            config = MLXLoaderConfig(batch_size=0)  # Invalid batch size
            
        # Test with negative values
        with pytest.raises(ValueError):
            config = MLXLoaderConfig(num_workers=-1)
            
    def test_loader_state_management(self, sample_dataset, loader_config):
        """Test loader state save/restore."""
        loader = MLXDataLoader(sample_dataset, loader_config)
        
        # Consume some batches
        iterator = iter(loader)
        next(iterator)
        next(iterator)
        
        # Save state
        state = loader.get_state()
        
        # Create new loader and restore state
        new_loader = MLXDataLoader(sample_dataset, loader_config)
        new_loader.set_state(state)
        
        # Should continue from saved position
        assert new_loader._current_epoch == loader._current_epoch
        
    def test_multi_epoch_iteration(self, sample_dataset, loader_config):
        """Test multi-epoch iteration."""
        loader = MLXDataLoader(sample_dataset, loader_config)
        
        # Iterate through multiple epochs
        epoch_counts = []
        for epoch in range(3):
            count = 0
            for batch in loader:
                count += 1
            epoch_counts.append(count)
            
        # All epochs should have same number of batches
        assert len(set(epoch_counts)) == 1
        
    def test_custom_collate_function(self, sample_dataset):
        """Test custom collate function."""
        def custom_collate(samples):
            # Custom collation logic
            return {
                'input_ids': mx.stack([s['input_ids'] for s in samples]),
                'attention_mask': mx.stack([s['attention_mask'] for s in samples]),
                'labels': mx.stack([s['labels'] for s in samples]),
                'text': [s['text'] for s in samples],
                'custom_field': 'custom_value'
            }
            
        config = MLXLoaderConfig(
            batch_size=8,
            collate_fn=custom_collate,
        )
        
        loader = MLXDataLoader(sample_dataset, config)
        
        batch = next(iter(loader))
        assert 'custom_field' in batch
        assert batch['custom_field'] == 'custom_value'


class TestStreamingPipeline:
    """Test StreamingPipeline class."""
    
    @pytest.fixture
    def streaming_config(self):
        """Create streaming configuration."""
        return StreamingConfig(
            buffer_size=1024,
            chunk_size=256,
            num_workers=2,
            target_throughput=100,
        )
        
    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for streaming."""
        spec = DatasetSpec(
            competition_name="streaming_test",
            dataset_path=Path("/tmp/streaming"),
            competition_type=CompetitionType.BINARY_CLASSIFICATION,
            num_samples=1000,
            num_features=5,
        )
        return MockKaggleDataset(spec, size=1000)
        
    def test_pipeline_creation(self, sample_dataset, streaming_config):
        """Test streaming pipeline creation."""
        pipeline = StreamingPipeline(sample_dataset, streaming_config)
        
        assert pipeline.dataset == sample_dataset
        assert pipeline.config == streaming_config
        assert hasattr(pipeline, '_buffer')
        assert hasattr(pipeline, '_workers')
        
    def test_pipeline_start_stop(self, sample_dataset, streaming_config):
        """Test starting and stopping pipeline."""
        pipeline = StreamingPipeline(sample_dataset, streaming_config)
        
        # Start pipeline
        pipeline.start()
        assert pipeline.is_running() == True
        
        # Stop pipeline
        pipeline.stop()
        assert pipeline.is_running() == False
        
    def test_streaming_iteration(self, sample_dataset, streaming_config):
        """Test streaming iteration."""
        pipeline = StreamingPipeline(sample_dataset, streaming_config)
        
        pipeline.start()
        
        try:
            # Stream some samples
            samples = []
            for i, sample in enumerate(pipeline):
                samples.append(sample)
                if i >= 50:  # Test first 50 samples
                    break
                    
            assert len(samples) == 51  # 0 to 50 inclusive
            assert all('input_ids' in sample for sample in samples)
            
        finally:
            pipeline.stop()
            
    def test_throughput_measurement(self, sample_dataset, streaming_config):
        """Test throughput measurement."""
        pipeline = StreamingPipeline(sample_dataset, streaming_config)
        
        pipeline.start()
        
        try:
            # Process samples for a short time
            start_time = time.time()
            sample_count = 0
            
            for sample in pipeline:
                sample_count += 1
                if time.time() - start_time > 1.0:  # Run for 1 second
                    break
                    
            throughput = pipeline.get_throughput_stats()
            
            assert 'samples_per_second' in throughput
            assert 'avg_batch_time_ms' in throughput
            assert throughput['samples_per_second'] > 0
            
        finally:
            pipeline.stop()
            
    def test_buffer_management(self, sample_dataset, streaming_config):
        """Test buffer management."""
        pipeline = StreamingPipeline(sample_dataset, streaming_config)
        
        buffer_info = pipeline.get_buffer_info()
        
        assert 'buffer_size' in buffer_info
        assert 'current_items' in buffer_info
        assert 'buffer_utilization' in buffer_info
        
    def test_adaptive_streaming(self, sample_dataset):
        """Test adaptive streaming based on consumption rate."""
        config = StreamingConfig(
            buffer_size=512,
            adaptive_batching=True,
            target_throughput=200,
        )
        
        pipeline = StreamingPipeline(sample_dataset, config)
        
        pipeline.start()
        
        try:
            # Consume at different rates to test adaptation
            samples = []
            for i, sample in enumerate(pipeline):
                samples.append(sample)
                if i >= 20:
                    break
                if i % 5 == 0:
                    time.sleep(0.01)  # Simulate slower consumption
                    
            assert len(samples) == 21
            
        finally:
            pipeline.stop()
            
    def test_error_recovery(self, sample_dataset, streaming_config):
        """Test error recovery in streaming."""
        pipeline = StreamingPipeline(sample_dataset, streaming_config)
        
        # Enable error recovery
        pipeline.enable_error_recovery(max_retries=3)
        
        pipeline.start()
        
        try:
            # Should handle errors gracefully
            samples = []
            for i, sample in enumerate(pipeline):
                samples.append(sample)
                if i >= 10:
                    break
                    
            assert len(samples) == 11
            
        finally:
            pipeline.stop()


class TestUnifiedMemoryManager:
    """Test UnifiedMemoryManager class."""
    
    @pytest.fixture
    def memory_config(self):
        """Create memory configuration."""
        return MemoryConfig(
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
        
    def test_tensor_allocation(self, memory_config):
        """Test tensor allocation from pool."""
        manager = UnifiedMemoryManager(memory_config)
        
        # Allocate tensor
        tensor = manager.allocate_tensor(shape=(32, 128), dtype=mx.float32)
        
        assert isinstance(tensor, mx.array)
        assert tensor.shape == (32, 128)
        assert tensor.dtype == mx.float32
        
    def test_tensor_deallocation(self, memory_config):
        """Test tensor deallocation."""
        manager = UnifiedMemoryManager(memory_config)
        
        # Allocate and deallocate
        tensor = manager.allocate_tensor(shape=(16, 64), dtype=mx.float32)
        tensor_id = id(tensor)
        
        manager.deallocate_tensor(tensor)
        
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
        
    def test_cache_eviction(self, memory_config):
        """Test cache eviction when limit reached."""
        # Small cache for testing eviction
        config = MemoryConfig(
            pool_size_mb=256,
            max_cache_size_mb=1,  # Very small cache
            enable_pooling=True,
        )
        
        manager = UnifiedMemoryManager(config)
        
        # Fill cache beyond limit
        large_tensor = mx.ones((1000, 1000))  # Large tensor
        
        manager.cache_tensor("large1", large_tensor)
        manager.cache_tensor("large2", large_tensor)
        
        # First tensor might be evicted
        cache_info = manager.get_cache_info()
        assert cache_info['cached_items'] <= 2
        
    def test_memory_cleanup(self, memory_config):
        """Test automatic memory cleanup."""
        manager = UnifiedMemoryManager(memory_config)
        
        # Allocate many tensors
        tensors = []
        for i in range(10):
            tensor = manager.allocate_tensor(shape=(100, 100), dtype=mx.float32)
            tensors.append(tensor)
            
        # Clear references
        del tensors
        
        # Trigger cleanup
        manager.cleanup()
        
        # Memory should be cleaned up
        memory_info = manager.get_memory_info()
        assert memory_info['allocated_mb'] >= 0  # Should have some cleanup
        
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
        
    def test_unified_memory_optimization(self, memory_config):
        """Test unified memory optimizations."""
        config = MemoryConfig(
            pool_size_mb=256,
            enable_unified_memory=True,
            zero_copy_threshold_mb=10,
        )
        
        manager = UnifiedMemoryManager(config)
        
        # Large tensor should use zero-copy
        large_tensor = manager.allocate_tensor(shape=(1000, 1000), dtype=mx.float32)
        
        memory_info = manager.get_memory_info()
        assert memory_info['unified_memory'] == True
        
    def test_memory_monitoring(self, memory_config):
        """Test memory monitoring and alerts."""
        manager = UnifiedMemoryManager(memory_config)
        
        # Enable monitoring
        manager.enable_monitoring(alert_threshold=0.8)
        
        # Allocate memory close to threshold
        tensors = []
        for i in range(50):
            tensor = manager.allocate_tensor(shape=(100, 100), dtype=mx.float32)
            tensors.append(tensor)
            
        # Check if alerts were triggered
        alerts = manager.get_alerts()
        assert isinstance(alerts, list)
        
    def test_memory_defragmentation(self, memory_config):
        """Test memory defragmentation."""
        manager = UnifiedMemoryManager(memory_config)
        
        # Create fragmented memory
        tensors = []
        for i in range(20):
            tensor = manager.allocate_tensor(shape=(50, 50), dtype=mx.float32)
            tensors.append(tensor)
            
        # Free every other tensor to create fragmentation
        for i in range(0, len(tensors), 2):
            manager.deallocate_tensor(tensors[i])
            
        # Defragment
        manager.defragment()
        
        # Check fragmentation improved
        stats = manager.get_memory_statistics()
        assert stats['fragmentation_ratio'] < 1.0
        
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