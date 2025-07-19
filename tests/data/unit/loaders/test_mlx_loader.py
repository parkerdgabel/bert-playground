"""Tests for MLX data loaders."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import time

import pandas as pd
import mlx.core as mx
import numpy as np

from data.loaders.mlx_loader import MLXDataLoader, MLXLoaderConfig
from tests.data.fixtures.datasets import MockKaggleDataset
from tests.data.fixtures.configs import (
    create_dataset_spec,
    create_mlx_loader_config,
)
from tests.data.fixtures.utils import (
    assert_batch_structure,
    check_memory_usage,
    measure_throughput,
)


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
        config = create_mlx_loader_config(
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
        config = create_mlx_loader_config(
            enable_gradient_accumulation=True,
            gradient_accumulation_steps=4,
            use_unified_memory=True,
            memory_pool_size_mb=512,
        )
        
        assert config.enable_gradient_accumulation == True
        assert config.gradient_accumulation_steps == 4
        assert config.use_unified_memory == True
        assert config.memory_pool_size_mb == 512

    def test_validation_constraints(self):
        """Test configuration validation."""
        # Test invalid batch size
        with pytest.raises(ValueError):
            config = MLXLoaderConfig(batch_size=0)
            
        # Test invalid workers
        with pytest.raises(ValueError):
            config = MLXLoaderConfig(num_workers=-1)
            
        # Test invalid prefetch size
        with pytest.raises(ValueError):
            config = MLXLoaderConfig(prefetch_size=-1)


class TestMLXDataLoader:
    """Test MLXDataLoader class."""
    
    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for testing."""
        spec = create_dataset_spec(
            num_samples=100,
            num_features=2,
        )
        return MockKaggleDataset(spec, size=100)
        
    @pytest.fixture
    def loader_config(self):
        """Create loader configuration."""
        return create_mlx_loader_config(
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
        assert loader.config.batch_size == 16
        
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
        
    def test_loader_iteration(self, sample_dataset, loader_config, assert_batch_structure):
        """Test loader iteration."""
        loader = MLXDataLoader(sample_dataset, loader_config)
        
        batches = list(loader)
        
        assert len(batches) > 0
        assert len(batches) == len(loader)
        
        # Check first batch structure using fixture
        assert_batch_structure(batches[0])
        
        # Check batch dimensions
        assert batches[0]['input_ids'].shape[0] <= loader_config.batch_size
        
    def test_loader_shuffle(self, sample_dataset):
        """Test loader shuffling."""
        config_shuffled = create_mlx_loader_config(batch_size=10, shuffle=True)
        config_no_shuffle = create_mlx_loader_config(batch_size=10, shuffle=False)
        
        loader_shuffled = MLXDataLoader(sample_dataset, config_shuffled)
        loader_no_shuffle = MLXDataLoader(sample_dataset, config_no_shuffle)
        
        # Get first batch from each
        batch_shuffled = next(iter(loader_shuffled))
        batch_no_shuffle = next(iter(loader_no_shuffle))
        
        # They should potentially be different due to shuffling
        assert isinstance(batch_shuffled['input_ids'], mx.array)
        assert isinstance(batch_no_shuffle['input_ids'], mx.array)
        
    def test_loader_drop_last(self, sample_dataset):
        """Test drop_last functionality."""
        # Dataset size is 100, use batch size that doesn't divide evenly
        config_drop = create_mlx_loader_config(batch_size=30, drop_last=True)
        config_no_drop = create_mlx_loader_config(batch_size=30, drop_last=False)
        
        loader_drop = MLXDataLoader(sample_dataset, config_drop)
        loader_no_drop = MLXDataLoader(sample_dataset, config_no_drop)
        
        assert len(loader_drop) == 3  # 100 // 30 = 3
        assert len(loader_no_drop) == 4  # 3 + 1 for remainder
        
    def test_get_batch(self, sample_dataset, loader_config, assert_batch_structure):
        """Test getting specific batch."""
        loader = MLXDataLoader(sample_dataset, loader_config)
        
        batch = loader.get_batch(0)
        
        assert_batch_structure(batch)
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
        config = create_mlx_loader_config(
            batch_size=8,
            enable_gradient_accumulation=True,
            gradient_accumulation_steps=4,
        )
        
        loader = MLXDataLoader(sample_dataset, config)
        
        # Effective batch size should be batch_size * accumulation_steps
        assert loader.effective_batch_size == 32
        
    def test_memory_management(self, sample_dataset, loader_config, check_memory_usage):
        """Test memory management features."""
        loader = MLXDataLoader(sample_dataset, loader_config)
        
        # Get memory info
        memory_info = loader.get_memory_info()
        
        assert 'allocated_mb' in memory_info
        assert 'cached_mb' in memory_info
        assert 'peak_allocated_mb' in memory_info
        assert 'unified_memory' in memory_info
        
        # Check memory usage is reasonable
        assert check_memory_usage(memory_info, max_allowed_mb=1000)
        
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
        config = create_mlx_loader_config(
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
            # Simple augmentation - add noise to inputs
            batch['input_ids'] = batch['input_ids'] + mx.random.randint(0, 2, batch['input_ids'].shape)
            return batch
            
        config = create_mlx_loader_config(
            batch_size=16,
            enable_augmentation=True,
            augmentation_fn=augment_batch,
        )
        
        loader = MLXDataLoader(sample_dataset, config)
        
        batch = next(iter(loader))
        assert batch is not None
        
    def test_mlx_optimization_settings(self, sample_dataset):
        """Test MLX-specific optimization settings."""
        config = create_mlx_loader_config(
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
            
        config = create_mlx_loader_config(
            batch_size=8,
            collate_fn=custom_collate,
        )
        
        loader = MLXDataLoader(sample_dataset, config)
        
        batch = next(iter(loader))
        assert 'custom_field' in batch
        assert batch['custom_field'] == 'custom_value'


@pytest.mark.integration
class TestMLXLoaderIntegration:
    """Integration tests for MLX data loader."""
    
    def test_loader_with_transforms(self, sample_dataset):
        """Test loader with data transforms."""
        transform_count = 0
        
        def transform_fn(sample):
            nonlocal transform_count
            transform_count += 1
            sample['transformed'] = True
            return sample
            
        # Apply transform to dataset
        dataset = MockKaggleDataset(sample_dataset.spec, transform=transform_fn)
        
        config = create_mlx_loader_config(batch_size=10)
        loader = MLXDataLoader(dataset, config)
        
        # Load all batches
        batches = list(loader)
        
        # Check transforms were applied
        assert transform_count > 0
        
    def test_loader_pipeline_end_to_end(self):
        """Test complete loader pipeline."""
        # Create dataset
        spec = create_dataset_spec(num_samples=64)
        dataset = MockKaggleDataset(spec, size=64)
        
        # Configure loader
        config = create_mlx_loader_config(
            batch_size=16,
            shuffle=True,
            num_workers=2,
            prefetch_size=2,
        )
        
        loader = MLXDataLoader(dataset, config)
        
        # Train simulation
        num_epochs = 2
        all_batches = []
        
        for epoch in range(num_epochs):
            epoch_batches = []
            for batch in loader:
                epoch_batches.append(batch)
                
            assert len(epoch_batches) == 4  # 64 / 16
            all_batches.extend(epoch_batches)
            
        assert len(all_batches) == 8  # 2 epochs * 4 batches
        
    def test_concurrent_loaders(self, sample_dataset):
        """Test multiple loaders on same dataset."""
        config1 = create_mlx_loader_config(batch_size=16)
        config2 = create_mlx_loader_config(batch_size=32)
        
        loader1 = MLXDataLoader(sample_dataset, config1)
        loader2 = MLXDataLoader(sample_dataset, config2)
        
        # Should work independently
        batch1 = next(iter(loader1))
        batch2 = next(iter(loader2))
        
        assert batch1['input_ids'].shape[0] == 16
        assert batch2['input_ids'].shape[0] == 32


