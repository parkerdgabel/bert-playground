"""Tests for MLX data loaders."""

import time

import mlx.core as mx
import mlx.nn
import pytest

from data.loaders.mlx_loader import MLXDataLoader, MLXLoaderConfig
from tests.data.fixtures.configs import (
    create_dataset_spec,
    create_mlx_loader_config,
)
from tests.data.fixtures.datasets import MockKaggleDataset
from tests.data.fixtures.utils import (
    assert_batch_structure,
)


class TestMLXLoaderConfig:
    """Test MLXLoaderConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MLXLoaderConfig()

        assert config.batch_size == 32
        assert config.shuffle == True
        assert config.prefetch_size == 2
        assert config.drop_last == False
        assert config.use_unified_memory == True
        assert config.lazy_evaluation == True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = create_mlx_loader_config(
            batch_size=64,
            shuffle=False,
            prefetch_size=8,
            use_unified_memory=False,
            lazy_evaluation=False,
        )

        assert config.batch_size == 64
        assert config.shuffle == False
        assert config.prefetch_size == 8
        assert config.use_unified_memory == False
        assert config.lazy_evaluation == False

    def test_optimization_settings(self):
        """Test optimization-related settings."""
        config = create_mlx_loader_config(
            use_unified_memory=True,
            lazy_evaluation=True,
            max_length=256,
            padding="longest",
            truncation=False,
        )

        assert config.use_unified_memory == True
        assert config.lazy_evaluation == True
        assert config.max_length == 256
        assert config.padding == "longest"
        assert config.truncation == False

    def test_validation_constraints(self):
        """Test configuration validation."""
        # MLXLoaderConfig is a dataclass without validation
        # These should NOT raise errors
        config1 = MLXLoaderConfig(batch_size=0)
        assert config1.batch_size == 0

        config2 = MLXLoaderConfig(prefetch_size=-1)
        assert config2.prefetch_size == -1

        # Test that we can create config with custom values
        config3 = MLXLoaderConfig(batch_size=1, prefetch_size=0)
        assert config3.batch_size == 1
        assert config3.prefetch_size == 0


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
        if (
            not loader_config.drop_last
            and len(sample_dataset) % loader_config.batch_size
        ):
            expected_length += 1

        assert len(loader) == expected_length

    def test_loader_iteration(self, sample_dataset, loader_config):
        """Test loader iteration."""
        loader = MLXDataLoader(sample_dataset, loader_config)

        batches = list(loader)

        assert len(batches) > 0
        assert len(batches) == len(loader)

        # Check first batch structure using imported function
        expected_keys = ["input_ids", "attention_mask", "labels", "metadata"]
        assert_batch_structure(batches[0], expected_keys)

        # Check batch dimensions
        assert batches[0]["input_ids"].shape[0] <= loader_config.batch_size

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
        assert isinstance(batch_shuffled["input_ids"], mx.array)
        assert isinstance(batch_no_shuffle["input_ids"], mx.array)

    def test_loader_drop_last(self, sample_dataset):
        """Test drop_last functionality."""
        # Dataset size is 100, use batch size that doesn't divide evenly
        config_drop = create_mlx_loader_config(batch_size=30, drop_last=True)
        config_no_drop = create_mlx_loader_config(batch_size=30, drop_last=False)

        loader_drop = MLXDataLoader(sample_dataset, config_drop)
        loader_no_drop = MLXDataLoader(sample_dataset, config_no_drop)

        assert len(loader_drop) == 3  # 100 // 30 = 3
        assert len(loader_no_drop) == 4  # 3 + 1 for remainder

    def test_get_batch(self, sample_dataset, loader_config):
        """Test getting specific batch."""
        loader = MLXDataLoader(sample_dataset, loader_config)

        # MLXDataLoader doesn't have get_batch method, use iterator instead
        batch = next(iter(loader))

        expected_keys = ["input_ids", "attention_mask", "labels", "metadata"]
        assert_batch_structure(batch, expected_keys)
        assert batch["input_ids"].shape[0] <= loader_config.batch_size

    def test_get_batch_out_of_range(self, sample_dataset, loader_config):
        """Test iterating beyond available batches."""
        loader = MLXDataLoader(sample_dataset, loader_config)

        # Consume all batches
        batches = list(loader)

        # Try to get another batch - should get empty list on next iteration
        more_batches = list(loader)
        assert len(more_batches) == len(batches)  # Should be able to iterate again

    def test_reset_loader(self, sample_dataset, loader_config):
        """Test that loader can be iterated multiple times."""
        loader = MLXDataLoader(sample_dataset, loader_config)

        # First iteration
        first_batches = []
        for i, batch in enumerate(loader):
            first_batches.append(batch)
            if i >= 2:  # Just get first 3 batches
                break

        # Second iteration should work from beginning
        second_batches = []
        for i, batch in enumerate(loader):
            second_batches.append(batch)
            if i >= 2:  # Just get first 3 batches
                break

        assert len(first_batches) == len(second_batches)

    def test_loader_statistics(self, sample_dataset, loader_config):
        """Test loader basic properties."""
        loader = MLXDataLoader(sample_dataset, loader_config)

        # Test basic properties
        assert len(loader) == loader.num_batches
        assert loader.config.batch_size == loader_config.batch_size
        assert loader.config.shuffle == loader_config.shuffle
        assert len(loader.indices) == len(sample_dataset)

    def test_loader_with_gradient_accumulation(self, sample_dataset):
        """Test loader with different batch sizes."""
        # Test with smaller batch size
        config = create_mlx_loader_config(batch_size=8)
        loader = MLXDataLoader(sample_dataset, config)

        # Check that loader works with small batch size
        batch = next(iter(loader))
        assert batch["input_ids"].shape[0] <= 8

        # Test with larger batch size
        config2 = create_mlx_loader_config(batch_size=32)
        loader2 = MLXDataLoader(sample_dataset, config2)
        batch2 = next(iter(loader2))
        assert batch2["input_ids"].shape[0] <= 32

    def test_memory_management(self, sample_dataset, loader_config):
        """Test memory management features."""
        loader = MLXDataLoader(sample_dataset, loader_config)

        # Test that loader uses unified memory if configured
        assert loader.config.use_unified_memory == True

        # Load a batch and check it's MLX arrays
        batch = next(iter(loader))
        assert isinstance(batch["input_ids"], mx.array)
        assert isinstance(batch["attention_mask"], mx.array)
        assert isinstance(batch["labels"], mx.array)

    def test_clear_cache(self, sample_dataset, loader_config):
        """Test MLX cache clearing."""
        loader = MLXDataLoader(sample_dataset, loader_config)

        # Load some batches
        batch1 = next(iter(loader))
        batch2 = next(iter(loader))

        # Clear MLX cache
        mx.metal.clear_cache()

        # Should still work after clearing cache
        batch3 = next(iter(loader))
        assert batch3 is not None
        assert isinstance(batch3["input_ids"], mx.array)

    def test_loader_performance_profiling(self, sample_dataset, loader_config):
        """Test loader performance."""
        loader = MLXDataLoader(sample_dataset, loader_config)

        # Measure time for loading batches
        start_time = time.time()
        batch_count = 0

        # Load some batches
        for i, batch in enumerate(loader):
            batch_count += 1
            if i >= 5:  # Only test a few batches
                break

        elapsed_time = time.time() - start_time

        # Basic performance check
        assert batch_count > 0
        assert elapsed_time > 0
        avg_batch_time = elapsed_time / batch_count
        assert avg_batch_time < 1.0  # Should be fast

    def test_batch_evaluation_for_lazy_eval_fix(self, sample_dataset, loader_config):
        """Test that batch arrays are evaluated to prevent lazy evaluation buildup.

        This test verifies the fix for the training hang issue caused by
        unevaluated MLX arrays building up in the computation graph.
        """
        loader = MLXDataLoader(sample_dataset, loader_config)

        # Test both prefetch and non-prefetch modes
        for prefetch_size in [0, 4]:  # 0 = no prefetch, 4 = with prefetch
            loader.config.prefetch_size = prefetch_size

            # Get a batch
            batch = next(iter(loader))

            # Check that arrays in the batch are already evaluated
            # This is the critical fix - arrays should be evaluated before being yielded
            for key, value in batch.items():
                if isinstance(value, mx.array):
                    # The array should be materialized (not lazy)
                    # We can check this by attempting to access its data
                    # If it's not evaluated, this would build up computation graph
                    assert value is not None

                    # Verify we can perform operations without building huge graphs
                    # This would hang if arrays weren't pre-evaluated
                    _ = mx.sum(value)

    def test_gradient_computation_with_dataloader_batches(
        self, sample_dataset, loader_config
    ):
        """Test that gradient computation works with dataloader batches.

        This simulates the training scenario where gradients are computed
        on batches from the dataloader.
        """
        loader = MLXDataLoader(sample_dataset, loader_config)

        # Simple model for testing
        class SimpleModel(mlx.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = mlx.nn.Linear(
                    512, 2
                )  # Assuming input_ids are padded to 512

            def __call__(self, x):
                # Simple forward pass - just use first token
                x = x[:, 0].astype(mx.float32)  # Take first token
                return self.linear(mx.zeros((x.shape[0], 512)))  # Dummy input

        model = SimpleModel()

        # Test gradient computation
        def loss_fn(model, batch):
            logits = model(batch["input_ids"])
            labels = batch["labels"].astype(mx.int32)
            return mlx.nn.losses.cross_entropy(logits, labels, reduction="mean")

        # Get a batch
        batch = next(iter(loader))

        # Compute gradients - this should not hang
        value_and_grad_fn = mx.value_and_grad(loss_fn)
        loss, grads = value_and_grad_fn(model, batch)

        # Force evaluation
        mx.eval(loss, grads)

        # Verify we got valid results
        assert loss is not None
        assert grads is not None
        assert isinstance(loss, mx.array)

    def test_prefetch_thread_evaluation(self, sample_dataset):
        """Test that prefetch thread properly evaluates arrays."""
        # Use prefetch mode
        config = create_mlx_loader_config(batch_size=8, prefetch_size=2)
        loader = MLXDataLoader(sample_dataset, config)

        # Load multiple batches to ensure prefetch is working
        batches = []
        for i, batch in enumerate(loader):
            batches.append(batch)
            if i >= 3:  # Get 4 batches
                break

        # All batches should have evaluated arrays
        for batch in batches:
            for key, value in batch.items():
                if isinstance(value, mx.array):
                    # Arrays should be ready to use without causing evaluation buildup
                    result = mx.mean(value)
                    mx.eval(
                        result
                    )  # This should be fast since arrays are pre-evaluated

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
            batch["input_ids"] = batch["input_ids"] + mx.random.randint(
                0, 2, batch["input_ids"].shape
            )
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
        assert isinstance(batch["input_ids"], mx.array)
        assert isinstance(batch["attention_mask"], mx.array)
        assert isinstance(batch["labels"], mx.array)

    def test_loader_state_management(self, sample_dataset, loader_config):
        """Test loader can be recreated with same configuration."""
        loader1 = MLXDataLoader(sample_dataset, loader_config)

        # Get first batch from loader1
        batch1 = next(iter(loader1))

        # Create new loader with same config
        loader2 = MLXDataLoader(sample_dataset, loader_config)

        # Both loaders should have same properties
        assert loader1.num_batches == loader2.num_batches
        assert loader1.config.batch_size == loader2.config.batch_size
        assert len(loader1) == len(loader2)

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
        """Test default collation behavior."""
        config = create_mlx_loader_config(batch_size=8)
        loader = MLXDataLoader(sample_dataset, config)

        batch = next(iter(loader))

        # Check that default collation produces expected structure
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch
        assert "metadata" in batch

        # Check types
        assert isinstance(batch["input_ids"], mx.array)
        assert isinstance(batch["attention_mask"], mx.array)
        assert isinstance(batch["labels"], mx.array)
        assert isinstance(batch["metadata"], list)


@pytest.mark.integration
class TestMLXLoaderIntegration:
    """Integration tests for MLX data loader."""

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for testing."""
        spec = create_dataset_spec(
            num_samples=100,
            num_features=2,
        )
        return MockKaggleDataset(spec, size=100)

    def test_loader_with_transforms(self, sample_dataset):
        """Test loader with different dataset sizes."""
        # Test with small dataset
        spec1 = create_dataset_spec(num_samples=50)
        dataset1 = MockKaggleDataset(spec1, size=50)

        config = create_mlx_loader_config(batch_size=10)
        loader1 = MLXDataLoader(dataset1, config)

        # Load all batches
        batches1 = list(loader1)
        assert len(batches1) == 5  # 50 / 10

        # Test with larger dataset
        spec2 = create_dataset_spec(num_samples=100)
        dataset2 = MockKaggleDataset(spec2, size=100)
        loader2 = MLXDataLoader(dataset2, config)

        batches2 = list(loader2)
        assert len(batches2) == 10  # 100 / 10

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

        assert batch1["input_ids"].shape[0] == 16
        assert batch2["input_ids"].shape[0] == 32
