"""Unit tests for adapter implementations."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import mlx.core as mx
import numpy as np
import pytest

from core.adapters.file_storage import FileStorageAdapter, ModelFileStorageAdapter
from core.adapters.mlx_adapter import MLXComputeAdapter, MLXNeuralOpsAdapter
from core.adapters.yaml_config import YAMLConfigAdapter, ConfigRegistryImpl
from core.adapters.loguru_monitoring import LoguruMonitoringAdapter
from core.ports.compute import DataType


class TestMLXComputeAdapter:
    """Test MLX compute backend adapter."""

    def test_adapter_properties(self):
        """Test adapter properties."""
        adapter = MLXComputeAdapter()
        
        assert adapter.name == "mlx"
        assert adapter.supports_compilation is True

    def test_array_creation(self):
        """Test array creation methods."""
        adapter = MLXComputeAdapter()
        
        # Test array creation
        arr = adapter.array([1, 2, 3])
        assert isinstance(arr, mx.array)
        assert arr.shape == (3,)
        
        # Test zeros
        zeros = adapter.zeros((2, 3))
        assert zeros.shape == (2, 3)
        assert mx.allclose(zeros, mx.zeros((2, 3)))
        
        # Test ones
        ones = adapter.ones((2, 3))
        assert ones.shape == (2, 3)
        assert mx.allclose(ones, mx.ones((2, 3)))

    def test_dtype_conversion(self):
        """Test data type conversion."""
        adapter = MLXComputeAdapter()
        
        # Test DataType enum conversion
        arr_f32 = adapter.array([1.0, 2.0], dtype=DataType.FLOAT32)
        assert arr_f32.dtype == mx.float32
        
        arr_i32 = adapter.array([1, 2], dtype=DataType.INT32)
        assert arr_i32.dtype == mx.int32

    def test_numpy_conversion(self):
        """Test numpy conversion."""
        adapter = MLXComputeAdapter()
        
        # Create MLX array
        mlx_arr = adapter.array([1, 2, 3, 4])
        
        # Convert to numpy
        np_arr = adapter.to_numpy(mlx_arr)
        assert isinstance(np_arr, np.ndarray)
        assert np.array_equal(np_arr, [1, 2, 3, 4])
        
        # Convert back to MLX
        mlx_arr2 = adapter.from_numpy(np_arr)
        assert isinstance(mlx_arr2, mx.array)
        assert mx.array_equal(mlx_arr, mlx_arr2)

    def test_array_properties(self):
        """Test array property access."""
        adapter = MLXComputeAdapter()
        
        arr = adapter.array([[1, 2], [3, 4]], dtype=DataType.FLOAT32)
        
        # Test shape
        assert adapter.shape(arr) == (2, 2)
        
        # Test dtype
        assert adapter.dtype(arr) == mx.float32
        
        # Test device (MLX always returns "gpu")
        assert adapter.device(arr) == "gpu"

    def test_random_with_seed(self):
        """Test deterministic random generation."""
        adapter = MLXComputeAdapter()
        
        # Generate with same seed
        arr1 = adapter.randn((3, 3), seed=42)
        arr2 = adapter.randn((3, 3), seed=42)
        
        # Should be identical
        assert mx.allclose(arr1, arr2)
        
        # Different seed should be different
        arr3 = adapter.randn((3, 3), seed=123)
        assert not mx.allclose(arr1, arr3)

    def test_compilation(self):
        """Test function compilation."""
        adapter = MLXComputeAdapter()
        
        def test_fn(x):
            return x * 2
        
        compiled_fn = adapter.compile(test_fn)
        
        # Test that compiled function works
        test_input = adapter.array([1, 2, 3])
        result = compiled_fn(test_input)
        expected = adapter.array([2, 4, 6])
        
        assert mx.allclose(result, expected)


class TestMLXNeuralOpsAdapter:
    """Test MLX neural operations adapter."""

    def test_linear_operation(self):
        """Test linear transformation."""
        adapter = MLXNeuralOpsAdapter()
        
        # Create test data
        input_arr = mx.array([[1.0, 2.0], [3.0, 4.0]])  # [2, 2]
        weight_arr = mx.array([[0.5, 0.5], [1.0, -1.0]])  # [2, 2]
        bias_arr = mx.array([0.1, 0.2])  # [2]
        
        # Test linear without bias
        output_no_bias = adapter.linear(input_arr, weight_arr)
        assert output_no_bias.shape == (2, 2)
        
        # Test linear with bias
        output_with_bias = adapter.linear(input_arr, weight_arr, bias_arr)
        assert output_with_bias.shape == (2, 2)
        
        # Verify bias was added
        assert not mx.array_equal(output_no_bias, output_with_bias)

    def test_embedding_lookup(self):
        """Test embedding lookup."""
        adapter = MLXNeuralOpsAdapter()
        
        # Create embedding table
        embedding_weight = mx.random.normal((10, 5))  # 10 vocab, 5 dim
        
        # Create indices
        indices = mx.array([0, 2, 5])
        
        # Perform lookup
        embeddings = adapter.embedding(indices, embedding_weight)
        
        assert embeddings.shape == (3, 5)
        
        # Verify correct embeddings were selected
        assert mx.allclose(embeddings[0], embedding_weight[0])
        assert mx.allclose(embeddings[1], embedding_weight[2])
        assert mx.allclose(embeddings[2], embedding_weight[5])

    def test_softmax(self):
        """Test softmax activation."""
        adapter = MLXNeuralOpsAdapter()
        
        # Create test logits
        logits = mx.array([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]])
        
        # Apply softmax
        probs = adapter.softmax(logits, dim=-1)
        
        assert probs.shape == logits.shape
        
        # Check that probabilities sum to 1
        prob_sums = mx.sum(probs, axis=-1)
        assert mx.allclose(prob_sums, mx.ones(2))


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestFileStorageAdapter:
    """Test file storage adapter."""

    def test_basic_operations(self, temp_dir):
        """Test basic file operations."""
        adapter = FileStorageAdapter(temp_dir)
        
        # Test save and load
        test_data = {"key": "value", "number": 42}
        adapter.save("test.json", test_data)
        
        loaded_data = adapter.load("test.json")
        assert loaded_data == test_data

    def test_metadata_handling(self, temp_dir):
        """Test metadata handling."""
        adapter = FileStorageAdapter(temp_dir)
        
        test_data = [1, 2, 3]
        metadata = {"type": "list", "length": 3}
        
        adapter.save("data.json", test_data, metadata)
        
        loaded_data = adapter.load("data.json")
        loaded_metadata = adapter.get_metadata("data.json")
        
        assert loaded_data == test_data
        assert loaded_metadata == metadata

    def test_file_formats(self, temp_dir):
        """Test different file formats."""
        adapter = FileStorageAdapter(temp_dir)
        
        # Test JSON
        json_data = {"test": "json"}
        adapter.save("test.json", json_data)
        assert adapter.load("test.json") == json_data
        
        # Test text
        text_data = "Hello, world!"
        adapter.save("test.txt", text_data)
        assert adapter.load("test.txt") == text_data

    def test_path_resolution(self, temp_dir):
        """Test path resolution."""
        adapter = FileStorageAdapter(temp_dir)
        
        # Test absolute path
        abs_path = temp_dir / "absolute.json"
        adapter.save(abs_path, {"absolute": True})
        assert adapter.exists(abs_path)
        
        # Test relative path
        adapter.save("relative.json", {"relative": True})
        assert adapter.exists("relative.json")

    def test_directory_operations(self, temp_dir):
        """Test directory operations."""
        adapter = FileStorageAdapter(temp_dir)
        
        # Create nested structure
        adapter.save("dir1/file1.json", {"file": 1})
        adapter.save("dir1/file2.json", {"file": 2})
        adapter.save("dir2/file3.json", {"file": 3})
        
        # List all keys
        all_keys = adapter.list_keys()
        assert len(all_keys) == 3
        
        # List with pattern
        dir1_keys = adapter.list_keys(pattern="dir1/*")
        assert len(dir1_keys) == 2

    def test_error_handling(self, temp_dir):
        """Test error handling."""
        adapter = FileStorageAdapter(temp_dir)
        
        # Load non-existent file
        with pytest.raises(KeyError):
            adapter.load("nonexistent.json")
        
        # Delete non-existent file
        with pytest.raises(KeyError):
            adapter.delete("nonexistent.json")

    def test_exists_and_delete(self, temp_dir):
        """Test existence checking and deletion."""
        adapter = FileStorageAdapter(temp_dir)
        
        # File doesn't exist initially
        assert not adapter.exists("test.json")
        
        # Save file
        adapter.save("test.json", {"test": True})
        assert adapter.exists("test.json")
        
        # Delete file
        adapter.delete("test.json")
        assert not adapter.exists("test.json")


class TestYAMLConfigAdapter:
    """Test YAML configuration adapter."""

    def test_basic_load_save(self, temp_dir):
        """Test basic YAML operations."""
        adapter = YAMLConfigAdapter()
        config_file = temp_dir / "config.yaml"
        
        # Test data
        config_data = {
            "model": {"name": "bert", "size": "base"},
            "training": {"epochs": 10, "lr": 0.001}
        }
        
        # Save config
        adapter.save(config_data, config_file)
        assert config_file.exists()
        
        # Load config
        loaded_config = adapter.load(config_file)
        assert loaded_config == config_data

    def test_environment_overrides(self, temp_dir):
        """Test environment-specific overrides."""
        adapter = YAMLConfigAdapter()
        config_file = temp_dir / "config.yaml"
        
        config_data = {
            "base_setting": "default",
            "environments": {
                "dev": {"base_setting": "dev_value", "debug": True},
                "prod": {"base_setting": "prod_value", "debug": False}
            }
        }
        
        # Save config
        adapter.save(config_data, config_file)
        
        # Load with environment
        dev_config = adapter.load(config_file, environment="dev")
        assert dev_config["base_setting"] == "dev_value"
        assert dev_config["debug"] is True
        assert "environments" not in dev_config  # Should be removed
        
        prod_config = adapter.load(config_file, environment="prod")
        assert prod_config["base_setting"] == "prod_value"
        assert prod_config["debug"] is False

    def test_config_merging(self):
        """Test configuration merging."""
        adapter = YAMLConfigAdapter()
        
        base_config = {
            "model": {"name": "bert", "layers": 12},
            "training": {"epochs": 10}
        }
        
        override_config = {
            "model": {"name": "modernbert"},  # Override name, keep layers
            "training": {"epochs": 20, "lr": 0.001}  # Override epochs, add lr
        }
        
        merged = adapter.merge(base_config, override_config)
        
        assert merged["model"]["name"] == "modernbert"
        assert merged["model"]["layers"] == 12  # Preserved from base
        assert merged["training"]["epochs"] == 20
        assert merged["training"]["lr"] == 0.001

    def test_dot_notation(self):
        """Test dot notation for nested keys."""
        adapter = YAMLConfigAdapter()
        
        config = {
            "model": {"bert": {"layers": 12, "hidden": 768}},
            "training": {"lr": 0.001}
        }
        
        # Test get
        assert adapter.get(config, "model.bert.layers") == 12
        assert adapter.get(config, "model.bert.hidden") == 768
        assert adapter.get(config, "nonexistent", default="default") == "default"
        
        # Test set
        updated = adapter.set(config, "model.bert.layers", 24)
        assert adapter.get(updated, "model.bert.layers") == 24
        
        # Test setting new key
        updated = adapter.set(config, "model.new.key", "value")
        assert adapter.get(updated, "model.new.key") == "value"

    def test_env_var_expansion(self):
        """Test environment variable expansion."""
        adapter = YAMLConfigAdapter()
        
        config = {
            "path": "${HOME}/models",
            "url": "https://${HOST}:${PORT}/api"
        }
        
        env_vars = {"HOME": "/home/user", "HOST": "localhost", "PORT": "8080"}
        expanded = adapter.expand_vars(config, env_vars)
        
        assert expanded["path"] == "/home/user/models"
        assert expanded["url"] == "https://localhost:8080/api"

    def test_flatten_unflatten(self):
        """Test configuration flattening and unflattening."""
        adapter = YAMLConfigAdapter()
        
        nested_config = {
            "model": {
                "bert": {"layers": 12, "hidden": 768},
                "head": {"dropout": 0.1}
            },
            "training": {"lr": 0.001, "epochs": 10}
        }
        
        # Flatten
        flat = adapter.to_flat(nested_config)
        
        expected_keys = {
            "model.bert.layers", "model.bert.hidden",
            "model.head.dropout", "training.lr", "training.epochs"
        }
        assert set(flat.keys()) == expected_keys
        
        # Unflatten
        reconstructed = adapter.from_flat(flat)
        assert reconstructed == nested_config


class TestConfigRegistryImpl:
    """Test configuration registry implementation."""

    def test_source_registration(self):
        """Test registering and managing configuration sources."""
        registry = ConfigRegistryImpl()
        
        # Create mock providers
        provider1 = Mock()
        provider2 = Mock()
        
        # Register sources
        registry.register_source("source1", provider1, priority=1)
        registry.register_source("source2", provider2, priority=2)
        
        # Test listing
        sources = registry.list_sources()
        assert len(sources) == 2
        assert ("source1", 1) in sources
        assert ("source2", 2) in sources
        
        # Test getting source
        assert registry.get_source("source1") == provider1
        assert registry.get_source("nonexistent") is None
        
        # Test unregistering
        registry.unregister_source("source1")
        assert registry.get_source("source1") is None
        assert len(registry.list_sources()) == 1


class TestLoguruMonitoringAdapter:
    """Test Loguru monitoring adapter."""

    def test_basic_logging(self):
        """Test basic logging functionality."""
        adapter = LoguruMonitoringAdapter()
        
        # Test different log levels
        adapter.debug("Debug message")
        adapter.info("Info message")
        adapter.warning("Warning message")
        adapter.error("Error message")
        
        # Test logging with context
        adapter.info("Message with context", key1="value1", key2="value2")

    def test_metrics(self):
        """Test metric recording."""
        adapter = LoguruMonitoringAdapter()
        
        # Record various metrics
        adapter.gauge("cpu_usage", 75.5)
        adapter.counter("requests", 10)
        adapter.histogram("response_time", 150.0)
        
        # Check that metrics are stored internally
        assert "gauge.cpu_usage" in adapter._metrics
        assert "counter.requests" in adapter._metrics
        assert "histogram.response_time" in adapter._metrics

    def test_context_management(self):
        """Test global context management."""
        adapter = LoguruMonitoringAdapter()
        
        # Set context
        adapter.set_context(user_id="123", session_id="abc")
        assert "user_id" in adapter._context
        assert "session_id" in adapter._context
        
        # Clear context
        adapter.clear_context()
        assert len(adapter._context) == 0

    def test_timer_context_manager(self):
        """Test timer context manager."""
        adapter = LoguruMonitoringAdapter()
        
        # Use timer
        with adapter.timer("test_operation") as timer:
            # Simulate some work
            import time
            time.sleep(0.01)  # 10ms
            
            # Timer should track elapsed time
            assert timer.elapsed > 0
        
        # Check that timer metric was recorded
        assert "timer.test_operation" in adapter._metrics

    def test_span_context_manager(self):
        """Test span context manager."""
        adapter = LoguruMonitoringAdapter()
        
        # Use span
        with adapter.span("test_span", {"operation": "test"}) as span:
            span.set_tag("tag_key", "tag_value")
            span.log("Operation in progress")
            span.set_status("success")
        
        # Span should complete without error