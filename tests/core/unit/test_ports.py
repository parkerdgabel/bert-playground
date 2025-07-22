"""Unit tests for port interfaces."""

import pytest
from pathlib import Path
from typing import Any, Callable, Sequence
from datetime import datetime
import tempfile
import numpy as np

from core.ports.compute import ComputeBackend, DataType, NeuralOps
from core.ports.config import ConfigurationProvider, ConfigRegistry
from core.ports.monitoring import MonitoringService, ExperimentTracker
from core.ports.storage import StorageService, ModelStorageService


class MockComputeBackend:
    """Mock implementation of ComputeBackend for testing."""

    @property
    def name(self) -> str:
        return "mock"

    @property
    def supports_compilation(self) -> bool:
        return True

    def array(self, data, dtype=None, device=None):
        return np.array(data)

    def zeros(self, shape, dtype=None, device=None):
        return np.zeros(shape)

    def ones(self, shape, dtype=None, device=None):
        return np.ones(shape)

    def randn(self, shape, dtype=None, device=None, seed=None):
        if seed:
            np.random.seed(seed)
        return np.random.randn(*shape)

    def to_numpy(self, array):
        return np.array(array)

    def from_numpy(self, array, dtype=None, device=None):
        return np.array(array)

    def shape(self, array):
        return array.shape

    def dtype(self, array):
        return array.dtype

    def device(self, array):
        return "cpu"

    def compile(self, function, static_argnums=None, static_argnames=None):
        return function  # No-op compilation

    def gradient(self, function, argnums=0):
        def grad_fn(*args):
            # Simple numerical gradient
            return np.ones_like(args[0]) * 0.1
        return grad_fn

    def value_and_gradient(self, function, argnums=0):
        def val_and_grad_fn(*args):
            value = function(*args)
            grad = np.ones_like(args[0]) * 0.1
            return value, grad
        return val_and_grad_fn


class MockStorageService:
    """Mock implementation of StorageService for testing."""

    def __init__(self):
        self._data = {}
        self._metadata = {}

    def save(self, key, value, metadata=None):
        self._data[str(key)] = value
        if metadata:
            self._metadata[str(key)] = metadata

    def load(self, key, expected_type=None):
        key = str(key)
        if key not in self._data:
            raise KeyError(f"Key not found: {key}")
        value = self._data[key]
        if expected_type and not isinstance(value, expected_type):
            raise TypeError(f"Expected {expected_type}, got {type(value)}")
        return value

    def exists(self, key):
        return str(key) in self._data

    def delete(self, key):
        key = str(key)
        if key not in self._data:
            raise KeyError(f"Key not found: {key}")
        del self._data[key]
        self._metadata.pop(key, None)

    def list_keys(self, prefix=None, pattern=None):
        keys = list(self._data.keys())
        if prefix:
            keys = [k for k in keys if k.startswith(str(prefix))]
        return keys

    def get_metadata(self, key):
        key = str(key)
        if key not in self._data:
            raise KeyError(f"Key not found: {key}")
        return self._metadata.get(key)


class TestComputeBackendProtocol:
    """Test ComputeBackend protocol compliance."""

    def test_mock_backend_implements_protocol(self):
        """Test that mock backend implements ComputeBackend protocol."""
        backend = MockComputeBackend()
        assert isinstance(backend, ComputeBackend)

    def test_basic_operations(self):
        """Test basic array operations."""
        backend = MockComputeBackend()
        
        # Test array creation
        arr = backend.array([1, 2, 3])
        assert backend.shape(arr) == (3,)
        
        # Test zeros/ones
        zeros = backend.zeros((2, 3))
        assert backend.shape(zeros) == (2, 3)
        assert np.all(zeros == 0)
        
        ones = backend.ones((2, 3))
        assert np.all(ones == 1)

    def test_random_operations(self):
        """Test random array generation."""
        backend = MockComputeBackend()
        
        # Test deterministic random
        arr1 = backend.randn((3, 3), seed=42)
        arr2 = backend.randn((3, 3), seed=42)
        assert np.allclose(arr1, arr2)

    def test_numpy_conversion(self):
        """Test conversion to/from numpy."""
        backend = MockComputeBackend()
        
        # Create array and convert to numpy
        arr = backend.array([1, 2, 3])
        np_arr = backend.to_numpy(arr)
        assert isinstance(np_arr, np.ndarray)
        
        # Convert back from numpy
        arr2 = backend.from_numpy(np_arr)
        assert np.allclose(arr, arr2)

    def test_compilation(self):
        """Test function compilation."""
        backend = MockComputeBackend()
        
        def test_fn(x):
            return x * 2
        
        compiled_fn = backend.compile(test_fn)
        assert callable(compiled_fn)
        
        # Mock just returns original function
        result = compiled_fn(5)
        assert result == 10

    def test_gradient_computation(self):
        """Test gradient computation."""
        backend = MockComputeBackend()
        
        def test_fn(x):
            return x * x
        
        grad_fn = backend.gradient(test_fn)
        grad = grad_fn(np.array([1.0, 2.0, 3.0]))
        
        assert grad is not None
        assert isinstance(grad, np.ndarray)


class TestStorageServiceProtocol:
    """Test StorageService protocol compliance."""

    def test_mock_storage_implements_protocol(self):
        """Test that mock storage implements StorageService protocol."""
        storage = MockStorageService()
        assert isinstance(storage, StorageService)

    def test_save_and_load(self):
        """Test basic save and load operations."""
        storage = MockStorageService()
        
        # Save data
        test_data = {"key": "value", "number": 42}
        storage.save("test_key", test_data)
        
        # Load data
        loaded_data = storage.load("test_key")
        assert loaded_data == test_data

    def test_save_with_metadata(self):
        """Test saving with metadata."""
        storage = MockStorageService()
        
        test_data = [1, 2, 3]
        metadata = {"created": "2023-01-01", "type": "list"}
        
        storage.save("test_key", test_data, metadata)
        
        loaded_data = storage.load("test_key")
        loaded_metadata = storage.get_metadata("test_key")
        
        assert loaded_data == test_data
        assert loaded_metadata == metadata

    def test_exists_and_delete(self):
        """Test existence checking and deletion."""
        storage = MockStorageService()
        
        # Test non-existent key
        assert not storage.exists("nonexistent")
        
        # Save and check existence
        storage.save("test_key", "test_value")
        assert storage.exists("test_key")
        
        # Delete and check
        storage.delete("test_key")
        assert not storage.exists("test_key")
        
        # Test deleting non-existent key
        with pytest.raises(KeyError):
            storage.delete("nonexistent")

    def test_list_keys(self):
        """Test key listing."""
        storage = MockStorageService()
        
        # Save multiple keys
        storage.save("prefix/key1", "value1")
        storage.save("prefix/key2", "value2")
        storage.save("other/key3", "value3")
        
        # List all keys
        all_keys = storage.list_keys()
        assert len(all_keys) == 3
        
        # List with prefix
        prefix_keys = storage.list_keys(prefix="prefix")
        assert len(prefix_keys) == 2
        assert all("prefix" in key for key in prefix_keys)

    def test_type_validation(self):
        """Test type validation on load."""
        storage = MockStorageService()
        
        storage.save("string_key", "test_string")
        storage.save("int_key", 42)
        
        # Load with correct type
        result = storage.load("string_key", expected_type=str)
        assert result == "test_string"
        
        # Load with incorrect type
        with pytest.raises(TypeError):
            storage.load("string_key", expected_type=int)

    def test_error_handling(self):
        """Test error handling for missing keys."""
        storage = MockStorageService()
        
        # Load non-existent key
        with pytest.raises(KeyError):
            storage.load("nonexistent")
        
        # Get metadata for non-existent key
        with pytest.raises(KeyError):
            storage.get_metadata("nonexistent")


class TestProtocolCompliance:
    """Test protocol compliance of mock implementations."""

    def test_compute_backend_compliance(self):
        """Test ComputeBackend protocol compliance."""
        backend = MockComputeBackend()
        
        # Check required properties
        assert hasattr(backend, 'name')
        assert hasattr(backend, 'supports_compilation')
        
        # Check required methods
        required_methods = [
            'array', 'zeros', 'ones', 'randn',
            'to_numpy', 'from_numpy', 'shape', 'dtype', 'device',
            'compile', 'gradient', 'value_and_gradient'
        ]
        
        for method in required_methods:
            assert hasattr(backend, method)
            assert callable(getattr(backend, method))

    def test_storage_service_compliance(self):
        """Test StorageService protocol compliance."""
        storage = MockStorageService()
        
        # Check required methods
        required_methods = [
            'save', 'load', 'exists', 'delete', 
            'list_keys', 'get_metadata'
        ]
        
        for method in required_methods:
            assert hasattr(storage, method)
            assert callable(getattr(storage, method))


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestPortIntegration:
    """Test integration between different ports."""

    def test_compute_storage_integration(self):
        """Test integration between compute and storage ports."""
        backend = MockComputeBackend()
        storage = MockStorageService()
        
        # Create array with compute backend
        arr = backend.array([1, 2, 3, 4, 5])
        
        # Convert to numpy for storage
        np_arr = backend.to_numpy(arr)
        
        # Store array
        storage.save("test_array", np_arr, {"dtype": str(np_arr.dtype)})
        
        # Load and convert back
        loaded_np = storage.load("test_array")
        loaded_arr = backend.from_numpy(loaded_np)
        
        # Verify round trip
        assert np.allclose(backend.to_numpy(arr), backend.to_numpy(loaded_arr))
        
        # Check metadata
        metadata = storage.get_metadata("test_array")
        assert "dtype" in metadata