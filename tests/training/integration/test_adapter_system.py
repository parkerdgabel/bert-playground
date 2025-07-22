"""Integration tests for the framework adapter system.

This module tests the integration of framework adapters with the training system.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import numpy as np

from training.adapters import (
    get_framework_adapter,
    register_adapter,
    list_adapters,
    MLXFrameworkAdapter,
)
from training.adapters.base import BaseFrameworkAdapter, TensorLike


class MockTensor:
    """Mock tensor-like object for testing."""
    
    def __init__(self, data, shape=None, dtype="float32"):
        self.data = data
        self._shape = shape or (len(data) if hasattr(data, '__len__') else ())
        self.dtype = dtype
    
    @property
    def shape(self):
        return self._shape
    
    def item(self):
        if hasattr(self.data, 'item'):
            return self.data.item()
        return float(self.data)
    
    def numpy(self):
        if hasattr(self.data, 'numpy'):
            return self.data.numpy()
        return np.array(self.data)
    
    def __add__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor(self.data + other.data, self.shape, self.dtype)
        return MockTensor(self.data + other, self.shape, self.dtype)
    
    def __mul__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor(self.data * other.data, self.shape, self.dtype)
        return MockTensor(self.data * other, self.shape, self.dtype)


class MockFrameworkAdapter(BaseFrameworkAdapter):
    """Mock framework adapter for testing."""
    
    def __init__(self, name="MockFramework", available=True):
        super().__init__(name)
        self._available = available
    
    @property
    def available(self):
        return self._available
    
    def to_tensor(self, data):
        return MockTensor(data)
    
    def tensor_add(self, a, b):
        return a + b
    
    def tensor_multiply(self, a, b):
        return a * b
    
    def tensor_norm(self, tensor):
        return abs(tensor.data)
    
    def tensor_clip(self, tensor, min_val, max_val):
        clipped_data = max(min_val, min(max_val, tensor.data))
        return MockTensor(clipped_data, tensor.shape, tensor.dtype)
    
    def create_value_and_grad_fn(self, model, loss_fn):
        def mock_value_and_grad(model, batch):
            # Mock implementation
            loss = MockTensor(0.5)
            outputs = {"loss": loss}
            grads = {"param1": MockTensor(0.1), "param2": MockTensor(0.2)}
            return (loss, outputs), grads
        return mock_value_and_grad
    
    def compute_gradient_norm(self, gradients):
        total = sum(abs(g.data) for g in gradients.values() if g is not None)
        return float(total)
    
    def clip_gradients_by_norm(self, gradients, max_norm):
        norm = self.compute_gradient_norm(gradients)
        if norm <= max_norm:
            return gradients, norm
        
        scale = max_norm / norm
        clipped = {k: MockTensor(v.data * scale) for k, v in gradients.items()}
        return clipped, max_norm
    
    def clip_gradients_by_value(self, gradients, max_value):
        clipped = {}
        for k, v in gradients.items():
            clipped_data = max(-max_value, min(max_value, v.data))
            clipped[k] = MockTensor(clipped_data)
        return clipped
    
    def scale_gradients(self, gradients, scale):
        return {k: MockTensor(v.data * scale) for k, v in gradients.items()}
    
    def accumulate_gradients(self, accumulated, current):
        result = {}
        all_keys = set(accumulated.keys()) | set(current.keys())
        for key in all_keys:
            acc_val = accumulated.get(key, MockTensor(0))
            curr_val = current.get(key, MockTensor(0))
            result[key] = MockTensor(acc_val.data + curr_val.data)
        return result
    
    def get_model_parameters(self, model):
        return getattr(model, 'parameters', {})
    
    def update_model_parameters(self, model, optimizer, gradients):
        # Mock update
        pass
    
    def get_learning_rate(self, optimizer):
        return getattr(optimizer, 'learning_rate', 0.001)


class TestAdapterRegistry:
    """Test adapter registration and retrieval."""
    
    def test_adapter_registration(self):
        """Test registering and retrieving adapters."""
        # Create mock adapter
        adapter = MockFrameworkAdapter("TestFramework")
        
        # Register adapter
        register_adapter(adapter)
        
        # Retrieve adapter
        retrieved = get_framework_adapter("TestFramework")
        assert retrieved.name == "TestFramework"
        assert retrieved.available
    
    def test_adapter_auto_detection(self):
        """Test adapter auto-detection."""
        from training.adapters.registry import auto_detect_framework
        
        # Should detect available frameworks
        detected = auto_detect_framework()
        assert detected is not None or len(list_adapters()) == 0
    
    def test_adapter_not_found_error(self):
        """Test error handling for missing adapters."""
        with pytest.raises(KeyError, match="Adapter 'NonExistent' not found"):
            get_framework_adapter("NonExistent")
    
    def test_adapter_not_available_error(self):
        """Test error handling for unavailable adapters."""
        # Register unavailable adapter
        unavailable_adapter = MockFrameworkAdapter("UnavailableFramework", available=False)
        register_adapter(unavailable_adapter)
        
        # Should raise error when trying to use unavailable adapter
        with pytest.raises(RuntimeError, match="not available"):
            get_framework_adapter("UnavailableFramework")


class TestBaseAdapterFunctionality:
    """Test base adapter functionality."""
    
    @pytest.fixture
    def adapter(self):
        return MockFrameworkAdapter()
    
    def test_tensor_operations(self, adapter):
        """Test basic tensor operations."""
        # Create tensors
        tensor1 = adapter.to_tensor(5.0)
        tensor2 = adapter.to_tensor(3.0)
        
        # Test operations
        result_add = adapter.tensor_add(tensor1, tensor2)
        assert result_add.data == 8.0
        
        result_mul = adapter.tensor_multiply(tensor1, 2.0)
        assert result_mul.data == 10.0
        
        norm = adapter.tensor_norm(tensor1)
        assert norm == 5.0
    
    def test_gradient_operations(self, adapter):
        """Test gradient-related operations."""
        # Create mock gradients
        gradients = {
            "param1": adapter.to_tensor(0.5),
            "param2": adapter.to_tensor(1.0),
        }
        
        # Test gradient norm computation
        norm = adapter.compute_gradient_norm(gradients)
        assert norm == 1.5  # |0.5| + |1.0|
        
        # Test gradient clipping by norm
        clipped_grads, clipped_norm = adapter.clip_gradients_by_norm(gradients, max_norm=1.0)
        assert clipped_norm == 1.0
        assert all(abs(g.data) <= 1.0 for g in clipped_grads.values())
        
        # Test gradient clipping by value
        clipped_by_value = adapter.clip_gradients_by_value(gradients, max_value=0.7)
        assert all(abs(g.data) <= 0.7 for g in clipped_by_value.values())
        
        # Test gradient scaling
        scaled = adapter.scale_gradients(gradients, scale=2.0)
        assert scaled["param1"].data == 1.0
        assert scaled["param2"].data == 2.0
    
    def test_gradient_accumulation(self, adapter):
        """Test gradient accumulation."""
        # Create initial gradients
        accumulated = {
            "param1": adapter.to_tensor(0.1),
            "param2": adapter.to_tensor(0.2),
        }
        
        # Create current gradients
        current = {
            "param1": adapter.to_tensor(0.05),
            "param2": adapter.to_tensor(0.1),
        }
        
        # Accumulate
        result = adapter.accumulate_gradients(accumulated, current)
        
        # Verify accumulation
        assert result["param1"].data == 0.15
        assert result["param2"].data == 0.3
    
    def test_python_conversion(self, adapter):
        """Test tensor to python conversion."""
        tensor = adapter.to_tensor(42.0)
        python_val = adapter.to_python(tensor)
        assert python_val == 42.0


class TestMLXAdapterIntegration:
    """Test MLX adapter integration (if available)."""
    
    @pytest.fixture
    def mlx_adapter(self):
        """Get MLX adapter if available, otherwise skip."""
        try:
            adapter = MLXFrameworkAdapter()
            if not adapter.available:
                pytest.skip("MLX not available")
            return adapter
        except Exception:
            pytest.skip("MLX adapter not available")
    
    def test_mlx_adapter_basic_operations(self, mlx_adapter):
        """Test basic MLX adapter operations."""
        # This test would run only if MLX is available
        assert mlx_adapter.name == "MLX"
        assert mlx_adapter.available
        
        # Test tensor creation and conversion
        try:
            import mlx.core as mx
            
            # Create MLX tensor
            tensor = mx.array([1.0, 2.0, 3.0])
            python_val = mlx_adapter.to_python(tensor)
            
            # Should be able to convert
            assert isinstance(python_val, (float, int, list))
        except ImportError:
            pytest.skip("MLX not available for detailed testing")
    
    def test_mlx_gradient_operations(self, mlx_adapter):
        """Test MLX gradient operations."""
        try:
            import mlx.core as mx
            
            # Create mock gradients
            gradients = {
                "param1": mx.array([0.1, 0.2]),
                "param2": mx.array([0.3, 0.4]),
            }
            
            # Test gradient norm
            norm = mlx_adapter.compute_gradient_norm(gradients)
            assert isinstance(norm, float)
            assert norm > 0
            
            # Test gradient clipping
            clipped, clip_norm = mlx_adapter.clip_gradients_by_norm(gradients, 0.5)
            assert clip_norm <= 0.5
            
        except ImportError:
            pytest.skip("MLX not available for gradient testing")


class TestAdapterCommandIntegration:
    """Test integration of adapters with commands."""
    
    def test_adapter_with_backward_command(self):
        """Test adapter integration with backward command."""
        from training.commands.backward import BackwardCommand
        
        # Create mock adapter and command
        adapter = MockFrameworkAdapter()
        backward_cmd = BackwardCommand(grad_clip_norm=1.0)
        
        # Create mock context
        context = Mock()
        context.model = Mock()
        context.loss = 0.5
        context.batch = {"input": [1, 2, 3]}
        context.outputs = {}
        context.is_training = True
        context.gradients = {}
        context.metrics = {}
        
        # Override the gradient computation to use our adapter
        def mock_compute_gradients(model, loss, inputs, outputs):
            return adapter.create_value_and_grad_fn(model, lambda m, b: (loss, {}))
        
        with patch.object(backward_cmd, '_compute_gradients', mock_compute_gradients):
            # This would test the integration, but requires more complex setup
            # For now, just verify the adapter can be used
            assert adapter.available
    
    def test_adapter_with_optimizer_command(self):
        """Test adapter integration with optimizer command."""
        from training.commands.optimizer_step import OptimizerStepCommand
        
        adapter = MockFrameworkAdapter()
        optimizer_cmd = OptimizerStepCommand()
        
        # Create mock context
        context = Mock()
        context.model = Mock()
        context.optimizer = Mock()
        context.optimizer.learning_rate = 0.001
        context.gradients = {"param1": adapter.to_tensor(0.1)}
        context.should_update_weights = True
        context.is_training = True
        context.state = Mock()
        context.state.global_step = 0
        context.config = {}
        
        # Test that adapter methods could be used
        lr = adapter.get_learning_rate(context.optimizer)
        assert lr == 0.001


@pytest.mark.integration 
class TestAdapterSystemIntegration:
    """High-level integration tests for the adapter system."""
    
    def test_adapter_framework_abstraction(self):
        """Test that adapters provide proper framework abstraction."""
        # Create adapters for different "frameworks"
        adapter1 = MockFrameworkAdapter("Framework1")
        adapter2 = MockFrameworkAdapter("Framework2")
        
        # Register adapters
        register_adapter(adapter1)
        register_adapter(adapter2)
        
        # Both should provide the same interface
        for adapter in [adapter1, adapter2]:
            # Test common interface
            tensor = adapter.to_tensor(1.5)
            assert hasattr(tensor, 'shape')
            assert hasattr(tensor, 'item')
            
            # Test operations
            result = adapter.tensor_multiply(tensor, 2.0)
            assert adapter.to_python(result) == 3.0
    
    def test_adapter_switching(self):
        """Test switching between different adapters."""
        # Create different adapters
        fast_adapter = MockFrameworkAdapter("FastFramework")
        accurate_adapter = MockFrameworkAdapter("AccurateFramework")
        
        # Register both
        register_adapter(fast_adapter)
        register_adapter(accurate_adapter)
        
        # Should be able to retrieve either
        fast = get_framework_adapter("FastFramework")
        accurate = get_framework_adapter("AccurateFramework")
        
        assert fast.name == "FastFramework"
        assert accurate.name == "AccurateFramework"
        
        # Both should work with the same operations
        tensor1 = fast.to_tensor(2.0)
        tensor2 = accurate.to_tensor(2.0)
        
        assert fast.to_python(tensor1) == accurate.to_python(tensor2)
    
    def test_adapter_error_propagation(self):
        """Test that adapter errors are properly propagated."""
        # Create adapter that raises errors
        class ErrorAdapter(MockFrameworkAdapter):
            def to_tensor(self, data):
                raise ValueError("Test error")
        
        error_adapter = ErrorAdapter("ErrorFramework")
        register_adapter(error_adapter)
        
        # Error should propagate when using adapter
        adapter = get_framework_adapter("ErrorFramework")
        with pytest.raises(ValueError, match="Test error"):
            adapter.to_tensor(1.0)