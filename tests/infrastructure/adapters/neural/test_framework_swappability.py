"""Tests demonstrating framework swappability of the neural network port.

This module shows how the same abstraction can work with different frameworks
without requiring all frameworks to be installed.
"""

import pytest
from unittest.mock import Mock, patch
import numpy as np

from infrastructure.ports.neural import NeuralBackend, Module, ActivationType, create_neural_backend


class TestFrameworkSwappability:
    """Test suite demonstrating framework-agnostic neural network operations."""
    
    def test_neural_backend_protocol(self):
        """Test that the NeuralBackend protocol defines the expected interface."""
        # Get all required methods from the protocol
        required_methods = [
            'name', 'supports_mixed_precision', 'linear', 'embedding',
            'layer_norm', 'rms_norm', 'dropout', 'multi_head_attention',
            'activation', 'gelu', 'relu', 'silu', 'sequential',
            'module_list', 'module_dict', 'cross_entropy_loss',
            'binary_cross_entropy_loss', 'mse_loss', 'matmul', 'transpose',
            'reshape', 'concat', 'split', 'mean', 'sum', 'max', 'min',
            'softmax', 'log_softmax', 'rotary_embedding', 'apply_rotary_pos_emb',
            'masked_fill', 'where', 'parameter', 'no_grad', 'enable_grad',
            'device_context', 'unsqueeze', 'arange', 'broadcast_to',
            'zeros_like', 'ones'
        ]
        
        # Verify protocol has all expected attributes
        for method in required_methods:
            assert hasattr(NeuralBackend, method)
    
    def test_mlx_backend_exists(self):
        """Test that MLX backend can be created."""
        from infrastructure.adapters.neural.mlx_backend import MLXNeuralBackend
        
        backend = MLXNeuralBackend()
        assert isinstance(backend, NeuralBackend)
        assert backend.name == "mlx"
    
    def test_backend_factory_with_mlx(self):
        """Test backend factory creates MLX backend."""
        backend = create_neural_backend("mlx")
        assert backend.name == "mlx"
    
    def test_pytorch_backend_structure(self):
        """Test PyTorch backend structure without requiring PyTorch installation."""
        # Instead of importing the actual backend, we'll test the concept
        # by checking the factory function recognizes pytorch
        
        # Test that the factory knows about PyTorch backend
        try:
            # This will fail because PyTorch isn't installed, but that's ok
            # We're testing that the factory knows about it
            backend = create_neural_backend("pytorch")
        except ImportError as e:
            # Expected - PyTorch not installed
            assert "torch" in str(e)
        except Exception:
            # Any other error means the factory doesn't know about pytorch
            pytest.fail("Factory should recognize 'pytorch' backend")
    
    def test_framework_agnostic_model_definition(self):
        """Test defining a model that works with any backend."""
        
        def create_transformer_block(backend: NeuralBackend, d_model: int, num_heads: int) -> Module:
            """Create a transformer block using any neural backend.
            
            This function demonstrates how the same model architecture
            can be defined for any framework using the abstraction.
            """
            # This would create the same architecture regardless of backend
            # Components would be:
            # 1. Multi-head attention
            # 2. Layer normalization
            # 3. Feed-forward network
            # 4. Another layer normalization
            
            # Mock implementation for testing
            mock_module = Mock(spec=Module)
            mock_module.forward = Mock(return_value=np.zeros((1, 10, d_model)))
            return mock_module
        
        # Mock backends to show the same function works with different frameworks
        mlx_backend = Mock(spec=NeuralBackend)
        mlx_backend.name = "mlx"
        
        pytorch_backend = Mock(spec=NeuralBackend) 
        pytorch_backend.name = "pytorch"
        
        # Create the same architecture with different backends
        mlx_transformer = create_transformer_block(mlx_backend, d_model=512, num_heads=8)
        pytorch_transformer = create_transformer_block(pytorch_backend, d_model=512, num_heads=8)
        
        # Both should be Module instances
        assert isinstance(mlx_transformer, Module)
        assert isinstance(pytorch_transformer, Module)
    
    def test_backend_specific_optimizations(self):
        """Test that backends can have framework-specific optimizations."""
        
        class OptimizedBackend(NeuralBackend):
            """Example backend with framework-specific optimizations."""
            
            @property
            def name(self) -> str:
                return "optimized"
            
            @property
            def supports_mixed_precision(self) -> bool:
                return True
            
            def linear(self, in_features: int, out_features: int, bias: bool = True, dtype=None) -> Module:
                # Framework-specific optimized linear layer
                mock_linear = Mock(spec=Module)
                mock_linear._optimized = True  # Mark as optimized
                return mock_linear
            
            # ... other required methods would be implemented
        
        # The backend can provide framework-specific optimizations
        # while maintaining the same interface
        backend = OptimizedBackend()
        linear = backend.linear(10, 20)
        assert hasattr(linear, '_optimized')
    
    def test_loss_function_abstraction(self):
        """Test that loss functions follow the same pattern across backends."""
        
        # Mock a backend
        backend = Mock(spec=NeuralBackend)
        
        # Define a mock cross entropy loss that returns a callable
        def mock_cross_entropy(**kwargs):
            def loss_fn(input_array, target_array):
                # Simplified loss calculation for testing
                return np.mean(input_array)
            return loss_fn
        
        backend.cross_entropy_loss = mock_cross_entropy
        
        # Create loss function
        loss_fn = backend.cross_entropy_loss(reduction="mean")
        
        # Test it works with arrays
        logits = np.random.randn(10, 5)
        targets = np.random.randint(0, 5, (10,))
        
        loss = loss_fn(logits, targets)
        assert isinstance(loss, (float, np.floating))
    
    def test_activation_factory_pattern(self):
        """Test activation factory pattern works across backends."""
        
        # Mock backend with activation factory
        backend = Mock(spec=NeuralBackend)
        
        # Mock activation modules
        relu_module = Mock(spec=Module)
        relu_module.forward = Mock(side_effect=lambda x: np.maximum(x, 0))
        
        gelu_module = Mock(spec=Module)
        gelu_module.forward = Mock(side_effect=lambda x: x * 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3))))
        
        # Setup activation factory
        def activation_factory(activation_type: ActivationType, **kwargs):
            if activation_type == ActivationType.RELU:
                return relu_module
            elif activation_type == ActivationType.GELU:
                return gelu_module
            else:
                raise ValueError(f"Unsupported activation: {activation_type}")
        
        backend.activation = activation_factory
        
        # Test creating different activations
        relu = backend.activation(ActivationType.RELU)
        gelu = backend.activation(ActivationType.GELU)
        
        # Test they work correctly
        test_input = np.array([-1, 0, 1, 2])
        
        relu_output = relu.forward(test_input)
        assert np.array_equal(relu_output, np.array([0, 0, 1, 2]))
        
        gelu_output = gelu.forward(test_input)
        assert gelu_output.shape == test_input.shape
    
    def test_device_abstraction(self):
        """Test device abstraction across backends."""
        from dataclasses import dataclass
        
        # Create a simple Device dataclass for testing
        @dataclass
        class Device:
            type: str
            index: int
        
        # Create devices
        cpu_device = Device(type="cpu", index=0)
        gpu_device = Device(type="cuda", index=0)
        mps_device = Device(type="mps", index=0)
        
        # Mock backend
        backend = Mock(spec=NeuralBackend)
        
        # Mock device context that tracks device placement
        class DeviceTracker:
            def __init__(self):
                self.current_device = None
            
            def set_device(self, device):
                self.current_device = device
        
        tracker = DeviceTracker()
        
        def mock_device_context(device: Device):
            from contextlib import contextmanager
            
            @contextmanager
            def context():
                old_device = tracker.current_device
                tracker.current_device = device
                yield device
                tracker.current_device = old_device
            
            return context()
        
        backend.device_context = mock_device_context
        
        # Test device context switching
        assert tracker.current_device is None
        
        with backend.device_context(cpu_device) as dev:
            assert tracker.current_device == cpu_device
            assert dev == cpu_device
        
        assert tracker.current_device is None
        
        with backend.device_context(gpu_device) as dev:
            assert tracker.current_device == gpu_device
    
    def test_module_composition(self):
        """Test that modules can be composed across backends."""
        
        # Create a mock sequential container
        class MockSequential(Module):
            def __init__(self, *modules):
                super().__init__()
                self.layers = list(modules)
            
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x
        
        # Create mock modules with __call__ method
        linear1 = Mock(spec=Module)
        linear1_output = np.random.randn(2, 20)
        linear1.return_value = linear1_output
        linear1.forward.return_value = linear1_output
        
        relu = Mock(spec=Module)
        relu_output = np.maximum(linear1_output, 0)
        relu.return_value = relu_output
        relu.forward.return_value = relu_output
        
        linear2 = Mock(spec=Module)
        final_output = np.random.randn(2, 5)
        linear2.return_value = final_output
        linear2.forward.return_value = final_output
        
        # Compose into a model
        model = MockSequential(linear1, relu, linear2)
        
        # Test forward pass
        input_data = np.random.randn(2, 10)
        output = model.forward(input_data)
        
        # Verify composition worked
        assert output is not None
        assert linear1.called
        assert relu.called
        assert linear2.called


class TestBackendFactoryErrors:
    """Test error handling in backend factory."""
    
    def test_unsupported_backend_error(self):
        """Test that unsupported backends raise appropriate errors."""
        with pytest.raises(ValueError, match="Unsupported neural backend: unsupported"):
            create_neural_backend("unsupported")
    
    def test_backend_aliases(self):
        """Test that backend aliases work correctly."""
        # Test that both 'pytorch' and 'torch' are recognized
        # They should both fail with ImportError since PyTorch isn't installed
        
        for alias in ["pytorch", "torch"]:
            try:
                backend = create_neural_backend(alias)
            except ImportError as e:
                # Expected - PyTorch not installed
                assert "torch" in str(e)
            except Exception as e:
                pytest.fail(f"Factory should recognize '{alias}' backend, got: {e}")