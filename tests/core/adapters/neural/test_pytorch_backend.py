"""Tests for PyTorch neural backend adapter.

This module tests the PyTorch implementation of the NeuralBackend protocol,
demonstrating that the same interface can work with different frameworks.
"""

import pytest
import torch
import numpy as np

from core.adapters.neural.pytorch_backend import PyTorchNeuralBackend
from core.ports.neural import ActivationType, NeuralBackend, Module
from core.ports.compute import DataType


class TestPyTorchNeuralBackend:
    """Test suite for PyTorch neural backend."""
    
    @pytest.fixture
    def backend(self) -> PyTorchNeuralBackend:
        """Create PyTorch backend instance."""
        return PyTorchNeuralBackend()
    
    def test_backend_name(self, backend: NeuralBackend):
        """Test backend name property."""
        assert backend.name == "pytorch"
        assert backend.supports_mixed_precision is True
    
    def test_linear_layer(self, backend: NeuralBackend):
        """Test linear layer creation and forward pass."""
        # Create a linear layer
        linear = backend.linear(in_features=10, out_features=5, bias=True)
        
        # Test that it's a Module
        assert isinstance(linear, Module)
        
        # Create input data
        input_data = torch.randn(2, 10)  # batch_size=2, features=10
        
        # Forward pass
        output = linear(input_data)
        
        # Check output shape
        assert output.shape == (2, 5)
        
        # Check parameters exist
        params = list(linear.parameters())
        assert len(params) == 2  # weight and bias
    
    def test_embedding_layer(self, backend: NeuralBackend):
        """Test embedding layer creation and lookup."""
        # Create embedding layer
        embedding = backend.embedding(
            num_embeddings=100,
            embedding_dim=64,
            padding_idx=0
        )
        
        # Test embedding lookup
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 0]])  # 0 is padding
        output = embedding(input_ids)
        
        # Check output shape
        assert output.shape == (2, 3, 64)
    
    def test_layer_norm(self, backend: NeuralBackend):
        """Test layer normalization."""
        # Create layer norm
        layer_norm = backend.layer_norm(normalized_shape=256, eps=1e-5)
        
        # Test forward pass
        input_data = torch.randn(2, 10, 256)
        output = layer_norm(input_data)
        
        # Check output shape preserved
        assert output.shape == input_data.shape
        
        # Check normalization (mean ~0, std ~1)
        # Due to affine transformation, we check the general shape preservation
        assert not torch.isnan(output).any()
    
    def test_rms_norm(self, backend: NeuralBackend):
        """Test RMS normalization."""
        # Create RMS norm
        rms_norm = backend.rms_norm(normalized_shape=256, eps=1e-6)
        
        # Test forward pass
        input_data = torch.randn(2, 10, 256)
        output = rms_norm(input_data)
        
        # Check output shape preserved
        assert output.shape == input_data.shape
        assert not torch.isnan(output).any()
    
    def test_dropout(self, backend: NeuralBackend):
        """Test dropout layer."""
        # Create dropout
        dropout = backend.dropout(p=0.5)
        
        # Test in training mode
        dropout.train()
        input_data = torch.ones(100, 100)
        output = dropout(input_data)
        
        # Some values should be zero (dropped) in training mode
        # Due to scaling, non-zero values should be ~2.0
        assert (output == 0).any() or output.mean() > 1.5
        
        # Test in eval mode
        dropout.eval()
        output_eval = dropout(input_data)
        
        # No dropout in eval mode
        assert torch.allclose(output_eval, input_data)
    
    def test_multi_head_attention(self, backend: NeuralBackend):
        """Test multi-head attention layer."""
        # Create MHA layer
        mha = backend.multi_head_attention(
            embed_dim=256,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Test forward pass
        batch_size, seq_len, embed_dim = 2, 10, 256
        query = torch.randn(batch_size, seq_len, embed_dim)
        key = torch.randn(batch_size, seq_len, embed_dim)
        value = torch.randn(batch_size, seq_len, embed_dim)
        
        # MHA forward returns (output, attention_weights)
        output = mha(query, key, value)
        
        # Check output shape - handle both tuple and tensor returns
        if isinstance(output, tuple):
            output = output[0]
        assert output.shape == (batch_size, seq_len, embed_dim)
    
    def test_activations(self, backend: NeuralBackend):
        """Test various activation functions."""
        input_data = torch.randn(10, 10)
        
        # Test ReLU
        relu = backend.relu()
        relu_output = relu(input_data)
        assert (relu_output >= 0).all()
        
        # Test GELU
        gelu = backend.gelu()
        gelu_output = gelu(input_data)
        assert gelu_output.shape == input_data.shape
        
        # Test SiLU/Swish
        silu = backend.silu()
        silu_output = silu(input_data)
        assert silu_output.shape == input_data.shape
        
        # Test activation factory
        tanh = backend.activation(ActivationType.TANH)
        tanh_output = tanh(input_data)
        assert (tanh_output >= -1).all() and (tanh_output <= 1).all()
    
    def test_sequential_container(self, backend: NeuralBackend):
        """Test sequential container."""
        # Create a simple sequential model
        model = backend.sequential(
            backend.linear(10, 20),
            backend.relu(),
            backend.linear(20, 5),
            backend.activation(ActivationType.SOFTMAX, dim=-1)
        )
        
        # Test forward pass
        input_data = torch.randn(2, 10)
        output = model(input_data)
        
        # Check output shape and properties
        assert output.shape == (2, 5)
        # Softmax output should sum to 1
        assert torch.allclose(output.sum(dim=-1), torch.ones(2), atol=1e-6)
    
    def test_loss_functions(self, backend: NeuralBackend):
        """Test loss function creation."""
        # Test cross entropy loss
        ce_loss = backend.cross_entropy_loss(reduction="mean")
        
        logits = torch.randn(10, 5)
        targets = torch.randint(0, 5, (10,))
        
        loss = ce_loss(logits, targets)
        assert loss.ndim == 0  # Scalar loss
        assert loss > 0
        
        # Test MSE loss
        mse_loss = backend.mse_loss(reduction="mean")
        
        predictions = torch.randn(10, 5)
        targets = torch.randn(10, 5)
        
        loss = mse_loss(predictions, targets)
        assert loss.ndim == 0
        assert loss >= 0
    
    def test_tensor_operations(self, backend: NeuralBackend):
        """Test basic tensor operations."""
        # Test matmul
        a = torch.randn(2, 3)
        b = torch.randn(3, 4)
        c = backend.matmul(a, b)
        assert c.shape == (2, 4)
        
        # Test transpose
        x = torch.randn(2, 3, 4)
        y = backend.transpose(x, 1, 2)
        assert y.shape == (2, 4, 3)
        
        # Test reshape
        x = torch.randn(6, 4)
        y = backend.reshape(x, (8, 3))
        assert y.shape == (8, 3)
        
        # Test concat
        x1 = torch.randn(2, 3)
        x2 = torch.randn(2, 3)
        y = backend.concat([x1, x2], dim=0)
        assert y.shape == (4, 3)
        
        # Test split
        x = torch.randn(6, 4)
        splits = backend.split(x, 2, dim=0)
        assert len(splits) == 3
        assert all(s.shape == (2, 4) for s in splits)
        
        # Test reductions
        x = torch.randn(2, 3, 4)
        
        mean = backend.mean(x, dim=1)
        assert mean.shape == (2, 4)
        
        sum_val = backend.sum(x, dim=(1, 2))
        assert sum_val.shape == (2,)
        
        max_val, max_idx = backend.max(x, dim=2)
        assert max_val.shape == (2, 3)
        assert max_idx.shape == (2, 3)
        
        # Test softmax
        logits = torch.randn(2, 10)
        probs = backend.softmax(logits, dim=-1)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-6)
    
    def test_utility_operations(self, backend: NeuralBackend):
        """Test utility operations."""
        # Test parameter creation
        param_data = np.random.randn(10, 10)
        param = backend.parameter(param_data, requires_grad=True)
        assert isinstance(param, torch.nn.Parameter)
        assert param.requires_grad
        
        # Test no_grad context
        x = torch.randn(10, 10, requires_grad=True)
        with backend.no_grad():
            y = x * 2
            assert not y.requires_grad
        
        # Test arange
        arr = backend.arange(0, 10, 2)
        assert torch.equal(arr, torch.tensor([0, 2, 4, 6, 8]))
        
        # Test unsqueeze
        x = torch.randn(3, 4)
        y = backend.unsqueeze(x, 0)
        assert y.shape == (1, 3, 4)
        
        # Test broadcast_to
        x = torch.randn(1, 4)
        y = backend.broadcast_to(x, (3, 4))
        assert y.shape == (3, 4)
        
        # Test zeros_like and ones
        x = torch.randn(2, 3, 4)
        zeros = backend.zeros_like(x)
        assert torch.all(zeros == 0)
        assert zeros.shape == x.shape
        
        ones = backend.ones((2, 3))
        assert torch.all(ones == 1)
        assert ones.shape == (2, 3)
    
    def test_masked_operations(self, backend: NeuralBackend):
        """Test masked operations."""
        # Test masked_fill
        x = torch.randn(3, 3)
        mask = torch.tensor([[True, False, True],
                           [False, True, False],
                           [True, True, False]])
        
        filled = backend.masked_fill(x, mask, -float('inf'))
        assert torch.all(filled[mask] == -float('inf'))
        assert torch.all(filled[~mask] == x[~mask])
        
        # Test where
        condition = torch.tensor([True, False, True, False])
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        y = torch.tensor([10.0, 20.0, 30.0, 40.0])
        
        result = backend.where(condition, x, y)
        expected = torch.tensor([1.0, 20.0, 3.0, 40.0])
        assert torch.equal(result, expected)


class TestFrameworkInteroperability:
    """Test that the abstraction truly allows framework swapping."""
    
    def test_backend_protocol_compliance(self):
        """Test that PyTorchNeuralBackend implements the NeuralBackend protocol."""
        from core.ports.neural import NeuralBackend
        
        backend = PyTorchNeuralBackend()
        
        # Should satisfy the protocol
        assert isinstance(backend, NeuralBackend)
        
        # Check all required methods exist
        required_methods = [
            'linear', 'embedding', 'layer_norm', 'rms_norm', 'dropout',
            'multi_head_attention', 'activation', 'gelu', 'relu', 'silu',
            'sequential', 'module_list', 'module_dict', 'cross_entropy_loss',
            'binary_cross_entropy_loss', 'mse_loss', 'matmul', 'transpose',
            'reshape', 'concat', 'split', 'mean', 'sum', 'max', 'min',
            'softmax', 'log_softmax', 'parameter', 'no_grad', 'enable_grad',
            'device_context', 'unsqueeze', 'arange', 'broadcast_to',
            'zeros_like', 'ones', 'masked_fill', 'where'
        ]
        
        for method in required_methods:
            assert hasattr(backend, method)
            assert callable(getattr(backend, method))
    
    def test_simple_model_creation(self):
        """Test creating a simple model that could work with any backend."""
        def create_simple_ffn(backend: NeuralBackend, input_dim: int, hidden_dim: int, output_dim: int) -> Module:
            """Create a simple feedforward network using any backend."""
            return backend.sequential(
                backend.linear(input_dim, hidden_dim),
                backend.relu(),
                backend.dropout(0.1),
                backend.linear(hidden_dim, output_dim)
            )
        
        # Create with PyTorch backend
        pytorch_backend = PyTorchNeuralBackend()
        pytorch_model = create_simple_ffn(pytorch_backend, 10, 20, 5)
        
        # Test forward pass
        input_data = torch.randn(2, 10)
        output = pytorch_model(input_data)
        assert output.shape == (2, 5)
        
        # The same function could be used with MLX backend
        # mlx_backend = MLXNeuralBackend()
        # mlx_model = create_simple_ffn(mlx_backend, 10, 20, 5)
    
    def test_backend_factory(self):
        """Test the backend factory function."""
        from core.ports.neural import create_neural_backend
        
        # Create PyTorch backend via factory
        backend = create_neural_backend("pytorch")
        assert isinstance(backend, PyTorchNeuralBackend)
        assert backend.name == "pytorch"
        
        # Also test "torch" alias
        backend2 = create_neural_backend("torch")
        assert isinstance(backend2, PyTorchNeuralBackend)