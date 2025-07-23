"""Tests for the MLX neural backend adapter."""

import pytest
import mlx.core as mx
import numpy as np

from infrastructure.adapters.neural import MLXNeuralBackend
from infrastructure.ports.neural import ActivationType, create_neural_backend


class TestMLXNeuralBackend:
    """Test suite for MLX neural backend implementation."""
    
    @pytest.fixture
    def backend(self):
        """Create MLX backend instance."""
        return MLXNeuralBackend()
    
    def test_backend_properties(self, backend):
        """Test backend properties."""
        assert backend.name == "mlx"
        assert backend.supports_mixed_precision is True
    
    def test_linear_layer(self, backend):
        """Test linear layer creation."""
        linear = backend.linear(10, 20, bias=True)
        
        # Test forward pass
        x = mx.random.normal((2, 10))
        output = linear(x)
        assert output.shape == (2, 20)
        
        # Check parameters
        params = list(linear.parameters())
        assert len(params) == 2  # weight and bias
    
    def test_embedding_layer(self, backend):
        """Test embedding layer."""
        embedding = backend.embedding(100, 32)
        
        # Test forward pass
        indices = mx.array([[1, 2, 3], [4, 5, 6]])
        output = embedding(indices)
        assert output.shape == (2, 3, 32)
    
    def test_layer_norm(self, backend):
        """Test layer normalization."""
        layer_norm = backend.layer_norm(64, eps=1e-5)
        
        # Test forward pass
        x = mx.random.normal((2, 10, 64))
        output = layer_norm(x)
        assert output.shape == x.shape
    
    def test_rms_norm(self, backend):
        """Test RMS normalization."""
        rms_norm = backend.rms_norm(64, eps=1e-6)
        
        # Test forward pass
        x = mx.random.normal((2, 10, 64))
        output = rms_norm(x)
        assert output.shape == x.shape
    
    def test_dropout(self, backend):
        """Test dropout layer."""
        dropout = backend.dropout(p=0.5)
        
        # Test forward pass
        x = mx.random.normal((2, 10, 64))
        output = dropout(x)
        assert output.shape == x.shape
    
    def test_multi_head_attention(self, backend):
        """Test multi-head attention."""
        mha = backend.multi_head_attention(
            embed_dim=64,
            num_heads=8,
            dropout=0.1
        )
        
        # Test forward pass
        x = mx.random.normal((2, 10, 64))
        output = mha(x)
        assert output.shape == x.shape
    
    def test_activations(self, backend):
        """Test various activation functions."""
        x = mx.random.normal((2, 10))
        
        # Test RELU
        relu = backend.relu()
        output = relu(x)
        assert output.shape == x.shape
        
        # Test GELU
        gelu = backend.gelu()
        output = gelu(x)
        assert output.shape == x.shape
        
        # Test SiLU
        silu = backend.silu()
        output = silu(x)
        assert output.shape == x.shape
        
        # Test activation factory
        tanh = backend.activation(ActivationType.TANH)
        output = tanh(x)
        assert output.shape == x.shape
    
    def test_sequential_container(self, backend):
        """Test sequential container."""
        seq = backend.sequential(
            backend.linear(10, 20),
            backend.relu(),
            backend.linear(20, 30)
        )
        
        x = mx.random.normal((2, 10))
        output = seq(x)
        assert output.shape == (2, 30)
    
    def test_loss_functions(self, backend):
        """Test loss function creation."""
        # Cross entropy loss
        ce_loss = backend.cross_entropy_loss(reduction="mean")
        logits = mx.random.normal((10, 5))
        targets = mx.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        loss = ce_loss(logits, targets)
        assert loss.shape == ()
        
        # MSE loss
        mse_loss = backend.mse_loss(reduction="mean")
        predictions = mx.random.normal((10, 5))
        targets = mx.random.normal((10, 5))
        loss = mse_loss(predictions, targets)
        assert loss.shape == ()
    
    def test_tensor_operations(self, backend):
        """Test various tensor operations."""
        # Test matmul
        a = mx.random.normal((2, 3, 4))
        b = mx.random.normal((2, 4, 5))
        result = backend.matmul(a, b)
        assert result.shape == (2, 3, 5)
        
        # Test transpose
        x = mx.random.normal((2, 3, 4, 5))
        result = backend.transpose(x, 1, 2)
        assert result.shape == (2, 4, 3, 5)
        
        # Test reshape
        x = mx.random.normal((2, 12))
        result = backend.reshape(x, (2, 3, 4))
        assert result.shape == (2, 3, 4)
        
        # Test concat
        arrays = [mx.random.normal((2, 3)) for _ in range(3)]
        result = backend.concat(arrays, dim=0)
        assert result.shape == (6, 3)
        
        # Test split
        x = mx.random.normal((10, 5))
        splits = backend.split(x, 2, dim=0)
        assert len(splits) == 5
        assert all(s.shape == (2, 5) for s in splits)
    
    def test_reduction_operations(self, backend):
        """Test reduction operations."""
        x = mx.random.normal((2, 3, 4))
        
        # Test mean
        result = backend.mean(x, dim=1)
        assert result.shape == (2, 4)
        
        result = backend.mean(x, dim=(1, 2))
        assert result.shape == (2,)
        
        # Test sum
        result = backend.sum(x, dim=1, keepdim=True)
        assert result.shape == (2, 1, 4)
        
        # Test max/min
        result, indices = backend.max(x, dim=1)
        assert result.shape == (2, 4)
        assert indices.shape == (2, 4)
    
    def test_softmax_operations(self, backend):
        """Test softmax operations."""
        x = mx.random.normal((2, 3, 4))
        
        # Test softmax
        result = backend.softmax(x, dim=-1)
        assert result.shape == x.shape
        # Check that softmax sums to 1
        assert mx.allclose(mx.sum(result, axis=-1), mx.ones((2, 3)), atol=1e-6)
        
        # Test log_softmax
        result = backend.log_softmax(x, dim=-1)
        assert result.shape == x.shape
    
    def test_utility_operations(self, backend):
        """Test utility operations."""
        # Test parameter creation
        data = mx.random.normal((10, 20))
        param = backend.parameter(data)
        assert isinstance(param, mx.array)
        assert param.shape == (10, 20)
        
        # Test arange
        result = backend.arange(0, 10, 2)
        assert result.shape == (5,)
        assert mx.array_equal(result, mx.array([0, 2, 4, 6, 8]))
        
        # Test unsqueeze
        x = mx.random.normal((2, 3))
        result = backend.unsqueeze(x, 1)
        assert result.shape == (2, 1, 3)
        
        # Test broadcast_to
        x = mx.random.normal((1, 3))
        result = backend.broadcast_to(x, (5, 3))
        assert result.shape == (5, 3)
        
        # Test zeros_like
        x = mx.random.normal((2, 3, 4))
        result = backend.zeros_like(x)
        assert result.shape == x.shape
        assert mx.all(result == 0)
        
        # Test ones
        result = backend.ones((2, 3, 4))
        assert result.shape == (2, 3, 4)
        assert mx.all(result == 1)
    
    def test_advanced_operations(self, backend):
        """Test advanced operations for BERT models."""
        # Test masked_fill
        x = mx.random.normal((2, 3, 4))
        mask = mx.array([[[True, False, True, False],
                         [False, True, False, True],
                         [True, True, False, False]]] * 2)
        result = backend.masked_fill(x, mask, -float('inf'))
        assert result.shape == x.shape
        # MLX doesn't support boolean indexing, so check differently
        masked_vals = mx.where(mask, result, mx.zeros_like(result))
        assert mx.all(mx.where(mask, masked_vals == -float('inf'), True))
        
        # Test where
        condition = mx.array([[True, False], [False, True]])
        x = mx.ones((2, 2))
        y = mx.zeros((2, 2))
        result = backend.where(condition, x, y)
        assert mx.array_equal(result, mx.array([[1, 0], [0, 1]]))
    
    def test_rotary_embeddings(self, backend):
        """Test rotary position embeddings."""
        rope = backend.rotary_embedding(
            dim=64,
            max_position_embeddings=512,
            base=10000.0
        )
        
        # Test forward pass
        x = mx.random.normal((2, 10, 8, 64))
        cos, sin = rope(x)
        assert cos.shape[0] >= 10
        assert sin.shape[0] >= 10
        
        # Test apply_rotary_pos_emb
        q = mx.random.normal((2, 8, 10, 64))
        k = mx.random.normal((2, 8, 10, 64))
        q_rot, k_rot = backend.apply_rotary_pos_emb(q, k, cos, sin)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
    
    def test_context_managers(self, backend):
        """Test context managers."""
        # Test no_grad context
        with backend.no_grad():
            x = mx.random.normal((2, 3))
            y = x * 2
            assert y.shape == (2, 3)
        
        # Test enable_grad context
        with backend.enable_grad():
            x = mx.random.normal((2, 3))
            y = x * 2
            assert y.shape == (2, 3)
        
        # Test device context
        with backend.device_context("cpu"):
            x = mx.random.normal((2, 3))
            assert x.shape == (2, 3)
    
    def test_create_neural_backend(self):
        """Test factory function."""
        backend = create_neural_backend("mlx")
        assert isinstance(backend, MLXNeuralBackend)
        assert backend.name == "mlx"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])