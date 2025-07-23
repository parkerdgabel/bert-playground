"""Tests for the neural network abstraction port."""

import pytest
from typing import Any

from core.ports.neural import (
    Module,
    NeuralBackend,
    ActivationType,
    NormalizationType,
    LossType,
)
from core.ports.neural import ModuleInfo
from core.ports.neural_types import (
    AttentionConfig,
    FeedForwardConfig,
    EmbeddingConfig,
    InitializationType,
)


class MockModule(Module):
    """Mock module for testing."""
    
    def __init__(self, name: str = "test"):
        super().__init__()
        self._name = name
        self.weight = "mock_weight"
        self.register_parameter("weight", self.weight)
    
    def forward(self, x):
        return x


class TestModule:
    """Test the base Module class."""
    
    def test_module_initialization(self):
        """Test module initialization."""
        module = MockModule()
        assert module.training is True
        assert module._name == "test"
        assert "weight" in module._parameters
    
    def test_module_train_eval_mode(self):
        """Test switching between train and eval mode."""
        module = MockModule()
        
        # Default is training mode
        assert module.training is True
        
        # Switch to eval mode
        module.eval()
        assert module.training is False
        
        # Switch back to training mode
        module.train()
        assert module.training is True
    
    def test_module_parameters(self):
        """Test parameter iteration."""
        module = MockModule()
        params = list(module.parameters())
        assert len(params) == 1
        assert params[0] == "mock_weight"
    
    def test_module_named_parameters(self):
        """Test named parameter iteration."""
        module = MockModule()
        named_params = dict(module.named_parameters())
        assert "weight" in named_params
        assert named_params["weight"] == "mock_weight"
    
    def test_module_add_module(self):
        """Test adding submodules."""
        parent = MockModule("parent")
        child = MockModule("child")
        
        parent.add_module("child", child)
        
        assert "child" in parent._modules
        assert parent._modules["child"] is child
    
    def test_module_get_info(self):
        """Test getting module info."""
        module = MockModule()
        info = module.get_info()
        
        assert isinstance(info, ModuleInfo)
        assert info.name == "test"
        assert info.module_type == "MockModule"
        assert info.trainable_params == 1
        assert info.total_params == 1


class TestNeuralTypes:
    """Test neural type definitions."""
    
    def test_activation_type_enum(self):
        """Test ActivationType enum."""
        assert ActivationType.RELU.value == "relu"
        assert ActivationType.GELU.value == "gelu"
        assert ActivationType.SILU.value == "silu"
    
    def test_normalization_type_enum(self):
        """Test NormalizationType enum."""
        assert NormalizationType.LAYER_NORM.value == "layer_norm"
        assert NormalizationType.RMS_NORM.value == "rms_norm"
    
    def test_loss_type_enum(self):
        """Test LossType enum."""
        assert LossType.CROSS_ENTROPY.value == "cross_entropy"
        assert LossType.MSE.value == "mse"
    
    def test_attention_config(self):
        """Test AttentionConfig creation."""
        config = AttentionConfig(
            hidden_size=768,
            num_attention_heads=12,
            attention_dropout=0.1
        )
        
        assert config.hidden_size == 768
        assert config.num_attention_heads == 12
        assert config.attention_dropout == 0.1
        assert config.use_bias is True  # default
        assert config.use_rope is False  # default
    
    def test_feedforward_config(self):
        """Test FeedForwardConfig creation."""
        config = FeedForwardConfig(
            hidden_size=768,
            intermediate_size=3072,
            activation="gelu",
            dropout=0.1
        )
        
        assert config.hidden_size == 768
        assert config.intermediate_size == 3072
        assert config.activation == "gelu"
        assert config.dropout == 0.1
        assert config.use_gate is False  # default
    
    def test_embedding_config(self):
        """Test EmbeddingConfig creation."""
        config = EmbeddingConfig(
            vocab_size=30522,
            embedding_dim=768,
            max_position_embeddings=512
        )
        
        assert config.vocab_size == 30522
        assert config.embedding_dim == 768
        assert config.max_position_embeddings == 512
        assert config.use_positional is True  # default
        assert config.use_token_type is True  # default
    
    def test_initialization_type_enum(self):
        """Test InitializationType enum."""
        assert InitializationType.XAVIER_UNIFORM.value == "xavier_uniform"
        assert InitializationType.KAIMING_NORMAL.value == "kaiming_normal"
        assert InitializationType.NORMAL.value == "normal"


class TestNeuralBackendProtocol:
    """Test that NeuralBackend protocol is well-defined."""
    
    def test_protocol_methods_exist(self):
        """Test that all required methods are defined in the protocol."""
        # Get all methods that should be in the protocol
        required_methods = [
            "name",
            "supports_mixed_precision",
            "linear",
            "embedding",
            "layer_norm",
            "rms_norm",
            "dropout",
            "multi_head_attention",
            "activation",
            "gelu",
            "relu",
            "silu",
            "sequential",
            "module_list",
            "module_dict",
            "cross_entropy_loss",
            "binary_cross_entropy_loss",
            "mse_loss",
            "matmul",
            "transpose",
            "reshape",
            "concat",
            "split",
            "mean",
            "sum",
            "max",
            "min",
            "softmax",
            "log_softmax",
            "rotary_embedding",
            "apply_rotary_pos_emb",
            "masked_fill",
            "where",
            "parameter",
            "no_grad",
            "enable_grad",
            "device_context",
            "unsqueeze",
            "arange",
            "broadcast_to",
            "zeros_like",
            "ones",
        ]
        
        # Check that NeuralBackend has all required attributes
        for method in required_methods:
            assert hasattr(NeuralBackend, method), f"NeuralBackend missing method: {method}"