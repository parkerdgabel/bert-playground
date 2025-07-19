"""
Pytest configuration and shared fixtures for model module tests.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import mlx.core as mx
import mlx.nn as nn

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.bert.config import BertConfig
from models.bert.modernbert_config import ModernBertConfig
from models.heads.config import ClassificationConfig, RegressionConfig
from models.lora.config import LoRAConfig


# Test configuration
@pytest.fixture(scope="session")
def test_config():
    """Shared test configuration."""
    return {
        "seed": 42,
        "batch_size": 4,
        "sequence_length": 128,
        "vocab_size": 30522,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
    }


# Temporary directories
@pytest.fixture
def tmp_model_dir():
    """Create temporary model directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def tmp_checkpoint_dir():
    """Create temporary checkpoint directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "checkpoints"


# Model configuration fixtures
@pytest.fixture
def bert_config():
    """Create standard BERT configuration."""
    return BertConfig(
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
    )


@pytest.fixture
def small_bert_config():
    """Create small BERT configuration for fast tests."""
    return BertConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=256,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=128,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
    )


@pytest.fixture
def modernbert_config():
    """Create ModernBERT configuration."""
    return ModernBertConfig(
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,  # ModernBERT uses different dropout
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=8192,  # Longer context
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        rope_theta=10000.0,  # RoPE specific
        attention_bias=False,
    )


@pytest.fixture
def classification_config():
    """Create classification head configuration."""
    return ClassificationConfig(
        input_size=768,
        output_size=2,
        num_classes=2,
        dropout_prob=0.1,
        pooling_type="cls",
        activation="tanh",
    )


@pytest.fixture
def regression_config():
    """Create regression head configuration."""
    return RegressionConfig(
        input_size=768,
        output_size=1,
        dropout_prob=0.1,
        pooling_type="mean",
        activation="tanh",
    )


@pytest.fixture
def lora_config():
    """Create LoRA configuration."""
    return LoRAConfig(
        rank=8,
        alpha=16,
        dropout=0.1,
        target_modules=["query", "value"],
        modules_to_save=["classifier"],
    )


# Mock model implementations
class MockBertModel(nn.Module):
    """Mock BERT model for testing."""
    
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.encoder = nn.Sequential(
            *[nn.Linear(config.hidden_size, config.hidden_size) 
              for _ in range(config.num_hidden_layers)]
        )
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
    
    def __call__(
        self, 
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        token_type_ids: Optional[mx.array] = None,
    ) -> Dict[str, mx.array]:
        """Forward pass."""
        # Get embeddings
        hidden_states = self.embeddings(input_ids)
        
        # Pass through encoder
        for layer in self.encoder:
            hidden_states = layer(hidden_states)
        
        # Pool
        pooled = self.pooler(hidden_states[:, 0])
        
        return {
            "last_hidden_state": hidden_states,
            "pooler_output": pooled,
        }


class MockClassificationHead(nn.Module):
    """Mock classification head for testing."""
    
    def __init__(self, config: ClassificationConfig):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.input_size, config.input_size)
        self.dropout = nn.Dropout(config.dropout_prob)
        self.classifier = nn.Linear(config.input_size, config.num_classes)
    
    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Forward pass."""
        # Pool based on config
        if self.config.pooling_type == "cls":
            pooled = hidden_states[:, 0]
        else:
            pooled = mx.mean(hidden_states, axis=1)
        
        # Classification layers
        pooled = self.dense(pooled)
        pooled = nn.tanh(pooled)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        
        return logits


# Import data generation functions
from tests.models.fixtures.data import (
    create_embeddings as _create_embeddings,
    create_attention_mask as _create_attention_mask,
    create_position_ids as _create_position_ids,
)

# Data generation fixtures
@pytest.fixture
def create_embeddings():
    """Create embeddings for testing."""
    return _create_embeddings

@pytest.fixture  
def create_position_ids():
    """Create position IDs for testing."""
    return _create_position_ids

@pytest.fixture
def create_test_batch():
    """Create test batch for model testing."""
    def _create(
        config: BertConfig,
        batch_size: int = 4,
        seq_length: int = 128,
    ) -> Dict[str, mx.array]:
        """Generate test batch."""
        input_ids = mx.random.randint(
            0, config.vocab_size, (batch_size, seq_length)
        )
        attention_mask = mx.ones((batch_size, seq_length))
        token_type_ids = mx.zeros((batch_size, seq_length), dtype=mx.int32)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
    return _create


@pytest.fixture
def create_random_inputs():
    """Create random inputs for testing."""
    def _create(
        batch_size: int = 4,
        seq_length: int = 128,
        vocab_size: int = 30522,
    ) -> Dict[str, mx.array]:
        """Generate random inputs."""
        return {
            "input_ids": mx.random.randint(0, vocab_size, (batch_size, seq_length)),
            "attention_mask": mx.ones((batch_size, seq_length)),
            "token_type_ids": mx.zeros((batch_size, seq_length), dtype=mx.int32),
            "labels": mx.random.randint(0, 2, (batch_size,)),
        }
    return _create


@pytest.fixture
def create_attention_mask():
    """Create attention mask for testing."""
    return _create_attention_mask


# Model comparison utilities
@pytest.fixture
def assert_models_equal():
    """Utility for comparing two models."""
    def _assert(model1: nn.Module, model2: nn.Module, rtol: float = 1e-5):
        """Assert two models have equal parameters."""
        params1 = model1.parameters()
        params2 = model2.parameters()
        
        # Flatten parameters
        flat1 = mx.tree_flatten(params1)
        flat2 = mx.tree_flatten(params2)
        
        assert len(flat1) == len(flat2), "Models have different number of parameters"
        
        for (k1, v1), (k2, v2) in zip(flat1, flat2):
            assert k1 == k2, f"Parameter names don't match: {k1} vs {k2}"
            assert mx.allclose(v1, v2, rtol=rtol), f"Parameter {k1} values don't match"
    
    return _assert


@pytest.fixture
def assert_outputs_close():
    """Utility for comparing model outputs."""
    def _assert(
        outputs1: Dict[str, mx.array], 
        outputs2: Dict[str, mx.array],
        rtol: float = 1e-5,
    ):
        """Assert two output dicts are close."""
        assert set(outputs1.keys()) == set(outputs2.keys()), "Output keys don't match"
        
        for key in outputs1:
            assert mx.allclose(outputs1[key], outputs2[key], rtol=rtol), \
                f"Output {key} values don't match"
    
    return _assert


# Gradient checking utilities
@pytest.fixture
def check_gradients():
    """Utility for checking gradient flow."""
    def _check(model: nn.Module, loss_fn, batch: Dict[str, mx.array]) -> bool:
        """Check if gradients flow through model."""
        # Compute loss and gradients
        loss, grads = mx.value_and_grad(loss_fn)(model, batch)
        
        # Check loss is valid
        assert mx.isfinite(loss), "Loss is not finite"
        
        # Check gradients exist and are finite
        flat_grads = mx.tree_flatten(grads)
        assert len(flat_grads) > 0, "No gradients computed"
        
        for name, grad in flat_grads:
            assert grad is not None, f"Gradient for {name} is None"
            assert mx.all(mx.isfinite(grad)), f"Gradient for {name} contains non-finite values"
        
        return True
    
    return _check


# Memory profiling utilities
@pytest.fixture
def memory_profiler():
    """Utility for profiling memory usage."""
    class MemoryProfiler:
        def __init__(self):
            self.initial_memory = None
            self.peak_memory = None
        
        def __enter__(self):
            mx.metal.clear_cache()
            self.initial_memory = mx.metal.get_active_memory()
            return self
        
        def __exit__(self, *args):
            self.peak_memory = mx.metal.get_peak_memory()
            mx.metal.clear_cache()
        
        def get_memory_used(self) -> int:
            """Get memory used during profiling."""
            if self.peak_memory is None or self.initial_memory is None:
                return 0
            return self.peak_memory - self.initial_memory
    
    return MemoryProfiler


# Performance benchmarking utilities
@pytest.fixture
def benchmark_model():
    """Utility for benchmarking model performance."""
    def _benchmark(
        model: nn.Module,
        batch: Dict[str, mx.array],
        num_iterations: int = 10,
        warmup: int = 3,
    ) -> Dict[str, float]:
        """Benchmark model forward pass."""
        import time
        
        # Warmup
        for _ in range(warmup):
            _ = model(**batch)
            mx.eval(model.parameters())
        
        # Time iterations
        times = []
        for _ in range(num_iterations):
            start = time.time()
            _ = model(**batch)
            mx.eval(model.parameters())
            times.append(time.time() - start)
        
        return {
            "mean_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
        }
    
    return _benchmark


# Error models for testing
class BrokenModel(nn.Module):
    """Model that raises errors for testing error handling."""
    
    def __init__(self, error_type: str = "forward"):
        super().__init__()
        self.error_type = error_type
        self.linear = nn.Linear(10, 10)
    
    def __call__(self, *args, **kwargs):
        if self.error_type == "forward":
            raise RuntimeError("Forward pass failed")
        return self.linear(mx.zeros((1, 10)))
    
    def save(self, path: Path):
        if self.error_type == "save":
            raise IOError("Save failed")
    
    def load(self, path: Path):
        if self.error_type == "load":
            raise IOError("Load failed")


class NaNModel(nn.Module):
    """Model that produces NaN values for testing."""
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
    
    def __call__(self, x: mx.array) -> mx.array:
        # Force NaN by dividing by zero
        return self.linear(x) / 0.0


# Model factory fixtures
@pytest.fixture
def mock_bert_model():
    """Create mock BERT model."""
    config = BertConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
    )
    return MockBertModel(config)


@pytest.fixture
def mock_classification_head():
    """Create mock classification head."""
    config = ClassificationConfig(
        input_size=128,
        output_size=2,
        num_classes=2,
        dropout_prob=0.1,
    )
    return MockClassificationHead(config)


@pytest.fixture
def broken_model():
    """Create broken model for error testing."""
    return BrokenModel()


@pytest.fixture
def nan_model():
    """Create model that produces NaN."""
    return NaNModel()


# Test markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "mlx: MLX-specific tests")
    config.addinivalue_line("markers", "quantization: Quantization tests")
    config.addinivalue_line("markers", "memory: Memory-intensive tests")
    config.addinivalue_line("markers", "benchmark: Performance benchmark tests")


# Pytest plugins
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        
        # Add MLX marker to all tests
        item.add_marker(pytest.mark.mlx)