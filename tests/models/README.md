# Model Module Test Suite

Comprehensive test suite for all model implementations including BERT, ModernBERT, classification heads, LoRA adapters, and model utilities.

## Test Structure

```
tests/models/
├── unit/                    # Unit tests for individual components
│   ├── bert/               # BERT model tests
│   │   ├── test_config.py
│   │   ├── test_core.py
│   │   ├── layers/
│   │   │   ├── test_activations.py
│   │   │   ├── test_attention.py
│   │   │   ├── test_embeddings.py
│   │   │   └── test_feedforward.py
│   │   └── test_model.py
│   ├── heads/              # Classification/regression head tests
│   │   ├── test_base.py
│   │   ├── test_classification.py
│   │   ├── test_regression.py
│   │   ├── layers/
│   │   │   └── test_pooling.py
│   │   └── utils/
│   │       ├── test_losses.py
│   │       └── test_metrics.py
│   ├── lora/               # LoRA adapter tests
│   │   ├── test_adapter.py
│   │   ├── test_config.py
│   │   └── test_layers.py
│   ├── test_factory.py     # Model factory tests
│   └── test_quantization.py # Quantization utilities
├── integration/            # Integration tests
│   ├── test_model_data_integration.py
│   ├── test_model_training_integration.py
│   └── test_model_composition.py
├── e2e/                    # End-to-end tests
│   ├── test_complete_model_workflow.py
│   └── test_model_export_import.py
├── fixtures/               # Shared test fixtures
│   ├── configs.py         # Model configurations
│   ├── models.py          # Mock and test models
│   ├── data.py           # Test data for models
│   └── utils.py          # Testing utilities
├── conftest.py            # Pytest configuration
└── README.md              # This file
```

## Running Tests

### Run All Tests
```bash
# Run all tests with coverage
pytest tests/models/ -v --cov=models --cov-report=html

# Run with markers
pytest tests/models/ -m unit      # Unit tests only
pytest tests/models/ -m integration # Integration tests only
pytest tests/models/ -m e2e        # End-to-end tests only
```

### Run Specific Test Categories
```bash
# BERT model tests
pytest tests/models/unit/bert/ -v

# Classification head tests
pytest tests/models/unit/heads/ -v

# LoRA adapter tests
pytest tests/models/unit/lora/ -v

# Layer-specific tests
pytest tests/models/unit/bert/layers/ -v
```

### Run Individual Test Files
```bash
# Test BERT configuration
pytest tests/models/unit/bert/test_config.py -v

# Test attention mechanisms
pytest tests/models/unit/bert/layers/test_attention.py -v

# Test model factory
pytest tests/models/unit/test_factory.py -v
```

### Run with Specific Markers
```bash
# Fast tests only (exclude slow tests)
pytest tests/models/ -v -m "not slow"

# MLX-specific tests
pytest tests/models/ -v -m mlx

# Quantization tests
pytest tests/models/ -v -m quantization
```

## Test Coverage

### Current Coverage Goals
- Unit tests: 90% coverage
- Integration tests: 80% coverage
- E2E tests: Core workflows covered
- Total: 85%+ coverage

### View Coverage Report
```bash
# Generate HTML coverage report
pytest tests/models/ --cov=models --cov-report=html

# Open report
open htmlcov/index.html
```

## Test Fixtures

### Configurations (`fixtures/configs.py`)
- `create_bert_config()`: Standard BERT configuration
- `create_modernbert_config()`: ModernBERT configuration
- `create_small_bert_config()`: Small BERT for fast tests
- `create_classification_config()`: Classification head config
- `create_lora_config()`: LoRA adapter configuration

### Models (`fixtures/models.py`)
- `MockBertModel`: Simple BERT implementation for testing
- `MockClassificationHead`: Mock classification head
- `MockLoRAAdapter`: Mock LoRA adapter
- `BrokenModel`: Model that raises errors (for error testing)
- `NaNModel`: Model that produces NaN values
- `create_test_model()`: Factory for test models

### Data (`fixtures/data.py`)
- `create_test_embeddings()`: Generate test embeddings
- `create_attention_mask()`: Generate attention masks
- `create_test_batch()`: Create test batches
- `create_random_inputs()`: Random input generation

### Utilities (`fixtures/utils.py`)
- Model comparison utilities
- Weight initialization helpers
- Gradient checking utilities
- Memory profiling helpers
- Performance benchmarking tools

## Writing New Tests

### Unit Test Template
```python
import pytest
import mlx.core as mx
from models.bert import BertCore
from tests.models.fixtures import create_bert_config, create_test_batch

class TestBertCore:
    """Test BertCore functionality."""
    
    def test_initialization(self):
        """Test model initialization."""
        config = create_bert_config()
        model = BertCore(config)
        
        assert model.config == config
        assert len(model.encoder.layers) == config.num_hidden_layers
    
    def test_forward_pass(self):
        """Test forward pass."""
        config = create_bert_config()
        model = BertCore(config)
        batch = create_test_batch(config)
        
        outputs = model(batch["input_ids"], batch["attention_mask"])
        
        assert outputs.shape == (batch["input_ids"].shape[0], 
                               batch["input_ids"].shape[1], 
                               config.hidden_size)
    
    def test_gradient_flow(self):
        """Test gradient flow through model."""
        config = create_bert_config()
        model = BertCore(config)
        
        # Test gradient computation
        def loss_fn(model, batch):
            outputs = model(batch["input_ids"], batch["attention_mask"])
            return mx.mean(outputs)
        
        batch = create_test_batch(config)
        loss, grads = mx.value_and_grad(loss_fn)(model, batch)
        
        assert loss.item() is not None
        assert all(g is not None for g in mx.tree_flatten(grads))
```

### Integration Test Template
```python
class TestModelDataIntegration:
    """Test model integration with data pipelines."""
    
    def test_model_with_dataloader(self, tmp_path):
        """Test model works with data loader."""
        from data.loaders import MLXDataLoader
        
        config = create_bert_config()
        model = BertCore(config)
        loader = create_test_dataloader()
        
        for batch in loader:
            outputs = model(batch["input_ids"], batch["attention_mask"])
            assert outputs.shape[0] == batch["input_ids"].shape[0]
```

### E2E Test Template
```python
class TestCompleteModelWorkflow:
    """Test complete model workflows."""
    
    def test_model_save_load_inference(self, tmp_path):
        """Test complete save/load/inference workflow."""
        # Create and train model
        config = create_bert_config()
        model = BertCore(config)
        
        # Save model
        save_path = tmp_path / "model"
        model.save(save_path)
        
        # Load model
        loaded_model = BertCore.load(save_path)
        
        # Test inference
        batch = create_test_batch(config)
        original_outputs = model(batch["input_ids"], batch["attention_mask"])
        loaded_outputs = loaded_model(batch["input_ids"], batch["attention_mask"])
        
        assert mx.allclose(original_outputs, loaded_outputs, atol=1e-6)
```

## Model-Specific Testing Guidelines

### BERT/ModernBERT Models
- Test all attention mechanisms (self-attention, cross-attention)
- Verify positional embeddings
- Check layer normalization
- Test with variable sequence lengths
- Verify gradient flow through all layers

### Classification Heads
- Test with different number of classes
- Verify loss computation
- Test pooling strategies
- Check dropout behavior
- Test metric computation

### LoRA Adapters
- Test rank adaptation
- Verify weight merging
- Test gradient accumulation
- Check memory efficiency
- Test adapter switching

### Quantization
- Test quantization accuracy
- Verify dequantization
- Test quantized inference speed
- Check memory savings
- Test mixed precision

## Common Testing Patterns

### Testing Model Initialization
```python
def test_model_initialization():
    """Test model initializes with correct architecture."""
    config = create_bert_config(num_hidden_layers=6)
    model = BertCore(config)
    
    # Check architecture
    assert len(model.encoder.layers) == 6
    assert model.embeddings.word_embeddings.weight.shape[0] == config.vocab_size
```

### Testing Forward Pass
```python
def test_forward_pass_shapes():
    """Test output shapes are correct."""
    config = create_bert_config()
    model = BertCore(config)
    
    batch_size, seq_len = 4, 128
    input_ids = mx.random.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = mx.ones((batch_size, seq_len))
    
    outputs = model(input_ids, attention_mask)
    assert outputs.shape == (batch_size, seq_len, config.hidden_size)
```

### Testing Memory Efficiency
```python
def test_memory_efficiency():
    """Test model memory usage."""
    import mlx.utils
    
    config = create_small_bert_config()
    model = BertCore(config)
    
    # Measure memory before and after forward pass
    initial_memory = mlx.utils.get_memory_info()
    
    batch = create_test_batch(config, batch_size=32)
    outputs = model(batch["input_ids"], batch["attention_mask"])
    
    final_memory = mlx.utils.get_memory_info()
    memory_used = final_memory - initial_memory
    
    # Assert reasonable memory usage
    assert memory_used < 1_000_000_000  # Less than 1GB
```

### Testing Save/Load
```python
def test_save_load(tmp_path):
    """Test model save and load functionality."""
    config = create_bert_config()
    model = BertCore(config)
    
    # Generate random weights
    model.apply(lambda m, k, v: mx.random.normal(v.shape))
    
    # Save
    save_path = tmp_path / "model.safetensors"
    model.save_safetensors(save_path)
    
    # Load
    loaded_model = BertCore(config)
    loaded_model.load_safetensors(save_path)
    
    # Compare weights
    for (k1, v1), (k2, v2) in zip(
        mx.tree_flatten(model.parameters()), 
        mx.tree_flatten(loaded_model.parameters())
    ):
        assert mx.allclose(v1, v2)
```

## Debugging Tests

### Run Single Test with Output
```bash
pytest tests/models/unit/bert/test_core.py::TestBertCore::test_forward_pass -v -s
```

### Run with Debugger
```bash
pytest tests/models/unit/bert/test_core.py --pdb
```

### Run with Logging
```bash
pytest tests/models/ -v --log-cli-level=DEBUG
```

## Performance Testing

### Benchmarking Template
```python
@pytest.mark.benchmark
def test_model_performance(benchmark):
    """Benchmark model performance."""
    config = create_bert_config()
    model = BertCore(config)
    batch = create_test_batch(config)
    
    def forward():
        return model(batch["input_ids"], batch["attention_mask"])
    
    result = benchmark(forward)
    
    # Assert performance requirements
    assert benchmark.stats["mean"] < 0.1  # Less than 100ms
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure project root is in PYTHONPATH
   - Check `conftest.py` adds correct paths

2. **MLX Device Issues**
   - Tests automatically use default device
   - Force CPU: `MLX_DEFAULT_DEVICE=cpu pytest`

3. **Memory Issues**
   - Use smaller models for unit tests
   - Clean up after tests with `mx.clear_cache()`

4. **Numerical Precision**
   - Use appropriate tolerances (`atol`, `rtol`)
   - Consider float32 vs float16 differences

5. **Slow Tests**
   - Mark slow tests with `@pytest.mark.slow`
   - Use smaller configurations for unit tests

## Contributing

When adding new model tests:
1. Follow existing test structure
2. Use appropriate fixtures
3. Test all model components
4. Add integration tests for new features
5. Ensure tests are deterministic
6. Document complex test scenarios
7. Add performance benchmarks for critical paths