# Neural Network Abstraction Port

This document describes the neural network abstraction port that allows BERT models to be written in a framework-agnostic way, supporting multiple ML frameworks (MLX, PyTorch, JAX) through a unified interface.

## Overview

The neural port provides a comprehensive abstraction layer for neural network operations, enabling:

- **Framework Independence**: Write models once, run with any supported backend
- **Type Safety**: Full type hints and protocol definitions
- **Modularity**: Clean separation between model logic and framework specifics
- **Extensibility**: Easy to add new backends or operations

## Architecture

### Core Components

1. **`neural.py`** - Main port interface
   - `Module`: Base class for all neural network modules
   - `NeuralBackend`: Protocol defining backend operations
   - Enums for activation types, normalization types, loss types

2. **`neural_types.py`** - Type definitions and utilities
   - Configuration classes (AttentionConfig, FeedForwardConfig, etc.)
   - Output types (TransformerOutput, LossOutput, etc.)
   - Helper types and utilities

3. **`neural_ops.py`** - High-level neural operations
   - Complex operations built on backend primitives
   - BERT-specific patterns (attention masks, pooling, etc.)

4. **`neural_example.py`** - Example implementation
   - Shows how to build BERT models using the abstraction

## Usage

### Basic Example

```python
from core.ports.neural import create_neural_backend, Module
from core.ports.neural_types import EmbeddingConfig

# Create a backend (MLX, PyTorch, or JAX)
backend = create_neural_backend("mlx")

# Create layers using the backend
embedding = backend.embedding(
    num_embeddings=30522,
    embedding_dim=768
)

linear = backend.linear(
    in_features=768,
    out_features=768,
    bias=True
)

# Use in a model
class MyModel(Module):
    def __init__(self, backend):
        super().__init__()
        self.embedding = backend.embedding(30522, 768)
        self.linear = backend.linear(768, 768)
        
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.linear(x)
        return x
```

### Creating a Custom Module

```python
from core.ports.neural import Module, NeuralBackend
from core.ports.neural_types import AttentionConfig

class CustomAttention(Module):
    def __init__(self, config: AttentionConfig, backend: NeuralBackend):
        super().__init__()
        self.config = config
        self.backend = backend
        
        # Create layers
        self.q_proj = backend.linear(
            config.hidden_size,
            config.hidden_size,
            bias=config.use_bias
        )
        self.k_proj = backend.linear(
            config.hidden_size,
            config.hidden_size,
            bias=config.use_bias
        )
        self.v_proj = backend.linear(
            config.hidden_size,
            config.hidden_size,
            bias=config.use_bias
        )
        
        # Register as submodules
        self.add_module("q_proj", self.q_proj)
        self.add_module("k_proj", self.k_proj)
        self.add_module("v_proj", self.v_proj)
    
    def forward(self, hidden_states, attention_mask=None):
        # Compute Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # ... attention computation ...
        
        return output
```

## Supported Operations

### Layer Types
- Linear/Dense layers
- Embedding layers
- Multi-head attention
- Layer normalization
- RMS normalization
- Dropout

### Activation Functions
- ReLU, GELU, SiLU (Swish)
- Tanh, Sigmoid
- Leaky ReLU, ELU
- Softmax, LogSoftmax

### Tensor Operations
- Matrix multiplication
- Transpose, reshape, concatenate, split
- Reduction operations (mean, sum, max, min)
- Broadcasting and unsqueezing
- Element-wise operations

### Advanced Operations
- Rotary position embeddings (RoPE)
- Attention masking
- Gradient checkpointing
- Mixed precision support

## Backend Implementation

To add a new backend, implement the `NeuralBackend` protocol:

```python
from core.ports.neural import NeuralBackend

class MyBackend:
    @property
    def name(self) -> str:
        return "my_backend"
    
    @property
    def supports_mixed_precision(self) -> bool:
        return True
    
    def linear(self, in_features: int, out_features: int, bias: bool = True):
        # Implementation using your framework
        ...
    
    # Implement all other required methods...
```

## Best Practices

1. **Use Type Hints**: Always specify types for better IDE support and type checking
2. **Register Submodules**: Use `add_module()` to register child modules
3. **Configuration Classes**: Use provided config classes for consistency
4. **Backend Agnostic**: Avoid framework-specific code in models
5. **Test with Multiple Backends**: Ensure models work with all supported backends

## Examples

See `neural_example.py` for a complete example of building BERT models using the abstraction, including:
- BertEmbeddings
- BertAttention
- BertFeedForward
- Complete BertModel

## Future Extensions

- Additional backends (TensorFlow, Flax)
- More specialized layers (convolutions, recurrent)
- Quantization support
- Distributed training abstractions
- Custom kernel support