# Neural Network Adapters

This directory contains framework-specific implementations of the `NeuralBackend` protocol defined in `core/ports/neural.py`. These adapters enable the core domain to work with different ML frameworks without changing the business logic.

## MLX Neural Backend

The MLX adapter (`mlx_backend.py`) provides a complete implementation of the neural network port using Apple's MLX framework.

### Features

- **Complete Neural Operations**: All operations defined in the `NeuralBackend` protocol are implemented
- **MLX-Optimized Modules**: Custom implementations for advanced features like FlashAttention, GeGLU, and Grouped Query Attention
- **Mixed Precision Support**: Native support for bfloat16 and other data types
- **Efficient Memory Usage**: Leverages MLX's unified memory architecture on Apple Silicon

### Key Components

1. **MLXModule**: Base wrapper class that adapts MLX modules to our framework-agnostic `Module` interface
2. **Layer Implementations**: 
   - Linear, Embedding, LayerNorm, RMSNorm, Dropout
   - Multi-head Attention with self-attention support
   - Container modules (Sequential, ModuleList, ModuleDict)
3. **Activation Functions**: All standard activations plus MLX-optimized implementations
4. **Loss Functions**: Cross-entropy, binary cross-entropy, MSE with customizable reductions
5. **Advanced Features**:
   - Rotary Position Embeddings (RoPE)
   - Flash Attention optimization
   - Grouped Query Attention (GQA)
   - GeGLU and SwiGLU activations
   - ALiBi positional biases

### Usage Example

```python
from infrastructure.ports.neural import create_neural_backend

# Create MLX backend
backend = create_neural_backend("mlx")

# Build a simple model
model = backend.sequential(
    backend.linear(768, 256),
    backend.gelu(),
    backend.dropout(0.1),
    backend.linear(256, 10)
)

# Use the model
import mlx.core as mx
x = mx.random.normal((32, 768))
output = model(x)
```

### MLX-Specific Considerations

1. **Array Operations**: MLX uses lazy evaluation, so operations are only executed when needed
2. **Device Management**: MLX automatically handles device placement on Apple Silicon
3. **Gradient Computation**: Use MLX's `grad` and `value_and_grad` functions for differentiation
4. **Memory Efficiency**: MLX's unified memory model eliminates CPU-GPU transfers

### Testing

The adapter includes comprehensive tests in `tests/core/adapters/test_mlx_neural_backend.py`:

```bash
uv run pytest tests/core/adapters/test_mlx_neural_backend.py -v
```

### Extending the Adapter

To add new MLX-specific features:

1. Implement the module in `mlx_modules.py` for complex operations
2. Add the factory method to `MLXNeuralBackend` class
3. Write tests to ensure compatibility with the protocol
4. Update this documentation

### Performance Tips

- Use batch sizes that are powers of 2 for optimal performance
- Enable MLX compilation with `mx.compile` for frequently-called functions
- Leverage MLX's automatic mixed precision for memory efficiency
- Use the specialized modules (FlashAttention, GQA) for transformer models

## Future Adapters

Placeholder for PyTorch and JAX adapters:

- `pytorch_backend.py`: PyTorch implementation (coming soon)
- `jax_backend.py`: JAX implementation (coming soon)

Each adapter will follow the same pattern: implementing the `NeuralBackend` protocol while leveraging framework-specific optimizations.