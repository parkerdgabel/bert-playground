"""Demo of using the MLX Neural Backend adapter.

This example shows how to use the framework-agnostic neural port
with the MLX implementation to build neural networks.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import numpy as np

from infrastructure.ports.neural import ActivationType, create_neural_backend


def demo_basic_layers():
    """Demonstrate basic layer creation with MLX backend."""
    print("=== Basic Layers Demo ===")
    
    # Create MLX backend
    backend = create_neural_backend("mlx")
    print(f"Using backend: {backend.name}")
    print(f"Supports mixed precision: {backend.supports_mixed_precision}")
    
    # Create layers
    linear = backend.linear(10, 20, bias=True)
    embedding = backend.embedding(100, 32)
    layer_norm = backend.layer_norm(32)
    dropout = backend.dropout(p=0.1)
    
    # Test forward passes
    x = mx.random.normal((2, 10))
    linear_out = linear(x)
    print(f"Linear output shape: {linear_out.shape}")
    
    indices = mx.array([[1, 2, 3], [4, 5, 6]])
    embed_out = embedding(indices)
    print(f"Embedding output shape: {embed_out.shape}")
    
    norm_out = layer_norm(embed_out)
    print(f"LayerNorm output shape: {norm_out.shape}")
    
    # Set to eval mode for consistent dropout
    dropout.eval()
    dropout_out = dropout(norm_out)
    print(f"Dropout output shape: {dropout_out.shape}")


def demo_attention():
    """Demonstrate attention mechanisms."""
    print("\n=== Attention Demo ===")
    
    backend = create_neural_backend("mlx")
    
    # Create multi-head attention
    mha = backend.multi_head_attention(
        embed_dim=64,
        num_heads=8,
        dropout=0.1
    )
    
    # Test self-attention
    x = mx.random.normal((2, 10, 64))  # [batch, seq_len, embed_dim]
    attn_out = mha(x)
    print(f"MHA output shape: {attn_out.shape}")
    
    # Create custom flash attention from modules
    from infrastructure.adapters.neural.mlx_modules import MLXFlashAttention
    
    flash_attn = MLXFlashAttention(
        embed_dim=64,
        num_heads=8,
        dropout=0.1
    )
    
    flash_out, attn_weights = flash_attn(x, is_causal=True)
    print(f"Flash attention output shape: {flash_out.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")


def demo_activations():
    """Demonstrate various activation functions."""
    print("\n=== Activations Demo ===")
    
    backend = create_neural_backend("mlx")
    
    x = mx.random.normal((2, 10))
    
    # Test different activations
    activations = {
        "ReLU": backend.relu(),
        "GELU": backend.gelu(),
        "SiLU": backend.silu(),
        "Tanh": backend.activation(ActivationType.TANH),
        "Sigmoid": backend.activation(ActivationType.SIGMOID),
    }
    
    for name, activation in activations.items():
        output = activation(x)
        print(f"{name} output shape: {output.shape}, "
              f"min: {mx.min(output).item():.3f}, "
              f"max: {mx.max(output).item():.3f}")


def demo_sequential_model():
    """Demonstrate building a sequential model."""
    print("\n=== Sequential Model Demo ===")
    
    backend = create_neural_backend("mlx")
    
    # Build a simple feedforward network
    model = backend.sequential(
        backend.linear(784, 256),
        backend.relu(),
        backend.dropout(0.2),
        backend.linear(256, 128),
        backend.relu(),
        backend.dropout(0.2),
        backend.linear(128, 10)
    )
    
    # Test forward pass
    x = mx.random.normal((32, 784))  # Batch of 32 flattened MNIST images
    output = model(x)
    print(f"Model output shape: {output.shape}")
    
    # Apply softmax for classification
    probs = backend.softmax(output, dim=-1)
    print(f"Probabilities shape: {probs.shape}")
    print(f"Sum of probabilities: {mx.sum(probs[0]).item():.6f}")


def demo_loss_functions():
    """Demonstrate loss function usage."""
    print("\n=== Loss Functions Demo ===")
    
    backend = create_neural_backend("mlx")
    
    # Classification loss
    ce_loss = backend.cross_entropy_loss(reduction="mean")
    logits = mx.random.normal((10, 5))
    targets = mx.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    loss = ce_loss(logits, targets)
    print(f"Cross entropy loss: {loss.item():.4f}")
    
    # Regression loss
    mse_loss = backend.mse_loss(reduction="mean")
    predictions = mx.random.normal((10, 3))
    targets = mx.random.normal((10, 3))
    loss = mse_loss(predictions, targets)
    print(f"MSE loss: {loss.item():.4f}")


def demo_advanced_features():
    """Demonstrate advanced features like RoPE."""
    print("\n=== Advanced Features Demo ===")
    
    backend = create_neural_backend("mlx")
    
    # Create rotary embeddings
    rope = backend.rotary_embedding(
        dim=64,
        max_position_embeddings=512,
        base=10000.0
    )
    
    # Generate query and key tensors
    batch_size, num_heads, seq_len, head_dim = 2, 8, 10, 64
    q = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
    k = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
    
    # Get cos/sin for positions
    dummy_input = mx.zeros((batch_size, seq_len, num_heads, head_dim))
    cos, sin = rope(dummy_input)
    
    # Apply rotary embeddings
    q_rot, k_rot = backend.apply_rotary_pos_emb(q, k, cos, sin)
    print(f"Rotated Q shape: {q_rot.shape}")
    print(f"Rotated K shape: {k_rot.shape}")
    
    # Test custom modules
    from infrastructure.adapters.neural.mlx_modules import MLXGeGLU, MLXGroupedQueryAttention
    
    # GeGLU activation
    geglu = MLXGeGLU(input_dim=64, hidden_dim=128)
    x = mx.random.normal((2, 10, 64))
    geglu_out = geglu(x)
    print(f"GeGLU output shape: {geglu_out.shape}")
    
    # Grouped Query Attention
    gqa = MLXGroupedQueryAttention(
        embed_dim=64,
        num_heads=8,
        num_kv_heads=2  # 4x compression
    )
    gqa_out = gqa(x)
    print(f"GQA output shape: {gqa_out.shape}")


if __name__ == "__main__":
    demo_basic_layers()
    demo_attention()
    demo_activations()
    demo_sequential_model()
    demo_loss_functions()
    demo_advanced_features()
    
    print("\n✅ All demos completed successfully!")