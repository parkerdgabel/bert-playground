"""Integration tests for MLX neural backend with the broader system."""

import pytest
import mlx.core as mx
import numpy as np

from core.ports.neural import create_neural_backend, Module, ActivationType
from core.adapters.neural.mlx_modules import (
    MLXFlashAttention,
    MLXGroupedQueryAttention,
    MLXGeGLU,
    MLXSwiGLU,
    MLXALiBi,
    MLXFusedLayerNorm
)


class TestMLXIntegration:
    """Integration tests for MLX backend with real-world scenarios."""
    
    def test_transformer_block(self):
        """Test building a complete transformer block."""
        backend = create_neural_backend("mlx")
        
        class TransformerBlock(Module):
            """Simple transformer block using MLX backend."""
            
            def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
                super().__init__()
                self.attention = MLXFlashAttention(dim, num_heads)
                self.norm1 = backend.layer_norm(dim)
                self.norm2 = backend.layer_norm(dim)
                
                mlp_dim = int(dim * mlp_ratio)
                self.mlp = backend.sequential(
                    backend.linear(dim, mlp_dim),
                    backend.gelu(),
                    backend.dropout(0.1),
                    backend.linear(mlp_dim, dim),
                    backend.dropout(0.1)
                )
            
            def forward(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
                # Self-attention with residual
                attn_out, _ = self.attention(x, attention_mask=mask)
                x = x + attn_out
                x = self.norm1(x)
                
                # MLP with residual
                mlp_out = self.mlp(x)
                x = x + mlp_out
                x = self.norm2(x)
                
                return x
        
        # Test the transformer block
        block = TransformerBlock(dim=256, num_heads=8)
        x = mx.random.normal((2, 32, 256))  # [batch, seq_len, dim]
        output = block(x)
        
        assert output.shape == x.shape
        # Check that output is different from input (processing happened)
        assert not mx.allclose(output, x)
    
    def test_bert_style_model(self):
        """Test building a BERT-style encoder."""
        backend = create_neural_backend("mlx")
        
        class BERTEncoder(Module):
            """Simplified BERT encoder using MLX backend."""
            
            def __init__(
                self, 
                vocab_size: int,
                hidden_size: int,
                num_layers: int,
                num_heads: int,
                max_position_embeddings: int = 512
            ):
                super().__init__()
                # Embeddings
                self.token_embeddings = backend.embedding(vocab_size, hidden_size)
                self.position_embeddings = backend.embedding(max_position_embeddings, hidden_size)
                self.layer_norm = backend.layer_norm(hidden_size)
                self.dropout = backend.dropout(0.1)
                
                # Transformer layers
                self.layers = backend.module_list([
                    self._make_layer(hidden_size, num_heads) 
                    for _ in range(num_layers)
                ])
                
                # Output
                self.pooler = backend.sequential(
                    backend.linear(hidden_size, hidden_size),
                    backend.activation(ActivationType.TANH)
                )
            
            def _make_layer(self, hidden_size: int, num_heads: int) -> Module:
                """Create a transformer layer."""
                return TransformerLayer(hidden_size, num_heads)
            
            def forward(self, input_ids: mx.array, attention_mask: mx.array | None = None) -> tuple[mx.array, mx.array]:
                seq_length = input_ids.shape[1]
                position_ids = mx.arange(seq_length)
                position_ids = mx.expand_dims(position_ids, 0)
                position_ids = mx.broadcast_to(position_ids, input_ids.shape)
                
                # Embed tokens and positions
                token_embeds = self.token_embeddings(input_ids)
                position_embeds = self.position_embeddings(position_ids)
                
                hidden_states = token_embeds + position_embeds
                hidden_states = self.layer_norm(hidden_states)
                hidden_states = self.dropout(hidden_states)
                
                # Apply transformer layers
                for layer in self.layers:
                    hidden_states = layer(hidden_states, attention_mask)
                
                # Pool the first token (CLS token)
                pooled_output = self.pooler(hidden_states[:, 0])
                
                return hidden_states, pooled_output
        
        class TransformerLayer(Module):
            """Transformer layer with MLX backend."""
            
            def __init__(self, hidden_size: int, num_heads: int):
                super().__init__()
                self.attention = backend.multi_head_attention(
                    hidden_size, num_heads, dropout=0.1
                )
                self.norm1 = backend.layer_norm(hidden_size)
                self.mlp = backend.sequential(
                    backend.linear(hidden_size, hidden_size * 4),
                    backend.gelu(),
                    backend.linear(hidden_size * 4, hidden_size)
                )
                self.norm2 = backend.layer_norm(hidden_size)
                self.dropout = backend.dropout(0.1)
            
            def forward(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
                # Attention block
                attn_out = self.attention(x)
                x = x + self.dropout(attn_out)
                x = self.norm1(x)
                
                # MLP block
                mlp_out = self.mlp(x)
                x = x + self.dropout(mlp_out)
                x = self.norm2(x)
                
                return x
        
        # Test the BERT encoder
        model = BERTEncoder(
            vocab_size=30522,
            hidden_size=256,
            num_layers=2,
            num_heads=8
        )
        
        # Mock input
        batch_size, seq_len = 4, 64
        input_ids = mx.random.randint(0, 30522, (batch_size, seq_len))
        
        hidden_states, pooled_output = model(input_ids)
        
        assert hidden_states.shape == (batch_size, seq_len, 256)
        assert pooled_output.shape == (batch_size, 256)
    
    def test_advanced_attention_variants(self):
        """Test advanced attention mechanisms."""
        # Test Grouped Query Attention
        gqa = MLXGroupedQueryAttention(
            embed_dim=256,
            num_heads=8,
            num_kv_heads=2,
            dropout=0.1
        )
        
        x = mx.random.normal((2, 32, 256))
        output = gqa(x, is_causal=True)
        assert output.shape == x.shape
        
        # Test ALiBi
        alibi = MLXALiBi(num_heads=8, max_positions=512)
        scores = mx.random.normal((2, 8, 32, 32))
        biased_scores = alibi(scores, seq_len=32)
        assert biased_scores.shape == scores.shape
    
    def test_advanced_activations(self):
        """Test advanced activation functions."""
        # Test GeGLU
        geglu = MLXGeGLU(input_dim=256, hidden_dim=512)
        x = mx.random.normal((2, 32, 256))
        output = geglu(x)
        assert output.shape == (2, 32, 512)
        
        # Test SwiGLU
        swiglu = MLXSwiGLU(input_dim=256, hidden_dim=512)
        output = swiglu(x)
        assert output.shape == (2, 32, 512)
    
    def test_custom_normalization(self):
        """Test custom normalization layers."""
        # Test fused layer norm
        fused_ln = MLXFusedLayerNorm(256, eps=1e-6)
        x = mx.random.normal((2, 32, 256))
        output = fused_ln(x)
        assert output.shape == x.shape
        
        # Verify normalization
        mean = mx.mean(output, axis=-1, keepdims=True)
        var = mx.var(output, axis=-1, keepdims=True)
        assert mx.allclose(mean, mx.zeros_like(mean), atol=1e-5)
        assert mx.allclose(var, mx.ones_like(var), atol=1e-5)
    
    def test_mixed_precision_workflow(self):
        """Test mixed precision training workflow."""
        backend = create_neural_backend("mlx")
        
        # Create a simple model
        model = backend.sequential(
            backend.linear(128, 64),
            backend.gelu(),
            backend.linear(64, 10)
        )
        
        # Test with different dtypes
        x_fp32 = mx.random.normal((16, 128), dtype=mx.float32)
        x_bf16 = x_fp32.astype(mx.bfloat16)
        
        output_fp32 = model(x_fp32)
        output_bf16 = model(x_bf16)
        
        # MLX uses automatic mixed precision, so outputs might be in float32
        # The important thing is that it accepts different input types
        assert output_fp32.shape == (16, 10)
        assert output_bf16.shape == (16, 10)
        
        # Results should be similar
        assert mx.allclose(
            output_fp32, 
            output_bf16.astype(mx.float32), 
            atol=1e-2
        )
    
    def test_gradient_computation(self):
        """Test gradient computation through the backend."""
        backend = create_neural_backend("mlx")
        
        # Simple model for testing
        linear = backend.linear(10, 1)
        
        def loss_fn(x, y):
            pred = linear(x)
            return mx.mean((pred - y) ** 2)
        
        # Generate data
        x = mx.random.normal((32, 10))
        y = mx.random.normal((32, 1))
        
        # Compute gradients using MLX
        grad_fn = mx.grad(loss_fn)
        grads = grad_fn(x, y)
        
        # Verify gradient shapes match input
        assert grads.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])