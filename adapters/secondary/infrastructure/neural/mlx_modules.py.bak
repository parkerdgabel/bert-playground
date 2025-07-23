"""Specialized MLX module implementations for complex operations.

This module provides custom implementations for neural network operations
that require special handling in MLX.
"""

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from core.ports.neural import Module
from .mlx_backend import MLXModule


class MLXFlashAttention(MLXModule):
    """Optimized attention implementation for MLX.
    
    This provides an efficient attention mechanism similar to FlashAttention,
    optimized for MLX's computation model.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        scale: Optional[float] = None,
        use_bias: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = scale or (self.head_dim ** -0.5)
        self.dropout = dropout
        
        # Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        
        if dropout > 0:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = None
        
        # Register submodules
        self.add_module("q_proj", MLXModule(self.q_proj))
        self.add_module("k_proj", MLXModule(self.k_proj))
        self.add_module("v_proj", MLXModule(self.v_proj))
        self.add_module("out_proj", MLXModule(self.out_proj))
        if self.dropout_layer:
            self.add_module("dropout", MLXModule(self.dropout_layer))
    
    def forward(
        self,
        query: mx.array,
        key: Optional[mx.array] = None,
        value: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        is_causal: bool = False
    ) -> Tuple[mx.array, Optional[mx.array]]:
        """Forward pass with optimized attention computation.
        
        Args:
            query: Query tensor [batch, seq_len, embed_dim]
            key: Key tensor (defaults to query)
            value: Value tensor (defaults to query)
            attention_mask: Optional attention mask
            is_causal: Whether to use causal masking
            
        Returns:
            Tuple of (output, attention_weights)
        """
        if key is None:
            key = query
        if value is None:
            value = query
        
        batch_size, seq_len, _ = query.shape
        
        # Project Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, -1, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, -1, self.num_heads, self.head_dim)
        
        # Transpose to [batch, heads, seq_len, head_dim]
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))
        
        # Scaled dot-product attention
        scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * self.scale
        
        # Apply masking
        if is_causal:
            causal_mask = mx.triu(mx.full((seq_len, seq_len), -float('inf')), k=1)
            scores = scores + causal_mask
        
        if attention_mask is not None:
            # Expand mask for all heads
            if attention_mask.ndim == 2:
                # [batch, seq_len] -> [batch, 1, 1, seq_len]
                attention_mask = mx.expand_dims(mx.expand_dims(attention_mask, 1), 1)
            elif attention_mask.ndim == 3:
                # [batch, seq_len, seq_len] -> [batch, 1, seq_len, seq_len]
                attention_mask = mx.expand_dims(attention_mask, 1)
            
            scores = mx.where(attention_mask == 0, -float('inf'), scores)
        
        # Softmax
        attn_weights = mx.softmax(scores, axis=-1)
        
        # Apply dropout
        if self.dropout_layer is not None and self.training:
            attn_weights = self.dropout_layer(attn_weights)
        
        # Apply attention to values
        attn_output = mx.matmul(attn_weights, v)
        
        # Transpose back and reshape
        attn_output = mx.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        
        # Output projection
        output = self.out_proj(attn_output)
        
        return output, attn_weights


class MLXGroupedQueryAttention(MLXModule):
    """Grouped Query Attention (GQA) implementation for MLX.
    
    This implements the GQA mechanism where key and value heads are shared
    across multiple query heads, reducing memory and computation.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float = 0.0,
        use_bias: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_heads
        self.kv_head_dim = embed_dim // num_kv_heads
        self.scale = self.head_dim ** -0.5
        
        # Ensure num_heads is divisible by num_kv_heads
        assert num_heads % num_kv_heads == 0, \
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        
        self.num_queries_per_kv = num_heads // num_kv_heads
        
        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=use_bias)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=use_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
    
    def forward(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        is_causal: bool = False
    ) -> mx.array:
        """Forward pass for grouped query attention.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, embed_dim]
            attention_mask: Optional attention mask
            is_causal: Whether to use causal masking
            
        Returns:
            Output tensor [batch, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project Q, K, V
        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)
        
        # Reshape for attention
        queries = queries.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        keys = keys.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        values = values.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Transpose to [batch, heads, seq_len, head_dim]
        queries = mx.transpose(queries, (0, 2, 1, 3))
        keys = mx.transpose(keys, (0, 2, 1, 3))
        values = mx.transpose(values, (0, 2, 1, 3))
        
        # Repeat KV heads for each query group
        if self.num_kv_heads < self.num_heads:
            keys = mx.repeat(keys, self.num_queries_per_kv, axis=1)
            values = mx.repeat(values, self.num_queries_per_kv, axis=1)
        
        # Compute attention scores
        scores = mx.matmul(queries, mx.transpose(keys, (0, 1, 3, 2))) * self.scale
        
        # Apply causal mask if needed
        if is_causal:
            causal_mask = mx.triu(mx.full((seq_len, seq_len), -float('inf')), k=1)
            scores = scores + causal_mask
        
        # Apply attention mask if provided
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                attention_mask = mx.expand_dims(mx.expand_dims(attention_mask, 1), 1)
            scores = mx.where(attention_mask == 0, -float('inf'), scores)
        
        # Softmax
        attn_weights = mx.softmax(scores, axis=-1)
        
        # Apply dropout
        if self.dropout is not None and self.training:
            attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = mx.matmul(attn_weights, values)
        
        # Transpose and reshape
        attn_output = mx.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        
        # Output projection
        output = self.out_proj(attn_output)
        
        return output


class MLXGeGLU(MLXModule):
    """Gated Linear Unit with GELU activation (GeGLU) for MLX.
    
    This implements the GeGLU activation function used in modern transformers,
    which combines gating with GELU activation.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, bias: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Two linear projections for gating
        self.gate_proj = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(input_dim, hidden_dim, bias=bias)
        
        # Register submodules
        self.add_module("gate_proj", MLXModule(self.gate_proj))
        self.add_module("up_proj", MLXModule(self.up_proj))
    
    def forward(self, x: mx.array) -> mx.array:
        """Apply GeGLU activation.
        
        Args:
            x: Input tensor
            
        Returns:
            Output after GeGLU activation
        """
        gate = nn.gelu(self.gate_proj(x))
        up = self.up_proj(x)
        return gate * up


class MLXSwiGLU(MLXModule):
    """SwiGLU activation for MLX.
    
    Similar to GeGLU but uses SiLU (Swish) activation instead of GELU.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, bias: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Two linear projections for gating
        self.gate_proj = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(input_dim, hidden_dim, bias=bias)
        
        # Register submodules
        self.add_module("gate_proj", MLXModule(self.gate_proj))
        self.add_module("up_proj", MLXModule(self.up_proj))
    
    def forward(self, x: mx.array) -> mx.array:
        """Apply SwiGLU activation.
        
        Args:
            x: Input tensor
            
        Returns:
            Output after SwiGLU activation
        """
        gate = nn.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return gate * up


class MLXALiBi(MLXModule):
    """Attention with Linear Biases (ALiBi) for MLX.
    
    This implements the ALiBi position encoding method that adds linear biases
    to attention scores based on relative positions.
    """
    
    def __init__(self, num_heads: int, max_positions: int = 8192):
        super().__init__()
        self.num_heads = num_heads
        self.max_positions = max_positions
        
        # Compute slopes for each head
        slopes = self._get_alibi_slopes(num_heads)
        self.slopes = mx.array(slopes)
        
        # Precompute bias matrix
        positions = mx.arange(max_positions)
        relative_positions = positions[None, :] - positions[:, None]
        alibi_bias = relative_positions[None, :, :] * slopes[:, None, None]
        self.alibi_bias = alibi_bias
    
    def _get_alibi_slopes(self, num_heads: int) -> mx.array:
        """Compute ALiBi slopes for each attention head."""
        # Following the ALiBi paper's slope computation
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(mx.log2(n) - 3)))
            ratio = start
            return [start * (ratio ** i) for i in range(n)]
        
        if mx.log2(num_heads).astype(mx.int32) == mx.log2(num_heads):
            return mx.array(get_slopes_power_of_2(num_heads))
        else:
            # For non-power-of-2 heads, interpolate
            closest_power_of_2 = 2 ** mx.floor(mx.log2(num_heads)).astype(mx.int32)
            return mx.array(get_slopes_power_of_2(closest_power_of_2) + 
                          get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:num_heads - closest_power_of_2])
    
    def forward(self, attention_scores: mx.array, seq_len: int) -> mx.array:
        """Add ALiBi biases to attention scores.
        
        Args:
            attention_scores: Raw attention scores [batch, heads, seq_len, seq_len]
            seq_len: Sequence length
            
        Returns:
            Attention scores with ALiBi biases added
        """
        # Extract the relevant portion of precomputed biases
        alibi_bias = self.alibi_bias[:, :seq_len, :seq_len]
        
        # Add biases to attention scores
        return attention_scores + alibi_bias


class MLXFusedLayerNorm(MLXModule):
    """Fused layer normalization with optional affine transformation.
    
    This provides an optimized implementation that fuses normalization
    with the affine transformation for better performance.
    """
    
    def __init__(
        self,
        normalized_shape: int | tuple[int, ...],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.use_bias = bias
        
        if elementwise_affine:
            self.weight = mx.ones(normalized_shape)
            if bias:
                self.bias = mx.zeros(normalized_shape)
            else:
                self.bias = None
            
            # Register parameters
            self.register_parameter("weight", self.weight)
            if self.bias is not None:
                self.register_parameter("bias", self.bias)
    
    def forward(self, x: mx.array) -> mx.array:
        """Apply fused layer normalization.
        
        Args:
            x: Input tensor
            
        Returns:
            Normalized tensor
        """
        # Compute mean and variance
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        
        # Normalize
        x_normalized = (x - mean) / mx.sqrt(var + self.eps)
        
        # Apply affine transformation if enabled
        if self.elementwise_affine:
            x_normalized = x_normalized * self.weight
            if self.bias is not None:
                x_normalized = x_normalized + self.bias
        
        return x_normalized


def register_buffer(module: Module, name: str, tensor: mx.array) -> None:
    """Helper to register a buffer (non-trainable parameter) on a module.
    
    Args:
        module: Module to register buffer on
        name: Name of the buffer
        tensor: Buffer tensor
    """
    # MLX doesn't distinguish between parameters and buffers
    # so we just store it as an attribute
    setattr(module, name, tensor)