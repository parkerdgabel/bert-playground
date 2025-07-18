"""
Rotary Positional Embeddings (RoPE) implementation for ModernBERT.

This module implements RoPE as described in "RoFormer: Enhanced Transformer with Rotary Position Embedding"
and used in ModernBERT for improved positional understanding.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import Optional, Tuple, Union
import math


class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embeddings (RoPE) implementation.
    
    RoPE encodes absolute positional information while naturally incorporating
    explicit relative position dependency in self-attention.
    """
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 8192,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
    ):
        """
        Initialize RoPE embeddings.
        
        Args:
            dim: Dimension of embeddings (should be head_dim)
            max_position_embeddings: Maximum sequence length
            base: Base frequency for rotary embeddings
            scaling_factor: Scaling factor for frequency adjustment
        """
        super().__init__()
        
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        
        # Pre-compute frequency inverse
        self.inv_freq = self._compute_inv_freq()
        
        # Pre-compute cos and sin values for efficiency
        self._cos_cached = None
        self._sin_cached = None
        self._cached_seq_len = 0
    
    def _compute_inv_freq(self) -> mx.array:
        """Compute inverse frequencies for RoPE."""
        # Create frequency inverse: 1 / (base^(2i/dim)) for i in [0, dim/2)
        exponent = mx.arange(0, self.dim, 2, dtype=mx.float32)
        inv_freq = 1.0 / (self.base ** (exponent / self.dim))
        
        # Apply scaling if needed
        if self.scaling_factor != 1.0:
            inv_freq = inv_freq / self.scaling_factor
        
        return inv_freq
    
    def _compute_cos_sin(self, seq_len: int) -> Tuple[mx.array, mx.array]:
        """
        Compute cosine and sine values for rotary embeddings.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Tuple of (cos, sin) arrays
        """
        # Create position tensor
        positions = mx.arange(seq_len, dtype=mx.float32)
        
        # Compute frequency matrix: positions * inv_freq
        # positions: [seq_len, 1], inv_freq: [dim/2] -> freqs: [seq_len, dim/2]
        freqs = mx.outer(positions, self.inv_freq)
        
        # Duplicate frequencies for cos and sin
        # freqs: [seq_len, dim/2] -> [seq_len, dim]
        cos_freqs = mx.concatenate([freqs, freqs], axis=-1)
        sin_freqs = mx.concatenate([freqs, freqs], axis=-1)
        
        # Compute cos and sin
        cos = mx.cos(cos_freqs)
        sin = mx.sin(sin_freqs)
        
        return cos, sin
    
    def forward(self, x: mx.array, seq_len: Optional[int] = None) -> Tuple[mx.array, mx.array]:
        """
        Forward pass to get cos and sin values.
        
        Args:
            x: Input tensor [batch_size, seq_len, ...]
            seq_len: Optional sequence length override
            
        Returns:
            Tuple of (cos, sin) arrays for position embeddings
        """
        if seq_len is None:
            seq_len = x.shape[1]
        
        # Use cached values if available and sequence length matches
        if (self._cos_cached is not None and 
            self._sin_cached is not None and 
            seq_len <= self._cached_seq_len):
            return self._cos_cached[:seq_len], self._sin_cached[:seq_len]
        
        # Compute new cos and sin values
        cos, sin = self._compute_cos_sin(seq_len)
        
        # Cache for future use
        self._cos_cached = cos
        self._sin_cached = sin
        self._cached_seq_len = seq_len
        
        return cos, sin
    
    def __call__(self, x: mx.array, seq_len: Optional[int] = None) -> Tuple[mx.array, mx.array]:
        """Make the module callable."""
        return self.forward(x, seq_len)


def rotate_half(x: mx.array) -> mx.array:
    """
    Rotate half the dimensions of the input tensor.
    
    This function rotates the last dimension by splitting it in half
    and swapping the halves with a sign flip.
    
    Args:
        x: Input tensor [..., dim]
        
    Returns:
        Rotated tensor [..., dim]
    """
    # Split the last dimension in half
    x1, x2 = mx.split(x, 2, axis=-1)
    
    # Rotate: [-x2, x1]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(
    q: mx.array, 
    k: mx.array, 
    cos: mx.array, 
    sin: mx.array,
    position_ids: Optional[mx.array] = None
) -> Tuple[mx.array, mx.array]:
    """
    Apply rotary positional embeddings to query and key tensors.
    
    Args:
        q: Query tensor [batch_size, num_heads, seq_len, head_dim]
        k: Key tensor [batch_size, num_heads, seq_len, head_dim]
        cos: Cosine values [seq_len, head_dim]
        sin: Sine values [seq_len, head_dim]
        position_ids: Optional position IDs for advanced positioning
        
    Returns:
        Tuple of (rotated_q, rotated_k)
    """
    # Handle position_ids if provided
    if position_ids is not None:
        # Gather cos and sin values for specific positions
        cos = mx.take(cos, position_ids, axis=0)
        sin = mx.take(sin, position_ids, axis=0)
    
    # Expand dimensions for broadcasting
    # cos, sin: [seq_len, head_dim] -> [1, 1, seq_len, head_dim]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    
    # Apply rotary embeddings
    # q_embed = q * cos + rotate_half(q) * sin
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


class RoPEAttention(nn.Module):
    """
    Self-attention with Rotary Positional Embeddings.
    
    This is a modified version of the standard attention mechanism
    that incorporates RoPE for better positional understanding.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        max_position_embeddings: int = 8192,
        rope_base: float = 10000.0,
        attention_dropout: float = 0.0,
        use_bias: bool = False,
    ):
        """
        Initialize RoPE attention.
        
        Args:
            hidden_size: Hidden size of the model
            num_attention_heads: Number of attention heads
            max_position_embeddings: Maximum sequence length
            rope_base: Base frequency for RoPE
            attention_dropout: Dropout rate for attention
            use_bias: Whether to use bias in linear layers
        """
        super().__init__()
        
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_attention_heads ({num_attention_heads})"
            )
        
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Linear projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        
        # RoPE embeddings
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_base,
        )
        
        # Dropout
        self.dropout = nn.Dropout(attention_dropout)
    
    def forward(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        output_attentions: bool = False,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        """
        Forward pass through RoPE attention.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len]
            output_attentions: Whether to return attention weights
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Linear projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        
        # Transpose to [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        
        # Apply RoPE
        cos, sin = self.rotary_emb(hidden_states, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        
        # Compute attention scores
        attn_scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Convert attention mask to attention scores format
            mask = attention_mask[:, None, None, :]  # [batch_size, 1, 1, seq_len]
            attn_scores = attn_scores + (1.0 - mask) * -1e9
        
        # Apply softmax
        attn_weights = nn.softmax(attn_scores, axis=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = mx.matmul(attn_weights, v)
        
        # Transpose back to [batch_size, seq_len, num_heads, head_dim]
        attn_output = attn_output.transpose(0, 2, 1, 3)
        
        # Reshape to [batch_size, seq_len, hidden_size]
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        
        # Final projection
        attn_output = self.o_proj(attn_output)
        
        return attn_output, attn_weights if output_attentions else None
    
    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        output_attentions: bool = False,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        """Make the module callable."""
        return self.forward(hidden_states, attention_mask, position_ids, output_attentions)


# Utility functions for RoPE scaling
def get_rope_scaling_config(
    scaling_type: str = "linear",
    scaling_factor: float = 1.0,
    **kwargs
) -> dict:
    """
    Get RoPE scaling configuration.
    
    Args:
        scaling_type: Type of scaling ("linear", "dynamic", "ntk")
        scaling_factor: Scaling factor
        **kwargs: Additional scaling parameters
        
    Returns:
        RoPE scaling configuration dictionary
    """
    config = {
        "type": scaling_type,
        "factor": scaling_factor,
    }
    config.update(kwargs)
    return config


def apply_rope_scaling(
    inv_freq: mx.array,
    scaling_config: dict,
    max_position_embeddings: int
) -> mx.array:
    """
    Apply RoPE scaling to inverse frequencies.
    
    Args:
        inv_freq: Inverse frequency tensor
        scaling_config: Scaling configuration
        max_position_embeddings: Maximum position embeddings
        
    Returns:
        Scaled inverse frequencies
    """
    scaling_type = scaling_config.get("type", "linear")
    scaling_factor = scaling_config.get("factor", 1.0)
    
    if scaling_type == "linear":
        # Linear scaling: simply divide by scaling factor
        return inv_freq / scaling_factor
    elif scaling_type == "dynamic":
        # Dynamic scaling based on sequence length
        # This would require more complex implementation
        return inv_freq / scaling_factor
    elif scaling_type == "ntk":
        # NTK-aware scaling
        # This would require more complex implementation
        return inv_freq / scaling_factor
    else:
        raise ValueError(f"Unknown scaling type: {scaling_type}")