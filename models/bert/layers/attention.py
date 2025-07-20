"""
Unified attention mechanisms for BERT models.

This module contains all attention-related components for both classic BERT
and ModernBERT models, including:
- Standard BERT self-attention
- RoPE (Rotary Positional Embeddings) attention
- Local sliding window attention
- Global attention
- Alternating attention patterns
"""

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..config import BertConfig

# ============================================================================
# Standard BERT Attention Components
# ============================================================================


class BertSelfAttention(nn.Module):
    """BERT self-attention mechanism.

    Implementation following the original BERT paper's multi-head attention.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Linear projections for Q, K, V
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # Dropout for attention probabilities
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # Scaling factor for attention
        self.scale = 1.0 / np.sqrt(self.attention_head_size)

    def transpose_for_scores(self, x: mx.array) -> mx.array:
        """Transpose to get attention scores.

        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]

        Returns:
            Tensor [batch_size, num_heads, seq_len, head_size]
        """
        batch_size, seq_len, hidden_size = x.shape
        x = x.reshape(
            batch_size, seq_len, self.num_attention_heads, self.attention_head_size
        )
        return x.transpose(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_size]

    def forward(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        output_attentions: bool = False,
    ) -> tuple[mx.array, mx.array | None]:
        """Forward pass through self-attention.

        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            output_attentions: Whether to output attention weights

        Returns:
            Tuple of (context_layer, attention_probs)
        """
        # Linear projections
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # Transpose for attention computation
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Compute attention scores
        attention_scores = mx.matmul(query_layer, key_layer.transpose(0, 1, 3, 2))
        attention_scores = attention_scores * self.scale

        # Apply attention mask if provided
        if attention_mask is not None:
            # Convert attention mask to attention scores format
            # attention_mask: [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
            attention_mask = attention_mask[:, None, None, :]

            # Apply mask (set masked positions to large negative value)
            attention_scores = attention_scores + (1.0 - attention_mask) * -1e9

        # Normalize attention scores to probabilities
        attention_probs = nn.softmax(attention_scores, axis=-1)

        # Apply dropout to attention probabilities
        attention_probs = self.dropout(attention_probs)

        # Apply attention to values
        context_layer = mx.matmul(attention_probs, value_layer)

        # Transpose back to original format
        context_layer = context_layer.transpose(0, 2, 1, 3)
        batch_size, seq_len, num_heads, head_size = context_layer.shape
        context_layer = context_layer.reshape(
            batch_size, seq_len, num_heads * head_size
        )

        outputs = (context_layer, attention_probs if output_attentions else None)
        return outputs

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        output_attentions: bool = False,
    ) -> tuple[mx.array, mx.array | None]:
        """Make the attention layer callable."""
        return self.forward(hidden_states, attention_mask, output_attentions)


class BertSelfOutput(nn.Module):
    """BERT self-attention output processing.

    Applies linear transformation, dropout, and layer normalization.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: mx.array, input_tensor: mx.array) -> mx.array:
        """Forward pass through self-attention output.

        Args:
            hidden_states: Output from self-attention [batch_size, seq_len, hidden_size]
            input_tensor: Input tensor for residual connection [batch_size, seq_len, hidden_size]

        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

    def __call__(self, hidden_states: mx.array, input_tensor: mx.array) -> mx.array:
        """Make the output layer callable."""
        return self.forward(hidden_states, input_tensor)


class BertAttention(nn.Module):
    """BERT attention layer.

    Combines self-attention and output processing.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        output_attentions: bool = False,
    ) -> tuple[mx.array, mx.array | None]:
        """Forward pass through attention.

        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            output_attentions: Whether to output attention weights

        Returns:
            Tuple of (attention_output, attention_probs)
        """
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            output_attentions,
        )

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[
            1:
        ]  # Add attentions if we output them
        return outputs

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        output_attentions: bool = False,
    ) -> tuple[mx.array, mx.array | None]:
        """Make the attention layer callable."""
        return self.forward(hidden_states, attention_mask, output_attentions)


# ============================================================================
# RoPE (Rotary Positional Embeddings) Components
# ============================================================================


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

    def _compute_cos_sin(self, seq_len: int) -> tuple[mx.array, mx.array]:
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

    def forward(
        self, x: mx.array, seq_len: int | None = None
    ) -> tuple[mx.array, mx.array]:
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
        if (
            self._cos_cached is not None
            and self._sin_cached is not None
            and seq_len <= self._cached_seq_len
        ):
            return self._cos_cached[:seq_len], self._sin_cached[:seq_len]

        # Compute new cos and sin values
        cos, sin = self._compute_cos_sin(seq_len)

        # Cache for future use
        self._cos_cached = cos
        self._sin_cached = sin
        self._cached_seq_len = seq_len

        return cos, sin

    def __call__(
        self, x: mx.array, seq_len: int | None = None
    ) -> tuple[mx.array, mx.array]:
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
    position_ids: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
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

    # Ensure cos and sin have the right number of dimensions for broadcasting
    if len(cos.shape) == 1:
        # If cos/sin are 1D (head_dim only), expand to [1, head_dim]
        cos = cos[None, :]
        sin = sin[None, :]
    
    if len(cos.shape) == 2:
        # cos, sin: [seq_len, head_dim] -> [1, 1, seq_len, head_dim]
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
    else:
        # Handle unexpected shapes
        raise ValueError(f"Unexpected cos shape: {cos.shape}, expected 1D or 2D")

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
        attention_mask: mx.array | None = None,
        position_ids: mx.array | None = None,
        output_attentions: bool = False,
    ) -> tuple[mx.array, mx.array | None]:
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
        attention_mask: mx.array | None = None,
        position_ids: mx.array | None = None,
        output_attentions: bool = False,
    ) -> tuple[mx.array, mx.array | None]:
        """Make the module callable."""
        return self.forward(
            hidden_states, attention_mask, position_ids, output_attentions
        )


# ============================================================================
# Local and Global Attention Components
# ============================================================================


class LocalAttention(nn.Module):
    """
    Local sliding window attention mechanism.

    This attention mechanism only attends to tokens within a local window,
    reducing computational complexity from O(n²) to O(n*w) where w is window size.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        window_size: int = 128,
        max_position_embeddings: int = 8192,
        rope_base: float = 10000.0,
        attention_dropout: float = 0.0,
        use_bias: bool = False,
    ):
        """
        Initialize local attention.

        Args:
            hidden_size: Hidden size of the model
            num_attention_heads: Number of attention heads
            window_size: Size of the local attention window
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
        self.window_size = window_size
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

    def _create_local_attention_mask(
        self, seq_len: int, attention_mask: mx.array | None = None
    ) -> mx.array:
        """
        Create local attention mask for sliding window attention.

        Args:
            seq_len: Sequence length
            attention_mask: Optional input attention mask

        Returns:
            Local attention mask
        """
        # Create sliding window mask
        # Each position can attend to positions in [i - window_size//2, i + window_size//2]
        half_window = self.window_size // 2

        # Create position indices
        i = mx.arange(seq_len)[:, None]  # [seq_len, 1]
        j = mx.arange(seq_len)[None, :]  # [1, seq_len]

        # Create local mask: |i - j| <= half_window
        local_mask = mx.abs(i - j) <= half_window

        # Convert to attention mask format (1 for valid, 0 for masked)
        local_mask = local_mask.astype(mx.float32)

        # Apply input attention mask if provided
        if attention_mask is not None:
            # attention_mask: [batch_size, seq_len] -> [batch_size, seq_len, seq_len]
            batch_size = attention_mask.shape[0]
            input_mask = attention_mask[:, None, :] * attention_mask[:, :, None]

            # Broadcast local mask to batch size
            local_mask = mx.broadcast_to(
                local_mask[None, :, :], (batch_size, seq_len, seq_len)
            )

            # Combine masks
            local_mask = local_mask * input_mask

        return local_mask

    def forward(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        position_ids: mx.array | None = None,
        output_attentions: bool = False,
    ) -> tuple[mx.array, mx.array | None]:
        """
        Forward pass through local attention.

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

        # Create and apply local attention mask
        local_mask = self._create_local_attention_mask(seq_len, attention_mask)

        # Expand mask for multi-head attention
        if len(local_mask.shape) == 2:
            local_mask = local_mask[None, None, :, :]  # [1, 1, seq_len, seq_len]
        elif len(local_mask.shape) == 3:
            local_mask = local_mask[:, None, :, :]  # [batch_size, 1, seq_len, seq_len]

        # Apply mask
        attn_scores = attn_scores + (1.0 - local_mask) * -1e9

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
        attention_mask: mx.array | None = None,
        position_ids: mx.array | None = None,
        output_attentions: bool = False,
    ) -> tuple[mx.array, mx.array | None]:
        """Make the module callable."""
        return self.forward(
            hidden_states, attention_mask, position_ids, output_attentions
        )


class GlobalAttention(RoPEAttention):
    """
    Global attention mechanism with RoPE.

    This is essentially the same as RoPEAttention but with a clearer name
    to distinguish it from local attention in the alternating pattern.
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
        Initialize global attention.

        Args:
            hidden_size: Hidden size of the model
            num_attention_heads: Number of attention heads
            max_position_embeddings: Maximum sequence length
            rope_base: Base frequency for RoPE
            attention_dropout: Dropout rate for attention
            use_bias: Whether to use bias in linear layers
        """
        super().__init__(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            rope_base=rope_base,
            attention_dropout=attention_dropout,
            use_bias=use_bias,
        )


class AlternatingAttention(nn.Module):
    """
    Alternating attention mechanism for ModernBERT.

    This module alternates between global and local attention patterns
    to balance computational efficiency with modeling capability.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        layer_idx: int,
        global_attention_every_n_layers: int = 3,
        local_attention_window: int = 128,
        max_position_embeddings: int = 8192,
        rope_base: float = 10000.0,
        attention_dropout: float = 0.0,
        use_bias: bool = False,
    ):
        """
        Initialize alternating attention.

        Args:
            hidden_size: Hidden size of the model
            num_attention_heads: Number of attention heads
            layer_idx: Index of the current layer (0-based)
            global_attention_every_n_layers: Global attention every N layers
            local_attention_window: Size of local attention window
            max_position_embeddings: Maximum sequence length
            rope_base: Base frequency for RoPE
            attention_dropout: Dropout rate for attention
            use_bias: Whether to use bias in linear layers
        """
        super().__init__()

        self.layer_idx = layer_idx
        self.global_attention_every_n_layers = global_attention_every_n_layers

        # Determine attention type for this layer
        self.is_global_layer = (layer_idx + 1) % global_attention_every_n_layers == 0

        if self.is_global_layer:
            # Global attention
            self.attention = GlobalAttention(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                max_position_embeddings=max_position_embeddings,
                rope_base=rope_base,
                attention_dropout=attention_dropout,
                use_bias=use_bias,
            )
        else:
            # Local attention
            self.attention = LocalAttention(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                window_size=local_attention_window,
                max_position_embeddings=max_position_embeddings,
                rope_base=rope_base,
                attention_dropout=attention_dropout,
                use_bias=use_bias,
            )

    def forward(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        position_ids: mx.array | None = None,
        output_attentions: bool = False,
    ) -> tuple[mx.array, mx.array | None]:
        """
        Forward pass through alternating attention.

        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len]
            output_attentions: Whether to return attention weights

        Returns:
            Tuple of (output, attention_weights)
        """
        return self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        position_ids: mx.array | None = None,
        output_attentions: bool = False,
    ) -> tuple[mx.array, mx.array | None]:
        """Make the module callable."""
        return self.forward(
            hidden_states, attention_mask, position_ids, output_attentions
        )

    def get_attention_type(self) -> str:
        """Get the attention type for this layer."""
        return "global" if self.is_global_layer else "local"


# ============================================================================
# Attention Factory Functions
# ============================================================================


def create_attention_layer(
    hidden_size: int, num_attention_heads: int, layer_idx: int, config, **kwargs
) -> nn.Module:
    """
    Create appropriate attention layer based on configuration.

    Args:
        hidden_size: Hidden size of the model
        num_attention_heads: Number of attention heads
        layer_idx: Index of the current layer
        config: Model configuration
        **kwargs: Additional arguments

    Returns:
        Attention layer module
    """
    # Check if using alternating attention (ModernBERT)
    if (
        hasattr(config, "use_alternating_attention")
        and config.use_alternating_attention
    ):
        return AlternatingAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            layer_idx=layer_idx,
            global_attention_every_n_layers=getattr(
                config, "global_attention_every_n_layers", 3
            ),
            local_attention_window=getattr(config, "local_attention_window", 128),
            max_position_embeddings=getattr(config, "max_position_embeddings", 8192),
            rope_base=getattr(config, "rope_base", 10000.0),
            attention_dropout=getattr(config, "attention_probs_dropout_prob", 0.0),
            use_bias=getattr(config, "use_bias", False),
            **kwargs,
        )
    # Check if using RoPE (ModernBERT)
    elif hasattr(config, "use_rope") and config.use_rope:
        return RoPEAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            max_position_embeddings=getattr(config, "max_position_embeddings", 8192),
            rope_base=getattr(config, "rope_base", 10000.0),
            attention_dropout=getattr(config, "attention_probs_dropout_prob", 0.0),
            use_bias=getattr(config, "use_bias", False),
            **kwargs,
        )
    else:
        # Use standard BERT attention
        return BertAttention(config)
