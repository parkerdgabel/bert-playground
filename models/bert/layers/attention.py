"""
Unified attention mechanisms for BERT models.

This module contains all attention-related components for both classic BERT
and ModernBERT models, including:
- Standard BERT self-attention
- RoPE (Rotary Positional Embeddings) attention
- Local sliding window attention
- Global attention
- Alternating attention patterns

All implementations use the neural abstraction port for framework-agnostic operations.
"""

import math
from typing import Any

import numpy as np

from core.ports.neural import Module, NeuralBackend
from ..config import BertConfig

# Temporary workaround: Import backend-specific operations
# TODO: These should be added to the NeuralBackend protocol
try:
    import mlx.core as mx
    _BACKEND_OPS = mx
except ImportError:
    _BACKEND_OPS = None

# ============================================================================
# Helper Functions for Operations Not in NeuralBackend Protocol
# ============================================================================

def backend_cos(x: Any) -> Any:
    """Compute cosine of x using backend operations."""
    if _BACKEND_OPS is not None:
        return _BACKEND_OPS.cos(x)
    else:
        # Fallback to numpy-like behavior
        return np.cos(x)


def backend_sin(x: Any) -> Any:
    """Compute sine of x using backend operations."""
    if _BACKEND_OPS is not None:
        return _BACKEND_OPS.sin(x)
    else:
        # Fallback to numpy-like behavior
        return np.sin(x)


def backend_abs(x: Any) -> Any:
    """Compute absolute value of x using backend operations."""
    if _BACKEND_OPS is not None:
        return _BACKEND_OPS.abs(x)
    else:
        # Fallback to numpy-like behavior
        return np.abs(x)


def backend_take(array: Any, indices: Any, axis: int = 0) -> Any:
    """Take elements from array along axis using indices."""
    if _BACKEND_OPS is not None:
        return _BACKEND_OPS.take(array, indices, axis=axis)
    else:
        # Fallback to numpy-like behavior
        return np.take(array, indices, axis=axis)


def backend_astype(x: Any, dtype: str) -> Any:
    """Cast x to specified dtype."""
    if _BACKEND_OPS is not None:
        if dtype == "float32":
            return x.astype(_BACKEND_OPS.float32)
        elif dtype == "int32":
            return x.astype(_BACKEND_OPS.int32)
        else:
            return x.astype(dtype)
    else:
        # Fallback to numpy-like behavior
        return x.astype(dtype)


def backend_outer(a: Any, b: Any) -> Any:
    """Compute outer product of a and b."""
    if _BACKEND_OPS is not None:
        # MLX doesn't have outer, so we use broadcasting
        a_expanded = _BACKEND_OPS.expand_dims(a, -1)  # [..., 1]
        b_expanded = _BACKEND_OPS.expand_dims(b, 0)   # [1, ...]
        return a_expanded * b_expanded
    else:
        # Fallback to numpy-like behavior
        return np.outer(a, b)


# ============================================================================
# Standard BERT Attention Components
# ============================================================================


class BertSelfAttention(Module):
    """BERT self-attention mechanism.

    Implementation following the original BERT paper's multi-head attention.
    """

    def __init__(self, config: BertConfig, backend: NeuralBackend):
        super().__init__()
        self.config = config
        self.backend = backend

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Linear projections for Q, K, V
        self.query = backend.linear(config.hidden_size, self.all_head_size)
        self.key = backend.linear(config.hidden_size, self.all_head_size)
        self.value = backend.linear(config.hidden_size, self.all_head_size)

        # Dropout for attention probabilities
        self.dropout = backend.dropout(config.attention_probs_dropout_prob)

        # Scaling factor for attention
        self.scale = 1.0 / np.sqrt(self.attention_head_size)

        # Register modules
        self.add_module("query", self.query)
        self.add_module("key", self.key)
        self.add_module("value", self.value)
        self.add_module("dropout", self.dropout)

    def transpose_for_scores(self, x: Any) -> Any:
        """Transpose to get attention scores.

        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]

        Returns:
            Tensor [batch_size, num_heads, seq_len, head_size]
        """
        batch_size, seq_len, hidden_size = x.shape
        x = self.backend.reshape(
            x, (batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
        )
        # Transpose from [batch, seq, heads, head_size] to [batch, heads, seq, head_size]
        x = self.backend.transpose(x, 1, 2)
        return x

    def forward(
        self,
        hidden_states: Any,
        attention_mask: Any | None = None,
        output_attentions: bool = False,
    ) -> tuple[Any, Any | None]:
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
        # Transpose key from [batch, heads, seq, head_size] to [batch, heads, head_size, seq]
        key_layer_transposed = self.backend.transpose(key_layer, 2, 3)
        attention_scores = self.backend.matmul(query_layer, key_layer_transposed)
        attention_scores = attention_scores * self.scale

        # Apply attention mask if provided
        if attention_mask is not None:
            # Convert attention mask to attention scores format
            # attention_mask: [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
            attention_mask = self.backend.unsqueeze(attention_mask, 1)
            attention_mask = self.backend.unsqueeze(attention_mask, 1)

            # Apply mask (set masked positions to large negative value)
            attention_scores = attention_scores + (1.0 - attention_mask) * -1e9

        # Normalize attention scores to probabilities
        attention_probs = self.backend.softmax(attention_scores, dim=-1)

        # Apply dropout to attention probabilities
        attention_probs = self.dropout(attention_probs)

        # Apply attention to values
        context_layer = self.backend.matmul(attention_probs, value_layer)

        # Transpose back to original format
        context_layer = self.backend.transpose(context_layer, 1, 2)
        batch_size, seq_len, num_heads, head_size = context_layer.shape
        context_layer = self.backend.reshape(
            context_layer, (batch_size, seq_len, num_heads * head_size)
        )

        outputs = (context_layer, attention_probs if output_attentions else None)
        return outputs


class BertSelfOutput(Module):
    """BERT self-attention output processing.

    Applies linear transformation, dropout, and layer normalization.
    """

    def __init__(self, config: BertConfig, backend: NeuralBackend):
        super().__init__()
        self.backend = backend
        self.dense = backend.linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = backend.layer_norm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = backend.dropout(config.hidden_dropout_prob)

        # Register modules
        self.add_module("dense", self.dense)
        self.add_module("LayerNorm", self.LayerNorm)
        self.add_module("dropout", self.dropout)

    def forward(self, hidden_states: Any, input_tensor: Any) -> Any:
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


class BertAttention(Module):
    """BERT attention layer.

    Combines self-attention and output processing.
    """

    def __init__(self, config: BertConfig, backend: NeuralBackend):
        super().__init__()
        self.self = BertSelfAttention(config, backend)
        self.output = BertSelfOutput(config, backend)

        # Register modules
        self.add_module("self", self.self)
        self.add_module("output", self.output)

    def forward(
        self,
        hidden_states: Any,
        attention_mask: Any | None = None,
        output_attentions: bool = False,
    ) -> tuple[Any, Any | None]:
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


# ============================================================================
# RoPE (Rotary Positional Embeddings) Components
# ============================================================================


class RotaryEmbedding(Module):
    """
    Rotary Positional Embeddings (RoPE) implementation.

    RoPE encodes absolute positional information while naturally incorporating
    explicit relative position dependency in self-attention.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int,
        base: float,
        scaling_factor: float,
        backend: NeuralBackend,
    ):
        """
        Initialize RoPE embeddings.

        Args:
            dim: Dimension of embeddings (should be head_dim)
            max_position_embeddings: Maximum sequence length
            base: Base frequency for rotary embeddings
            scaling_factor: Scaling factor for frequency adjustment
            backend: Neural backend for operations
        """
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        self.backend = backend

        # Pre-compute frequency inverse
        self.inv_freq = self._compute_inv_freq()

        # Pre-compute cos and sin values for efficiency
        self._cos_cached = None
        self._sin_cached = None
        self._cached_seq_len = 0

    def _compute_inv_freq(self) -> Any:
        """Compute inverse frequencies for RoPE."""
        # Create frequency inverse: 1 / (base^(2i/dim)) for i in [0, dim/2)
        exponent = self.backend.arange(0, self.dim, 2, dtype="float32")
        inv_freq = 1.0 / (self.base ** (exponent / self.dim))

        # Apply scaling if needed
        if self.scaling_factor != 1.0:
            inv_freq = inv_freq / self.scaling_factor

        return inv_freq

    def _compute_cos_sin(self, seq_len: int) -> tuple[Any, Any]:
        """
        Compute cosine and sine values for rotary embeddings.

        Args:
            seq_len: Sequence length

        Returns:
            Tuple of (cos, sin) arrays
        """
        # Create position tensor
        positions = self.backend.arange(seq_len, dtype="float32")

        # Compute frequency matrix: positions * inv_freq
        # We need to create outer product manually
        positions_expanded = self.backend.unsqueeze(positions, -1)  # [seq_len, 1]
        inv_freq_expanded = self.backend.unsqueeze(self.inv_freq, 0)  # [1, dim/2]
        freqs = positions_expanded * inv_freq_expanded  # [seq_len, dim/2]

        # Duplicate frequencies for cos and sin
        # freqs: [seq_len, dim/2] -> [seq_len, dim]
        freqs_doubled = self.backend.concat([freqs, freqs], dim=-1)

        # Compute cos and sin using backend operations
        cos = backend_cos(freqs_doubled)
        sin = backend_sin(freqs_doubled)

        return cos, sin

    def forward(
        self, x: Any, seq_len: int | None = None
    ) -> tuple[Any, Any]:
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
            # Slice cached values to current sequence length
            return self._cos_cached[:seq_len], self._sin_cached[:seq_len]

        # Compute new cos and sin values
        cos, sin = self._compute_cos_sin(seq_len)

        # Cache for future use
        self._cos_cached = cos
        self._sin_cached = sin
        self._cached_seq_len = seq_len

        return cos, sin


def rotate_half(x: Any, backend: NeuralBackend) -> Any:
    """
    Rotate half the dimensions of the input tensor.

    This function rotates the last dimension by splitting it in half
    and swapping the halves with a sign flip.

    Args:
        x: Input tensor [..., dim]
        backend: Neural backend for operations

    Returns:
        Rotated tensor [..., dim]
    """
    # Split the last dimension in half
    x1, x2 = backend.split(x, 2, dim=-1)

    # Rotate: [-x2, x1]
    neg_x2 = -x2
    return backend.concat([neg_x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: Any,
    k: Any,
    cos: Any,
    sin: Any,
    backend: NeuralBackend,
    position_ids: Any | None = None,
) -> tuple[Any, Any]:
    """
    Apply rotary positional embeddings to query and key tensors.

    Args:
        q: Query tensor [batch_size, num_heads, seq_len, head_dim]
        k: Key tensor [batch_size, num_heads, seq_len, head_dim]
        cos: Cosine values [seq_len, head_dim]
        sin: Sine values [seq_len, head_dim]
        backend: Neural backend for operations
        position_ids: Optional position IDs for advanced positioning

    Returns:
        Tuple of (rotated_q, rotated_k)
    """
    # Handle position_ids if provided
    if position_ids is not None:
        # Gather cos and sin values for specific positions
        cos = backend_take(cos, position_ids, axis=0)
        sin = backend_take(sin, position_ids, axis=0)

    # Ensure cos and sin have the right number of dimensions for broadcasting
    if len(cos.shape) == 1:
        # If cos/sin are 1D (head_dim only), expand to [1, head_dim]
        cos = backend.unsqueeze(cos, 0)
        sin = backend.unsqueeze(sin, 0)

    if len(cos.shape) == 2:
        # cos, sin: [seq_len, head_dim] -> [1, 1, seq_len, head_dim]
        cos = backend.unsqueeze(cos, 0)
        cos = backend.unsqueeze(cos, 0)
        sin = backend.unsqueeze(sin, 0)
        sin = backend.unsqueeze(sin, 0)
    else:
        # Handle unexpected shapes
        raise ValueError(f"Unexpected cos shape: {cos.shape}, expected 1D or 2D")

    # Apply rotary embeddings
    # q_embed = q * cos + rotate_half(q) * sin
    q_rotated = rotate_half(q, backend)
    q_embed = (q * cos) + (q_rotated * sin)

    k_rotated = rotate_half(k, backend)
    k_embed = (k * cos) + (k_rotated * sin)

    return q_embed, k_embed


class RoPEAttention(Module):
    """
    Self-attention with Rotary Positional Embeddings.

    This is a modified version of the standard attention mechanism
    that incorporates RoPE for better positional understanding.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        max_position_embeddings: int,
        rope_base: float,
        attention_dropout: float,
        use_bias: bool,
        backend: NeuralBackend,
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
            backend: Neural backend for operations
        """
        super().__init__()

        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_attention_heads ({num_attention_heads})"
            )

        self.backend = backend
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Linear projections
        self.q_proj = backend.linear(hidden_size, hidden_size, bias=use_bias)
        self.k_proj = backend.linear(hidden_size, hidden_size, bias=use_bias)
        self.v_proj = backend.linear(hidden_size, hidden_size, bias=use_bias)
        self.o_proj = backend.linear(hidden_size, hidden_size, bias=use_bias)

        # RoPE embeddings
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_base,
            scaling_factor=1.0,
            backend=backend,
        )

        # Dropout
        self.dropout = backend.dropout(attention_dropout)

        # Register modules
        self.add_module("q_proj", self.q_proj)
        self.add_module("k_proj", self.k_proj)
        self.add_module("v_proj", self.v_proj)
        self.add_module("o_proj", self.o_proj)
        self.add_module("rotary_emb", self.rotary_emb)
        self.add_module("dropout", self.dropout)

    def forward(
        self,
        hidden_states: Any,
        attention_mask: Any | None = None,
        position_ids: Any | None = None,
        output_attentions: bool = False,
    ) -> tuple[Any, Any | None]:
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
        q = self.backend.reshape(q, (batch_size, seq_len, self.num_attention_heads, self.head_dim))
        k = self.backend.reshape(k, (batch_size, seq_len, self.num_attention_heads, self.head_dim))
        v = self.backend.reshape(v, (batch_size, seq_len, self.num_attention_heads, self.head_dim))

        # Transpose to [batch_size, num_heads, seq_len, head_dim]
        q = self.backend.transpose(q, 1, 2)
        k = self.backend.transpose(k, 1, 2)
        v = self.backend.transpose(v, 1, 2)

        # Apply RoPE
        cos, sin = self.rotary_emb(hidden_states, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, self.backend, position_ids)

        # Compute attention scores
        k_transposed = self.backend.transpose(k, 2, 3)
        attn_scores = self.backend.matmul(q, k_transposed) * self.scale

        # Apply attention mask if provided
        if attention_mask is not None:
            # Convert attention mask to attention scores format
            mask = self.backend.unsqueeze(attention_mask, 1)
            mask = self.backend.unsqueeze(mask, 1)
            attn_scores = attn_scores + (1.0 - mask) * -1e9

        # Apply softmax
        attn_weights = self.backend.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = self.backend.matmul(attn_weights, v)

        # Transpose back to [batch_size, seq_len, num_heads, head_dim]
        attn_output = self.backend.transpose(attn_output, 1, 2)

        # Reshape to [batch_size, seq_len, hidden_size]
        attn_output = self.backend.reshape(attn_output, (batch_size, seq_len, self.hidden_size))

        # Final projection
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights if output_attentions else None


# ============================================================================
# Local and Global Attention Components
# ============================================================================


class LocalAttention(Module):
    """
    Local sliding window attention mechanism.

    This attention mechanism only attends to tokens within a local window,
    reducing computational complexity from O(n²) to O(n*w) where w is window size.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        window_size: int,
        max_position_embeddings: int,
        rope_base: float,
        attention_dropout: float,
        use_bias: bool,
        backend: NeuralBackend,
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
            backend: Neural backend for operations
        """
        super().__init__()

        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_attention_heads ({num_attention_heads})"
            )

        self.backend = backend
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.window_size = window_size
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Linear projections
        self.q_proj = backend.linear(hidden_size, hidden_size, bias=use_bias)
        self.k_proj = backend.linear(hidden_size, hidden_size, bias=use_bias)
        self.v_proj = backend.linear(hidden_size, hidden_size, bias=use_bias)
        self.o_proj = backend.linear(hidden_size, hidden_size, bias=use_bias)

        # RoPE embeddings
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_base,
            scaling_factor=1.0,
            backend=backend,
        )

        # Dropout
        self.dropout = backend.dropout(attention_dropout)

        # Register modules
        self.add_module("q_proj", self.q_proj)
        self.add_module("k_proj", self.k_proj)
        self.add_module("v_proj", self.v_proj)
        self.add_module("o_proj", self.o_proj)
        self.add_module("rotary_emb", self.rotary_emb)
        self.add_module("dropout", self.dropout)

    def _create_local_attention_mask(
        self, seq_len: int, attention_mask: Any | None = None
    ) -> Any:
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
        i = self.backend.arange(seq_len)
        i = self.backend.unsqueeze(i, -1)  # [seq_len, 1]
        j = self.backend.arange(seq_len)
        j = self.backend.unsqueeze(j, 0)  # [1, seq_len]

        # Create local mask: |i - j| <= half_window
        diff = i - j
        local_mask = backend_abs(diff) <= half_window
        local_mask = backend_astype(local_mask, "float32")

        # Apply input attention mask if provided
        if attention_mask is not None:
            # attention_mask: [batch_size, seq_len] -> [batch_size, seq_len, seq_len]
            batch_size = attention_mask.shape[0]
            mask_expanded_1 = self.backend.unsqueeze(attention_mask, 1)
            mask_expanded_2 = self.backend.unsqueeze(attention_mask, 2)
            input_mask = mask_expanded_1 * mask_expanded_2

            # Broadcast local mask to batch size
            local_mask = self.backend.unsqueeze(local_mask, 0)
            local_mask = self.backend.broadcast_to(
                local_mask, (batch_size, seq_len, seq_len)
            )

            # Combine masks
            local_mask = local_mask * input_mask

        return local_mask

    def forward(
        self,
        hidden_states: Any,
        attention_mask: Any | None = None,
        position_ids: Any | None = None,
        output_attentions: bool = False,
    ) -> tuple[Any, Any | None]:
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
        q = self.backend.reshape(q, (batch_size, seq_len, self.num_attention_heads, self.head_dim))
        k = self.backend.reshape(k, (batch_size, seq_len, self.num_attention_heads, self.head_dim))
        v = self.backend.reshape(v, (batch_size, seq_len, self.num_attention_heads, self.head_dim))

        # Transpose to [batch_size, num_heads, seq_len, head_dim]
        q = self.backend.transpose(q, 1, 2)
        k = self.backend.transpose(k, 1, 2)
        v = self.backend.transpose(v, 1, 2)

        # Apply RoPE
        cos, sin = self.rotary_emb(hidden_states, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, self.backend, position_ids)

        # Compute attention scores
        k_transposed = self.backend.transpose(k, 2, 3)
        attn_scores = self.backend.matmul(q, k_transposed) * self.scale

        # Create and apply local attention mask
        local_mask = self._create_local_attention_mask(seq_len, attention_mask)

        # Expand mask for multi-head attention
        if len(local_mask.shape) == 2:
            local_mask = self.backend.unsqueeze(local_mask, 0)
            local_mask = self.backend.unsqueeze(local_mask, 0)
        elif len(local_mask.shape) == 3:
            local_mask = self.backend.unsqueeze(local_mask, 1)

        # Apply mask
        attn_scores = attn_scores + (1.0 - local_mask) * -1e9

        # Apply softmax
        attn_weights = self.backend.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = self.backend.matmul(attn_weights, v)

        # Transpose back to [batch_size, seq_len, num_heads, head_dim]
        attn_output = self.backend.transpose(attn_output, 1, 2)

        # Reshape to [batch_size, seq_len, hidden_size]
        attn_output = self.backend.reshape(attn_output, (batch_size, seq_len, self.hidden_size))

        # Final projection
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights if output_attentions else None


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
        max_position_embeddings: int,
        rope_base: float,
        attention_dropout: float,
        use_bias: bool,
        backend: NeuralBackend,
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
            backend: Neural backend for operations
        """
        super().__init__(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            rope_base=rope_base,
            attention_dropout=attention_dropout,
            use_bias=use_bias,
            backend=backend,
        )


class AlternatingAttention(Module):
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
        global_attention_every_n_layers: int,
        local_attention_window: int,
        max_position_embeddings: int,
        rope_base: float,
        attention_dropout: float,
        use_bias: bool,
        backend: NeuralBackend,
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
            backend: Neural backend for operations
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
                backend=backend,
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
                backend=backend,
            )

        # Register module
        self.add_module("attention", self.attention)

    def forward(
        self,
        hidden_states: Any,
        attention_mask: Any | None = None,
        position_ids: Any | None = None,
        output_attentions: bool = False,
    ) -> tuple[Any, Any | None]:
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

    def get_attention_type(self) -> str:
        """Get the attention type for this layer."""
        return "global" if self.is_global_layer else "local"


# ============================================================================
# Attention Factory Functions
# ============================================================================


def create_attention_layer(
    hidden_size: int, 
    num_attention_heads: int, 
    layer_idx: int, 
    config, 
    backend: NeuralBackend,
    **kwargs
) -> Module:
    """
    Create appropriate attention layer based on configuration.

    Args:
        hidden_size: Hidden size of the model
        num_attention_heads: Number of attention heads
        layer_idx: Index of the current layer
        config: Model configuration
        backend: Neural backend for operations
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
            backend=backend,
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
            backend=backend,
            **kwargs,
        )
    else:
        # Use standard BERT attention
        return BertAttention(config, backend)