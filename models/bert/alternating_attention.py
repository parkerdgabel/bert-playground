"""
Alternating Attention mechanism for ModernBERT.

This module implements the alternating global/local attention pattern
used in ModernBERT for improved efficiency on long sequences.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple
import math

from .rope import RoPEAttention, apply_rotary_pos_emb


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
        from .rope import RotaryEmbedding
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_base,
        )
        
        # Dropout
        self.dropout = nn.Dropout(attention_dropout)
    
    def _create_local_attention_mask(
        self, 
        seq_len: int, 
        attention_mask: Optional[mx.array] = None
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
            local_mask = mx.broadcast_to(local_mask[None, :, :], (batch_size, seq_len, seq_len))
            
            # Combine masks
            local_mask = local_mask * input_mask
        
        return local_mask
    
    def forward(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        output_attentions: bool = False,
    ) -> Tuple[mx.array, Optional[mx.array]]:
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
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        output_attentions: bool = False,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        """Make the module callable."""
        return self.forward(hidden_states, attention_mask, position_ids, output_attentions)


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
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        output_attentions: bool = False,
    ) -> Tuple[mx.array, Optional[mx.array]]:
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
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        output_attentions: bool = False,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        """Make the module callable."""
        return self.forward(hidden_states, attention_mask, position_ids, output_attentions)
    
    def get_attention_type(self) -> str:
        """Get the attention type for this layer."""
        return "global" if self.is_global_layer else "local"


def create_attention_layer(
    hidden_size: int,
    num_attention_heads: int,
    layer_idx: int,
    config,
    **kwargs
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
    if config.use_alternating_attention:
        return AlternatingAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            layer_idx=layer_idx,
            global_attention_every_n_layers=config.global_attention_every_n_layers,
            local_attention_window=config.local_attention_window,
            max_position_embeddings=config.max_position_embeddings,
            rope_base=config.rope_base,
            attention_dropout=config.attention_probs_dropout_prob,
            use_bias=config.use_bias,
            **kwargs
        )
    else:
        # Use global attention for all layers
        return GlobalAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            rope_base=config.rope_base,
            attention_dropout=config.attention_probs_dropout_prob,
            use_bias=config.use_bias,
            **kwargs
        )