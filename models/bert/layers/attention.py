"""
BERT attention mechanisms.

This module contains the attention-related components for BERT models,
including self-attention, multi-head attention, and output processing.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple
import numpy as np

from ..config import BertConfig


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
        x = x.reshape(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
        return x.transpose(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_size]
    
    def forward(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        output_attentions: bool = False,
    ) -> Tuple[mx.array, Optional[mx.array]]:
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
        context_layer = context_layer.reshape(batch_size, seq_len, num_heads * head_size)
        
        outputs = (context_layer, attention_probs if output_attentions else None)
        return outputs
    
    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        output_attentions: bool = False,
    ) -> Tuple[mx.array, Optional[mx.array]]:
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
        attention_mask: Optional[mx.array] = None,
        output_attentions: bool = False,
    ) -> Tuple[mx.array, Optional[mx.array]]:
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
        
        outputs = (attention_output,) + self_outputs[1:]  # Add attentions if we output them
        return outputs
    
    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        output_attentions: bool = False,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        """Make the attention layer callable."""
        return self.forward(hidden_states, attention_mask, output_attentions)