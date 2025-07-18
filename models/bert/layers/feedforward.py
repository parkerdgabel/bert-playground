"""
BERT feedforward network components.

This module contains the feedforward network layers for BERT models,
including the intermediate layer and output layer.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional

from ..config import BertConfig


class BertIntermediate(nn.Module):
    """BERT intermediate layer (first part of FFN).
    
    Applies linear transformation and GELU activation.
    """
    
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        
        # Use GELU activation as per BERT paper
        self.intermediate_act_fn = nn.GELU()
    
    def forward(self, hidden_states: mx.array) -> mx.array:
        """Forward pass through intermediate layer.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            
        Returns:
            Output tensor [batch_size, seq_len, intermediate_size]
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
    
    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Make the intermediate layer callable."""
        return self.forward(hidden_states)


class BertOutput(nn.Module):
    """BERT FFN output layer (second part of FFN).
    
    Applies linear transformation, dropout, and layer normalization.
    Note: This class is named BertOutput to match the original BERT naming,
    but it's specifically for the FFN output (distinct from model output).
    """
    
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, hidden_states: mx.array, input_tensor: mx.array) -> mx.array:
        """Forward pass through FFN output layer.
        
        Args:
            hidden_states: Output from intermediate layer [batch_size, seq_len, intermediate_size]
            input_tensor: Input tensor for residual connection [batch_size, seq_len, hidden_size]
            
        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    
    def __call__(self, hidden_states: mx.array, input_tensor: mx.array) -> mx.array:
        """Make the FFN output layer callable."""
        return self.forward(hidden_states, input_tensor)


class BertFeedForward(nn.Module):
    """Complete BERT feedforward network.
    
    Combines the intermediate layer and output layer into a single module.
    This is a convenience wrapper that encapsulates the full FFN.
    """
    
    def __init__(self, config: BertConfig):
        super().__init__()
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
    
    def forward(self, hidden_states: mx.array) -> mx.array:
        """Forward pass through complete FFN.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            
        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        input_tensor = hidden_states
        
        # First part: hidden_size -> intermediate_size with GELU
        hidden_states = self.intermediate(hidden_states)
        
        # Second part: intermediate_size -> hidden_size with residual connection
        hidden_states = self.output(hidden_states, input_tensor)
        
        return hidden_states
    
    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Make the feedforward network callable."""
        return self.forward(hidden_states)