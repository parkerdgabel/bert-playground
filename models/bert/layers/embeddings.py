"""
BERT embeddings layer.

This module contains the embeddings components for BERT models,
including token, position, and token type embeddings.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional

from ..config import BertConfig


class BertEmbeddings(nn.Module):
    """BERT embeddings layer.
    
    This layer combines token embeddings, position embeddings, and token type embeddings,
    applies layer normalization and dropout as per the original BERT architecture.
    """
    
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Position embeddings (learned, not sinusoidal like in Transformer)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Token type embeddings (for NSP task - sentence A vs sentence B)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        
        # Layer normalization and dropout
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Register position_ids as a buffer (similar to PyTorch implementation)
        self.register_buffer_persistent = False
        self.position_ids = mx.arange(config.max_position_embeddings)[None, :]  # [1, max_position_embeddings]
    
    def forward(
        self,
        input_ids: mx.array,
        token_type_ids: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        """Forward pass through embeddings.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len]
            
        Returns:
            Embeddings [batch_size, seq_len, hidden_size]
        """
        input_shape = input_ids.shape
        seq_length = input_shape[1]
        
        # Get position IDs
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        
        # Get token type IDs
        if token_type_ids is None:
            token_type_ids = mx.zeros(input_shape, dtype=mx.int32)
        
        # Get embeddings
        inputs_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)
        
        # Combine embeddings
        embeddings = inputs_embeds + position_embeds + token_type_embeds
        
        # Apply layer normalization and dropout
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings
    
    def __call__(self, input_ids: mx.array, token_type_ids: Optional[mx.array] = None, position_ids: Optional[mx.array] = None) -> mx.array:
        """Make the embeddings layer callable."""
        return self.forward(input_ids, token_type_ids, position_ids)


class BertPooler(nn.Module):
    """BERT pooler layer.
    
    This layer pools the sequence representation into a single vector by taking
    the hidden state corresponding to the first token ([CLS]) and applying a
    linear transformation and tanh activation.
    """
    
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
    
    def forward(self, hidden_states: mx.array) -> mx.array:
        """Forward pass through pooler.
        
        Args:
            hidden_states: Hidden states [batch_size, seq_len, hidden_size]
            
        Returns:
            Pooled output [batch_size, hidden_size]
        """
        # Take the hidden state corresponding to the first token (CLS)
        first_token_tensor = hidden_states[:, 0]
        
        # Apply linear transformation and activation
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        
        return pooled_output
    
    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Make the pooler layer callable."""
        return self.forward(hidden_states)