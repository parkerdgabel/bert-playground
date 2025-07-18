"""
ModernBERT embeddings implementation.

This module implements the enhanced embedding layer for ModernBERT,
including additional normalization and streamlined architecture.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional

from .modernbert_config import ModernBertConfig
from .activations import get_normalization_function


class ModernBertEmbeddings(nn.Module):
    """
    ModernBERT embeddings layer.
    
    Key differences from Classic BERT:
    - No learned positional embeddings (RoPE is used instead)
    - Additional normalization layer after embeddings
    - Streamlined architecture without bias terms
    - Support for extended vocabulary size
    """
    
    def __init__(self, config: ModernBertConfig):
        """
        Initialize ModernBERT embeddings.
        
        Args:
            config: ModernBERT configuration
        """
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.word_embeddings = nn.Embedding(
            config.vocab_size, 
            config.hidden_size,
            # No padding_idx for ModernBERT
        )
        
        # Token type embeddings (for NSP task - sentence A vs sentence B)
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, 
            config.hidden_size
        )
        
        # Note: No position embeddings - RoPE is used instead
        
        # Layer normalization after embeddings (Classic BERT style)
        self.LayerNorm = get_normalization_function(
            norm_name="layer_norm",
            hidden_size=config.hidden_size,
            eps=config.layer_norm_eps,
        )
        
        # Additional post-embedding normalization (ModernBERT improvement)
        if config.use_post_embedding_norm:
            self.post_embedding_norm = get_normalization_function(
                norm_name="layer_norm",
                hidden_size=config.hidden_size,
                eps=config.post_embedding_norm_eps,
            )
        else:
            self.post_embedding_norm = None
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embedding weights."""
        # Initialize token embeddings with normal distribution
        std = self.config.initializer_range
        
        # Word embeddings
        nn.init.normal(self.word_embeddings.weight, std=std)
        
        # Token type embeddings
        nn.init.normal(self.token_type_embeddings.weight, std=std)
    
    def forward(
        self,
        input_ids: mx.array,
        token_type_ids: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,  # Not used in ModernBERT
    ) -> mx.array:
        """
        Forward pass through ModernBERT embeddings.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len] (ignored in ModernBERT)
            
        Returns:
            Embeddings [batch_size, seq_len, hidden_size]
        """
        input_shape = input_ids.shape
        seq_length = input_shape[1]
        
        # Get token type IDs
        if token_type_ids is None:
            token_type_ids = mx.zeros(input_shape, dtype=mx.int32)
        
        # Get token embeddings
        inputs_embeds = self.word_embeddings(input_ids)
        
        # Get token type embeddings
        token_type_embeds = self.token_type_embeddings(token_type_ids)
        
        # Combine embeddings (no positional embeddings in ModernBERT)
        embeddings = inputs_embeds + token_type_embeds
        
        # Apply layer normalization
        embeddings = self.LayerNorm(embeddings)
        
        # Apply post-embedding normalization if enabled
        if self.post_embedding_norm is not None:
            embeddings = self.post_embedding_norm(embeddings)
        
        # Apply dropout
        embeddings = self.dropout(embeddings)
        
        return embeddings
    
    def __call__(
        self,
        input_ids: mx.array,
        token_type_ids: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        """Make the embeddings layer callable."""
        return self.forward(input_ids, token_type_ids, position_ids)


class ModernBertPooler(nn.Module):
    """
    ModernBERT pooler layer.
    
    Similar to Classic BERT pooler but with streamlined architecture
    and optional bias terms.
    """
    
    def __init__(self, config: ModernBertConfig):
        """
        Initialize ModernBERT pooler.
        
        Args:
            config: ModernBERT configuration
        """
        super().__init__()
        self.config = config
        
        # Dense layer for pooling
        self.dense = nn.Linear(
            config.hidden_size,
            config.pooler_hidden_size or config.hidden_size,
            bias=config.use_bias,
        )
        
        # Activation function
        if config.pooler_activation == "tanh":
            self.activation = nn.Tanh()
        elif config.pooler_activation == "gelu":
            self.activation = nn.GELU()
        elif config.pooler_activation == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Tanh()  # Default
        
        # Dropout
        self.dropout = nn.Dropout(config.pooler_dropout)
    
    def forward(self, hidden_states: mx.array) -> mx.array:
        """
        Pool the [CLS] token representation.
        
        Args:
            hidden_states: Hidden states from encoder [batch_size, seq_len, hidden_size]
            
        Returns:
            Pooled representation [batch_size, pooler_hidden_size]
        """
        # Extract [CLS] token representation (first token)
        first_token_tensor = hidden_states[:, 0, :]  # [batch_size, hidden_size]
        
        # Apply linear transformation
        pooled_output = self.dense(first_token_tensor)
        
        # Apply activation
        pooled_output = self.activation(pooled_output)
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        return pooled_output
    
    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Make the pooler callable."""
        return self.forward(hidden_states)


class ModernBertEmbeddingOutput(nn.Module):
    """
    ModernBERT embedding output layer.
    
    This layer can be used for tasks that require embedding outputs,
    such as masked language modeling.
    """
    
    def __init__(self, config: ModernBertConfig):
        """
        Initialize ModernBERT embedding output.
        
        Args:
            config: ModernBERT configuration
        """
        super().__init__()
        self.config = config
        
        # Dense layer to project back to vocab size
        self.dense = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=config.use_bias,
        )
        
        # Layer normalization
        self.LayerNorm = get_normalization_function(
            norm_name="layer_norm",
            hidden_size=config.vocab_size,
            eps=config.layer_norm_eps,
        )
        
        # Tie weights with input embeddings if requested
        self.tie_weights = True  # Common practice in language models
    
    def forward(self, hidden_states: mx.array) -> mx.array:
        """
        Forward pass through embedding output.
        
        Args:
            hidden_states: Hidden states [batch_size, seq_len, hidden_size]
            
        Returns:
            Logits [batch_size, seq_len, vocab_size]
        """
        # Project to vocabulary size
        logits = self.dense(hidden_states)
        
        # Apply layer normalization
        logits = self.LayerNorm(logits)
        
        return logits
    
    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Make the embedding output callable."""
        return self.forward(hidden_states)


def create_modernbert_embeddings(config: ModernBertConfig) -> ModernBertEmbeddings:
    """
    Create ModernBERT embeddings from configuration.
    
    Args:
        config: ModernBERT configuration
        
    Returns:
        ModernBertEmbeddings instance
    """
    return ModernBertEmbeddings(config)


def create_modernbert_pooler(config: ModernBertConfig) -> ModernBertPooler:
    """
    Create ModernBERT pooler from configuration.
    
    Args:
        config: ModernBERT configuration
        
    Returns:
        ModernBertPooler instance
    """
    return ModernBertPooler(config)


# Utility functions for embeddings
def get_embedding_size(config: ModernBertConfig) -> int:
    """Get the embedding size for ModernBERT."""
    return config.hidden_size


def get_vocab_size(config: ModernBertConfig) -> int:
    """Get the vocabulary size for ModernBERT."""
    return config.vocab_size


def get_max_sequence_length(config: ModernBertConfig) -> int:
    """Get the maximum sequence length for ModernBERT."""
    return config.max_position_embeddings