"""
Unified embeddings for BERT models.

This module contains all embedding-related components for both classic BERT
and ModernBERT models, including:
- Standard BERT embeddings with positional embeddings
- ModernBERT embeddings without positional embeddings (uses RoPE)
- Pooler components for both architectures
- Factory functions for creating appropriate embeddings
"""

import mlx.core as mx
import mlx.nn as nn

from ..config import BertConfig

# ============================================================================
# Standard BERT Embeddings
# ============================================================================


class BertEmbeddings(nn.Module):
    """BERT embeddings layer.

    Constructs embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            # padding_idx=config.pad_token_id if hasattr(config, 'pad_token_id') else None
        )

        # Position embeddings (learned)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

        # Token type embeddings (for NSP task - sentence A vs sentence B)
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )

        # Layer normalization
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Position IDs (1, len position emb) is contiguous in memory and exported when serialized
        self.position_ids = mx.arange(config.max_position_embeddings).reshape(1, -1)

    def forward(
        self,
        input_ids: mx.array,
        token_type_ids: mx.array | None = None,
        position_ids: mx.array | None = None,
    ) -> mx.array:
        """Forward pass through BERT embeddings.

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

        # Apply layer normalization
        embeddings = self.LayerNorm(embeddings)

        # Apply dropout
        embeddings = self.dropout(embeddings)

        return embeddings

    def __call__(
        self,
        input_ids: mx.array,
        token_type_ids: mx.array | None = None,
        position_ids: mx.array | None = None,
    ) -> mx.array:
        """Make the embeddings layer callable."""
        return self.forward(input_ids, token_type_ids, position_ids)


class BertPooler(nn.Module):
    """BERT pooler layer.

    We "pool" the model by simply taking the hidden state corresponding
    to the first token.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config

        # Dense layer for pooling
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        # Activation function
        self.activation = nn.Tanh()

    def forward(self, hidden_states: mx.array) -> mx.array:
        """Pool the [CLS] token representation.

        Args:
            hidden_states: Hidden states from encoder [batch_size, seq_len, hidden_size]

        Returns:
            Pooled representation [batch_size, hidden_size]
        """
        # Extract [CLS] token representation (first token)
        first_token_tensor = hidden_states[:, 0, :]

        # Apply linear transformation and activation
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)

        return pooled_output

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Make the pooler callable."""
        return self.forward(hidden_states)


# ============================================================================
# ModernBERT Embeddings (without positional embeddings)
# ============================================================================


class ModernBertEmbeddings(nn.Module):
    """
    ModernBERT embeddings layer.

    Key differences from Classic BERT:
    - No learned positional embeddings (RoPE is used instead)
    - Additional normalization layer after embeddings
    - Streamlined architecture without bias terms
    - Support for extended vocabulary size
    """

    def __init__(self, config):
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
            config.type_vocab_size, config.hidden_size
        )

        # Note: No position embeddings - RoPE is used instead

        # Layer normalization after embeddings (Classic BERT style)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Additional post-embedding normalization (ModernBERT improvement)
        if (
            hasattr(config, "use_post_embedding_norm")
            and config.use_post_embedding_norm
        ):
            self.post_embedding_norm = nn.LayerNorm(
                config.hidden_size,
                eps=getattr(config, "post_embedding_norm_eps", config.layer_norm_eps),
            )
        else:
            self.post_embedding_norm = None

        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids: mx.array,
        token_type_ids: mx.array | None = None,
        position_ids: mx.array | None = None,  # Not used in ModernBERT
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
        token_type_ids: mx.array | None = None,
        position_ids: mx.array | None = None,
    ) -> mx.array:
        """Make the embeddings layer callable."""
        return self.forward(input_ids, token_type_ids, position_ids)


class ModernBertPooler(nn.Module):
    """
    ModernBERT pooler layer.

    Similar to Classic BERT pooler but with streamlined architecture
    and optional bias terms.
    """

    def __init__(self, config):
        """
        Initialize ModernBERT pooler.

        Args:
            config: ModernBERT configuration
        """
        super().__init__()
        self.config = config

        # Dense layer for pooling
        pooler_hidden_size = getattr(config, "pooler_hidden_size", config.hidden_size)
        use_bias = getattr(config, "use_bias", True)

        self.dense = nn.Linear(
            config.hidden_size,
            pooler_hidden_size,
            bias=use_bias,
        )

        # Activation function
        pooler_activation = getattr(config, "pooler_activation", "tanh")
        if pooler_activation == "tanh":
            self.activation = nn.Tanh()
        elif pooler_activation == "gelu":
            self.activation = nn.GELU()
        elif pooler_activation == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Tanh()  # Default

        # Dropout
        pooler_dropout = getattr(config, "pooler_dropout", 0.0)
        self.dropout = nn.Dropout(pooler_dropout)

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


# ============================================================================
# Embedding Factory Functions
# ============================================================================


def create_embeddings(config) -> BertEmbeddings | ModernBertEmbeddings:
    """
    Create appropriate embeddings based on configuration.

    Args:
        config: Model configuration

    Returns:
        Embeddings layer
    """
    # Check if this is ModernBERT (no positional embeddings)
    if hasattr(config, "use_rope") and config.use_rope:
        return ModernBertEmbeddings(config)
    else:
        return BertEmbeddings(config)


def create_pooler(config) -> BertPooler | ModernBertPooler:
    """
    Create appropriate pooler based on configuration.

    Args:
        config: Model configuration

    Returns:
        Pooler layer
    """
    # Check if this is ModernBERT
    if hasattr(config, "use_rope") and config.use_rope:
        return ModernBertPooler(config)
    else:
        return BertPooler(config)


# ============================================================================
# Utility Functions
# ============================================================================


def get_embedding_size(config) -> int:
    """Get the embedding size for the model."""
    return config.hidden_size


def get_vocab_size(config) -> int:
    """Get the vocabulary size for the model."""
    return config.vocab_size


def get_max_sequence_length(config) -> int:
    """Get the maximum sequence length for the model."""
    return config.max_position_embeddings
