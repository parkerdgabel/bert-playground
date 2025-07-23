"""
Unified embeddings for BERT models using neural abstraction.

This module contains all embedding-related components for both classic BERT
and ModernBERT models, implemented using the framework-agnostic neural port:
- Standard BERT embeddings with positional embeddings
- ModernBERT embeddings without positional embeddings (uses RoPE)
- Pooler components for both architectures
- Factory functions for creating appropriate embeddings

The implementation is framework-agnostic and can work with MLX, PyTorch, or JAX backends.
"""

from typing import Optional

from core.ports.compute import Array
from core.ports.neural import Module, NeuralBackend, ActivationType
from core.ports.neural_types import EmbeddingConfig
from ..config import BertConfig

# ============================================================================
# Standard BERT Embeddings
# ============================================================================


class BertEmbeddings(Module):
    """BERT embeddings layer using neural abstraction.

    Constructs embeddings from word, position and token_type embeddings.
    Framework-agnostic implementation.
    """

    def __init__(self, config: BertConfig, backend: NeuralBackend):
        super().__init__()
        self.config = config
        self.backend = backend

        # Token embeddings
        self.word_embeddings = backend.embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=getattr(config, 'pad_token_id', None)
        )

        # Position embeddings (learned)
        self.position_embeddings = backend.embedding(
            num_embeddings=config.max_position_embeddings,
            embedding_dim=config.hidden_size
        )

        # Token type embeddings (for NSP task - sentence A vs sentence B)
        self.token_type_embeddings = backend.embedding(
            num_embeddings=config.type_vocab_size,
            embedding_dim=config.hidden_size
        )

        # Layer normalization
        self.layer_norm = backend.layer_norm(
            normalized_shape=config.hidden_size,
            eps=config.layer_norm_eps
        )

        # Dropout
        self.dropout = backend.dropout(p=config.hidden_dropout_prob)

        # Register submodules
        self.add_module("word_embeddings", self.word_embeddings)
        self.add_module("position_embeddings", self.position_embeddings)
        self.add_module("token_type_embeddings", self.token_type_embeddings)
        self.add_module("layer_norm", self.layer_norm)
        self.add_module("dropout", self.dropout)

        # Position IDs will be created during forward pass
        self._max_position_embeddings = config.max_position_embeddings

    def forward(
        self,
        input_ids: Array,
        token_type_ids: Optional[Array] = None,
        position_ids: Optional[Array] = None,
    ) -> Array:
        """Forward pass through BERT embeddings.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len]

        Returns:
            Embeddings [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_length = input_ids.shape

        # Get position IDs
        if position_ids is None:
            # Create position IDs [0, 1, 2, ..., seq_length-1]
            position_ids = self.backend.arange(seq_length)
            # Expand to match batch size [batch_size, seq_length]
            position_ids = self.backend.unsqueeze(position_ids, 0)
            position_ids = self.backend.broadcast_to(
                position_ids, (batch_size, seq_length)
            )

        # Get token type IDs
        if token_type_ids is None:
            token_type_ids = self.backend.zeros_like(input_ids)

        # Get embeddings
        inputs_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)

        # Combine embeddings
        embeddings = inputs_embeds + position_embeds + token_type_embeds

        # Apply layer normalization
        embeddings = self.layer_norm(embeddings)

        # Apply dropout
        embeddings = self.dropout(embeddings)

        return embeddings



class BertPooler(Module):
    """BERT pooler layer using neural abstraction.

    We "pool" the model by simply taking the hidden state corresponding
    to the first token. Framework-agnostic implementation.
    """

    def __init__(self, config: BertConfig, backend: NeuralBackend):
        super().__init__()
        self.config = config
        self.backend = backend

        # Dense layer for pooling
        self.dense = backend.linear(
            in_features=config.hidden_size,
            out_features=config.hidden_size,
            bias=True
        )

        # Activation function
        self.activation = backend.activation(ActivationType.TANH)

        # Register submodules
        self.add_module("dense", self.dense)
        self.add_module("activation", self.activation)

    def forward(self, hidden_states: Array) -> Array:
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


# ============================================================================
# ModernBERT Embeddings (without positional embeddings)
# ============================================================================


class ModernBertEmbeddings(Module):
    """
    ModernBERT embeddings layer using neural abstraction.

    Key differences from Classic BERT:
    - No learned positional embeddings (RoPE is used instead)
    - Additional normalization layer after embeddings
    - Streamlined architecture without bias terms
    - Support for extended vocabulary size
    - Framework-agnostic implementation
    """

    def __init__(self, config: BertConfig, backend: NeuralBackend):
        """
        Initialize ModernBERT embeddings.

        Args:
            config: ModernBERT configuration
            backend: Neural backend for framework-agnostic operations
        """
        super().__init__()
        self.config = config
        self.backend = backend

        # Token embeddings
        self.word_embeddings = backend.embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size
            # No padding_idx for ModernBERT
        )

        # Token type embeddings (for NSP task - sentence A vs sentence B)
        self.token_type_embeddings = backend.embedding(
            num_embeddings=config.type_vocab_size,
            embedding_dim=config.hidden_size
        )

        # Note: No position embeddings - RoPE is used instead

        # Layer normalization after embeddings (Classic BERT style)
        self.layer_norm = backend.layer_norm(
            normalized_shape=config.hidden_size,
            eps=config.layer_norm_eps
        )

        # Additional post-embedding normalization (ModernBERT improvement)
        if (
            hasattr(config, "use_post_embedding_norm")
            and config.use_post_embedding_norm
        ):
            self.post_embedding_norm = backend.layer_norm(
                normalized_shape=config.hidden_size,
                eps=getattr(config, "post_embedding_norm_eps", config.layer_norm_eps)
            )
        else:
            self.post_embedding_norm = None

        # Dropout
        self.dropout = backend.dropout(p=config.hidden_dropout_prob)

        # Register submodules
        self.add_module("word_embeddings", self.word_embeddings)
        self.add_module("token_type_embeddings", self.token_type_embeddings)
        self.add_module("layer_norm", self.layer_norm)
        if self.post_embedding_norm is not None:
            self.add_module("post_embedding_norm", self.post_embedding_norm)
        self.add_module("dropout", self.dropout)

    def forward(
        self,
        input_ids: Array,
        token_type_ids: Optional[Array] = None,
        position_ids: Optional[Array] = None,  # Not used in ModernBERT
    ) -> Array:
        """
        Forward pass through ModernBERT embeddings.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len] (ignored in ModernBERT)

        Returns:
            Embeddings [batch_size, seq_len, hidden_size]
        """
        # Get token type IDs
        if token_type_ids is None:
            token_type_ids = self.backend.zeros_like(input_ids)

        # Get token embeddings
        inputs_embeds = self.word_embeddings(input_ids)

        # Get token type embeddings
        token_type_embeds = self.token_type_embeddings(token_type_ids)

        # Combine embeddings (no positional embeddings in ModernBERT)
        embeddings = inputs_embeds + token_type_embeds

        # Apply layer normalization
        embeddings = self.layer_norm(embeddings)

        # Apply post-embedding normalization if enabled
        if self.post_embedding_norm is not None:
            embeddings = self.post_embedding_norm(embeddings)

        # Apply dropout
        embeddings = self.dropout(embeddings)

        return embeddings



class ModernBertPooler(Module):
    """
    ModernBERT pooler layer using neural abstraction.

    Similar to Classic BERT pooler but with streamlined architecture
    and optional bias terms. Framework-agnostic implementation.
    """

    def __init__(self, config: BertConfig, backend: NeuralBackend):
        """
        Initialize ModernBERT pooler.

        Args:
            config: ModernBERT configuration
            backend: Neural backend for framework-agnostic operations
        """
        super().__init__()
        self.config = config
        self.backend = backend

        # Dense layer for pooling
        pooler_hidden_size = getattr(config, "pooler_hidden_size", config.hidden_size)
        use_bias = getattr(config, "use_bias", True)

        self.dense = backend.linear(
            in_features=config.hidden_size,
            out_features=pooler_hidden_size,
            bias=use_bias
        )

        # Activation function
        pooler_activation = getattr(config, "pooler_activation", "tanh")
        activation_map = {
            "tanh": ActivationType.TANH,
            "gelu": ActivationType.GELU,
            "relu": ActivationType.RELU,
        }
        activation_type = activation_map.get(pooler_activation, ActivationType.TANH)
        self.activation = backend.activation(activation_type)

        # Dropout
        pooler_dropout = getattr(config, "pooler_dropout", 0.0)
        self.dropout = backend.dropout(p=pooler_dropout)

        # Register submodules
        self.add_module("dense", self.dense)
        self.add_module("activation", self.activation)
        self.add_module("dropout", self.dropout)

    def forward(self, hidden_states: Array) -> Array:
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


# ============================================================================
# Embedding Factory Functions
# ============================================================================


def create_embeddings(
    config: BertConfig, backend: NeuralBackend
) -> BertEmbeddings | ModernBertEmbeddings:
    """
    Create appropriate embeddings based on configuration.

    Args:
        config: Model configuration
        backend: Neural backend for framework-agnostic operations

    Returns:
        Embeddings layer
    """
    # Check if this is ModernBERT (no positional embeddings)
    if hasattr(config, "use_rope") and config.use_rope:
        return ModernBertEmbeddings(config, backend)
    else:
        return BertEmbeddings(config, backend)


def create_pooler(
    config: BertConfig, backend: NeuralBackend
) -> BertPooler | ModernBertPooler:
    """
    Create appropriate pooler based on configuration.

    Args:
        config: Model configuration
        backend: Neural backend for framework-agnostic operations

    Returns:
        Pooler layer
    """
    # Check if this is ModernBERT
    if hasattr(config, "use_rope") and config.use_rope:
        return ModernBertPooler(config, backend)
    else:
        return BertPooler(config, backend)


# ============================================================================
# Utility Functions
# ============================================================================


def get_embedding_size(config: BertConfig) -> int:
    """Get the embedding size for the model."""
    return config.hidden_size


def get_vocab_size(config: BertConfig) -> int:
    """Get the vocabulary size for the model."""
    return config.vocab_size


def get_max_sequence_length(config: BertConfig) -> int:
    """Get the maximum sequence length for the model."""
    return config.max_position_embeddings


def create_embedding_config_from_bert(
    config: BertConfig,
    use_positional: bool = True,
    use_token_type: bool = True
) -> EmbeddingConfig:
    """Create EmbeddingConfig from BertConfig for neural port compatibility.
    
    Args:
        config: BERT configuration
        use_positional: Whether to use positional embeddings
        use_token_type: Whether to use token type embeddings
        
    Returns:
        EmbeddingConfig for neural port
    """
    return EmbeddingConfig(
        vocab_size=config.vocab_size,
        embedding_dim=config.hidden_size,
        padding_idx=getattr(config, 'pad_token_id', None),
        max_position_embeddings=config.max_position_embeddings,
        use_positional=use_positional and not (hasattr(config, 'use_rope') and config.use_rope),
        use_token_type=use_token_type,
        type_vocab_size=config.type_vocab_size
    )
