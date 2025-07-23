"""Example of how to use the neural network abstraction port.

This example demonstrates how BERT models can be written in a 
framework-agnostic way using the neural port abstraction.
"""

from typing import Any

from .compute import Array
from .neural import Module, NeuralBackend
from .neural_types import (
    AttentionConfig,
    EmbeddingConfig,
    FeedForwardConfig,
    TransformerOutput,
)


class BertEmbeddings(Module):
    """BERT embeddings using the neural abstraction."""
    
    def __init__(self, config: EmbeddingConfig, backend: NeuralBackend):
        super().__init__()
        self.config = config
        self.backend = backend
        
        # Create embedding layers
        self.word_embeddings = backend.embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_dim,
            padding_idx=config.padding_idx
        )
        
        if config.use_positional:
            self.position_embeddings = backend.embedding(
                num_embeddings=config.max_position_embeddings,
                embedding_dim=config.embedding_dim
            )
        
        if config.use_token_type:
            self.token_type_embeddings = backend.embedding(
                num_embeddings=config.type_vocab_size,
                embedding_dim=config.embedding_dim
            )
        
        # Normalization and dropout
        self.layer_norm = backend.layer_norm(
            normalized_shape=config.embedding_dim,
            eps=1e-12
        )
        self.dropout = backend.dropout(p=0.1)
        
        # Register as submodules
        self.add_module("word_embeddings", self.word_embeddings)
        if config.use_positional:
            self.add_module("position_embeddings", self.position_embeddings)
        if config.use_token_type:
            self.add_module("token_type_embeddings", self.token_type_embeddings)
        self.add_module("layer_norm", self.layer_norm)
        self.add_module("dropout", self.dropout)
    
    def forward(
        self,
        input_ids: Array,
        token_type_ids: Array | None = None,
        position_ids: Array | None = None
    ) -> Array:
        """Forward pass through embeddings."""
        # Get word embeddings
        embeddings = self.word_embeddings(input_ids)
        
        # Add position embeddings if configured
        if self.config.use_positional:
            if position_ids is None:
                seq_length = input_ids.shape[1]
                position_ids = self.backend.arange(seq_length)
                position_ids = self.backend.broadcast_to(
                    position_ids, input_ids.shape
                )
            embeddings = embeddings + self.position_embeddings(position_ids)
        
        # Add token type embeddings if configured
        if self.config.use_token_type:
            if token_type_ids is None:
                token_type_ids = self.backend.zeros_like(input_ids)
            embeddings = embeddings + self.token_type_embeddings(token_type_ids)
        
        # Apply layer norm and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class BertAttention(Module):
    """BERT attention layer using the neural abstraction."""
    
    def __init__(self, config: AttentionConfig, backend: NeuralBackend):
        super().__init__()
        self.config = config
        self.backend = backend
        
        # Multi-head attention
        self.attention = backend.multi_head_attention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            bias=config.use_bias
        )
        
        # Output projection
        self.output_projection = backend.linear(
            in_features=config.hidden_size,
            out_features=config.hidden_size,
            bias=config.use_bias
        )
        
        # Layer norm and dropout
        self.layer_norm = backend.layer_norm(
            normalized_shape=config.hidden_size,
            eps=1e-12
        )
        self.dropout = backend.dropout(p=config.attention_dropout)
        
        # Register submodules
        self.add_module("attention", self.attention)
        self.add_module("output_projection", self.output_projection)
        self.add_module("layer_norm", self.layer_norm)
        self.add_module("dropout", self.dropout)
    
    def forward(
        self,
        hidden_states: Array,
        attention_mask: Array | None = None,
        output_attentions: bool = False
    ) -> tuple[Array, Array | None]:
        """Forward pass through attention."""
        # Self-attention
        attn_output, attn_weights = self.attention(
            hidden_states,
            hidden_states,
            hidden_states,
            attention_mask=attention_mask
        )
        
        # Output projection
        attn_output = self.output_projection(attn_output)
        attn_output = self.dropout(attn_output)
        
        # Residual connection and layer norm
        output = self.layer_norm(hidden_states + attn_output)
        
        return output, attn_weights if output_attentions else None


class BertFeedForward(Module):
    """BERT feed-forward network using the neural abstraction."""
    
    def __init__(self, config: FeedForwardConfig, backend: NeuralBackend):
        super().__init__()
        self.config = config
        self.backend = backend
        
        # Feed-forward layers
        self.intermediate = backend.linear(
            in_features=config.hidden_size,
            out_features=config.intermediate_size,
            bias=config.use_bias
        )
        
        self.output = backend.linear(
            in_features=config.intermediate_size,
            out_features=config.hidden_size,
            bias=config.use_bias
        )
        
        # Activation
        from .neural import ActivationType
        activation_map = {
            "gelu": ActivationType.GELU,
            "relu": ActivationType.RELU,
            "silu": ActivationType.SILU,
            "tanh": ActivationType.TANH,
            "sigmoid": ActivationType.SIGMOID,
        }
        activation_type = activation_map.get(config.activation, ActivationType.GELU)
        self.activation = backend.activation(activation_type=activation_type)
        
        # Layer norm and dropout
        self.layer_norm = backend.layer_norm(
            normalized_shape=config.hidden_size,
            eps=1e-12
        )
        self.dropout = backend.dropout(p=config.dropout)
        
        # Register submodules
        self.add_module("intermediate", self.intermediate)
        self.add_module("output", self.output)
        self.add_module("activation", self.activation)
        self.add_module("layer_norm", self.layer_norm)
        self.add_module("dropout", self.dropout)
    
    def forward(self, hidden_states: Array) -> Array:
        """Forward pass through feed-forward network."""
        # Intermediate projection with activation
        intermediate = self.intermediate(hidden_states)
        intermediate = self.activation(intermediate)
        
        # Output projection
        output = self.output(intermediate)
        output = self.dropout(output)
        
        # Residual connection and layer norm
        output = self.layer_norm(hidden_states + output)
        
        return output


class BertLayer(Module):
    """Single BERT transformer layer using the neural abstraction."""
    
    def __init__(
        self,
        attention_config: AttentionConfig,
        ffn_config: FeedForwardConfig,
        backend: NeuralBackend
    ):
        super().__init__()
        self.backend = backend
        
        # Attention and feed-forward sublayers
        self.attention = BertAttention(attention_config, backend)
        self.feed_forward = BertFeedForward(ffn_config, backend)
        
        # Register submodules
        self.add_module("attention", self.attention)
        self.add_module("feed_forward", self.feed_forward)
    
    def forward(
        self,
        hidden_states: Array,
        attention_mask: Array | None = None,
        output_attentions: bool = False
    ) -> tuple[Array, Array | None]:
        """Forward pass through transformer layer."""
        # Self-attention
        attn_output, attn_weights = self.attention(
            hidden_states,
            attention_mask,
            output_attentions
        )
        
        # Feed-forward
        output = self.feed_forward(attn_output)
        
        return output, attn_weights


class BertModel(Module):
    """Complete BERT model using the neural abstraction."""
    
    def __init__(
        self,
        num_layers: int,
        embedding_config: EmbeddingConfig,
        attention_config: AttentionConfig,
        ffn_config: FeedForwardConfig,
        backend: NeuralBackend
    ):
        super().__init__()
        self.num_layers = num_layers
        self.backend = backend
        
        # Embeddings
        self.embeddings = BertEmbeddings(embedding_config, backend)
        
        # Transformer layers
        self.layers = backend.module_list([
            BertLayer(attention_config, ffn_config, backend)
            for _ in range(num_layers)
        ])
        
        # Pooler
        self.pooler = backend.sequential(
            backend.linear(
                embedding_config.embedding_dim,
                embedding_config.embedding_dim
            ),
            backend.activation("tanh")
        )
        
        # Register submodules
        self.add_module("embeddings", self.embeddings)
        self.add_module("layers", self.layers)
        self.add_module("pooler", self.pooler)
    
    def forward(
        self,
        input_ids: Array,
        attention_mask: Array | None = None,
        token_type_ids: Array | None = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False
    ) -> TransformerOutput:
        """Forward pass through BERT model."""
        # Get embeddings
        hidden_states = self.embeddings(
            input_ids,
            token_type_ids=token_type_ids
        )
        
        # Initialize outputs
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        # Pass through transformer layers
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            hidden_states, attn_weights = layer(
                hidden_states,
                attention_mask,
                output_attentions
            )
            
            if output_attentions:
                all_attentions = all_attentions + (attn_weights,)
        
        # Add final hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        # Pooler output (CLS token)
        pooler_output = self.pooler(hidden_states[:, 0, :])
        
        return TransformerOutput(
            last_hidden_state=hidden_states,
            pooler_output=pooler_output,
            hidden_states=all_hidden_states,
            attentions=all_attentions
        )


def create_bert_model(
    vocab_size: int = 30522,
    hidden_size: int = 768,
    num_layers: int = 12,
    num_attention_heads: int = 12,
    intermediate_size: int = 3072,
    backend_name: str = "mlx"
) -> BertModel:
    """Factory function to create a BERT model with any backend.
    
    Args:
        vocab_size: Vocabulary size
        hidden_size: Hidden dimension
        num_layers: Number of transformer layers
        num_attention_heads: Number of attention heads
        intermediate_size: Feed-forward intermediate size
        backend_name: Name of the backend to use
        
    Returns:
        BERT model instance
    """
    from .neural import create_neural_backend
    
    # Create backend
    backend = create_neural_backend(backend_name)
    
    # Create configurations
    embedding_config = EmbeddingConfig(
        vocab_size=vocab_size,
        embedding_dim=hidden_size,
        max_position_embeddings=512,
        use_positional=True,
        use_token_type=True
    )
    
    attention_config = AttentionConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        attention_dropout=0.1
    )
    
    ffn_config = FeedForwardConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation="gelu",
        dropout=0.1
    )
    
    # Create model
    return BertModel(
        num_layers=num_layers,
        embedding_config=embedding_config,
        attention_config=attention_config,
        ffn_config=ffn_config,
        backend=backend
    )