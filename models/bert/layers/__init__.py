"""
BERT layers module.

This module contains the core building blocks for BERT models,
organized into focused components for better maintainability.
"""

from .activations import (
    RMSNorm,
    create_activation_layer,
    create_normalization_layer,
    get_activation_function,
    get_normalization_function,
)
from .attention import (
    AlternatingAttention,
    BertAttention,
    BertSelfAttention,
    BertSelfOutput,
    GlobalAttention,
    LocalAttention,
    RoPEAttention,
    RotaryEmbedding,
    create_attention_layer,
)
from .embeddings import (
    BertEmbeddings,
    BertPooler,
    ModernBertEmbeddings,
    ModernBertPooler,
    create_embeddings,
    create_pooler,
)
from .feedforward import (
    BertFeedForward,
    BertIntermediate,
    BertOutput,
    GeGLU,
    GeGLUMLP,
    ModernBertFeedForward,
    NeoBertFeedForward,
    SwiGLU,
    SwiGLUMLP,
    create_feedforward_layer,
)

__all__ = [
    # Classic BERT components
    "BertAttention",
    "BertSelfAttention",
    "BertSelfOutput",
    "BertFeedForward",
    "BertIntermediate",
    "BertOutput",
    "BertEmbeddings",
    "BertPooler",
    # ModernBERT components
    "RoPEAttention",
    "LocalAttention",
    "GlobalAttention",
    "AlternatingAttention",
    "RotaryEmbedding",
    "GeGLU",
    "SwiGLU",
    "GeGLUMLP",
    "SwiGLUMLP",
    "ModernBertFeedForward",
    "NeoBertFeedForward",
    "ModernBertEmbeddings",
    "ModernBertPooler",
    "RMSNorm",
    # Factory functions
    "create_attention_layer",
    "create_feedforward_layer",
    "create_embeddings",
    "create_pooler",
    "get_activation_function",
    "get_normalization_function",
    "create_activation_layer",
    "create_normalization_layer",
]
