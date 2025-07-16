"""Compatibility wrapper for modernbert_optimized.py - redirects to unified implementation."""

import warnings
from .modernbert import (
    ModernBertConfig,
    OptimizedEmbeddings,
    FusedMultiHeadAttention,
    TransformerBlock,
    ModernBertModel,
    create_model,
)

warnings.warn(
    "modernbert_optimized.py is deprecated. Please import from models.modernbert instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export all symbols for backward compatibility
__all__ = [
    "ModernBertConfig",
    "OptimizedEmbeddings",
    "FusedMultiHeadAttention",
    "TransformerBlock",
    "ModernBertModel",
    "create_model",
]
