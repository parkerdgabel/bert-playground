"""
MLX Embeddings Integration Package

This package provides adapters and utilities for integrating mlx-embeddings
library with our existing ModernBERT implementation.
"""

from embeddings.mlx_adapter import MLXEmbeddingsAdapter
from embeddings.tokenizer_wrapper import TokenizerWrapper
from embeddings.model_wrapper import MLXEmbeddingModel
from embeddings.config import MLXEmbeddingsConfig, get_default_config

__all__ = [
    "MLXEmbeddingsAdapter",
    "TokenizerWrapper", 
    "MLXEmbeddingModel",
    "MLXEmbeddingsConfig",
    "get_default_config",
]