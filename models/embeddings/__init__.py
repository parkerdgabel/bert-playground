"""
Embeddings Module

Pure embedding extraction functionality for various model types.
Separated from classification logic for better architecture.
"""

from .mlx_adapter import MLXEmbeddingsAdapter
from .tokenizer_wrapper import TokenizerWrapper
from .embedding_model import EmbeddingModel

__all__ = [
    "MLXEmbeddingsAdapter",
    "TokenizerWrapper", 
    "EmbeddingModel",
]