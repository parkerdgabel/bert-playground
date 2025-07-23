"""Infrastructure adapters for data layer.

This package contains the concrete implementations of domain interfaces
using specific technologies like MLX, file system, caching, etc.
"""

# Adapter implementations
from .data_adapter import (
    FileSystemDataRepository,
    MLXTokenizerAdapter, 
    MLXDataLoaderAdapter,
    DataValidationAdapter,
    SimpleDatasetFactory,
    SimpleFileSystemAdapter,
    # Factory functions
    create_data_repository,
    create_tokenizer_adapter,
    create_data_validator,
    create_dataset_factory,
    create_filesystem_adapter
)

# Infrastructure components  
from .loaders.mlx_loader import MLXDataLoader, MLXLoaderConfig
from .tokenizers.mlx_tokenizer import MLXTokenizer
from .cache.factory import create_cache
from .preprocessing.tokenizer_cache import TokenizerCache, PreTokenizedDataset

__all__ = [
    # Adapter implementations
    "FileSystemDataRepository",
    "MLXTokenizerAdapter",
    "MLXDataLoaderAdapter", 
    "DataValidationAdapter",
    "SimpleDatasetFactory",
    "SimpleFileSystemAdapter",
    
    # Factory functions
    "create_data_repository",
    "create_tokenizer_adapter",
    "create_data_validator", 
    "create_dataset_factory",
    "create_filesystem_adapter",
    
    # Infrastructure components
    "MLXDataLoader",
    "MLXLoaderConfig",
    "MLXTokenizer",
    "create_cache",
    "TokenizerCache",
    "PreTokenizedDataset"
]