"""BERT module - Modular BERT implementation with clean head interface.

This module provides:
- BertCore: The core BERT model with standardized outputs
- BertOutput: Standard output format for BERT models
- BertWithHead: Combines BERT with any task-specific head
- Factory functions for easy model creation
"""

from .config import (
    BertConfig,
    ModernBertConfig,
    CNNHybridConfig,
    get_base_config,
    get_large_config,
    get_mini_config,
    get_cnn_hybrid_config,
)

from .core import (
    BertCore,
    BertOutput,
    create_bert_core,
)

from .model import (
    BertWithHead,
    create_bert_with_head,
    create_bert_for_competition,
)

__all__ = [
    # Configuration
    "BertConfig",
    "ModernBertConfig",  # Backward compatibility
    "CNNHybridConfig",   # Backward compatibility
    
    # Core classes
    "BertCore",
    "BertOutput",
    "BertWithHead",
    
    # Factory functions
    "create_bert_core",
    "create_bert_with_head",
    "create_bert_for_competition",
    
    # Config utilities
    "get_base_config",
    "get_large_config",
    "get_mini_config",
    "get_cnn_hybrid_config",
]