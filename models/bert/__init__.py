"""BERT module - Modular BERT implementation with clean head interface.

This module provides:
- BertCore: The core BERT model with standardized outputs
- BertOutput: Standard output format for BERT models
- BertWithHead: Combines BERT with any task-specific head
- Factory functions for easy model creation
"""

from .config import (
    BertConfig,
    get_base_config,
    get_large_config,
    get_mini_config,
    get_modernbert_base_config,
    get_modernbert_large_config,
    get_neobert_config,
    get_neobert_mini_config,
)
from .core import (
    BertCore,
    BertOutput,
    ModernBertCore,
    create_bert_core,
    create_model_core,
    create_modernbert_base,
    create_modernbert_core,
    create_modernbert_large,
    create_neobert,
    create_neobert_core,
    create_neobert_mini,
)
from .core_base import (
    BaseBertModel,
    BertLayer,
    BertModelOutput,
)
from .model import (
    BertWithHead,
    create_bert_for_competition,
    create_bert_with_head,
)

# ModernBERT imports
from .modernbert_config import (
    MODERNBERT_BASE_CONFIG,
    MODERNBERT_LARGE_CONFIG,
    ModernBertConfig,
    get_modernbert_config,
)

__all__ = [
    # Configuration
    "BertConfig",
    "ModernBertConfig",
    # Base classes
    "BaseBertModel",
    "BertModelOutput",
    "BertLayer",
    # Core classes
    "BertCore",
    "BertOutput",
    "BertWithHead",
    "ModernBertCore",
    # Factory functions - Classic BERT
    "create_bert_core",
    "create_bert_with_head",
    "create_bert_for_competition",
    # Factory functions - ModernBERT
    "create_modernbert_core",
    "create_modernbert_base",
    "create_modernbert_large",
    # Factory functions - neoBERT
    "create_neobert_core",
    "create_neobert",
    "create_neobert_mini",
    # Factory functions - Generic
    "create_model_core",
    # Config utilities - Classic BERT
    "get_base_config",
    "get_large_config",
    "get_mini_config",
    # Config utilities - ModernBERT
    "get_modernbert_config",
    "get_modernbert_base_config",
    "get_modernbert_large_config",
    "MODERNBERT_BASE_CONFIG",
    "MODERNBERT_LARGE_CONFIG",
    # Config utilities - neoBERT
    "get_neobert_config",
    "get_neobert_mini_config",
]
