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

# ModernBERT imports
from .modernbert_config import (
    ModernBertConfig,
    get_modernbert_config,
    MODERNBERT_BASE_CONFIG,
    MODERNBERT_LARGE_CONFIG,
)

from .modernbert_core import (
    ModernBertCore,
    create_modernbert_core,
    create_modernbert_base,
    create_modernbert_large,
)

__all__ = [
    # Configuration
    "BertConfig",
    "ModernBertConfig",
    
    # Core classes
    "BertCore",
    "BertOutput",
    "BertWithHead",
    "ModernBertCore",
    
    # Factory functions
    "create_bert_core",
    "create_bert_with_head",
    "create_bert_for_competition",
    "create_modernbert_core",
    "create_modernbert_base",
    "create_modernbert_large",
    
    # Config utilities
    "get_base_config",
    "get_large_config",
    "get_mini_config",
    "get_modernbert_config",
    "MODERNBERT_BASE_CONFIG",
    "MODERNBERT_LARGE_CONFIG",
]