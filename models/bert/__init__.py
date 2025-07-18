"""BERT module - Modular BERT implementation with clean head interface.

This module provides:
- BertCore: The core BERT model with standardized outputs
- BertOutput: Standard output format for BERT models
- BertWithHead: Combines BERT with any task-specific head
- Factory functions for easy model creation
"""

from .core import (
    BertCore,
    BertOutput,
    ModernBertConfig,
    create_bert_core,
)

from .model import (
    BertWithHead,
    create_bert_with_head,
    create_bert_for_competition,
)

__all__ = [
    # Core classes
    "BertCore",
    "BertOutput",
    "ModernBertConfig",
    "BertWithHead",
    
    # Factory functions
    "create_bert_core",
    "create_bert_with_head",
    "create_bert_for_competition",
]