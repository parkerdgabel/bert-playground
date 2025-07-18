"""LoRA (Low-Rank Adaptation) module for efficient BERT fine-tuning.

This module provides LoRA and QLoRA implementations optimized for MLX,
enabling efficient fine-tuning of large BERT models for Kaggle competitions.
"""

from .config import (
    LoRAConfig,
    QLoRAConfig,
    MultiLoRAConfig,
    LoRATrainingConfig,
    KAGGLE_LORA_PRESETS,
    get_lora_preset,
)
from .layers import (
    LoRALinear,
    QLoRALinear,
    MultiLoRALinear,
)
from .adapter import (
    LoRAAdapter,
    MultiAdapterManager,
)

__all__ = [
    # Config classes
    "LoRAConfig",
    "QLoRAConfig", 
    "MultiLoRAConfig",
    "LoRATrainingConfig",
    "KAGGLE_LORA_PRESETS",
    "get_lora_preset",
    # Layer implementations
    "LoRALinear",
    "QLoRALinear",
    "MultiLoRALinear",
    # Adapter utilities
    "LoRAAdapter",
    "MultiAdapterManager",
]