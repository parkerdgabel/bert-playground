"""LoRA (Low-Rank Adaptation) module for efficient BERT fine-tuning.

This module provides LoRA and QLoRA implementations optimized for MLX,
enabling efficient fine-tuning of large BERT models for Kaggle competitions.
"""

from .adapter import (
    LoRAAdapter,
    MultiAdapterManager,
)
from .config import (
    KAGGLE_LORA_PRESETS,
    LoRAConfig,
    LoRATrainingConfig,
    MultiLoRAConfig,
    QLoRAConfig,
    get_lora_preset,
)
from .layers import (
    LoRALinear,
    MultiLoRALinear,
    QLoRALinear,
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
