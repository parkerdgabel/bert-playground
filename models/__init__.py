"""Models package for BERT implementations with LoRA support."""

from .bert import (
    BertConfig,
    ModernBertConfig,
    BertCore,
    ModernBertCore,
    BaseBertModel,
    BertModel,
    BertWithHead,
)
from .factory import (
    create_bert_core,
    create_bert_with_head,
    MODEL_REGISTRY,
    HEAD_REGISTRY,
)
from .heads import (
    BaseHead,
    BinaryClassificationHead,
    MulticlassClassificationHead,
    RegressionHead,
)
from .lora import (
    LoRAConfig,
    QLoRAConfig,
    MultiLoRAConfig,
    LoRATrainingConfig,
    KAGGLE_LORA_PRESETS,
    get_lora_preset,
    LoRALinear,
    QLoRALinear,
    MultiLoRALinear,
    LoRAAdapter,
    MultiAdapterManager,
)

__all__ = [
    # Configs
    "BertConfig",
    "ModernBertConfig",
    # Core models
    "BertCore",
    "ModernBertCore",
    "BaseBertModel",
    "BertModel",
    "BertWithHead",
    # Factory
    "create_bert_core",
    "create_bert_with_head",
    "MODEL_REGISTRY",
    "HEAD_REGISTRY",
    # Heads
    "BaseHead",
    "BinaryClassificationHead",
    "MulticlassClassificationHead",
    "RegressionHead",
    # LoRA
    "LoRAConfig",
    "QLoRAConfig",
    "MultiLoRAConfig",
    "LoRATrainingConfig",
    "KAGGLE_LORA_PRESETS",
    "get_lora_preset",
    "LoRALinear",
    "QLoRALinear",
    "MultiLoRALinear",
    "LoRAAdapter",
    "MultiAdapterManager",
]