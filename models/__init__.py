"""Models package for BERT implementations with LoRA support."""

from .bert import (
    BaseBertModel,
    BertConfig,
    BertCore,
    BertWithHead,
    ModernBertConfig,
    ModernBertCore,
)
from .bert import (
    create_bert_core,
    create_bert_with_head,
)
from .factory import (
    MODEL_REGISTRY,
)
from .heads import (
    BaseHead,
    BinaryClassificationHead,
    MulticlassClassificationHead,
    RegressionHead,
)
from .lora import (
    KAGGLE_LORA_PRESETS,
    LoRAAdapter,
    LoRAConfig,
    LoRALinear,
    LoRATrainingConfig,
    MultiAdapterManager,
    MultiLoRAConfig,
    MultiLoRALinear,
    QLoRAConfig,
    QLoRALinear,
    get_lora_preset,
)

__all__ = [
    # Configs
    "BertConfig",
    "ModernBertConfig",
    # Base classes
    "BaseBertModel",
    # Core models
    "BertCore",
    "ModernBertCore",
    "BertWithHead",
    # Factory
    "create_bert_core",
    "create_bert_with_head",
    "MODEL_REGISTRY",
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
