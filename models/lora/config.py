"""LoRA and QLoRA configuration classes for BERT models.

This module provides configuration classes for Low-Rank Adaptation (LoRA) and
Quantized LoRA (QLoRA) fine-tuning of BERT models. These configurations extend
the existing BERT configuration system to support efficient fine-tuning for
Kaggle competitions.
"""

from dataclasses import dataclass, field


@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation) parameters.

    LoRA enables efficient fine-tuning by adding trainable rank decomposition
    matrices to frozen pretrained weights. This is particularly useful for
    Kaggle competitions where we want to adapt large models with limited
    computational resources.

    Attributes:
        r: Rank of the LoRA decomposition. Higher ranks allow more expressivity
           but increase parameters. Common values: 4, 8, 16, 32.
        alpha: LoRA scaling parameter. The scaling factor is alpha/r.
               Common practice: set alpha = r for alpha/r = 1.
        dropout: Dropout probability for LoRA layers. Helps prevent overfitting
                on small Kaggle datasets.
        target_modules: Set of module names to apply LoRA to. Default targets
                       key projection layers in BERT.
        modules_to_save: Modules to train fully (not with LoRA). Useful for
                        task-specific layers like classification heads.
        bias: How to handle biases. Options:
              - "none": Don't train biases
              - "all": Train all biases
              - "lora_only": Only train biases in LoRA layers
        use_rslora: Use Rank-Stabilized LoRA for better multi-rank training.
        use_dora: Use Weight-Decomposed LoRA (DoRA) for enhanced performance.
        lora_bias_trainable: Whether LoRA bias terms are trainable.
        init_lora_weights: Initialization method for LoRA weights:
                          - "gaussian": Standard Gaussian initialization
                          - "uniform": Uniform initialization
                          - "bert": BERT-style initialization
        layer_specific_config: Per-layer LoRA configuration for fine-grained control.
    """

    # Core LoRA parameters
    r: int = 8
    alpha: int = 8
    dropout: float = 0.1

    # Target modules for LoRA adaptation
    target_modules: set[str] = field(
        default_factory=lambda: {
            "query",
            "key",
            "value",
            "dense",  # Attention layers
            "intermediate.dense",
            "output.dense",  # FFN layers
        }
    )

    # Modules to save/train normally (without LoRA)
    modules_to_save: set[str] = field(
        default_factory=lambda: {"classifier", "regression_head", "pooler"}
    )

    # Bias handling
    bias: str = "none"  # Options: "none", "all", "lora_only"

    # Advanced LoRA variants
    use_rslora: bool = False  # Rank-Stabilized LoRA
    use_dora: bool = False  # Weight-Decomposed LoRA

    # Training configuration
    lora_bias_trainable: bool = False

    # Initialization
    init_lora_weights: str = "gaussian"  # Options: "gaussian", "uniform", "bert"

    # Layer-specific configuration (layer_name -> custom config)
    layer_specific_config: dict[str, dict[str, int | float]] = field(
        default_factory=dict
    )

    def get_layer_config(self, layer_name: str) -> dict[str, int | float]:
        """Get configuration for a specific layer.

        Args:
            layer_name: Name of the layer to get config for

        Returns:
            Dictionary with layer-specific configuration
        """
        if layer_name in self.layer_specific_config:
            # Merge with default config
            config = {"r": self.r, "alpha": self.alpha, "dropout": self.dropout}
            config.update(self.layer_specific_config[layer_name])
            return config
        return {"r": self.r, "alpha": self.alpha, "dropout": self.dropout}

    @property
    def scaling(self) -> float:
        """Get the LoRA scaling factor (alpha/r)."""
        return self.alpha / self.r

    def should_apply_lora(self, module_name: str) -> bool:
        """Check if LoRA should be applied to a module.

        Args:
            module_name: Full name of the module

        Returns:
            Whether LoRA should be applied
        """
        # Check if any target module pattern matches
        for target in self.target_modules:
            if target in module_name:
                # But not if it's in modules_to_save
                for save_module in self.modules_to_save:
                    if save_module in module_name:
                        return False
                return True
        return False


@dataclass
class QLoRAConfig(LoRAConfig):
    """Configuration for QLoRA (Quantized LoRA) parameters.

    QLoRA combines 4-bit quantization of the base model with LoRA adapters
    in higher precision. This enables fine-tuning of very large models on
    consumer GPUs, perfect for Kaggle competitions with compute constraints.

    Additional Attributes:
        bnb_4bit_compute_dtype: Computation dtype for 4-bit operations.
                               Default: "float16" for efficiency.
        bnb_4bit_use_double_quant: Use double quantization to save more memory.
        bnb_4bit_quant_type: Quantization type. Options: "nf4", "int4".
        gradient_checkpointing: Enable gradient checkpointing to save memory.
        max_memory_mb: Maximum memory to use for model (helps prevent OOM).
        offload_to_cpu: Offload frozen layers to CPU to save GPU memory.
    """

    # Quantization configuration
    bnb_4bit_compute_dtype: str = "float16"  # or "bfloat16", "float32"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"  # or "int4"

    # Memory optimization
    gradient_checkpointing: bool = True
    max_memory_mb: int | None = None  # Set to limit GPU memory usage
    offload_to_cpu: bool = False

    # QLoRA typically uses lower rank due to memory constraints
    r: int = 4
    alpha: int = 4


@dataclass
class MultiLoRAConfig:
    """Configuration for multi-adapter LoRA setups.

    Useful for Kaggle competitions where you want to:
    1. Train multiple LoRA adapters for different tasks
    2. Ensemble multiple LoRA adapters
    3. Switch between adapters dynamically

    Attributes:
        adapters: Dictionary mapping adapter names to their configs
        active_adapters: List of currently active adapter names
        combination_type: How to combine multiple adapters:
                         - "concatenate": Concatenate adapter outputs
                         - "average": Average adapter outputs
                         - "weighted": Weighted combination
        adapter_weights: Weights for weighted combination
    """

    adapters: dict[str, LoRAConfig] = field(default_factory=dict)
    active_adapters: list[str] = field(default_factory=list)
    combination_type: str = "average"  # Options: "concatenate", "average", "weighted"
    adapter_weights: dict[str, float] | None = None

    def add_adapter(self, name: str, config: LoRAConfig) -> None:
        """Add a new LoRA adapter configuration.

        Args:
            name: Name of the adapter
            config: LoRA configuration for the adapter
        """
        self.adapters[name] = config

    def activate_adapter(self, name: str) -> None:
        """Activate a LoRA adapter.

        Args:
            name: Name of the adapter to activate
        """
        if name in self.adapters and name not in self.active_adapters:
            self.active_adapters.append(name)

    def deactivate_adapter(self, name: str) -> None:
        """Deactivate a LoRA adapter.

        Args:
            name: Name of the adapter to deactivate
        """
        if name in self.active_adapters:
            self.active_adapters.remove(name)


@dataclass
class LoRATrainingConfig:
    """Configuration specific to LoRA training strategies.

    Attributes:
        learning_rate: Learning rate for LoRA parameters
        base_model_lr: Optional different LR for base model params (if any are trainable)
        lora_lr_multiplier: Multiply base LR by this for LoRA params
        warmup_ratio: Portion of training for LR warmup
        weight_decay: Weight decay for regularization
        gradient_accumulation_steps: Steps to accumulate gradients
        mixed_precision: Enable mixed precision training
        layer_wise_lr_decay: Apply exponential LR decay through layers
        freeze_base_model: Whether to freeze the base model completely
    """

    learning_rate: float = 5e-4  # Higher than typical full fine-tuning
    base_model_lr: float | None = None
    lora_lr_multiplier: float = 1.0
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = True
    layer_wise_lr_decay: float | None = None
    freeze_base_model: bool = True


# Preset configurations for common Kaggle scenarios
KAGGLE_LORA_PRESETS = {
    "efficient": LoRAConfig(
        r=4,
        alpha=4,
        dropout=0.1,
        target_modules={"query", "value"},  # Only Q,V for maximum efficiency
    ),
    "balanced": LoRAConfig(
        r=8,
        alpha=8,
        dropout=0.1,
        target_modules={"query", "key", "value", "dense"},
    ),
    "expressive": LoRAConfig(
        r=16,
        alpha=16,
        dropout=0.05,
        target_modules={
            "query",
            "key",
            "value",
            "dense",
            "intermediate.dense",
            "output.dense",
        },
    ),
    "qlora_memory": QLoRAConfig(
        r=4,
        alpha=4,
        dropout=0.1,
        target_modules={"query", "value"},
        gradient_checkpointing=True,
        bnb_4bit_use_double_quant=True,
    ),
    "qlora_quality": QLoRAConfig(
        r=8,
        alpha=8,
        dropout=0.05,
        target_modules={"query", "key", "value", "dense"},
        gradient_checkpointing=True,
        bnb_4bit_compute_dtype="float32",
    ),
}


def get_lora_preset(name: str) -> LoRAConfig | QLoRAConfig:
    """Get a preset LoRA configuration for Kaggle competitions.

    Args:
        name: Name of the preset configuration

    Returns:
        LoRA or QLoRA configuration

    Raises:
        ValueError: If preset name is not found
    """
    if name not in KAGGLE_LORA_PRESETS:
        raise ValueError(
            f"Unknown preset '{name}'. Available presets: {list(KAGGLE_LORA_PRESETS.keys())}"
        )
    return KAGGLE_LORA_PRESETS[name]
