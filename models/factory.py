"""MASTER MODEL FACTORY - Single source for all model creation.

This factory creates models using the modular BERT architecture with pluggable heads.
It supports:
- BertCore creation
- Head attachment via BertWithHead
- Competition-specific model creation
- Automatic dataset analysis and optimization
- LoRA/QLoRA adapter injection for efficient fine-tuning
"""

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional

import mlx.nn as nn
import pandas as pd
from loguru import logger

# Import BERT configs and classes
# Import Classic BERT architecture
# Import ModernBERT architecture
from .bert import (
    BertConfig,
    BertCore,
    BertWithHead,
    ModernBertConfig,
    create_bert_core,
    create_bert_for_competition,
    create_bert_with_head,
    create_modernbert_core,
)
from .heads.base import HeadConfig

# Import Kaggle heads if available
try:
    from .classification.kaggle_heads import create_kaggle_head  # noqa: F401

    KAGGLE_HEADS_AVAILABLE = True
except ImportError:
    KAGGLE_HEADS_AVAILABLE = False
    logger.info("Kaggle heads not available")

# Import dataset analysis for automatic optimization
try:
    from data.dataset_spec import KaggleDatasetSpec

    DATASET_ANALYSIS_AVAILABLE = True
except ImportError:
    DATASET_ANALYSIS_AVAILABLE = False
    logger.warning("Dataset analysis not available")

# Import LoRA components
from .lora import (
    LoRAAdapter,
    LoRAConfig,
    MultiAdapterManager,
    QLoRAConfig,
    get_lora_preset,
)

ModelType = Literal[
    "bert_core",
    "bert_with_head",
    "modernbert_core",
    "modernbert_with_head",
    "bert_with_lora",
    "modernbert_with_lora",
]


class CompetitionType(Enum):
    """Types of Kaggle competitions (absorbed from kaggle_competition_factory)."""

    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"
    REGRESSION = "regression"
    ORDINAL_REGRESSION = "ordinal_regression"
    TIME_SERIES = "time_series"
    RANKING = "ranking"
    STRUCTURED_PREDICTION = "structured_prediction"
    GENERATIVE = "generative"
    UNKNOWN = "unknown"


@dataclass
class CompetitionAnalysis:
    """Analysis results for a Kaggle competition dataset (absorbed from kaggle_competition_factory)."""

    # Basic characteristics
    competition_type: CompetitionType
    num_samples: int
    num_features: int

    # Target characteristics
    target_column: str | None = None
    num_classes: int | None = None
    class_distribution: dict[str, int] | None = None
    is_balanced: bool = True

    # Feature characteristics
    categorical_columns: list[str] = None
    numerical_columns: list[str] = None
    text_columns: list[str] = None

    # Optimization recommendations
    recommended_model_type: str = "generic"
    recommended_head_type: str = "binary"
    recommended_batch_size: int = 32
    recommended_learning_rate: float = 2e-5

    def __post_init__(self):
        if self.categorical_columns is None:
            self.categorical_columns = []
        if self.numerical_columns is None:
            self.numerical_columns = []
        if self.text_columns is None:
            self.text_columns = []


def create_model(
    model_type: ModelType = "bert_with_head",
    config: dict[str, Any] | BertConfig | ModernBertConfig | None = None,
    pretrained_path: str | Path | None = None,
    head_type: str | None = None,
    head_config: HeadConfig | dict | None = None,
    **kwargs,
) -> nn.Module:
    """
    Create a model based on type and configuration.

    Args:
        model_type: Type of model to create ("bert_core", "bert_with_head", "modernbert_core", "modernbert_with_head")
        config: Model configuration (dict or Config object)
        pretrained_path: Path to pretrained weights
        head_type: Type of head to attach (for bert_with_head/modernbert_with_head)
        head_config: Head configuration (for bert_with_head/modernbert_with_head)
        **kwargs: Additional arguments passed to model constructor

    Returns:
        Initialized model
    """
    # Determine config type and create appropriate config
    if model_type in ["modernbert_core", "modernbert_with_head"]:
        # ModernBERT models
        if isinstance(config, dict):
            # Create ModernBertConfig from dict, applying only valid config kwargs
            config_dict = config.copy()
            # Filter out non-config arguments
            valid_config_keys = {
                f.name for f in ModernBertConfig.__dataclass_fields__.values()
            }
            config_kwargs = {k: v for k, v in kwargs.items() if k in valid_config_keys}
            config_dict.update(config_kwargs)
            config = ModernBertConfig(**config_dict)
        elif config is None:
            model_size = kwargs.pop("model_size", "base")
            # Filter out non-config arguments
            valid_config_keys = {
                f.name for f in ModernBertConfig.__dataclass_fields__.values()
            }
            config_kwargs = {k: v for k, v in kwargs.items() if k in valid_config_keys}
            config = ModernBertConfig(model_size=model_size, **config_kwargs)
        elif isinstance(config, BertConfig):
            # Convert BertConfig to ModernBertConfig
            config = ModernBertConfig.from_bert_config(config)
    else:
        # Classic BERT models
        if isinstance(config, dict):
            config = BertConfig(**config)
        elif config is None:
            config = BertConfig()

    # Create base model
    if model_type == "bert_core":
        # Create Classic BERT core
        model = create_bert_core(config=config, **kwargs)
        logger.info("Created Classic BertCore model")
    elif model_type == "modernbert_core":
        # Create ModernBERT core
        model = create_modernbert_core(config=config, **kwargs)
        logger.info("Created ModernBertCore model")
    elif model_type == "bert_with_head":
        # Create Classic BERT with attached head
        num_labels = kwargs.pop("num_labels", 2)
        freeze_bert = kwargs.pop("freeze_bert", False)
        freeze_bert_layers = kwargs.pop("freeze_bert_layers", None)

        model = create_bert_with_head(
            bert_config=config,
            head_config=head_config,
            head_type=head_type,
            num_labels=num_labels,
            freeze_bert=freeze_bert,
            freeze_bert_layers=freeze_bert_layers,
            **kwargs,
        )
        logger.info(f"Created BertWithHead model (head_type: {head_type})")
    elif model_type == "modernbert_with_head":
        # Create ModernBERT with attached head
        num_labels = kwargs.pop("num_labels", 2)
        freeze_bert = kwargs.pop("freeze_bert", False)
        freeze_bert_layers = kwargs.pop("freeze_bert_layers", None)

        # Create ModernBERT core first
        modernbert_core = create_modernbert_core(config=config, **kwargs)

        # Create head configuration
        if head_config is None:
            if head_type is None:
                raise ValueError("Either head_config or head_type must be provided")

            # Convert string to enum if needed
            # Keep head_type as string

            # Get default config for head type
            from .heads.base import get_default_config_for_head_type

            head_config = get_default_config_for_head_type(
                head_type,
                input_size=modernbert_core.get_hidden_size(),
                output_size=num_labels,
            )

        # Convert dict to HeadConfig if needed
        if isinstance(head_config, dict):
            head_config = HeadConfig(**head_config)

        # Create head using the factory function
        from .heads import create_head

        # Extract config dict and remove duplicates
        head_config_dict = head_config.__dict__.copy()
        head_type_str = head_config_dict.pop("head_type", None)

        head = create_head(
            head_type=head_type_str,
            **head_config_dict,
        )

        # Create ModernBERT with head (reuse BertWithHead wrapper)
        model = BertWithHead(
            bert=modernbert_core,
            head=head,
            freeze_bert=freeze_bert,
            freeze_bert_layers=freeze_bert_layers,
        )
        logger.info(f"Created ModernBertWithHead model (head_type: {head_type})")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load pretrained weights if provided
    if pretrained_path:
        load_pretrained_weights(model, pretrained_path)

    return model


def create_model_with_lora(
    base_model: nn.Module | None = None,
    model_type: ModelType | None = None,
    lora_config: LoRAConfig | QLoRAConfig | str | dict | None = None,
    inject_lora: bool = True,
    verbose: bool = False,
    **model_kwargs,
) -> tuple[nn.Module, LoRAAdapter]:
    """
    Create a model with LoRA adapters for efficient fine-tuning.

    Args:
        base_model: Existing model to add LoRA to (optional)
        model_type: Type of model to create if base_model not provided
        lora_config: LoRA configuration (config object, preset name, or dict)
        inject_lora: Whether to inject LoRA adapters immediately
        verbose: Print injection details
        **model_kwargs: Arguments for model creation

    Returns:
        Tuple of (model, lora_adapter)
    """
    # Create base model if not provided
    if base_model is None:
        if model_type is None:
            model_type = "bert_with_head"
        base_model = create_model(model_type, **model_kwargs)

    # Process LoRA config
    if lora_config is None:
        lora_config = LoRAConfig()  # Default config
    elif isinstance(lora_config, str):
        # Load preset
        lora_config = get_lora_preset(lora_config)
    elif isinstance(lora_config, dict):
        # Determine if it's QLoRA based on config
        if any(k.startswith("bnb_") for k in lora_config):
            lora_config = QLoRAConfig(**lora_config)
        else:
            lora_config = LoRAConfig(**lora_config)

    # Create LoRA adapter
    lora_adapter = LoRAAdapter(base_model, lora_config)

    # Inject adapters if requested
    if inject_lora:
        stats = lora_adapter.inject_adapters(verbose=verbose)
        if verbose:
            total_params = sum(p.size for p in base_model.parameters())
            lora_params = sum(stats.values())
            reduction = (1 - lora_params / total_params) * 100
            logger.info(f"Parameter reduction: {reduction:.1f}%")

    return base_model, lora_adapter


def create_bert_with_lora(
    head_type: str | None = None,
    lora_preset: str = "balanced",
    num_labels: int = 2,
    freeze_bert: bool = True,
    **kwargs,
) -> tuple[BertWithHead, LoRAAdapter]:
    """
    Create a BERT model with head and LoRA adapters.

    Convenience function for common use case of BERT + head + LoRA.

    Args:
        head_type: Type of head to attach
        lora_preset: LoRA preset name from KAGGLE_LORA_PRESETS
        num_labels: Number of output labels
        freeze_bert: Whether to freeze BERT (recommended with LoRA)
        **kwargs: Additional arguments

    Returns:
        Tuple of (model, lora_adapter)
    """
    return create_model_with_lora(
        model_type="bert_with_head",
        lora_config=lora_preset,
        head_type=head_type,
        num_labels=num_labels,
        freeze_bert=freeze_bert,
        inject_lora=True,
        **kwargs,
    )


def create_modernbert_with_lora(
    head_type: str | None = None,
    lora_preset: str = "balanced",
    model_size: str = "base",
    num_labels: int = 2,
    freeze_bert: bool = True,
    **kwargs,
) -> tuple[nn.Module, LoRAAdapter]:
    """
    Create a ModernBERT model with head and LoRA adapters.

    Args:
        head_type: Type of head to attach
        lora_preset: LoRA preset name
        model_size: Model size ("base" or "large")
        num_labels: Number of output labels
        freeze_bert: Whether to freeze ModernBERT
        **kwargs: Additional arguments

    Returns:
        Tuple of (model, lora_adapter)
    """
    return create_model_with_lora(
        model_type="modernbert_with_head",
        lora_config=lora_preset,
        head_type=head_type,
        model_size=model_size,
        num_labels=num_labels,
        freeze_bert=freeze_bert,
        inject_lora=True,
        **kwargs,
    )


def create_qlora_model(
    model_type: ModelType = "bert_with_head",
    qlora_preset: str = "qlora_memory",
    quantize_base: bool = True,
    **kwargs,
) -> tuple[nn.Module, LoRAAdapter]:
    """
    Create a model with QLoRA (quantized base + LoRA adapters).

    Args:
        model_type: Type of model to create
        qlora_preset: QLoRA preset name
        quantize_base: Whether to quantize the base model
        **kwargs: Additional arguments

    Returns:
        Tuple of (model, lora_adapter)
    """
    # Get QLoRA config
    qlora_config = get_lora_preset(qlora_preset)
    if not isinstance(qlora_config, QLoRAConfig):
        # Convert to QLoRA config if needed
        qlora_config = QLoRAConfig(**qlora_config.__dict__)

    # Create base model
    base_model = create_model(model_type, **kwargs)

    # Quantize base model if requested
    if quantize_base:
        from .quantization_utils import ModelQuantizer, QuantizationConfig

        quant_config = QuantizationConfig(
            bits=4,
            quantization_type=qlora_config.bnb_4bit_quant_type,
            use_double_quant=qlora_config.bnb_4bit_use_double_quant,
        )

        quantizer = ModelQuantizer(quant_config)
        base_model = quantizer.quantize_model(base_model)
        logger.info("Quantized base model to 4-bit")

    # Create QLoRA adapter
    lora_adapter = LoRAAdapter(base_model, qlora_config)
    lora_adapter.inject_adapters(verbose=True)

    return base_model, lora_adapter


def create_kaggle_lora_model(
    competition_type: str | CompetitionType,
    data_path: str | None = None,
    lora_preset: str | None = None,
    auto_select_preset: bool = True,
    **kwargs,
) -> tuple[nn.Module, LoRAAdapter]:
    """
    Create an optimized LoRA model for a Kaggle competition.

    This function automatically selects the best LoRA configuration
    based on the competition type and dataset characteristics.

    Args:
        competition_type: Type of competition
        data_path: Optional path to data for analysis
        lora_preset: LoRA preset (auto-selected if None)
        auto_select_preset: Whether to auto-select preset
        **kwargs: Additional arguments

    Returns:
        Tuple of (model, lora_adapter)
    """
    # Convert string to CompetitionType if needed
    if isinstance(competition_type, str):
        competition_type = CompetitionType(competition_type)

    # Auto-select LoRA preset based on competition type
    if lora_preset is None and auto_select_preset:
        preset_map = {
            CompetitionType.BINARY_CLASSIFICATION: "balanced",
            CompetitionType.MULTICLASS_CLASSIFICATION: "balanced",
            CompetitionType.MULTILABEL_CLASSIFICATION: "expressive",
            CompetitionType.REGRESSION: "efficient",
            CompetitionType.ORDINAL_REGRESSION: "balanced",
            CompetitionType.TIME_SERIES: "expressive",
            CompetitionType.RANKING: "expressive",
        }
        lora_preset = preset_map.get(competition_type, "balanced")
        logger.info(f"Auto-selected LoRA preset: {lora_preset}")

    # Analyze dataset if provided
    if data_path and DATASET_ANALYSIS_AVAILABLE:
        target_column = kwargs.pop("target_column", "target")
        analysis = analyze_competition_dataset(data_path, target_column)

        # Adjust preset based on dataset size
        if analysis.num_samples > 100000 and lora_preset == "expressive":
            lora_preset = "balanced"  # Use smaller rank for large datasets
            logger.info("Adjusted to 'balanced' preset for large dataset")
        elif analysis.num_samples < 5000 and lora_preset == "efficient":
            lora_preset = "balanced"  # Use larger rank for small datasets
            logger.info("Adjusted to 'balanced' preset for small dataset")

        # Use analysis results
        kwargs["num_labels"] = analysis.num_classes

    # Map competition type to head type
    head_type_map = {
        CompetitionType.BINARY_CLASSIFICATION: "binary_classification",
        CompetitionType.MULTICLASS_CLASSIFICATION: "multiclass_classification",
        CompetitionType.MULTILABEL_CLASSIFICATION: "multilabel_classification",
        CompetitionType.REGRESSION: "regression",
        CompetitionType.ORDINAL_REGRESSION: "ordinal_regression",
        CompetitionType.TIME_SERIES: "time_series",
        CompetitionType.RANKING: "ranking",
    }

    head_type = head_type_map.get(competition_type, "binary_classification")

    # Create model with LoRA
    return create_bert_with_lora(head_type=head_type, lora_preset=lora_preset, **kwargs)


def create_multi_adapter_model(
    base_model: nn.Module | None = None,
    adapter_configs: dict[str, LoRAConfig | dict | str] | None = None,
    **model_kwargs,
) -> tuple[nn.Module, MultiAdapterManager]:
    """
    Create a model with multiple LoRA adapters for multi-task learning.

    Args:
        base_model: Base model (created if None)
        adapter_configs: Dict mapping adapter names to configs
        **model_kwargs: Arguments for model creation

    Returns:
        Tuple of (model, multi_adapter_manager)
    """
    # Create base model if needed
    if base_model is None:
        base_model = create_model(**model_kwargs)

    # Create multi-adapter manager
    manager = MultiAdapterManager(base_model)

    # Add adapters
    if adapter_configs:
        for name, config in adapter_configs.items():
            # Process config
            if isinstance(config, str):
                config = get_lora_preset(config)
            elif isinstance(config, dict):
                if any(k.startswith("bnb_") for k in config):
                    config = QLoRAConfig(**config)
                else:
                    config = LoRAConfig(**config)

            manager.add_adapter(name, config)
            logger.info(f"Added adapter '{name}'")

    return base_model, manager


def create_model_from_checkpoint(checkpoint_path: str | Path) -> nn.Module:
    """
    Create and load a model from a checkpoint directory.

    Args:
        checkpoint_path: Path to checkpoint directory

    Returns:
        Loaded model
    """
    checkpoint_path = Path(checkpoint_path)

    # Load config
    config_path = checkpoint_path / "config.json"
    if not config_path.exists():
        raise ValueError(f"No config.json found in {checkpoint_path}")

    with open(config_path) as f:
        config_dict = json.load(f)

    # Determine model type from config
    # Check if this is a BertWithHead model by looking for head metadata
    metadata_path = checkpoint_path / "model_metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        if metadata.get("model_type") == "BertWithHead":
            # Load as BertWithHead
            from .bert.model import BertWithHead

            return BertWithHead.from_pretrained(checkpoint_path)

    # Default to bert_core
    model_type = "bert_core"

    # Create model
    model = create_model(model_type, config_dict)

    # Load weights
    weights_path = checkpoint_path / "model.safetensors"
    if weights_path.exists():
        from safetensors.mlx import load_model

        load_model(model, str(weights_path))
        logger.info(f"Loaded weights from {weights_path}")
    else:
        logger.warning(f"No weights found at {weights_path}")

    return model


def load_pretrained_weights(model: nn.Module, weights_path: str | Path):
    """
    Load pretrained weights into a model.

    Args:
        model: Model to load weights into
        weights_path: Path to weights file or directory
    """
    weights_path = Path(weights_path)

    if weights_path.is_dir():
        # Load from directory
        safetensors_path = weights_path / "model.safetensors"
        if safetensors_path.exists():
            from safetensors.mlx import load_model

            load_model(model, str(safetensors_path))
            logger.info(f"Loaded weights from {safetensors_path}")
        else:
            raise ValueError(f"No model.safetensors found in {weights_path}")
    elif weights_path.suffix == ".safetensors":
        # Load safetensors file directly
        from safetensors.mlx import load_model

        load_model(model, str(weights_path))
        logger.info(f"Loaded weights from {weights_path}")
    else:
        raise ValueError(f"Unsupported weights format: {weights_path}")


def get_model_config(model_type: ModelType = "bert_with_head", **kwargs) -> BertConfig:
    """
    Get default configuration for a model type.

    Args:
        model_type: Type of model
        **kwargs: Configuration overrides

    Returns:
        Model configuration object
    """
    return BertConfig(**kwargs)


# Model registry for easy access
MODEL_REGISTRY = {
    # Classic BERT models
    "bert-core": lambda **kwargs: create_model("bert_core", **kwargs),
    "bert-binary": lambda **kwargs: create_model(
        "bert_with_head", head_type="binary_classification", **kwargs
    ),
    "bert-multiclass": lambda **kwargs: create_model(
        "bert_with_head", head_type="multiclass_classification", **kwargs
    ),
    "bert-multilabel": lambda **kwargs: create_model(
        "bert_with_head", head_type="multilabel_classification", **kwargs
    ),
    "bert-regression": lambda **kwargs: create_model(
        "bert_with_head", head_type="regression", **kwargs
    ),
    # ModernBERT models
    "modernbert-core": lambda **kwargs: create_model("modernbert_core", **kwargs),
    "modernbert-base": lambda **kwargs: create_model(
        "modernbert_core", model_size="base", **kwargs
    ),
    "modernbert-large": lambda **kwargs: create_model(
        "modernbert_core", model_size="large", **kwargs
    ),
    "modernbert-binary": lambda **kwargs: create_model(
        "modernbert_with_head", head_type="binary_classification", **kwargs
    ),
    "modernbert-multiclass": lambda **kwargs: create_model(
        "modernbert_with_head", head_type="multiclass_classification", **kwargs
    ),
    "modernbert-multilabel": lambda **kwargs: create_model(
        "modernbert_with_head", head_type="multilabel_classification", **kwargs
    ),
    "modernbert-regression": lambda **kwargs: create_model(
        "modernbert_with_head", head_type="regression", **kwargs
    ),
    # LoRA models
    "bert-lora-binary": lambda **kwargs: create_bert_with_lora(
        head_type="binary_classification", **kwargs
    ),
    "bert-lora-multiclass": lambda **kwargs: create_bert_with_lora(
        head_type="multiclass_classification", **kwargs
    ),
    "bert-lora-regression": lambda **kwargs: create_bert_with_lora(
        head_type="regression", **kwargs
    ),
    "modernbert-lora-binary": lambda **kwargs: create_modernbert_with_lora(
        head_type="binary_classification", **kwargs
    ),
    "modernbert-lora-multiclass": lambda **kwargs: create_modernbert_with_lora(
        head_type="multiclass_classification", **kwargs
    ),
    "modernbert-lora-regression": lambda **kwargs: create_modernbert_with_lora(
        head_type="regression", **kwargs
    ),
    # QLoRA models (memory efficient)
    "bert-qlora-binary": lambda **kwargs: create_qlora_model(
        model_type="bert_with_head", head_type="binary_classification", **kwargs
    ),
    "modernbert-qlora-binary": lambda **kwargs: create_qlora_model(
        model_type="modernbert_with_head", head_type="binary_classification", **kwargs
    ),
    # Competition-specific models
    "titanic-bert": lambda **kwargs: create_bert_for_task(
        "binary_classification", num_labels=2, **kwargs
    ),
    "titanic-modernbert": lambda **kwargs: create_model(
        "modernbert_with_head",
        head_type="binary_classification",
        num_labels=2,
        **kwargs,
    ),
    "titanic-bert-lora": lambda **kwargs: create_kaggle_lora_model(
        "binary_classification", num_labels=2, **kwargs
    ),
    "titanic-modernbert-lora": lambda **kwargs: create_modernbert_with_lora(
        head_type="binary_classification",
        lora_preset="balanced",
        num_labels=2,
        **kwargs,
    ),
}


def list_available_models() -> list[str]:
    """List all available model types in the registry."""
    return list(MODEL_REGISTRY.keys())


def create_from_registry(model_name: str, **kwargs) -> nn.Module:
    """
    Create a model from the registry by name.

    Args:
        model_name: Name of model in registry
        **kwargs: Model configuration parameters

    Returns:
        Initialized model
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {', '.join(list_available_models())}"
        )

    return MODEL_REGISTRY[model_name](**kwargs)


# === ADVANCED CLASSIFICATION FUNCTIONS (absorbed from classification/factory.py) ===


def create_kaggle_classifier(
    task_type: str,
    num_classes: int | None = None,
    dataset_spec: Optional["KaggleDatasetSpec"] = None,
    **kwargs,
) -> nn.Module:
    """
    Create classifier optimized for Kaggle competitions using BertWithHead.

    This is the main entry point for creating classifiers. It automatically
    selects the best configuration based on the task type and dataset characteristics.

    Args:
        task_type: Type of classification task ("binary", "multiclass", "regression",
                   "titanic", "multilabel", "ordinal", "hierarchical", "ensemble",
                   "time_series", "ranking", "contrastive", "multi_task", "metric_learning")
        num_classes: Number of classes/labels
        dataset_spec: Optional dataset specification for optimization
        **kwargs: Additional arguments

    Returns:
        Configured classifier instance
    """
    # Map task types to head types
    task_to_head_map = {
        "binary": "binary_classification",
        "multiclass": "multiclass_classification",
        "multilabel": "multilabel_classification",
        "regression": "regression",
        "ordinal": "ordinal_regression",
        "time_series": "time_series",
        "ranking": "ranking",
    }

    head_type = task_to_head_map.get(task_type, "multiclass_classification")

    # Use dataset spec to optimize configuration if available
    if dataset_spec and hasattr(dataset_spec, "optimization_profile"):
        logger.info(
            f"Optimizing for dataset: {dataset_spec.name} (profile: {dataset_spec.optimization_profile})"
        )

        # Auto-configure based on dataset characteristics
        if dataset_spec.optimization_profile.value == "competition":
            kwargs.setdefault("dropout_prob", 0.1)
            kwargs.setdefault("use_layer_norm", True)
            kwargs.setdefault("pooling_type", "attention")
        elif dataset_spec.optimization_profile.value == "development":
            kwargs.setdefault("dropout_prob", 0.2)
            kwargs.setdefault("use_layer_norm", False)
            kwargs.setdefault("pooling_type", "mean")

    return create_bert_with_head(
        head_type=head_type, num_labels=num_classes or 2, **kwargs
    )


def create_competition_classifier(
    data_path: str,
    target_column: str,
    model_name: str = "answerdotai/ModernBERT-base",
    auto_optimize: bool = True,
    **kwargs,
) -> tuple[nn.Module, CompetitionAnalysis]:
    """
    Automatically analyze dataset and create optimized classifier.

    This function combines dataset analysis with classifier creation to provide
    a fully automated solution for Kaggle competitions.

    Args:
        data_path: Path to the training data CSV
        target_column: Name of the target column
        model_name: Name of the embedding model to use
        auto_optimize: Whether to automatically optimize configuration
        **kwargs: Additional arguments

    Returns:
        Tuple of (classifier, analysis_results)
    """
    if not DATASET_ANALYSIS_AVAILABLE:
        logger.warning("Dataset analysis not available, using basic classifier")
        return create_model("classifier", **kwargs), None

    # Analyze dataset
    analysis = analyze_competition_dataset(data_path, target_column)

    # Create optimized classifier based on analysis
    if auto_optimize:
        # Override kwargs with optimized settings
        kwargs.update(
            {
                "dropout_prob": 0.1 if analysis.num_samples > 10000 else 0.2,
                "use_layer_norm": analysis.num_features > 50,
                "pooling_type": "attention" if analysis.num_samples > 5000 else "mean",
                "batch_size": analysis.recommended_batch_size,
            }
        )

        logger.info(f"Auto-optimized configuration: {kwargs}")

    # Determine task type from analysis
    task_type_map = {
        CompetitionType.BINARY_CLASSIFICATION: "binary",
        CompetitionType.MULTICLASS_CLASSIFICATION: "multiclass",
        CompetitionType.MULTILABEL_CLASSIFICATION: "multilabel",
        CompetitionType.REGRESSION: "regression",
        CompetitionType.ORDINAL_REGRESSION: "ordinal",
        CompetitionType.TIME_SERIES: "time_series",
        CompetitionType.RANKING: "ranking",
    }

    task_type = task_type_map.get(analysis.competition_type, "multiclass")

    # Create classifier
    classifier = create_kaggle_classifier(
        task_type=task_type,
        model_name=model_name,
        num_classes=analysis.num_classes,
        **kwargs,
    )

    logger.info(f"Created {task_type} classifier for {analysis.competition_type.value}")
    return classifier, analysis


def analyze_competition_dataset(
    data_path: str, target_column: str
) -> CompetitionAnalysis:
    """
    Analyze a competition dataset and return optimization recommendations.

    Args:
        data_path: Path to the training data CSV
        target_column: Name of the target column

    Returns:
        CompetitionAnalysis with recommendations
    """
    # Load and analyze data
    df = pd.read_csv(data_path)

    # Basic characteristics
    num_samples = len(df)
    num_features = len(df.columns) - 1  # Exclude target

    # Analyze target column
    target_series = df[target_column]
    num_classes = target_series.nunique()
    class_distribution = target_series.value_counts().to_dict()

    # Determine competition type
    if target_series.dtype in ["object", "category"]:
        if num_classes == 2:
            competition_type = CompetitionType.BINARY_CLASSIFICATION
        else:
            competition_type = CompetitionType.MULTICLASS_CLASSIFICATION
    else:
        # Check if it's regression or classification with numeric labels
        if num_classes <= 10 and target_series.dtype in ["int64", "int32"]:
            competition_type = CompetitionType.MULTICLASS_CLASSIFICATION
        else:
            competition_type = CompetitionType.REGRESSION

    # Analyze features
    categorical_columns = []
    numerical_columns = []
    text_columns = []

    for col in df.columns:
        if col == target_column:
            continue

        series = df[col]

        if series.dtype == "object":
            # If most values are unique, likely text
            if series.nunique() / len(series) > 0.9:
                text_columns.append(col)
            else:
                categorical_columns.append(col)
        elif pd.api.types.is_numeric_dtype(series):
            # If low cardinality, might be categorical
            if series.nunique() <= 10 and series.dtype in ["int64", "int32"]:
                categorical_columns.append(col)
            else:
                numerical_columns.append(col)
        else:
            categorical_columns.append(col)

    # Check class balance
    if competition_type in [
        CompetitionType.BINARY_CLASSIFICATION,
        CompetitionType.MULTICLASS_CLASSIFICATION,
    ]:
        min_class_count = min(class_distribution.values())
        max_class_count = max(class_distribution.values())
        is_balanced = (min_class_count / max_class_count) > 0.1
    else:
        is_balanced = True

    # Recommendations
    recommended_batch_size = 32
    if num_samples > 100000:
        recommended_batch_size = 64
    elif num_samples < 5000:
        recommended_batch_size = 16

    recommended_learning_rate = 2e-5
    if not is_balanced:
        recommended_learning_rate = 1e-5  # Lower LR for imbalanced datasets

    return CompetitionAnalysis(
        competition_type=competition_type,
        num_samples=num_samples,
        num_features=num_features,
        target_column=target_column,
        num_classes=num_classes,
        class_distribution=class_distribution,
        is_balanced=is_balanced,
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        text_columns=text_columns,
        recommended_batch_size=recommended_batch_size,
        recommended_learning_rate=recommended_learning_rate,
        recommended_model_type="generic",
        recommended_head_type="binary"
        if competition_type == CompetitionType.BINARY_CLASSIFICATION
        else "multiclass",
    )


# === CONVENIENCE FUNCTIONS (absorbed from classification/factory.py) ===


def create_titanic_classifier_optimized(**kwargs) -> nn.Module:
    """Create optimized Titanic classifier using BertWithHead."""
    return create_bert_with_head(
        head_type="binary_classification", num_labels=2, **kwargs
    )


def create_multilabel_classifier_optimized(num_labels: int, **kwargs) -> nn.Module:
    """Create optimized multilabel classifier using BertWithHead."""
    return create_bert_with_head(
        head_type="multilabel_classification", num_labels=num_labels, **kwargs
    )


def create_ensemble_classifier_optimized(num_classes: int, **kwargs) -> nn.Module:
    """Create optimized ensemble classifier using BertWithHead."""
    return create_bert_with_head(head_type="ensemble", num_labels=num_classes, **kwargs)


# === BACKWARD COMPATIBILITY ALIASES ===

# For models/classification/factory.py compatibility
create_classifier_advanced = create_kaggle_classifier

# For factories/kaggle_competition_factory.py compatibility
create_competition_model = create_competition_classifier
analyze_dataset = analyze_competition_dataset

# Legacy function names
create_enhanced_classifier = create_kaggle_classifier


# === BACKWARD COMPATIBILITY FOR LORA ===

# Export LoRA creation functions
__all__ = [
    # Core creation functions
    "create_model",
    "create_model_from_checkpoint",
    "create_from_registry",
    "list_available_models",
    # LoRA creation functions
    "create_model_with_lora",
    "create_bert_with_lora",
    "create_modernbert_with_lora",
    "create_qlora_model",
    "create_kaggle_lora_model",
    "create_multi_adapter_model",
    # Competition functions
    "create_kaggle_classifier",
    "create_competition_classifier",
    "analyze_competition_dataset",
    # Legacy compatibility
    "create_bert_for_task",
    "create_bert_from_dataset",
]


# === NEW MODULAR BERT ARCHITECTURE FUNCTIONS ===


def create_modular_bert(
    pretrained_name: str | None = None,
    config: BertConfig | dict | None = None,
    **kwargs,
) -> BertCore:
    """Create a modular BERT core model.

    Args:
        pretrained_name: Optional pretrained model name
        config: Optional configuration
        **kwargs: Additional configuration parameters

    Returns:
        BertCore model
    """
    return create_bert_core(model_name=pretrained_name, config=config, **kwargs)


def create_bert_for_task(
    task: str | CompetitionType,
    pretrained_name: str | None = None,
    num_labels: int = 2,
    freeze_bert: bool = False,
    **kwargs,
) -> BertWithHead:
    """Create a BERT model with appropriate head for a specific task.

    Args:
        task: Task type (string or CompetitionType)
        pretrained_name: Optional pretrained model name
        num_labels: Number of output labels
        freeze_bert: Whether to freeze BERT parameters
        **kwargs: Additional arguments

    Returns:
        BertWithHead model
    """
    # Convert task to head type string if needed
    if isinstance(task, str):
        # Try to parse as CompetitionType
        try:
            comp_type = CompetitionType(task)
            # Map competition type to head type
            comp_to_head_map = {
                CompetitionType.BINARY_CLASSIFICATION: "binary_classification",
                CompetitionType.MULTICLASS_CLASSIFICATION: "multiclass_classification",
                CompetitionType.MULTILABEL_CLASSIFICATION: "multilabel_classification",
                CompetitionType.REGRESSION: "regression",
                CompetitionType.ORDINAL_REGRESSION: "ordinal_regression",
                CompetitionType.TIME_SERIES: "time_series",
                CompetitionType.RANKING: "ranking",
            }
            head_type = comp_to_head_map.get(comp_type, "binary_classification")
        except ValueError:
            # Task is already a string head type
            head_type = task
    elif isinstance(task, CompetitionType):
        # Map competition type to head type
        comp_to_head_map = {
            CompetitionType.BINARY_CLASSIFICATION: "binary_classification",
            CompetitionType.MULTICLASS_CLASSIFICATION: "multiclass_classification",
            CompetitionType.MULTILABEL_CLASSIFICATION: "multilabel_classification",
            CompetitionType.REGRESSION: "regression",
            CompetitionType.ORDINAL_REGRESSION: "ordinal_regression",
            CompetitionType.TIME_SERIES: "time_series",
            CompetitionType.RANKING: "ranking",
        }
        head_type = comp_to_head_map.get(task, "binary_classification")
    else:
        head_type = task

    return create_bert_with_head(
        bert_name=pretrained_name,
        head_type=head_type,
        num_labels=num_labels,
        freeze_bert=freeze_bert,
        **kwargs,
    )


def create_bert_from_dataset(
    dataset_path: str | Path,
    pretrained_name: str | None = None,
    auto_analyze: bool = True,
    **kwargs,
) -> BertWithHead:
    """Create a BERT model optimized for a specific dataset.

    Args:
        dataset_path: Path to dataset file
        pretrained_name: Optional pretrained model name
        auto_analyze: Whether to automatically analyze dataset
        **kwargs: Additional arguments

    Returns:
        BertWithHead model optimized for the dataset
    """
    if auto_analyze and DATASET_ANALYSIS_AVAILABLE:
        # Analyze dataset
        # For now, we need to provide target_column
        # In a real implementation, this would be auto-detected
        target_column = kwargs.pop("target_column", "target")
        analysis = analyze_competition_dataset(str(dataset_path), target_column)

        # Get competition type
        comp_type = analysis.competition_type
        num_labels = analysis.num_classes or 2

        logger.info(f"Dataset analysis: {comp_type.value}, {num_labels} classes")

        # Create optimized model
        return create_bert_for_competition(
            competition_type=comp_type,
            bert_name=pretrained_name,
            num_labels=num_labels,
            **kwargs,
        )
    else:
        # Default to binary classification
        return create_bert_for_task(
            task="binary_classification", pretrained_name=pretrained_name, **kwargs
        )
