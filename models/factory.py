"""MASTER MODEL FACTORY - Single source for all model creation.

This factory creates models using the modular BERT architecture with pluggable heads.
It supports:
- BertCore creation
- Head attachment via BertWithHead
- Competition-specific model creation
- Automatic dataset analysis and optimization
- LoRA/QLoRA adapter injection for efficient fine-tuning

NOTE: This module now delegates to focused builders for better separation of concerns.
The API remains the same for backward compatibility.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import pandas as pd
from loguru import logger

# Import MLX for type annotations (TODO: abstract through ports)
import mlx.nn as nn

# Import advanced logging features
from utils.logging_utils import (
    catch_and_log,
    log_timing,
    lazy_debug,
    bind_context
)

# Import dependency injection and ports
from core.bootstrap import get_service
from core.ports.compute import ComputeBackend

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

# Kaggle heads are not currently available
KAGGLE_HEADS_AVAILABLE = False

# Import dataset analysis for automatic optimization
try:
    from data.dataset_spec import KaggleDatasetSpec
    from data.core.base import CompetitionType

    DATASET_ANALYSIS_AVAILABLE = True
except ImportError:
    DATASET_ANALYSIS_AVAILABLE = False
    logger.debug("Dataset analysis not available")
    # Define CompetitionType locally if not available
    from enum import Enum
    
    class CompetitionType(Enum):
        """Types of Kaggle competitions."""
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
]


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


@catch_and_log(
    ValueError,
    "Model creation failed",
    reraise=True
)
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
    # Create logger with context
    log = bind_context(model_type=model_type, head_type=head_type)
    log.info(f"Creating model: {model_type}")
    
    # Use lazy debug for expensive config serialization
    lazy_debug(
        "Model configuration",
        lambda: {
            "config": config.__dict__ if hasattr(config, "__dict__") else config,
            "head_config": head_config.__dict__ if hasattr(head_config, "__dict__") else head_config,
            "kwargs": kwargs
        }
    )
    
    # Get services through hexagonal architecture
    compute_backend = get_service(ComputeBackend)
    
    # Delegate to appropriate creation method
    if model_type in ["bert_core", "modernbert_core"]:
        # Create core model using compute backend
        model = _create_core_model(model_type, config, compute_backend, **kwargs)
    elif model_type in ["bert_with_head", "modernbert_with_head"]:
        # Extract common parameters
        num_labels = kwargs.pop("num_labels", 2)
        freeze_bert = kwargs.pop("freeze_bert", False)
        freeze_bert_layers = kwargs.pop("freeze_bert_layers", None)
        
        # Create model with head using compute backend
        model = _create_model_with_head(
            model_type, config, compute_backend, head_type, head_config, 
            num_labels=num_labels, freeze_bert=freeze_bert, 
            freeze_bert_layers=freeze_bert_layers, **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load pretrained weights if provided
    if pretrained_path:
        with log_timing("load_pretrained_weights", path=str(pretrained_path)):
            # TODO: Implement weight loading through ports
            logger.debug(f"Loading pretrained weights from {pretrained_path}")
    
    # Log model statistics
    if hasattr(model, "num_parameters"):
        param_count = model.num_parameters()
        log.info(f"Model created with {param_count:,} parameters")

    return model


# Parameter breakdown function removed - now handled by ModelBuilder


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


@catch_and_log(
    Exception,
    "Failed to load model from checkpoint",
    reraise=True
)
def create_model_from_checkpoint(checkpoint_path: str | Path) -> nn.Module:
    """
    Create and load a model from a checkpoint directory.

    Args:
        checkpoint_path: Path to checkpoint directory

    Returns:
        Loaded model
    """
    checkpoint_path = Path(checkpoint_path)
    log = bind_context(checkpoint=str(checkpoint_path))
    
    with log_timing("load_model_from_checkpoint", checkpoint=str(checkpoint_path)):
        log.info(f"Loading model from checkpoint: {checkpoint_path}")

        # Look for metadata.json instead of config.json
        metadata_path = checkpoint_path / "metadata.json"

        # Try to find training configuration in parent directories
        training_config_path = None
        current_path = checkpoint_path
        for _ in range(3):  # Search up to 3 levels up
            current_path = current_path.parent
            potential_config = current_path / "training_config.json"
            if potential_config.exists():
                training_config_path = potential_config
                break

        # Load training config if available
        if training_config_path:
            with open(training_config_path) as f:
                training_config = json.load(f)
            logger.info(f"Found training config at {training_config_path}")
        else:
            # Use defaults
            training_config = {"model": "answerdotai/ModernBERT-base", "model_type": "base"}
            logger.warning("No training config found, using defaults")

        # Check if we have training state for more model info
        training_state_path = checkpoint_path / "training_state.json"
        if training_state_path.exists():
            with open(training_state_path) as f:
                training_state = json.load(f)
            logger.info(f"Found training state at {training_state_path}")
        else:
            training_state = {}

        # Load weights first to infer model architecture
        weights_path = checkpoint_path / "model.safetensors"
        if not weights_path.exists():
            raise ValueError(f"No model.safetensors found in {checkpoint_path}")

        import mlx.core as mx

        weights = mx.load(str(weights_path))
        weight_keys = list(weights.keys())

        # Infer model type from weight keys by checking the number of encoder layers
        encoder_layer_keys = [k for k in weight_keys if "encoder_layers" in k]
        if encoder_layer_keys:
            # Extract layer numbers
            layer_numbers = set()
            for key in encoder_layer_keys:
                parts = key.split('.')
                for i, part in enumerate(parts):
                    if part == "encoder_layers" and i + 1 < len(parts):
                        try:
                            layer_num = int(parts[i + 1])
                            layer_numbers.add(layer_num)
                        except ValueError:
                            pass
            
            max_layer = max(layer_numbers) if layer_numbers else 0
            
            if max_layer > 11:  # ModernBERT has 22 layers (0-21)
                model_type = "modernbert_with_head"
                logger.info(f"Detected ModernBERT architecture (found {max_layer + 1} layers)")
            else:  # Classic BERT has 12 layers (0-11)
                model_type = "bert_with_head"
                logger.info(f"Detected classic BERT architecture (found {max_layer + 1} layers)")
        else:
            # Default to classic BERT for backward compatibility
            model_type = "bert_with_head"
            logger.warning(
                "Could not determine architecture from weights, defaulting to classic BERT"
            )

        # Create model with appropriate architecture
        model = create_model(
            model_type=model_type,
            head_type="binary_classification",
            num_labels=2,
            model_size=training_config.get("model_type", "base"),
        )

        # Load weights into model using tree_unflatten to restore hierarchical structure
        from mlx.utils import tree_unflatten
        unflattened_weights = tree_unflatten(list(weights.items()))
        model.update(unflattened_weights)
        logger.info(f"Loaded weights from {weights_path}")

        return model


def load_pretrained_weights(model: nn.Module, weights_path: str | Path):
    """
    Load pretrained weights into a model.

    Args:
        model: Model to load weights into
        weights_path: Path to weights file or directory
    """
    # Delegate to model builder
    container = get_container()
    model_builder = container.get_model_builder()
    model_builder.load_pretrained_weights(model, weights_path)


def get_model_config(**kwargs) -> BertConfig:
    """
    Get default configuration for a model.

    Args:
        **kwargs: Configuration overrides

    Returns:
        Model configuration object
    """
    return BertConfig(**kwargs)


# Model registry is now managed by ModelRegistry class
# This dictionary is maintained for backward compatibility
MODEL_REGISTRY = {}

def _init_model_registry():
    """Initialize the backward-compatible model registry."""
    global MODEL_REGISTRY
    
    # Get registry from DI container
    try:
        from core.di.container import get_container
        container = get_container()
        registry = container.get_registry()
    except Exception:
        # Fallback for when DI container is not available
        logger.debug("DI container not available, skipping model registry initialization")
        return
    
    # Add LoRA models to registry (not in default registration)
    registry.register(
        "bert-lora-binary",
        lambda **kwargs: create_bert_with_lora(
            head_type="binary_classification", **kwargs
        ),
        description="BERT with LoRA for binary classification",
        category="lora",
        tags=["bert", "lora", "binary"],
    )
    
    registry.register(
        "bert-lora-multiclass",
        lambda **kwargs: create_bert_with_lora(
            head_type="multiclass_classification", **kwargs
        ),
        description="BERT with LoRA for multi-class classification",
        category="lora",
        tags=["bert", "lora", "multiclass"],
    )
    
    registry.register(
        "bert-lora-regression",
        lambda **kwargs: create_bert_with_lora(
            head_type="regression", **kwargs
        ),
        description="BERT with LoRA for regression",
        category="lora",
        tags=["bert", "lora", "regression"],
    )
    
    registry.register(
        "modernbert-lora-binary",
        lambda **kwargs: create_modernbert_with_lora(
            head_type="binary_classification", **kwargs
        ),
        description="ModernBERT with LoRA for binary classification",
        category="lora",
        tags=["modernbert", "lora", "binary"],
    )
    
    registry.register(
        "modernbert-lora-multiclass",
        lambda **kwargs: create_modernbert_with_lora(
            head_type="multiclass_classification", **kwargs
        ),
        description="ModernBERT with LoRA for multi-class classification",
        category="lora",
        tags=["modernbert", "lora", "multiclass"],
    )
    
    registry.register(
        "modernbert-lora-regression",
        lambda **kwargs: create_modernbert_with_lora(
            head_type="regression", **kwargs
        ),
        description="ModernBERT with LoRA for regression",
        category="lora",
        tags=["modernbert", "lora", "regression"],
    )
    
    # QLoRA models
    registry.register(
        "bert-qlora-binary",
        lambda **kwargs: create_qlora_model(
            model_type="bert_with_head", head_type="binary_classification", **kwargs
        ),
        description="BERT with QLoRA for binary classification",
        category="qlora",
        tags=["bert", "qlora", "binary", "memory-efficient"],
    )
    
    registry.register(
        "modernbert-qlora-binary",
        lambda **kwargs: create_qlora_model(
            model_type="modernbert_with_head", head_type="binary_classification", **kwargs
        ),
        description="ModernBERT with QLoRA for binary classification",
        category="qlora",
        tags=["modernbert", "qlora", "binary", "memory-efficient"],
    )
    
    # Additional specialized models
    registry.register(
        "bert-multilabel",
        lambda **kwargs: create_model(
            "bert_with_head", head_type="multilabel_classification", **kwargs
        ),
        description="BERT for multi-label classification",
        category="classification",
        tags=["bert", "classification", "multilabel"],
    )
    
    registry.register(
        "modernbert-multilabel",
        lambda **kwargs: create_model(
            "modernbert_with_head", head_type="multilabel_classification", **kwargs
        ),
        description="ModernBERT for multi-label classification",
        category="classification",
        tags=["modernbert", "classification", "multilabel"],
    )
    
    registry.register(
        "modernbert-base",
        lambda **kwargs: create_model(
            "modernbert_core", model_size="base", **kwargs
        ),
        description="ModernBERT base size core model",
        category="core",
        tags=["modernbert", "core", "base"],
    )
    
    registry.register(
        "modernbert-large",
        lambda **kwargs: create_model(
            "modernbert_core", model_size="large", **kwargs
        ),
        description="ModernBERT large size core model",
        category="core",
        tags=["modernbert", "core", "large"],
    )
    
    # Build backward-compatible dictionary
    for name in registry.list_models():
        MODEL_REGISTRY[name] = lambda n=name, **kwargs: registry.create(n, **kwargs)

# Initialize on module import
_init_model_registry()


def list_available_models() -> list[str]:
    """List all available model types in the registry."""
    container = get_container()
    registry = container.get_registry()
    return registry.list_models()


def create_from_registry(model_name: str, **kwargs) -> nn.Module:
    """
    Create a model from the registry by name.

    Args:
        model_name: Name of model in registry
        **kwargs: Model configuration parameters

    Returns:
        Initialized model
    """
    container = get_container()
    registry = container.get_registry()
    return registry.create(model_name, **kwargs)


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
    # Get head factory from container
    container = get_container()
    head_factory = container.get_head_factory()
    
    # Get appropriate head type for task
    head_type = head_factory.get_head_for_task(task_type)

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

    # Get head factory from container
    container = get_container()
    head_factory = container.get_head_factory()
    
    # Get appropriate head type from competition type
    head_type = head_factory.get_head_for_competition_type(analysis.competition_type)
    
    # Map to task type for create_kaggle_classifier
    head_to_task_map = {
        "binary_classification": "binary",
        "multiclass_classification": "multiclass",
        "multilabel_classification": "multilabel",
        "regression": "regression",
        "ordinal_regression": "ordinal",
        "time_series": "time_series",
        "ranking": "ranking",
    }
    
    task_type = head_to_task_map.get(head_type, "multiclass")

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




# === BACKWARD COMPATIBILITY ALIASES ===

# For models/classification/factory.py compatibility
create_classifier_advanced = create_kaggle_classifier

# For factories/kaggle_competition_factory.py compatibility
create_competition_model = create_competition_classifier
analyze_dataset = analyze_competition_dataset


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
    # Get head factory from container
    container = get_container()
    head_factory = container.get_head_factory()
    
    # Convert task to head type string if needed
    if isinstance(task, str):
        # Try to parse as CompetitionType
        try:
            comp_type = CompetitionType(task)
            head_type = head_factory.get_head_for_competition_type(comp_type)
        except ValueError:
            # Task is already a string head type
            head_type = task
    elif isinstance(task, CompetitionType):
        head_type = head_factory.get_head_for_competition_type(task)
    else:
        head_type = str(task)

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


# Helper functions for hexagonal architecture

def _create_core_model(model_type: str, config: Any, compute_backend: ComputeBackend, **kwargs):
    """Create a core BERT model using the compute backend."""
    from .bert import create_bert_core, create_modernbert_core
    
    if model_type == "bert_core":
        return create_bert_core(config=config, **kwargs)
    elif model_type == "modernbert_core":
        return create_modernbert_core(config=config, **kwargs)
    else:
        raise ValueError(f"Unsupported core model type: {model_type}")


def _create_model_with_head(
    model_type: str,
    config: Any,
    compute_backend: ComputeBackend,
    head_type: str,
    head_config: Any,
    **kwargs
):
    """Create a BERT model with head using the compute backend."""
    from .bert import create_bert_with_head, create_modernbert_with_head
    
    if model_type == "bert_with_head":
        return create_bert_with_head(
            config=config,
            head_type=head_type,
            head_config=head_config,
            **kwargs
        )
    elif model_type == "modernbert_with_head":
        return create_modernbert_with_head(
            config=config,
            head_type=head_type,
            head_config=head_config,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported model with head type: {model_type}")


class ModelFactory:
    """
    Factory class that uses dependency injection and hexagonal architecture.
    
    This replaces direct function calls with a service that can be injected
    and uses ports/adapters for external dependencies.
    """
    
    def __init__(self):
        self.compute_backend = get_service(ComputeBackend)
    
    def create_model(
        self,
        model_name: str,
        model_type: str = "modernbert_with_head",
        head_type: str = "binary_classification",
        num_labels: int = 2,
        **kwargs
    ):
        """
        Create a model using dependency injection and hexagonal architecture.
        
        Args:
            model_name: Name/path of the pretrained model
            model_type: Type of model architecture
            head_type: Type of task head
            num_labels: Number of output labels
            **kwargs: Additional configuration
            
        Returns:
            Created model
        """
        logger.info(f"Creating {model_type} model: {model_name}")
        
        # Use the global create_model function but with DI services
        return create_model(
            model_type=model_type,
            pretrained_path=model_name,
            head_type=head_type,
            num_labels=num_labels,
            **kwargs
        )
