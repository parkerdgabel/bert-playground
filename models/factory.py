"""Model factory with decomposed architecture using focused builders.

This factory maintains backward compatibility while internally using focused builders
following the Single Responsibility Principle and hexagonal architecture.

The monolithic factory has been decomposed into:
- CoreModelBuilder: Core model creation
- ModelWithHeadBuilder: Models with task heads
- LoRABuilder: LoRA adapter management
- CheckpointManager: Model loading/saving
- CompetitionBuilder: Competition-optimized models

All original functions are preserved for backward compatibility.
"""

# Import the new factory facade
from .factory_facade import (
    ModelFactory,
    create_model as _create_model,
    create_model_with_lora as _create_model_with_lora,
    create_model_from_checkpoint as _create_model_from_checkpoint,
    create_kaggle_classifier as _create_kaggle_classifier,
    create_competition_classifier as _create_competition_classifier,
    create_kaggle_lora_model as _create_kaggle_lora_model,
    analyze_competition_dataset as _analyze_competition_dataset,
)

# Import types for backward compatibility
from .factory_facade import ModelType
from .builders import CompetitionAnalysis
from .bert import BertConfig, BertCore, BertWithHead, ModernBertConfig
from .heads.base import HeadConfig
from .lora import LoRAAdapter, LoRAConfig, MultiAdapterManager, QLoRAConfig, get_lora_preset
from data.core.base import CompetitionType

from pathlib import Path
from typing import Any, Optional
from loguru import logger

# Global factory instance for backward compatibility
_factory = ModelFactory()

# Dataset analysis availability check
try:
    from data.dataset_spec import KaggleDatasetSpec
    DATASET_ANALYSIS_AVAILABLE = True
except ImportError:
    DATASET_ANALYSIS_AVAILABLE = False
    logger.debug("Dataset analysis not available")


# === MAIN API FUNCTIONS (delegate to facade) ===

def create_model(
    model_type: ModelType = "bert_with_head",
    config: dict[str, Any] | BertConfig | ModernBertConfig | None = None,
    pretrained_path: str | Path | None = None,
    head_type: str | None = None,
    head_config: HeadConfig | dict | None = None,
    **kwargs,
):
    """Create a model based on type and configuration.
    
    Delegates to the new factory facade for focused builder architecture.
    """
    return _create_model(
        model_type=model_type,
        config=config,
        pretrained_path=pretrained_path,
        head_type=head_type,
        head_config=head_config,
        **kwargs,
    )


def create_model_with_lora(
    base_model=None,
    model_type=None,
    lora_config=None,
    inject_lora: bool = True,
    verbose: bool = False,
    **model_kwargs,
):
    """Create a model with LoRA adapters.
    
    Delegates to the new factory facade.
    """
    return _create_model_with_lora(
        base_model=base_model,
        model_type=model_type,
        lora_config=lora_config,
        inject_lora=inject_lora,
        verbose=verbose,
        **model_kwargs,
    )


def create_bert_with_lora(
    head_type: str | None = None,
    lora_preset: str = "balanced",
    num_labels: int = 2,
    freeze_bert: bool = True,
    **kwargs,
):
    """Create a BERT model with head and LoRA adapters."""
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
):
    """Create a ModernBERT model with head and LoRA adapters."""
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
):
    """Create a model with QLoRA (quantized base + LoRA adapters)."""
    # Create base model first
    base_model = create_model(model_type, **kwargs)
    
    # Use LoRA builder to create QLoRA
    return _factory.lora_builder.build_qlora_model(
        base_model, qlora_preset, quantize_base, verbose=True
    )


def create_kaggle_lora_model(
    competition_type: str | CompetitionType,
    data_path: str | None = None,
    lora_preset: str | None = None,
    auto_select_preset: bool = True,
    **kwargs,
):
    """Create an optimized LoRA model for a Kaggle competition."""
    return _create_kaggle_lora_model(
        competition_type=competition_type,
        data_path=data_path,
        lora_preset=lora_preset,
        auto_select_preset=auto_select_preset,
        **kwargs,
    )


def create_multi_adapter_model(
    base_model=None,
    adapter_configs: dict[str, LoRAConfig | dict | str] | None = None,
    **model_kwargs,
):
    """Create a model with multiple LoRA adapters for multi-task learning."""
    # Create base model if needed
    if base_model is None:
        base_model = create_model(**model_kwargs)

    return _factory.lora_builder.build_multi_adapter_model(
        base_model, adapter_configs
    )


def create_model_from_checkpoint(checkpoint_path: str | Path):
    """Create and load a model from a checkpoint directory."""
    return _create_model_from_checkpoint(checkpoint_path)


def load_pretrained_weights(model, weights_path: str | Path):
    """Load pretrained weights into a model."""
    _factory.checkpoint_manager.load_pretrained_weights(model, weights_path)


def get_model_config(**kwargs) -> BertConfig:
    """Get default configuration for a model."""
    return BertConfig(**kwargs)


def create_kaggle_classifier(
    task_type: str,
    num_classes: int | None = None,
    **kwargs,
):
    """Create classifier optimized for Kaggle competitions."""
    return _create_kaggle_classifier(
        task_type=task_type, num_classes=num_classes, **kwargs
    )


def create_competition_classifier(
    data_path: str,
    target_column: str,
    model_name: str = "answerdotai/ModernBERT-base",
    auto_optimize: bool = True,
    **kwargs,
):
    """Automatically analyze dataset and create optimized classifier."""
    return _create_competition_classifier(
        data_path=data_path,
        target_column=target_column,
        model_name=model_name,
        auto_optimize=auto_optimize,
        **kwargs,
    )


def analyze_competition_dataset(data_path: str, target_column: str):
    """Analyze a competition dataset and return optimization recommendations."""
    return _analyze_competition_dataset(data_path, target_column)


# === REGISTRY AND CONVENIENCE FUNCTIONS ===

def list_available_models() -> list[str]:
    """List all available model types in the registry."""
    # This would need to be implemented in the registry builder
    return ["bert_core", "bert_with_head", "modernbert_core", "modernbert_with_head"]


def create_from_registry(model_name: str, **kwargs):
    """Create a model from the registry by name."""
    # Map common registry names to model types
    name_to_type = {
        "bert-binary": ("bert_with_head", {"head_type": "binary_classification"}),
        "bert-multiclass": ("bert_with_head", {"head_type": "multiclass_classification"}),
        "modernbert-binary": ("modernbert_with_head", {"head_type": "binary_classification"}),
        "modernbert-multiclass": ("modernbert_with_head", {"head_type": "multiclass_classification"}),
    }
    
    if model_name in name_to_type:
        model_type, extra_kwargs = name_to_type[model_name]
        kwargs.update(extra_kwargs)
        return create_model(model_type, **kwargs)
    else:
        raise ValueError(f"Unknown model name in registry: {model_name}")


# === CONVENIENCE FUNCTIONS FOR MODULAR BERT ===

def create_modular_bert(
    pretrained_name: str | None = None,
    config: BertConfig | dict | None = None,
    **kwargs,
):
    """Create a modular BERT core model."""
    return create_model("bert_core", config=config, pretrained_path=pretrained_name, **kwargs)


def create_bert_for_task(
    task: str | CompetitionType,
    pretrained_name: str | None = None,
    num_labels: int = 2,
    freeze_bert: bool = False,
    **kwargs,
):
    """Create a BERT model with appropriate head for a specific task."""
    # Convert task to head type string if needed
    if isinstance(task, str):
        try:
            comp_type = CompetitionType(task)
            head_type = _get_head_for_competition_type(comp_type)
        except ValueError:
            head_type = task
    elif isinstance(task, CompetitionType):
        head_type = _get_head_for_competition_type(task)
    else:
        head_type = str(task)

    return create_model(
        "bert_with_head",
        pretrained_path=pretrained_name,
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
):
    """Create a BERT model optimized for a specific dataset."""
    if auto_analyze and DATASET_ANALYSIS_AVAILABLE:
        target_column = kwargs.pop("target_column", "target")
        analysis = analyze_competition_dataset(str(dataset_path), target_column)

        comp_type = analysis.competition_type
        num_labels = analysis.num_classes or 2

        logger.info(f"Dataset analysis: {comp_type.value}, {num_labels} classes")

        head_type = _get_head_for_competition_type(comp_type)
        
        return create_model(
            "bert_with_head",
            pretrained_path=pretrained_name,
            head_type=head_type,
            num_labels=num_labels,
            **kwargs,
        )
    else:
        return create_bert_for_task(
            task="binary_classification", pretrained_name=pretrained_name, **kwargs
        )


# === HELPER FUNCTIONS ===

def _get_head_for_competition_type(comp_type: CompetitionType) -> str:
    """Map competition type to head type."""
    head_type_map = {
        CompetitionType.BINARY_CLASSIFICATION: "binary_classification",
        CompetitionType.MULTICLASS_CLASSIFICATION: "multiclass_classification",
        CompetitionType.MULTILABEL_CLASSIFICATION: "multilabel_classification",
        CompetitionType.REGRESSION: "regression",
        CompetitionType.ORDINAL_REGRESSION: "ordinal_regression",
        CompetitionType.TIME_SERIES: "time_series",
        CompetitionType.RANKING: "ranking",
    }
    return head_type_map.get(comp_type, "binary_classification")


# === MODEL REGISTRY (maintained for backward compatibility) ===

MODEL_REGISTRY = {}

def _init_model_registry():
    """Initialize the backward-compatible model registry."""
    global MODEL_REGISTRY
    
    # Basic models
    MODEL_REGISTRY.update({
        "bert-binary": lambda **kwargs: create_model(
            "bert_with_head", head_type="binary_classification", **kwargs
        ),
        "bert-multiclass": lambda **kwargs: create_model(
            "bert_with_head", head_type="multiclass_classification", **kwargs
        ),
        "modernbert-binary": lambda **kwargs: create_model(
            "modernbert_with_head", head_type="binary_classification", **kwargs
        ),
        "modernbert-multiclass": lambda **kwargs: create_model(
            "modernbert_with_head", head_type="multiclass_classification", **kwargs
        ),
        # LoRA models
        "bert-lora-binary": lambda **kwargs: create_bert_with_lora(
            head_type="binary_classification", **kwargs
        ),
        "modernbert-lora-binary": lambda **kwargs: create_modernbert_with_lora(
            head_type="binary_classification", **kwargs
        ),
    })

# Initialize registry on module import
_init_model_registry()


# === BACKWARD COMPATIBILITY ALIASES ===

# For models/classification/factory.py compatibility
create_classifier_advanced = create_kaggle_classifier

# For factories/kaggle_competition_factory.py compatibility
create_competition_model = create_competition_classifier
analyze_dataset = analyze_competition_dataset

# For bert factory functions
create_bert_for_competition = create_bert_for_task


# === EXPORTS ===

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
    "create_modular_bert",
    # Types
    "ModelType",
    "CompetitionAnalysis",
    # Registry
    "MODEL_REGISTRY",
]