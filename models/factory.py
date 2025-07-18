"""MASTER MODEL FACTORY - Single source for all model creation.

This is the SINGLE factory for the entire codebase. It merges:
- Basic model creation (ModernBERT, CNN-hybrid)
- Classification model creation (from models/classification/factory.py)
- Kaggle competition automation (from factories/kaggle_competition_factory.py)
- MLX embeddings support
- Automatic dataset analysis and optimization

Replace all other factory imports with this single factory.
"""

from typing import Dict, Any, Optional, Union, Literal, List, Tuple
from pathlib import Path
import json
from loguru import logger
import mlx.nn as nn
import pandas as pd
from enum import Enum
from dataclasses import dataclass
import numpy as np

# Import unified modules
from .modernbert import ModernBertModel, ModernBertConfig
from .modernbert_cnn_hybrid import CNNEnhancedModernBERT, CNNHybridConfig
from .classification import TitanicClassifier, create_classifier

# Import new modular BERT architecture
from .bert import (
    BertCore, BertWithHead, BertOutput,
    create_bert_core, create_bert_with_head, create_bert_for_competition
)
from .heads.base_head import HeadType, HeadConfig
from .heads.head_registry import HeadRegistry

# Import all classification factories
try:
    from .classification.factory import (
        create_classifier as create_advanced_classifier,
        create_multilabel_classifier,
        create_ordinal_classifier,
        create_hierarchical_classifier,
        create_ensemble_classifier,
        create_titanic_classifier,
    )
    from .classification.generic_classifier import GenericClassifier
    from .classification.kaggle_heads import KAGGLE_HEAD_REGISTRY, create_kaggle_head
    ADVANCED_CLASSIFICATION_AVAILABLE = True
except ImportError:
    ADVANCED_CLASSIFICATION_AVAILABLE = False
    logger.warning("Advanced classification modules not available")

# Import new architecture modules
try:
    from .embeddings import EmbeddingModel
    NEW_ARCHITECTURE_AVAILABLE = True
except ImportError:
    NEW_ARCHITECTURE_AVAILABLE = False
    logger.warning("New architecture modules not available")

# Import old MLX embeddings support for backward compatibility
try:
    from embeddings.model_wrapper import MLXEmbeddingModel
    from embeddings.config import MLXEmbeddingsConfig
    OLD_MLX_EMBEDDINGS_AVAILABLE = True
except ImportError:
    OLD_MLX_EMBEDDINGS_AVAILABLE = False
    logger.warning("Old MLX embeddings not available")

# Import dataset analysis for automatic optimization
try:
    from data.dataset_spec import KaggleDatasetSpec
    from data.universal_loader import UniversalKaggleLoader
    DATASET_ANALYSIS_AVAILABLE = True
except ImportError:
    DATASET_ANALYSIS_AVAILABLE = False
    logger.warning("Dataset analysis not available")


ModelType = Literal["standard", "cnn_hybrid", "classifier", "mlx_embedding", "new_mlx_embedding", "bert_core", "bert_with_head"]


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
    target_column: Optional[str] = None
    num_classes: Optional[int] = None
    class_distribution: Optional[Dict[str, int]] = None
    is_balanced: bool = True
    
    # Feature characteristics
    categorical_columns: List[str] = None
    numerical_columns: List[str] = None
    text_columns: List[str] = None
    
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
    model_type: ModelType = "standard",
    config: Optional[Union[Dict[str, Any], ModernBertConfig, "MLXEmbeddingsConfig"]] = None,
    pretrained_path: Optional[Union[str, Path]] = None,
    use_mlx_embeddings: bool = False,
    head_type: Optional[Union[HeadType, str]] = None,
    head_config: Optional[Union[HeadConfig, Dict]] = None,
    **kwargs,
) -> nn.Module:
    """
    Create a model based on type and configuration.

    Args:
        model_type: Type of model to create ("standard", "cnn_hybrid", "classifier", 
                   "mlx_embedding", "new_mlx_embedding", "bert_core", "bert_with_head")
        config: Model configuration (dict or Config object)
        pretrained_path: Path to pretrained weights
        use_mlx_embeddings: Whether to use MLX embeddings backend
        head_type: Type of head to attach (for bert_with_head)
        head_config: Head configuration (for bert_with_head)
        **kwargs: Additional arguments passed to model constructor

    Returns:
        Initialized model
    """
    # Convert config dict to Config object if needed
    if isinstance(config, dict):
        if model_type == "new_mlx_embedding":
            # New architecture doesn't need special config handling
            config = config
        elif model_type == "mlx_embedding" or use_mlx_embeddings:
            if OLD_MLX_EMBEDDINGS_AVAILABLE:
                config = MLXEmbeddingsConfig(**config)
            else:
                raise RuntimeError("Old MLX embeddings requested but not available")
        elif model_type == "cnn_hybrid" or "cnn_kernel_sizes" in config:
            config = CNNHybridConfig(**config)
        else:
            config = ModernBertConfig(**config)
    elif config is None:
        # Use default config
        if model_type == "new_mlx_embedding":
            # New architecture uses simple dict config
            config = {}
        elif model_type == "mlx_embedding" or use_mlx_embeddings:
            if OLD_MLX_EMBEDDINGS_AVAILABLE:
                config = MLXEmbeddingsConfig()
            else:
                raise RuntimeError("Old MLX embeddings requested but not available")
        elif model_type == "cnn_hybrid":
            config = CNNHybridConfig()
        else:
            config = ModernBertConfig()

    # Create base model
    if model_type == "new_mlx_embedding":
        # Create new architecture MLX embedding model
        if not NEW_ARCHITECTURE_AVAILABLE:
            raise RuntimeError("New architecture requested but not available")
        
        model_name = kwargs.pop("model_name", "mlx-community/answerdotai-ModernBERT-base-4bit")
        model = EmbeddingModel.from_pretrained(model_name, **kwargs)
        logger.info(f"Created new architecture MLX embedding model: {model_name}")
    elif model_type == "mlx_embedding" or (use_mlx_embeddings and OLD_MLX_EMBEDDINGS_AVAILABLE):
        # Create old MLX embedding model (backward compatibility)
        model_name = kwargs.pop("model_name", config.model_name if hasattr(config, "model_name") else "answerdotai/ModernBERT-base")
        num_labels = kwargs.pop("num_labels", None)
        model = MLXEmbeddingModel(
            model_name=model_name,
            num_labels=num_labels,
            hidden_size=config.hidden_size if hasattr(config, "hidden_size") else 768,
            dropout_rate=config.dropout_rate if hasattr(config, "dropout_rate") else 0.1,
            pooling_strategy=config.pooling_strategy if hasattr(config, "pooling_strategy") else "mean",
            use_mlx_embeddings=True,
            **kwargs
        )
        logger.info(f"Created old MLX embedding model: {model_name}")
    elif model_type == "cnn_hybrid":
        model = CNNEnhancedModernBERT(config, **kwargs)
        logger.info("Created CNN-enhanced ModernBERT model")
    elif model_type == "standard":
        model = ModernBertModel(config, **kwargs)
        logger.info("Created standard ModernBERT model")
    elif model_type == "classifier":
        # Create base model first
        base_model = ModernBertModel(config)
        # Wrap with classifier
        loss_type = kwargs.pop("loss_type", "cross_entropy")
        model = create_classifier(base_model, loss_type=loss_type, **kwargs)
        logger.info(f"Created ModernBERT with {loss_type} classifier")
    elif model_type == "bert_core":
        # Create new modular BERT core
        model = create_bert_core(config=config, **kwargs)
        logger.info("Created modular BertCore model")
    elif model_type == "bert_with_head":
        # Create BERT with attached head
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
            **kwargs
        )
        logger.info(f"Created BertWithHead model (head_type: {head_type})")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load pretrained weights if provided
    if pretrained_path:
        load_pretrained_weights(model, pretrained_path)

    return model


def create_model_from_checkpoint(checkpoint_path: Union[str, Path]) -> nn.Module:
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
    if "cnn_kernel_sizes" in config_dict:
        model_type = "cnn_hybrid"
    else:
        model_type = "standard"

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


def load_pretrained_weights(model: nn.Module, weights_path: Union[str, Path]):
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


def get_model_config(
    model_type: ModelType = "standard", 
    use_mlx_embeddings: bool = False,
    **kwargs
) -> Union[ModernBertConfig, CNNHybridConfig, "MLXEmbeddingsConfig"]:
    """
    Get default configuration for a model type.

    Args:
        model_type: Type of model
        use_mlx_embeddings: Whether to use MLX embeddings
        **kwargs: Configuration overrides

    Returns:
        Model configuration object
    """
    if model_type == "mlx_embedding" or use_mlx_embeddings:
        if MLX_EMBEDDINGS_AVAILABLE:
            config = MLXEmbeddingsConfig(**kwargs)
        else:
            raise RuntimeError("MLX embeddings requested but not available")
    elif model_type == "cnn_hybrid":
        config = CNNHybridConfig(**kwargs)
    else:
        config = ModernBertConfig(**kwargs)

    return config


def create_titanic_model(
    model_type: ModelType = "standard",
    loss_type: str = "cross_entropy",
    pretrained_path: Optional[Union[str, Path]] = None,
    **kwargs,
) -> TitanicClassifier:
    """
    Create a model specifically for Titanic classification.

    Args:
        model_type: Base model type ("standard" or "cnn_hybrid")
        loss_type: Loss function type
        pretrained_path: Path to pretrained weights
        **kwargs: Additional model configuration

    Returns:
        TitanicClassifier model
    """
    # Create base model
    if model_type == "cnn_hybrid":
        # CNN hybrid already has built-in classifier
        return create_model("cnn_hybrid", pretrained_path=pretrained_path, **kwargs)
    else:
        # Create standard model and wrap with classifier
        base_model = create_model("standard", **kwargs)

        # Load pretrained weights if provided
        if pretrained_path:
            load_pretrained_weights(base_model, pretrained_path)

        # Wrap with classifier
        return create_classifier(base_model, loss_type=loss_type, **kwargs)


# Model registry for easy access
MODEL_REGISTRY = {
    "modernbert-base": lambda **kwargs: create_model("standard", **kwargs),
    "modernbert-cnn": lambda **kwargs: create_model("cnn_hybrid", **kwargs),
    "modernbert-classifier": lambda **kwargs: create_model("classifier", **kwargs),
    "titanic-base": lambda **kwargs: create_titanic_model("standard", **kwargs),
    "titanic-cnn": lambda **kwargs: create_titanic_model("cnn_hybrid", **kwargs),
}

# Add new architecture MLX embedding models if available
if NEW_ARCHITECTURE_AVAILABLE:
    MODEL_REGISTRY.update({
        "new-mlx-modernbert-base": lambda **kwargs: create_model(
            "new_mlx_embedding", 
            model_name="mlx-community/answerdotai-ModernBERT-base-4bit",
            **kwargs
        ),
        "new-mlx-modernbert-large": lambda **kwargs: create_model(
            "new_mlx_embedding",
            model_name="mlx-community/answerdotai-ModernBERT-large-4bit",
            **kwargs
        ),
        "new-mlx-minilm": lambda **kwargs: create_model(
            "new_mlx_embedding",
            model_name="mlx-community/all-MiniLM-L6-v2-4bit",
            **kwargs
        ),
        "new-titanic-mlx": lambda **kwargs: create_titanic_classifier(
            model_name="mlx-community/answerdotai-ModernBERT-base-4bit",
            **kwargs
        ),
    })

# Add old MLX embedding models if available (backward compatibility)
if OLD_MLX_EMBEDDINGS_AVAILABLE:
    MODEL_REGISTRY.update({
        "mlx-modernbert-base": lambda **kwargs: create_model(
            "mlx_embedding", 
            model_name="answerdotai/ModernBERT-base",
            **kwargs
        ),
        "mlx-modernbert-large": lambda **kwargs: create_model(
            "mlx_embedding",
            model_name="answerdotai/ModernBERT-large",
            **kwargs
        ),
        "mlx-minilm": lambda **kwargs: create_model(
            "mlx_embedding",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            **kwargs
        ),
    })


def list_available_models() -> List[str]:
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
    model_name: str,
    num_classes: Optional[int] = None,
    dataset_spec: Optional["KaggleDatasetSpec"] = None,
    **kwargs
) -> nn.Module:
    """
    Create classifier optimized for Kaggle competitions.
    
    This is the main entry point for creating classifiers. It automatically
    selects the best configuration based on the task type and dataset characteristics.
    
    Args:
        task_type: Type of classification task ("binary", "multiclass", "regression", 
                   "titanic", "multilabel", "ordinal", "hierarchical", "ensemble",
                   "time_series", "ranking", "contrastive", "multi_task", "metric_learning")
        model_name: Name of the embedding model to use
        num_classes: Number of classes/labels
        dataset_spec: Optional dataset specification for optimization
        **kwargs: Additional arguments
        
    Returns:
        Configured classifier instance
    """
    if not ADVANCED_CLASSIFICATION_AVAILABLE:
        logger.warning("Advanced classification not available, falling back to basic classifier")
        return create_model("classifier", **kwargs)
    
    # Use dataset spec to optimize configuration if available
    if dataset_spec and hasattr(dataset_spec, 'optimization_profile'):
        logger.info(f"Optimizing for dataset: {dataset_spec.name} (profile: {dataset_spec.optimization_profile})")
        
        # Auto-configure based on dataset characteristics
        if dataset_spec.optimization_profile.value == "competition":
            kwargs.setdefault("dropout_prob", 0.1)
            kwargs.setdefault("use_layer_norm", True)
            kwargs.setdefault("pooling_type", "attention")
        elif dataset_spec.optimization_profile.value == "development":
            kwargs.setdefault("dropout_prob", 0.2)
            kwargs.setdefault("use_layer_norm", False)
            kwargs.setdefault("pooling_type", "mean")
    
    return create_advanced_classifier(
        task_type=task_type,
        model_name=model_name,
        num_classes=num_classes,
        **kwargs
    )


def create_competition_classifier(
    data_path: str,
    target_column: str,
    model_name: str = "answerdotai/ModernBERT-base",
    auto_optimize: bool = True,
    **kwargs
) -> Tuple[nn.Module, CompetitionAnalysis]:
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
        kwargs.update({
            "dropout_prob": 0.1 if analysis.num_samples > 10000 else 0.2,
            "use_layer_norm": analysis.num_features > 50,
            "pooling_type": "attention" if analysis.num_samples > 5000 else "mean",
            "batch_size": analysis.recommended_batch_size,
        })
        
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
        **kwargs
    )
    
    logger.info(f"Created {task_type} classifier for {analysis.competition_type.value}")
    return classifier, analysis


def analyze_competition_dataset(data_path: str, target_column: str) -> CompetitionAnalysis:
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
    if target_series.dtype in ['object', 'category']:
        if num_classes == 2:
            competition_type = CompetitionType.BINARY_CLASSIFICATION
        else:
            competition_type = CompetitionType.MULTICLASS_CLASSIFICATION
    else:
        # Check if it's regression or classification with numeric labels
        if num_classes <= 10 and target_series.dtype in ['int64', 'int32']:
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
        
        if series.dtype == 'object':
            # If most values are unique, likely text
            if series.nunique() / len(series) > 0.9:
                text_columns.append(col)
            else:
                categorical_columns.append(col)
        elif pd.api.types.is_numeric_dtype(series):
            # If low cardinality, might be categorical
            if series.nunique() <= 10 and series.dtype in ['int64', 'int32']:
                categorical_columns.append(col)
            else:
                numerical_columns.append(col)
        else:
            categorical_columns.append(col)
    
    # Check class balance
    if competition_type in [CompetitionType.BINARY_CLASSIFICATION, CompetitionType.MULTICLASS_CLASSIFICATION]:
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
        recommended_head_type="binary" if competition_type == CompetitionType.BINARY_CLASSIFICATION else "multiclass"
    )


# === CONVENIENCE FUNCTIONS (absorbed from classification/factory.py) ===

def create_titanic_classifier_optimized(
    model_name: str = "answerdotai/ModernBERT-base",
    **kwargs
) -> nn.Module:
    """Create optimized Titanic classifier."""
    if ADVANCED_CLASSIFICATION_AVAILABLE:
        return create_titanic_classifier(model_name=model_name, **kwargs)
    else:
        return create_model("classifier", **kwargs)


def create_multilabel_classifier_optimized(
    model_name: str,
    num_labels: int,
    **kwargs
) -> nn.Module:
    """Create optimized multilabel classifier."""
    if ADVANCED_CLASSIFICATION_AVAILABLE:
        return create_multilabel_classifier(model_name=model_name, num_labels=num_labels, **kwargs)
    else:
        return create_kaggle_classifier("multilabel", model_name, num_labels, **kwargs)


def create_ensemble_classifier_optimized(
    model_name: str,
    num_classes: int,
    **kwargs
) -> nn.Module:
    """Create optimized ensemble classifier."""
    if ADVANCED_CLASSIFICATION_AVAILABLE:
        return create_ensemble_classifier(model_name=model_name, num_classes=num_classes, **kwargs)
    else:
        return create_kaggle_classifier("ensemble", model_name, num_classes, **kwargs)


# === BACKWARD COMPATIBILITY ALIASES ===

# For models/classification/factory.py compatibility
create_classifier_advanced = create_kaggle_classifier

# For factories/kaggle_competition_factory.py compatibility
create_competition_model = create_competition_classifier
analyze_dataset = analyze_competition_dataset

# Legacy function names
create_enhanced_classifier = create_kaggle_classifier


# === NEW MODULAR BERT ARCHITECTURE FUNCTIONS ===

def create_modular_bert(
    pretrained_name: Optional[str] = None,
    config: Optional[Union[ModernBertConfig, Dict]] = None,
    **kwargs
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
    task: Union[str, HeadType, CompetitionType],
    pretrained_name: Optional[str] = None,
    num_labels: int = 2,
    freeze_bert: bool = False,
    **kwargs
) -> BertWithHead:
    """Create a BERT model with appropriate head for a specific task.
    
    Args:
        task: Task type (string, HeadType, or CompetitionType)
        pretrained_name: Optional pretrained model name
        num_labels: Number of output labels
        freeze_bert: Whether to freeze BERT parameters
        **kwargs: Additional arguments
        
    Returns:
        BertWithHead model
    """
    # Convert task to HeadType if needed
    if isinstance(task, str):
        # Try to parse as HeadType first
        try:
            head_type = HeadType(task)
        except ValueError:
            # Try to parse as CompetitionType
            try:
                comp_type = CompetitionType(task)
                # Map competition type to head type
                comp_to_head_map = {
                    CompetitionType.BINARY_CLASSIFICATION: HeadType.BINARY_CLASSIFICATION,
                    CompetitionType.MULTICLASS_CLASSIFICATION: HeadType.MULTICLASS_CLASSIFICATION,
                    CompetitionType.MULTILABEL_CLASSIFICATION: HeadType.MULTILABEL_CLASSIFICATION,
                    CompetitionType.REGRESSION: HeadType.REGRESSION,
                    CompetitionType.ORDINAL_REGRESSION: HeadType.ORDINAL_REGRESSION,
                    CompetitionType.TIME_SERIES: HeadType.TIME_SERIES,
                    CompetitionType.RANKING: HeadType.RANKING,
                }
                head_type = comp_to_head_map.get(comp_type, HeadType.BINARY_CLASSIFICATION)
            except ValueError:
                # Default to binary classification
                logger.warning(f"Unknown task type: {task}, defaulting to binary classification")
                head_type = HeadType.BINARY_CLASSIFICATION
    elif isinstance(task, CompetitionType):
        # Map competition type to head type
        comp_to_head_map = {
            CompetitionType.BINARY_CLASSIFICATION: HeadType.BINARY_CLASSIFICATION,
            CompetitionType.MULTICLASS_CLASSIFICATION: HeadType.MULTICLASS_CLASSIFICATION,
            CompetitionType.MULTILABEL_CLASSIFICATION: HeadType.MULTILABEL_CLASSIFICATION,
            CompetitionType.REGRESSION: HeadType.REGRESSION,
            CompetitionType.ORDINAL_REGRESSION: HeadType.ORDINAL_REGRESSION,
            CompetitionType.TIME_SERIES: HeadType.TIME_SERIES,
            CompetitionType.RANKING: HeadType.RANKING,
        }
        head_type = comp_to_head_map.get(task, HeadType.BINARY_CLASSIFICATION)
    else:
        head_type = task
    
    return create_bert_with_head(
        bert_name=pretrained_name,
        head_type=head_type,
        num_labels=num_labels,
        freeze_bert=freeze_bert,
        **kwargs
    )


def create_bert_from_dataset(
    dataset_path: Union[str, Path],
    pretrained_name: Optional[str] = None,
    auto_analyze: bool = True,
    **kwargs
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
        analysis = analyze_competition_dataset(dataset_path)
        
        # Get competition type
        comp_type = analysis.competition_type
        num_labels = analysis.num_classes or 2
        
        logger.info(f"Dataset analysis: {comp_type.value}, {num_labels} classes")
        
        # Create optimized model
        return create_bert_for_competition(
            competition_type=comp_type,
            bert_name=pretrained_name,
            num_labels=num_labels,
            **kwargs
        )
    else:
        # Default to binary classification
        return create_bert_for_task(
            task="binary_classification",
            pretrained_name=pretrained_name,
            **kwargs
        )
