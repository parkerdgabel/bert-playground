"""
Classification Factory

Factory functions for creating different types of classifiers with optimal configurations.
"""

from typing import Dict, Optional, Union, List, Any
from loguru import logger

from models.embeddings import EmbeddingModel
from models.embeddings.embedding_model import EmbeddingModel as BertEmbeddingModel
from .heads import BinaryClassificationHead, MultiClassificationHead, RegressionHead
from .titanic_classifier import TitanicClassifier
from .generic_classifier import GenericClassifier, ClassificationTask
from .kaggle_heads import KAGGLE_HEAD_REGISTRY, create_kaggle_head


def create_classifier(
    task_type: str,
    model_name: str,
    num_classes: Optional[int] = None,
    hidden_dim: Optional[Union[int, List[int]]] = None,
    dropout_prob: float = 0.1,
    use_layer_norm: bool = False,
    activation: str = "relu",
    freeze_embeddings: bool = False,
    pooling_type: str = "mean",
    label_smoothing: float = 0.0,
    head_config: Optional[Dict[str, Any]] = None,
    auxiliary_heads: Optional[Dict[str, Dict[str, Any]]] = None,
    **kwargs
) -> Union[TitanicClassifier, GenericClassifier]:
    """
    Factory function to create classifiers with optimal settings.
    
    Args:
        task_type: Type of classification task ("binary", "multiclass", "regression", 
                   "titanic", "multilabel", "ordinal", "hierarchical", "ensemble",
                   "time_series", "ranking", "contrastive", "multi_task", "metric_learning")
        model_name: Name of the embedding model to use
        num_classes: Number of classes/labels
        hidden_dim: Hidden dimension(s) for classification head
        dropout_prob: Dropout probability
        use_layer_norm: Whether to use layer normalization
        activation: Activation function name
        freeze_embeddings: Whether to freeze embedding model
        pooling_type: Type of pooling ("mean", "max", "cls", "attention", "weighted", "learned")
        label_smoothing: Label smoothing factor
        head_config: Additional configuration for classification head
        auxiliary_heads: Configuration for auxiliary heads (multi-task learning)
        **kwargs: Additional arguments
        
    Returns:
        Configured classifier instance
    """
    # Default configurations for different tasks
    default_configs = {
        "binary": {
            "dropout_prob": 0.1,
            "use_layer_norm": False,
            "activation": "relu",
            "pooling_type": "mean",
        },
        "multiclass": {
            "dropout_prob": 0.1,
            "use_layer_norm": True,
            "activation": "gelu",
            "pooling_type": "mean",
        },
        "regression": {
            "dropout_prob": 0.1,
            "use_layer_norm": False,
            "activation": "relu",
            "pooling_type": "mean",
        },
        "titanic": {
            "dropout_prob": 0.1,
            "use_layer_norm": False,
            "activation": "relu",
            "pooling_type": "mean",
        },
        "multilabel": {
            "dropout_prob": 0.1,
            "use_layer_norm": True,
            "activation": "gelu",
            "pooling_type": "mean",
            "label_smoothing": 0.1,
        },
        "ordinal": {
            "dropout_prob": 0.1,
            "use_layer_norm": True,
            "activation": "gelu",
            "pooling_type": "attention",
            "head_config": {"temperature": 1.0},
        },
        "hierarchical": {
            "dropout_prob": 0.1,
            "use_layer_norm": True,
            "activation": "gelu",
            "pooling_type": "weighted",
            "head_config": {"consistency_weight": 1.0},
        },
        "ensemble": {
            "dropout_prob": 0.1,
            "use_layer_norm": True,
            "activation": "gelu",
            "pooling_type": "mean",
            "head_config": {
                "num_heads": 3,
                "ensemble_method": "attention",
                "temperature": 1.0,
            },
        },
        "time_series": {
            "dropout_prob": 0.2,
            "use_layer_norm": True,
            "activation": "gelu",
            "pooling_type": "learned",
            "head_config": {
                "hidden_dim": 256,
                "num_lstm_layers": 2,
                "use_attention": True,
                "bidirectional": True,
            },
        },
        "ranking": {
            "dropout_prob": 0.1,
            "use_layer_norm": True,
            "activation": "relu",
            "pooling_type": "attention",
            "head_config": {
                "hidden_dim": 256,
                "num_hidden_layers": 2,
                "ranking_loss": "listnet",
                "temperature": 1.0,
            },
        },
        "contrastive": {
            "dropout_prob": 0.1,
            "use_layer_norm": False,
            "activation": "relu",
            "pooling_type": "mean",
            "head_config": {
                "embedding_dim": 128,
                "temperature": 0.07,
                "normalize_embeddings": True,
            },
        },
        "multi_task": {
            "dropout_prob": 0.1,
            "use_layer_norm": True,
            "activation": "relu",
            "pooling_type": "mean",
            "head_config": {
                "shared_hidden_dim": 256,
            },
        },
        "metric_learning": {
            "dropout_prob": 0.1,
            "use_layer_norm": False,
            "activation": "relu",
            "pooling_type": "mean",
            "head_config": {
                "embedding_dim": 128,
                "margin": 0.5,
                "distance_metric": "cosine",
                "normalize_embeddings": True,
            },
        },
    }
    
    # Get default config for task type
    config = default_configs.get(task_type, default_configs["binary"])
    
    # Override with provided values
    config["dropout_prob"] = dropout_prob
    config["use_layer_norm"] = use_layer_norm
    config["activation"] = activation
    config["pooling_type"] = pooling_type
    config["label_smoothing"] = label_smoothing
    
    # Merge head_config
    if head_config:
        config["head_config"] = {**config.get("head_config", {}), **head_config}
    
    # Override with additional kwargs
    config.update(kwargs)
    
    # Create embedding model
    if isinstance(model_name, str):
        embedding_model = EmbeddingModel.from_pretrained(model_name)
    else:
        # Allow passing pre-initialized embedding model
        embedding_model = model_name
    
    # Ensure it's wrapped as BertEmbeddingModel if needed
    if not isinstance(embedding_model, BertEmbeddingModel):
        # Wrap it if needed (this assumes EmbeddingModel is compatible)
        embedding_model = embedding_model
    
    # Create classifier based on task type
    if task_type == "titanic":
        classifier = TitanicClassifier(
            embedding_model=embedding_model,
            hidden_dim=hidden_dim,
            dropout_prob=config.get("dropout_prob", dropout_prob),
            use_layer_norm=config.get("use_layer_norm", use_layer_norm),
            activation=config.get("activation", activation),
            freeze_embeddings=freeze_embeddings,
        )
    elif task_type in ["binary", "multiclass", "regression", "multilabel", 
                       "ordinal", "hierarchical", "ensemble", "time_series", 
                       "ranking", "contrastive", "multi_task", "metric_learning"]:
        # Use GenericClassifier for all modern task types
        if task_type == "regression":
            # For regression, num_classes represents output dimensions
            num_classes = kwargs.get("output_dim", num_classes or 1)
        elif task_type in ["binary"]:
            num_classes = 2
        elif num_classes is None:
            raise ValueError(f"num_classes must be specified for {task_type} classification")
        
        classifier = GenericClassifier(
            embedding_model=embedding_model,
            num_classes=num_classes,
            task_type=task_type,
            pooling_type=config.get("pooling_type", pooling_type),
            hidden_dims=hidden_dim,
            activation=config.get("activation", activation),
            dropout_rate=config.get("dropout_prob", dropout_prob),
            use_layer_norm=config.get("use_layer_norm", use_layer_norm),
            use_batch_norm=config.get("use_batch_norm", False),
            freeze_embeddings=freeze_embeddings,
            head_config=config.get("head_config"),
            class_names=kwargs.get("class_names"),
            label_smoothing=config.get("label_smoothing", label_smoothing),
            auxiliary_heads=auxiliary_heads,
        )
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    logger.info(f"Created {task_type} classifier with {model_name if isinstance(model_name, str) else 'pre-loaded model'}")
    return classifier


def create_multilabel_classifier(
    model_name: str,
    num_labels: int,
    label_names: Optional[List[str]] = None,
    pos_weights: Optional[List[float]] = None,
    **kwargs
) -> GenericClassifier:
    """
    Create a multilabel classifier.
    
    Args:
        model_name: Name of the embedding model
        num_labels: Number of labels (each can be 0 or 1)
        label_names: Optional names for labels
        pos_weights: Optional positive class weights for each label
        **kwargs: Additional arguments
        
    Returns:
        Configured multilabel classifier
    """
    head_config = kwargs.pop("head_config", {})
    if pos_weights:
        head_config["pos_weights"] = pos_weights
    
    return create_classifier(
        task_type="multilabel",
        model_name=model_name,
        num_classes=num_labels,
        class_names=label_names,
        head_config=head_config,
        **kwargs
    )


def create_ordinal_classifier(
    model_name: str,
    num_classes: int,
    class_names: Optional[List[str]] = None,
    temperature: float = 1.0,
    **kwargs
) -> GenericClassifier:
    """
    Create an ordinal regression classifier.
    
    Args:
        model_name: Name of the embedding model
        num_classes: Number of ordered classes
        class_names: Optional names for classes
        temperature: Temperature for ordinal thresholds
        **kwargs: Additional arguments
        
    Returns:
        Configured ordinal classifier
    """
    head_config = kwargs.pop("head_config", {})
    head_config["temperature"] = temperature
    
    return create_classifier(
        task_type="ordinal",
        model_name=model_name,
        num_classes=num_classes,
        class_names=class_names,
        head_config=head_config,
        pooling_type=kwargs.pop("pooling_type", "attention"),
        **kwargs
    )


def create_hierarchical_classifier(
    model_name: str,
    hierarchy: Dict[str, List[str]],
    label_to_idx: Dict[str, int],
    consistency_weight: float = 1.0,
    **kwargs
) -> GenericClassifier:
    """
    Create a hierarchical classifier.
    
    Args:
        model_name: Name of the embedding model
        hierarchy: Dict mapping parent labels to children
        label_to_idx: Mapping from label names to indices
        consistency_weight: Weight for hierarchical consistency
        **kwargs: Additional arguments
        
    Returns:
        Configured hierarchical classifier
    """
    head_config = kwargs.pop("head_config", {})
    head_config["hierarchy"] = hierarchy
    head_config["label_to_idx"] = label_to_idx
    head_config["consistency_weight"] = consistency_weight
    
    return create_classifier(
        task_type="hierarchical",
        model_name=model_name,
        num_classes=len(label_to_idx),
        head_config=head_config,
        pooling_type=kwargs.pop("pooling_type", "weighted"),
        **kwargs
    )


def create_ensemble_classifier(
    model_name: str,
    num_classes: int,
    num_heads: int = 3,
    ensemble_method: str = "attention",
    **kwargs
) -> GenericClassifier:
    """
    Create an ensemble classifier.
    
    Args:
        model_name: Name of the embedding model
        num_classes: Number of classes
        num_heads: Number of ensemble heads
        ensemble_method: Method for combining heads
        **kwargs: Additional arguments
        
    Returns:
        Configured ensemble classifier
    """
    head_config = kwargs.pop("head_config", {})
    head_config["num_heads"] = num_heads
    head_config["ensemble_method"] = ensemble_method
    
    return create_classifier(
        task_type="ensemble",
        model_name=model_name,
        num_classes=num_classes,
        head_config=head_config,
        **kwargs
    )


def create_titanic_classifier(
    model_name: str = "mlx-community/answerdotai-ModernBERT-base-4bit",
    **kwargs
) -> TitanicClassifier:
    """
    Convenience function to create a Titanic classifier.
    
    Args:
        model_name: Name of the embedding model to use
        **kwargs: Additional arguments for the classifier
        
    Returns:
        TitanicClassifier instance
    """
    return create_classifier(
        task_type="titanic",
        model_name=model_name,
        **kwargs
    )


# Backward compatibility
def create_enhanced_classifier(*args, **kwargs):
    """Backward compatibility alias."""
    logger.warning("create_enhanced_classifier is deprecated. Use create_classifier instead.")
    return create_classifier(*args, **kwargs)