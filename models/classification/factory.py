"""
Classification Factory

Factory functions for creating different types of classifiers with optimal configurations.
"""

from typing import Dict, Optional, Union
from loguru import logger

from models.embeddings import EmbeddingModel
from .heads import BinaryClassificationHead, MultiClassificationHead, RegressionHead
from .titanic_classifier import TitanicClassifier


def create_classifier(
    task_type: str,
    model_name: str,
    num_classes: Optional[int] = None,
    hidden_dim: Optional[int] = None,
    dropout_prob: float = 0.1,
    use_layer_norm: bool = False,
    activation: str = "relu",
    freeze_embeddings: bool = False,
    **kwargs
) -> Union[TitanicClassifier, "BaseClassifier"]:
    """
    Factory function to create classifiers with optimal settings.
    
    Args:
        task_type: Type of classification task ("binary", "multiclass", "regression", "titanic")
        model_name: Name of the embedding model to use
        num_classes: Number of classes (for multiclass)
        hidden_dim: Hidden dimension for classification head
        dropout_prob: Dropout probability
        use_layer_norm: Whether to use layer normalization
        activation: Activation function name
        freeze_embeddings: Whether to freeze embedding model
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
        },
        "multiclass": {
            "dropout_prob": 0.1,
            "use_layer_norm": True,
            "activation": "gelu",
        },
        "regression": {
            "dropout_prob": 0.1,
            "use_layer_norm": False,
            "activation": "relu",
        },
        "titanic": {
            "dropout_prob": 0.1,
            "use_layer_norm": False,
            "activation": "relu",
        }
    }
    
    # Get default config for task type
    config = default_configs.get(task_type, default_configs["binary"])
    
    # Override with provided kwargs
    config.update(kwargs)
    
    # Create embedding model
    embedding_model = EmbeddingModel.from_pretrained(model_name)
    
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
    elif task_type == "binary":
        # Generic binary classifier (can be extended)
        classifier = _create_generic_classifier(
            embedding_model=embedding_model,
            head_type="binary",
            hidden_dim=hidden_dim,
            dropout_prob=config.get("dropout_prob", dropout_prob),
            use_layer_norm=config.get("use_layer_norm", use_layer_norm),
            activation=config.get("activation", activation),
            freeze_embeddings=freeze_embeddings,
        )
    elif task_type == "multiclass":
        if num_classes is None:
            raise ValueError("num_classes must be specified for multiclass classification")
        
        classifier = _create_generic_classifier(
            embedding_model=embedding_model,
            head_type="multiclass",
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout_prob=config.get("dropout_prob", dropout_prob),
            use_layer_norm=config.get("use_layer_norm", use_layer_norm),
            activation=config.get("activation", activation),
            freeze_embeddings=freeze_embeddings,
        )
    elif task_type == "regression":
        output_dim = kwargs.get("output_dim", 1)
        classifier = _create_generic_classifier(
            embedding_model=embedding_model,
            head_type="regression",
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dropout_prob=config.get("dropout_prob", dropout_prob),
            use_layer_norm=config.get("use_layer_norm", use_layer_norm),
            activation=config.get("activation", activation),
            freeze_embeddings=freeze_embeddings,
        )
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    logger.info(f"Created {task_type} classifier with {model_name}")
    return classifier


def _create_generic_classifier(
    embedding_model: EmbeddingModel,
    head_type: str,
    num_classes: Optional[int] = None,
    output_dim: Optional[int] = None,
    hidden_dim: Optional[int] = None,
    dropout_prob: float = 0.1,
    use_layer_norm: bool = False,
    activation: str = "relu",
    freeze_embeddings: bool = False,
):
    """
    Create a generic classifier with the specified head type.
    
    This is a placeholder for future generic classifier implementations.
    For now, it creates the appropriate head but doesn't provide a full classifier wrapper.
    """
    # Create the appropriate head
    if head_type == "binary":
        head = BinaryClassificationHead(
            input_dim=embedding_model.hidden_size,
            hidden_dim=hidden_dim,
            dropout_prob=dropout_prob,
            use_layer_norm=use_layer_norm,
            activation=activation,
        )
    elif head_type == "multiclass":
        head = MultiClassificationHead(
            input_dim=embedding_model.hidden_size,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout_prob=dropout_prob,
            use_layer_norm=use_layer_norm,
            activation=activation,
        )
    elif head_type == "regression":
        head = RegressionHead(
            input_dim=embedding_model.hidden_size,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dropout_prob=dropout_prob,
            use_layer_norm=use_layer_norm,
            activation=activation,
        )
    else:
        raise ValueError(f"Unknown head type: {head_type}")
    
    # For now, return a simple wrapper
    # In the future, this could be expanded to a full generic classifier
    class GenericClassifier:
        def __init__(self, embedding_model, head):
            self.embedding_model = embedding_model
            self.head = head
    
    return GenericClassifier(embedding_model, head)


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