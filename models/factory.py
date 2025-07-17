"""Unified model factory for creating all model variants."""

from typing import Dict, Any, Optional, Union, Literal, List
from pathlib import Path
import json
from loguru import logger
import mlx.nn as nn

# Import unified modules
from .modernbert import ModernBertModel, ModernBertConfig
from .modernbert_cnn_hybrid import CNNEnhancedModernBERT, CNNHybridConfig
from .classification import TitanicClassifier, create_classifier

# Import new architecture modules
try:
    from .embeddings import EmbeddingModel
    from .classification import create_titanic_classifier
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


ModelType = Literal["standard", "cnn_hybrid", "classifier", "mlx_embedding", "new_mlx_embedding"]


def create_model(
    model_type: ModelType = "standard",
    config: Optional[Union[Dict[str, Any], ModernBertConfig, "MLXEmbeddingsConfig"]] = None,
    pretrained_path: Optional[Union[str, Path]] = None,
    use_mlx_embeddings: bool = False,
    **kwargs,
) -> nn.Module:
    """
    Create a model based on type and configuration.

    Args:
        model_type: Type of model to create ("standard", "cnn_hybrid", "classifier", "mlx_embedding", "new_mlx_embedding")
        config: Model configuration (dict or Config object)
        pretrained_path: Path to pretrained weights
        use_mlx_embeddings: Whether to use MLX embeddings backend
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
