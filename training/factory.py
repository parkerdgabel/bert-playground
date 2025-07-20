"""
Factory for creating trainers with preset configurations.
"""

from typing import Dict, Any, Optional, Type, Union, List
from pathlib import Path
from loguru import logger

from .core import BaseTrainer, BaseTrainerConfig, Trainer
from .core.config import (
    get_quick_test_config,
    get_development_config,
    get_production_config,
    get_kaggle_competition_config,
)
from .kaggle import KaggleTrainer, KaggleTrainerConfig, get_competition_config, CompetitionProfile
from .core.protocols import Model
from .callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler,
    ProgressBar,
    MLflowCallback,
    MetricsLogger,
)


# Registry of trainer types
TRAINER_REGISTRY: Dict[str, Type[Trainer]] = {
    "base": BaseTrainer,
    "kaggle": KaggleTrainer,
}

# Registry of preset configurations
CONFIG_REGISTRY: Dict[str, callable] = {
    # Base configs
    "quick_test": get_quick_test_config,
    "development": get_development_config,
    "production": get_production_config,
    "kaggle": get_kaggle_competition_config,
    # Kaggle competition configs
    "titanic": lambda: get_competition_config(CompetitionProfile.TITANIC),
    "house_prices": lambda: get_competition_config(CompetitionProfile.HOUSE_PRICES),
    "nlp_disaster": lambda: get_competition_config(CompetitionProfile.NLP_DISASTER),
}


def register_trainer(name: str, trainer_class: Type[Trainer]) -> None:
    """
    Register a custom trainer class.
    
    Args:
        name: Name to register the trainer under
        trainer_class: Trainer class to register
    """
    TRAINER_REGISTRY[name] = trainer_class
    logger.info(f"Registered trainer: {name}")


def register_config(name: str, config_fn: callable) -> None:
    """
    Register a custom configuration function.
    
    Args:
        name: Name to register the config under
        config_fn: Function that returns a trainer config
    """
    CONFIG_REGISTRY[name] = config_fn
    logger.info(f"Registered config: {name}")


def list_trainers() -> List[str]:
    """List available trainer types."""
    return list(TRAINER_REGISTRY.keys())


def list_configs() -> List[str]:
    """List available preset configurations."""
    return list(CONFIG_REGISTRY.keys())


def get_trainer_config(
    config_name: Optional[str] = None,
    config_path: Optional[Path] = None,
    **kwargs,
) -> BaseTrainerConfig:
    """
    Get a trainer configuration.
    
    Args:
        config_name: Name of preset configuration
        config_path: Path to configuration file
        **kwargs: Override configuration values
        
    Returns:
        Trainer configuration
    """
    # Load from file if path provided
    if config_path:
        config_path = Path(config_path)
        if config_name == "kaggle" or config_path.name.startswith("kaggle"):
            config = KaggleTrainerConfig.load(config_path)
        else:
            config = BaseTrainerConfig.load(config_path)
        logger.info(f"Loaded config from {config_path}")
    
    # Use preset if name provided
    elif config_name:
        if config_name not in CONFIG_REGISTRY:
            raise ValueError(
                f"Unknown config: {config_name}. "
                f"Available: {', '.join(list_configs())}"
            )
        config = CONFIG_REGISTRY[config_name]()
        logger.info(f"Using preset config: {config_name}")
    
    # Default config
    else:
        config = BaseTrainerConfig()
        logger.info("Using default config")
    
    # Apply overrides
    if kwargs:
        _apply_config_overrides(config, kwargs)
    
    return config


def create_trainer(
    model: Model,
    trainer_type: str = "base",
    config: Optional[Union[BaseTrainerConfig, str, Path]] = None,
    callbacks: Optional[List[Callback]] = None,
    add_default_callbacks: bool = True,
    **kwargs,
) -> Trainer:
    """
    Create a trainer instance.
    
    Args:
        model: Model to train
        trainer_type: Type of trainer to create
        config: Configuration (object, preset name, or file path)
        callbacks: List of callbacks to use
        add_default_callbacks: Whether to add default callbacks
        **kwargs: Additional arguments for trainer or config overrides
        
    Returns:
        Trainer instance
    """
    # Get trainer class
    if trainer_type not in TRAINER_REGISTRY:
        raise ValueError(
            f"Unknown trainer type: {trainer_type}. "
            f"Available: {', '.join(list_trainers())}"
        )
    
    trainer_class = TRAINER_REGISTRY[trainer_type]
    
    # Get configuration
    if isinstance(config, (str, Path)):
        config = get_trainer_config(config_name=config if isinstance(config, str) else None,
                                   config_path=config if isinstance(config, Path) else None,
                                   **kwargs)
    elif config is None:
        # Use default config for trainer type
        if trainer_type == "kaggle":
            config = get_kaggle_competition_config()
        else:
            config = get_production_config()
    
    # Validate config type
    if trainer_type == "kaggle" and not isinstance(config, KaggleTrainerConfig):
        raise ValueError("KaggleTrainer requires KaggleTrainerConfig")
    
    # Create callback list
    if callbacks is None:
        callbacks = []
    
    # Add default callbacks if requested
    if add_default_callbacks:
        callbacks.extend(_get_default_callbacks(config))
    
    # Extract trainer-specific kwargs
    trainer_kwargs = {}
    if trainer_type == "kaggle":
        # Extract test_dataloader if provided
        if "test_dataloader" in kwargs:
            trainer_kwargs["test_dataloader"] = kwargs.pop("test_dataloader")
    
    # Create trainer
    trainer = trainer_class(
        model=model,
        config=config,
        callbacks=callbacks,
        **trainer_kwargs,
    )
    
    logger.info(f"Created {trainer_type} trainer with {len(callbacks)} callbacks")
    
    return trainer


def create_trainer_from_yaml(
    model: Model,
    yaml_path: Path,
    trainer_type: Optional[str] = None,
    **kwargs,
) -> Trainer:
    """
    Create a trainer from a YAML configuration file.
    
    Args:
        model: Model to train
        yaml_path: Path to YAML config file
        trainer_type: Override trainer type from config
        **kwargs: Additional arguments
        
    Returns:
        Trainer instance
    """
    # Load config
    config = get_trainer_config(config_path=yaml_path)
    
    # Determine trainer type
    if trainer_type is None:
        # Infer from config type
        if isinstance(config, KaggleTrainerConfig):
            trainer_type = "kaggle"
        else:
            trainer_type = "base"
    
    return create_trainer(
        model=model,
        trainer_type=trainer_type,
        config=config,
        **kwargs,
    )


def _get_default_callbacks(config: BaseTrainerConfig) -> List[Callback]:
    """Get default callbacks based on configuration."""
    callbacks = []
    
    # Get callback configs from custom config if available
    callback_config = config.custom.get("callbacks", {})
    
    # Progress bar
    progress_config = callback_config.get("progress_bar", {})
    if progress_config.get("show_batch_progress", True) or progress_config.get("show_epoch_progress", True):
        callbacks.append(ProgressBar(
            update_freq=progress_config.get("update_freq", 1),
            show_batch_progress=progress_config.get("show_batch_progress", True),
            show_epoch_progress=progress_config.get("show_epoch_progress", True),
        ))
    
    # Early stopping
    if config.training.early_stopping:
        callbacks.append(
            EarlyStopping(
                patience=config.training.early_stopping_patience,
                min_delta=config.training.early_stopping_threshold,
                monitor=config.training.best_metric,
                mode=config.training.best_metric_mode,
            )
        )
    
    # Model checkpoint
    if config.training.save_strategy != "no":
        callbacks.append(
            ModelCheckpoint(
                monitor=config.training.best_metric,
                save_best_only=config.training.save_best_only,
                mode=config.training.best_metric_mode,
            )
        )
    
    # Learning rate scheduler
    if config.scheduler.type != "none":
        lr_config = callback_config.get("lr_scheduler", {})
        callbacks.append(LearningRateScheduler(
            verbose=lr_config.get("verbose", True),
            update_freq=lr_config.get("update_freq", "step"),
        ))
    
    # MLflow
    if "mlflow" in config.training.report_to:
        mlflow_config = callback_config.get("mlflow", {})
        callbacks.append(MLflowCallback(
            log_every_n_steps=mlflow_config.get("log_every_n_steps", 1),
            log_model_checkpoints=mlflow_config.get("log_model_checkpoints", False),
        ))
    
    # Metrics logger
    metrics_config = callback_config.get("metrics_logger", {})
    callbacks.append(MetricsLogger(
        save_format=metrics_config.get("save_format", "json"),
        plot_freq=metrics_config.get("plot_freq", "epoch"),
        aggregate_metrics=metrics_config.get("aggregate_metrics", True),
    ))
    
    return callbacks


def _apply_config_overrides(config: BaseTrainerConfig, overrides: Dict[str, Any]) -> None:
    """Apply configuration overrides."""
    for key, value in overrides.items():
        # Handle nested configs
        if "." in key:
            parts = key.split(".")
            obj = config
            
            # Navigate to nested object
            for part in parts[:-1]:
                obj = getattr(obj, part)
            
            # Set value
            setattr(obj, parts[-1], value)
        else:
            # Direct attribute
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                # Add to custom config
                config.custom[key] = value


# Convenience functions for common scenarios
def create_quick_trainer(model: Model, **kwargs) -> BaseTrainer:
    """Create a trainer for quick testing."""
    config = get_quick_test_config()
    return create_trainer(model, config=config, add_default_callbacks=False, **kwargs)


def create_kaggle_trainer(
    model: Model,
    competition: str,
    test_dataloader=None,
    **kwargs,
) -> KaggleTrainer:
    """Create a trainer for Kaggle competition."""
    # Get competition config
    if competition in CONFIG_REGISTRY:
        config = CONFIG_REGISTRY[competition]()
    else:
        # Create generic Kaggle config
        config = KaggleTrainerConfig(
            kaggle__competition_name=competition,
            **kwargs,
        )
    
    return create_trainer(
        model,
        trainer_type="kaggle",
        config=config,
        test_dataloader=test_dataloader,
        **kwargs,
    )


def create_production_trainer(model: Model, **kwargs) -> BaseTrainer:
    """Create a trainer for production training."""
    config = get_production_config()
    return create_trainer(model, config=config, **kwargs)