"""Configuration fixtures for testing."""

from pathlib import Path
from typing import Dict, Any, Optional
import json
import yaml

from training.core.config import (
    BaseTrainerConfig,
    OptimizerConfig,
    SchedulerConfig,
    DataConfig,
    TrainingConfig,
    EnvironmentConfig,
)
from training.kaggle.config import KaggleConfig, KaggleTrainerConfig


def create_test_config(
    output_dir: Optional[Path] = None,
    num_epochs: int = 2,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    **kwargs
) -> BaseTrainerConfig:
    """Create test configuration with sensible defaults."""
    if output_dir is None:
        output_dir = Path("/tmp/test_training")
    
    config_dict = {
        "optimizer": {
            "type": "adam",
            "learning_rate": learning_rate,
            "weight_decay": 0.0,
        },
        "scheduler": {
            "type": "none",
        },
        "data": {
            "batch_size": batch_size,
            "eval_batch_size": batch_size,
            "num_workers": 0,  # Single-threaded for tests
        },
        "training": {
            "num_epochs": num_epochs,
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
            "logging_steps": 10,
            "eval_steps": 50,
            "save_steps": 50,
            "gradient_accumulation_steps": 1,
            "early_stopping": False,
            "log_level": "info",
        },
        "environment": {
            "output_dir": output_dir,
            "seed": 42,
        },
    }
    
    # Update with any additional kwargs
    for key, value in kwargs.items():
        if key in config_dict:
            if isinstance(config_dict[key], dict) and isinstance(value, dict):
                config_dict[key].update(value)
            else:
                config_dict[key] = value
    
    return BaseTrainerConfig(**config_dict)


def create_kaggle_config(
    output_dir: Optional[Path] = None,
    competition_name: str = "titanic",
    cv_folds: int = 5,
    **kwargs
) -> KaggleTrainerConfig:
    """Create Kaggle-specific test configuration."""
    if output_dir is None:
        output_dir = Path("/tmp/test_kaggle")
    
    # Create full config dict with kaggle settings
    config_dict = {
        "optimizer": {
            "type": "adam",
            "learning_rate": kwargs.get("learning_rate", 1e-3),
        },
        "training": {
            "num_epochs": kwargs.get("num_epochs", 2),
            "eval_strategy": "epoch",
        },
        "environment": {
            "output_dir": output_dir,
        },
        "kaggle": {
            "competition_name": competition_name,
            "competition_type": "binary_classification", 
            "competition_metric": "accuracy",
            "cv_folds": cv_folds,
            "enable_ensemble": False,
            "enable_tta": False,
            "auto_submit": False,
        }
    }
    
    return KaggleTrainerConfig(**config_dict)


def create_minimal_config(output_dir: Optional[Path] = None) -> BaseTrainerConfig:
    """Create minimal configuration for fast tests."""
    return create_test_config(
        output_dir=output_dir,
        num_epochs=1,
        batch_size=2,
        training={
            "eval_strategy": "steps",
            "eval_steps": 2,
            "save_strategy": "no",
            "logging_steps": 1,
        },
    )


def create_distributed_config(output_dir: Optional[Path] = None) -> BaseTrainerConfig:
    """Create configuration for distributed training tests."""
    return create_test_config(
        output_dir=output_dir,
        training={
            "distributed": True,
            "world_size": 2,
            "local_rank": 0,
        },
    )


def create_gradient_accumulation_config(
    output_dir: Optional[Path] = None,
    accumulation_steps: int = 4,
) -> BaseTrainerConfig:
    """Create configuration with gradient accumulation."""
    return create_test_config(
        output_dir=output_dir,
        batch_size=2,
        training={
            "gradient_accumulation_steps": accumulation_steps,
            "logging_steps": 1,
        },
    )


def create_early_stopping_config(
    output_dir: Optional[Path] = None,
    patience: int = 3,
) -> BaseTrainerConfig:
    """Create configuration with early stopping."""
    return create_test_config(
        output_dir=output_dir,
        training={
            "early_stopping": True,
            "early_stopping_patience": patience,
            "eval_strategy": "steps",
            "eval_steps": 5,
        },
    )


def create_scheduler_config(
    output_dir: Optional[Path] = None,
    scheduler_type: str = "cosine",
) -> BaseTrainerConfig:
    """Create configuration with specific scheduler."""
    scheduler_configs = {
        "cosine": {
            "type": "cosine",
            "warmup_ratio": 0.1,
            "num_cycles": 0.5,
        },
        "linear": {
            "type": "linear",
            "warmup_ratio": 0.1,
        },
        "exponential": {
            "type": "exponential",
            "gamma": 0.95,
        },
        "reduce_on_plateau": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 2,
            "threshold": 0.01,
        },
    }
    
    return create_test_config(
        output_dir=output_dir,
        scheduler=scheduler_configs.get(scheduler_type, {"type": scheduler_type}),
    )


def save_config_to_file(config: BaseTrainerConfig, file_path: Path, format: str = "json"):
    """Save configuration to file for testing."""
    config_dict = config.to_dict()
    
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "json":
        with open(file_path, "w") as f:
            json.dump(config_dict, f, indent=2)
    elif format == "yaml":
        with open(file_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    else:
        raise ValueError(f"Unknown format: {format}")


def create_invalid_config_dict() -> Dict[str, Any]:
    """Create invalid configuration dictionary for error testing."""
    return {
        "optimizer": {
            "type": "invalid_optimizer",
            "learning_rate": -1.0,  # Invalid negative LR
        },
        "training": {
            "num_epochs": 0,  # Invalid zero epochs
            "batch_size": -4,  # Invalid negative batch size
        },
    }


def create_config_variations() -> Dict[str, BaseTrainerConfig]:
    """Create various configuration variations for testing."""
    return {
        "minimal": create_minimal_config(),
        "standard": create_test_config(),
        "gradient_accumulation": create_gradient_accumulation_config(),
        "early_stopping": create_early_stopping_config(),
        "cosine_scheduler": create_scheduler_config(scheduler_type="cosine"),
        "linear_scheduler": create_scheduler_config(scheduler_type="linear"),
    }