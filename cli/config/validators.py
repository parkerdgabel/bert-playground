"""Configuration validators for k-bert CLI.

This module provides validation functions to ensure configuration
values are valid and consistent.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .schemas import KBertConfig, CompetitionConfig


class ConfigValidationError(Exception):
    """Configuration validation error."""
    
    def __init__(self, errors: List[str]):
        """Initialize with list of errors."""
        self.errors = errors
        message = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        super().__init__(message)


def validate_config(config: KBertConfig) -> None:
    """Validate a complete configuration.
    
    Args:
        config: Configuration to validate
        
    Raises:
        ConfigValidationError: If validation fails
    """
    errors = []
    
    # Validate Kaggle configuration
    errors.extend(_validate_kaggle_config(config.kaggle))
    
    # Validate model configuration
    errors.extend(_validate_model_config(config.models))
    
    # Validate training configuration
    errors.extend(_validate_training_config(config.training))
    
    # Validate data configuration
    errors.extend(_validate_data_config(config.data))
    
    # Validate MLflow configuration
    errors.extend(_validate_mlflow_config(config.mlflow))
    
    # Validate logging configuration
    errors.extend(_validate_logging_config(config.logging))
    
    if errors:
        raise ConfigValidationError(errors)


def _validate_kaggle_config(kaggle_config) -> List[str]:
    """Validate Kaggle configuration."""
    errors = []
    
    # Check if API credentials are provided together
    if kaggle_config.username and not kaggle_config.key:
        errors.append("Kaggle API key is required when username is provided")
    elif kaggle_config.key and not kaggle_config.username:
        errors.append("Kaggle username is required when API key is provided")
    
    # Validate API key format if provided
    if kaggle_config.key and not _is_valid_kaggle_key(kaggle_config.key):
        errors.append("Invalid Kaggle API key format")
    
    # Check competitions directory
    if kaggle_config.competitions_dir:
        comp_dir = Path(kaggle_config.competitions_dir)
        if comp_dir.exists() and not comp_dir.is_dir():
            errors.append(f"Competitions directory is not a directory: {comp_dir}")
    
    return errors


def _validate_model_config(model_config) -> List[str]:
    """Validate model configuration."""
    errors = []
    
    # Check cache directory
    cache_dir = Path(model_config.cache_dir).expanduser()
    if cache_dir.exists() and not cache_dir.is_dir():
        errors.append(f"Model cache directory is not a directory: {cache_dir}")
    
    # Validate architecture
    valid_architectures = ["bert", "modernbert", "neobert"]
    if model_config.default_architecture not in valid_architectures:
        errors.append(
            f"Invalid default architecture: {model_config.default_architecture}. "
            f"Must be one of: {valid_architectures}"
        )
    
    # Validate LoRA preset
    valid_lora_presets = ["minimal", "balanced", "aggressive"]
    if model_config.lora_preset not in valid_lora_presets:
        errors.append(
            f"Invalid LoRA preset: {model_config.lora_preset}. "
            f"Must be one of: {valid_lora_presets}"
        )
    
    return errors


def _validate_training_config(training_config) -> List[str]:
    """Validate training configuration."""
    errors = []
    
    # Check output directory
    output_dir = Path(training_config.output_dir)
    if output_dir.exists() and not output_dir.is_dir():
        errors.append(f"Output directory is not a directory: {output_dir}")
    
    # Validate numeric ranges
    if training_config.default_batch_size < 1:
        errors.append("Default batch size must be at least 1")
    
    if training_config.default_epochs < 1:
        errors.append("Default epochs must be at least 1")
    
    if training_config.default_learning_rate <= 0:
        errors.append("Default learning rate must be positive")
    
    if training_config.warmup_ratio < 0 or training_config.warmup_ratio > 1:
        errors.append("Warmup ratio must be between 0 and 1")
    
    if training_config.max_grad_norm <= 0:
        errors.append("Max gradient norm must be positive")
    
    if training_config.gradient_accumulation_steps < 1:
        errors.append("Gradient accumulation steps must be at least 1")
    
    if training_config.early_stopping_patience < 0:
        errors.append("Early stopping patience must be non-negative")
    
    return errors


def _validate_data_config(data_config) -> List[str]:
    """Validate data configuration."""
    errors = []
    
    # Check cache directory
    cache_dir = Path(data_config.cache_dir).expanduser()
    if cache_dir.exists() and not cache_dir.is_dir():
        errors.append(f"Data cache directory is not a directory: {cache_dir}")
    
    # Validate max length
    if data_config.max_length < 1:
        errors.append("Max length must be at least 1")
    elif data_config.max_length > 8192:
        errors.append("Max length cannot exceed 8192 (ModernBERT limit)")
    
    # Validate workers and prefetch
    if data_config.num_workers < 0:
        errors.append("Number of workers must be non-negative")
    
    if data_config.prefetch_size < 0:
        errors.append("Prefetch size must be non-negative")
    
    if data_config.mlx_prefetch_size is not None and data_config.mlx_prefetch_size < 0:
        errors.append("MLX prefetch size must be non-negative")
    
    # Validate tokenizer backend
    valid_backends = ["auto", "mlx", "huggingface", "transformers"]
    if data_config.tokenizer_backend not in valid_backends:
        errors.append(
            f"Invalid tokenizer backend: {data_config.tokenizer_backend}. "
            f"Must be one of: {valid_backends}"
        )
    
    return errors


def _validate_mlflow_config(mlflow_config) -> List[str]:
    """Validate MLflow configuration."""
    errors = []
    
    # Validate tracking URI format
    tracking_uri = mlflow_config.tracking_uri
    if tracking_uri.startswith("file://"):
        # Check if path is valid
        path = tracking_uri.replace("file://", "")
        expanded_path = Path(path).expanduser()
        if expanded_path.exists() and not expanded_path.is_dir():
            errors.append(f"MLflow tracking path is not a directory: {expanded_path}")
    elif not (
        tracking_uri.startswith("http://") or 
        tracking_uri.startswith("https://") or
        tracking_uri.startswith("sqlite://") or
        tracking_uri.startswith("postgresql://") or
        tracking_uri.startswith("mysql://")
    ):
        errors.append(f"Invalid MLflow tracking URI format: {tracking_uri}")
    
    return errors


def _validate_logging_config(logging_config) -> List[str]:
    """Validate logging configuration."""
    errors = []
    
    # Check log directory
    if logging_config.file_output:
        log_dir = Path(logging_config.file_dir).expanduser()
        if log_dir.exists() and not log_dir.is_dir():
            errors.append(f"Log directory is not a directory: {log_dir}")
    
    # Validate log format
    valid_formats = ["structured", "simple"]
    if logging_config.format not in valid_formats:
        errors.append(
            f"Invalid log format: {logging_config.format}. "
            f"Must be one of: {valid_formats}"
        )
    
    # Validate rotation format
    if not _is_valid_rotation(logging_config.rotation):
        errors.append(f"Invalid log rotation format: {logging_config.rotation}")
    
    # Validate retention format
    if not _is_valid_retention(logging_config.retention):
        errors.append(f"Invalid log retention format: {logging_config.retention}")
    
    return errors


def validate_competition_config(config: CompetitionConfig) -> List[str]:
    """Validate competition configuration."""
    errors = []
    
    # Check data directory
    if not config.data_dir.exists():
        errors.append(f"Competition data directory does not exist: {config.data_dir}")
    elif not config.data_dir.is_dir():
        errors.append(f"Competition data path is not a directory: {config.data_dir}")
    
    # Check required files exist
    train_path = config.data_dir / config.train_file
    if not train_path.exists():
        errors.append(f"Training file not found: {train_path}")
    
    test_path = config.data_dir / config.test_file
    if not test_path.exists():
        errors.append(f"Test file not found: {test_path}")
    
    # Validate metrics
    valid_metrics = [
        "accuracy", "f1", "precision", "recall", "auc", "log_loss",
        "rmse", "mae", "mse", "r2", "mape"
    ]
    for metric in config.metrics:
        if metric not in valid_metrics:
            errors.append(f"Invalid metric: {metric}. Must be one of: {valid_metrics}")
    
    return errors


def _is_valid_kaggle_key(key: str) -> bool:
    """Check if Kaggle API key has valid format."""
    # Kaggle keys are typically 32 character hex strings
    if len(key) != 32:
        return False
    
    try:
        int(key, 16)
        return True
    except ValueError:
        return False


def _is_valid_rotation(rotation: str) -> bool:
    """Check if log rotation format is valid."""
    # Examples: "500 MB", "1 GB", "10 MB", "daily", "weekly"
    if rotation in ["daily", "weekly", "monthly"]:
        return True
    
    parts = rotation.split()
    if len(parts) != 2:
        return False
    
    try:
        size = float(parts[0])
        unit = parts[1].upper()
        return size > 0 and unit in ["B", "KB", "MB", "GB", "TB"]
    except (ValueError, IndexError):
        return False


def _is_valid_retention(retention: str) -> bool:
    """Check if log retention format is valid."""
    # Examples: "30 days", "1 week", "6 months"
    parts = retention.split()
    if len(parts) != 2:
        return False
    
    try:
        count = int(parts[0])
        unit = parts[1].lower()
        return count > 0 and unit in ["day", "days", "week", "weeks", "month", "months", "year", "years"]
    except (ValueError, IndexError):
        return False


def validate_path(
    path: Path,
    must_exist: bool = False,
    must_be_file: bool = False,
    must_be_dir: bool = False,
    create_parents: bool = False
) -> Optional[str]:
    """Validate a path with various constraints.
    
    Args:
        path: Path to validate
        must_exist: Path must exist
        must_be_file: Path must be a file
        must_be_dir: Path must be a directory
        create_parents: Create parent directories if they don't exist
        
    Returns:
        Error message if invalid, None if valid
    """
    if must_exist and not path.exists():
        return f"Path does not exist: {path}"
    
    if path.exists():
        if must_be_file and not path.is_file():
            return f"Path is not a file: {path}"
        if must_be_dir and not path.is_dir():
            return f"Path is not a directory: {path}"
    
    if create_parents and not path.exists():
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            return f"Failed to create parent directories: {e}"
    
    return None


def validate_cli_overrides(overrides: Dict[str, Any]) -> List[str]:
    """Validate CLI override values.
    
    Args:
        overrides: Dictionary of CLI overrides
        
    Returns:
        List of validation errors
    """
    errors = []
    
    # Validate specific override patterns
    for key, value in overrides.items():
        if key == "batch_size" and (not isinstance(value, int) or value < 1):
            errors.append("Batch size must be a positive integer")
        elif key == "learning_rate" and (not isinstance(value, (int, float)) or value <= 0):
            errors.append("Learning rate must be a positive number")
        elif key == "epochs" and (not isinstance(value, int) or value < 1):
            errors.append("Epochs must be a positive integer")
    
    return errors