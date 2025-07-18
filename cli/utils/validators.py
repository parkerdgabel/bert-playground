"""Input validators for CLI commands."""

from pathlib import Path
from typing import Optional, List, Union
import typer

def validate_path(
    path: Path,
    must_exist: bool = True,
    must_be_file: bool = False,
    must_be_dir: bool = False,
    extensions: Optional[List[str]] = None
) -> Path:
    """Validate a file or directory path."""
    if must_exist and not path.exists():
        raise typer.BadParameter(f"Path does not exist: {path}")
    
    if must_be_file and path.exists() and not path.is_file():
        raise typer.BadParameter(f"Path is not a file: {path}")
    
    if must_be_dir and path.exists() and not path.is_dir():
        raise typer.BadParameter(f"Path is not a directory: {path}")
    
    if extensions and path.is_file():
        if not any(path.suffix == ext for ext in extensions):
            raise typer.BadParameter(
                f"Invalid file extension. Expected one of: {', '.join(extensions)}"
            )
    
    return path

def validate_batch_size(value: int) -> int:
    """Validate batch size is positive and reasonable."""
    if value <= 0:
        raise typer.BadParameter("Batch size must be positive")
    
    if value > 1024:
        raise typer.BadParameter(
            "Batch size > 1024 is not recommended. Use gradient accumulation instead."
        )
    
    # Warn if not power of 2
    if value & (value - 1) != 0:
        typer.echo(
            f"Warning: Batch size {value} is not a power of 2. "
            "Performance may be suboptimal.",
            err=True
        )
    
    return value

def validate_model_type(value: str) -> str:
    """Validate model type against available models."""
    valid_types = [
        "modernbert",
        "modernbert-cnn",
        "mlx-bert",
        "answerdotai/ModernBERT-base",
        "answerdotai/ModernBERT-large",
        "bert-base-uncased",
        "bert-large-uncased",
    ]
    
    if value not in valid_types:
        raise typer.BadParameter(
            f"Invalid model type. Choose from: {', '.join(valid_types)}"
        )
    
    return value

def validate_learning_rate(value: float) -> float:
    """Validate learning rate is in reasonable range."""
    if value <= 0:
        raise typer.BadParameter("Learning rate must be positive")
    
    if value > 1.0:
        raise typer.BadParameter("Learning rate > 1.0 is not recommended")
    
    if value < 1e-8:
        raise typer.BadParameter("Learning rate < 1e-8 may cause underflow")
    
    return value

def validate_epochs(value: int) -> int:
    """Validate number of epochs."""
    if value <= 0:
        raise typer.BadParameter("Number of epochs must be positive")
    
    if value > 1000:
        raise typer.BadParameter("Number of epochs > 1000 is excessive")
    
    return value

def validate_port(value: int) -> int:
    """Validate network port number."""
    if value < 1 or value > 65535:
        raise typer.BadParameter("Port must be between 1 and 65535")
    
    if value < 1024:
        typer.echo(
            f"Warning: Port {value} requires root privileges on most systems",
            err=True
        )
    
    return value

def validate_percentage(value: float, name: str = "value") -> float:
    """Validate percentage value between 0 and 1."""
    if value < 0 or value > 1:
        raise typer.BadParameter(f"{name} must be between 0 and 1")
    
    return value

def validate_kaggle_competition(value: str) -> str:
    """Validate Kaggle competition name format."""
    import re
    
    # Kaggle competition names are lowercase with hyphens
    pattern = r'^[a-z0-9-]+$'
    
    if not re.match(pattern, value):
        raise typer.BadParameter(
            "Invalid competition name. Use lowercase letters, numbers, and hyphens only."
        )
    
    return value

def validate_output_format(value: str) -> str:
    """Validate output format."""
    valid_formats = ["human", "json", "csv", "yaml"]
    
    if value not in valid_formats:
        raise typer.BadParameter(
            f"Invalid output format. Choose from: {', '.join(valid_formats)}"
        )
    
    return value