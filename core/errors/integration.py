"""Integration examples for k-bert error handling framework.

This shows how to integrate the error handling framework with existing k-bert code.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import typer
from rich.console import Console

from core.errors import (
    CLIError,
    CLIErrorHandler,
    ConfigurationError,
    DataError,
    ModelError,
    TrainingError,
    setup_default_handlers,
    with_error_handling,
)


# Example: Integration with CLI commands
def create_cli_app() -> typer.Typer:
    """Create CLI app with integrated error handling."""
    app = typer.Typer()
    console = Console()
    
    # Set up error handlers
    setup_default_handlers(console)
    
    @app.command()
    def train(
        config: Path = typer.Option(..., help="Path to configuration file"),
        resume: Optional[Path] = typer.Option(None, help="Resume from checkpoint"),
    ):
        """Train a BERT model with error handling."""
        try:
            # Validate config exists
            if not config.exists():
                raise ConfigurationError.missing_required_field(
                    "config",
                    config,
                ).with_suggestion(
                    "Use 'k-bert config init' to create a default configuration"
                )
            
            # Load and train
            train_with_config(config, resume)
            
        except CLIError:
            # CLI errors are handled by CLIErrorHandler
            raise
        except Exception as e:
            # Convert other errors to CLI errors
            raise CLIError.from_exception(
                e,
                "Training failed",
            ).with_suggestion(
                "Check the logs for more details"
            )
    
    return app


# Example: Integration with configuration loading
@with_error_handling(
    error_type=ConfigurationError,
    error_message="Failed to load configuration",
)
def load_config_with_validation(config_path: Path) -> Dict[str, Any]:
    """Load configuration with comprehensive validation."""
    import yaml
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Validate required fields
    required_fields = [
        ("models.type", str),
        ("training.epochs", int),
        ("training.batch_size", int),
        ("data.path", str),
    ]
    
    for field_path, expected_type in required_fields:
        value = get_nested_value(config, field_path)
        
        if value is None:
            raise ConfigurationError.missing_required_field(
                field_path,
                config_path,
            )
        
        if not isinstance(value, expected_type):
            raise ConfigurationError.invalid_value(
                field_path,
                value,
                expected_type.__name__,
                config_path,
            )
    
    return config


def get_nested_value(config: Dict[str, Any], path: str) -> Any:
    """Get nested value from config."""
    keys = path.split(".")
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return None
    
    return value


# Example: Integration with model loading
def load_model_with_recovery(
    model_config: Dict[str, Any],
    checkpoint_path: Optional[Path] = None,
) -> Any:
    """Load model with error handling and recovery."""
    try:
        if checkpoint_path and not checkpoint_path.exists():
            raise ModelError.checkpoint_not_found(checkpoint_path)
        
        # Load model (simplified)
        model_type = model_config.get("type", "modernbert")
        
        if model_type not in ["bert", "modernbert", "roberta"]:
            raise ModelError.unsupported_model_type(model_type)
        
        # Create model...
        return f"Model({model_type})"
        
    except ModelError as e:
        # Add recovery suggestions based on error
        if e.error_code == "MODEL_CHECKPOINT_NOT_FOUND":
            e.with_recovery("Train from scratch instead")
            e.with_suggestion(f"Remove --checkpoint argument")
        
        raise


# Example: Integration with data loading
def load_dataset_with_validation(
    data_config: Dict[str, Any],
) -> Any:
    """Load dataset with comprehensive error handling."""
    data_path = Path(data_config["path"])
    
    if not data_path.exists():
        raise DataError.file_not_found(data_path)
    
    # Check format
    if data_path.suffix not in [".csv", ".parquet", ".json"]:
        raise DataError.invalid_format(
            data_path,
            "csv, parquet, or json",
            data_path.suffix[1:],  # Remove dot
        )
    
    try:
        # Load data (simplified)
        if data_path.suffix == ".csv":
            import pandas as pd
            data = pd.read_csv(data_path)
        else:
            data = None
        
        # Validate columns
        required_columns = data_config.get("required_columns", [])
        if data is not None and hasattr(data, "columns"):
            missing = set(required_columns) - set(data.columns)
            if missing:
                raise DataError.missing_column(
                    list(missing)[0],
                    list(data.columns),
                    data_path,
                )
        
        return data
        
    except Exception as e:
        # Wrap unexpected errors
        raise DataError.from_exception(
            e,
            f"Failed to load data from {data_path}",
        ).with_context(
            format=data_path.suffix,
            size=data_path.stat().st_size if data_path.exists() else None,
        )


# Example: Integration with training loop
def train_with_config(
    config_path: Path,
    resume_checkpoint: Optional[Path] = None,
) -> None:
    """Train model with comprehensive error handling."""
    # Load configuration
    config = load_config_with_validation(config_path)
    
    # Load model
    model = load_model_with_recovery(
        config["models"],
        resume_checkpoint,
    )
    
    # Load data
    dataset = load_dataset_with_validation(config["data"])
    
    # Training loop with error handling
    epoch = 0
    step = 0
    
    try:
        for epoch in range(config["training"]["epochs"]):
            for step, batch in enumerate(get_batches(dataset, config)):
                # Simulate training
                loss = train_step(model, batch)
                
                # Check for NaN
                if loss != loss:  # NaN check
                    raise TrainingError.nan_loss(epoch, step)
                
                # Check for OOM (simulated)
                if config["training"]["batch_size"] > 128:
                    raise TrainingError.out_of_memory(
                        config["training"]["batch_size"],
                        config["models"].get("size", "base"),
                    )
    
    except TrainingError as e:
        # Add context about training state
        e.with_context(
            total_epochs=config["training"]["epochs"],
            completed_epochs=epoch,
            model_type=config["models"]["type"],
        )
        
        # Suggest recovery based on error type
        if e.error_code == "TRAINING_NAN_LOSS":
            e.with_recovery("Resume with lower learning rate")
            e.with_suggestion("Check for numerical instabilities in your data")
        
        raise


def get_batches(dataset: Any, config: Dict[str, Any]):
    """Dummy batch generator."""
    batch_size = config["training"]["batch_size"]
    for i in range(10):  # Dummy batches
        yield f"batch_{i}"


def train_step(model: Any, batch: Any) -> float:
    """Dummy training step."""
    import random
    # Simulate occasional NaN
    if random.random() < 0.1:
        return float("nan")
    return random.random()


# Example: Error reporting for users
def format_error_for_user(error: Exception, verbose: bool = False) -> str:
    """Format error for user-friendly display."""
    from core.errors import KBertError
    
    if isinstance(error, KBertError):
        return error.format_for_cli(verbose=verbose)
    
    # Format other exceptions
    lines = [
        f"[red]Error[/red]: {type(error).__name__}: {str(error)}",
    ]
    
    if verbose:
        import traceback
        lines.append("\n[dim]Stack Trace:[/dim]")
        lines.extend(traceback.format_tb(error.__traceback__))
    
    return "\n".join(lines)


# Example: Integration with DI container
from typing import Protocol


class ErrorHandlerProtocol(Protocol):
    """Protocol for error handlers in DI container."""
    
    def handle(self, error: Exception) -> Optional[Any]:
        """Handle an error."""
        ...


def create_error_handler_factory(console: Console):
    """Create error handler factory for DI container."""
    def factory() -> ErrorHandlerProtocol:
        handler = CLIErrorHandler(console)
        return handler
    
    return factory


# Example usage
if __name__ == "__main__":
    # Set up console
    console = Console()
    
    # Test error formatting
    error = ConfigurationError(
        "Invalid configuration",
        config_path=Path("config.yaml"),
    ).with_suggestion(
        "Check your configuration file"
    ).with_recovery(
        "Use default configuration"
    )
    
    print("Simple format:")
    print(format_error_for_user(error, verbose=False))
    
    print("\n\nVerbose format:")
    print(format_error_for_user(error, verbose=True))