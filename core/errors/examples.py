"""Examples of using the k-bert error handling framework."""

from pathlib import Path
from typing import Any, Dict

from core.errors import (
    ConfigurationError,
    DataError,
    ModelError,
    TrainingError,
    attempt_recovery,
    register_recovery_strategy,
    with_error_handling,
    with_recovery,
    RetryStrategy,
    ResourceReductionStrategy,
)


# Example 1: Basic error handling with context
def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration with error handling."""
    if not config_path.exists():
        raise ConfigurationError(
            f"Configuration file not found: {config_path}",
            config_path=config_path,
            error_code="CONFIG_FILE_NOT_FOUND",
        ).with_suggestion(
            f"Create a configuration file at {config_path}"
        ).with_suggestion(
            "Use 'k-bert config init' to create a default configuration"
        )
    
    # Load and validate config...
    return {}


# Example 2: Using the decorator for automatic error wrapping
@with_error_handling(
    error_type=DataError,
    error_message="Failed to load dataset",
)
def load_dataset(data_path: Path) -> Any:
    """Load dataset with automatic error handling."""
    # This will automatically wrap exceptions in DataError
    with open(data_path) as f:
        return f.read()


# Example 3: Recovery with retry strategy
@with_recovery(strategies=[RetryStrategy(max_attempts=3)])
def fetch_remote_data(url: str) -> bytes:
    """Fetch data with automatic retry."""
    import requests
    response = requests.get(url)
    response.raise_for_status()
    return response.content


# Example 4: Complex error with recovery actions
def train_model(config: Dict[str, Any]) -> None:
    """Train model with comprehensive error handling."""
    try:
        # Training logic...
        batch_size = config["training"]["batch_size"]
        
        # Simulate OOM error
        if batch_size > 128:
            raise TrainingError.out_of_memory(
                batch_size=batch_size,
                model_size=config["model"]["size"],
            )
        
    except TrainingError as e:
        # Attempt recovery
        result = attempt_recovery(e, context=config)
        
        if result.success and result.modifications:
            # Apply modifications and retry
            config.update(result.modifications)
            train_model(config)  # Retry with modified config
        else:
            raise


# Example 5: Custom recovery strategy
class CustomModelRecovery(ResourceReductionStrategy):
    """Custom recovery for model-specific errors."""
    
    def recover(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Custom recovery logic."""
        result = super().recover(error, context)
        
        if result.success:
            # Add model-specific modifications
            result.modifications["use_mixed_precision"] = False
            result.modifications["compile_model"] = False
        
        return result


# Example 6: Error chaining and context enrichment
def process_data_pipeline(data_path: Path) -> Any:
    """Process data with rich error context."""
    try:
        # Step 1: Load data
        data = load_raw_data(data_path)
        
    except Exception as e:
        raise DataError.from_exception(
            e,
            f"Failed to load data from {data_path}",
        ).with_context(
            pipeline_stage="data_loading",
            data_path=str(data_path),
            data_format="csv",
        ).with_suggestion(
            "Check that the file exists and is readable"
        ).with_recovery(
            "Try alternative data format (parquet, json)"
        )
    
    try:
        # Step 2: Validate data
        validate_data(data)
        
    except ValueError as e:
        # Chain errors with context
        error = DataError(
            "Data validation failed",
            cause=e,
        ).with_context(
            pipeline_stage="validation",
            row_count=len(data),
            columns=list(data.columns) if hasattr(data, "columns") else None,
        )
        
        # Add specific suggestions based on the error
        if "missing columns" in str(e):
            error.with_suggestion("Check column names in your configuration")
        
        raise error
    
    return data


def load_raw_data(path: Path) -> Any:
    """Dummy function for example."""
    raise FileNotFoundError(f"File not found: {path}")


def validate_data(data: Any) -> None:
    """Dummy function for example."""
    raise ValueError("missing columns: target")


# Example 7: Using error groups for batch operations
from core.errors import ErrorGroup


def process_multiple_files(file_paths: list[Path]) -> list[Any]:
    """Process multiple files with error grouping."""
    results = []
    errors = []
    
    for path in file_paths:
        try:
            result = process_single_file(path)
            results.append(result)
        except Exception as e:
            errors.append(
                DataError.from_exception(
                    e,
                    f"Failed to process {path}",
                ).with_context(file_path=str(path))
            )
    
    if errors:
        raise ErrorGroup(
            f"Failed to process {len(errors)} out of {len(file_paths)} files",
            errors,
        ).with_suggestion(
            "Check the error details for each failed file"
        ).with_recovery(
            "Process files individually with error handling"
        )
    
    return results


def process_single_file(path: Path) -> Any:
    """Dummy function for example."""
    if "bad" in str(path):
        raise ValueError(f"Invalid file: {path}")
    return f"Processed: {path}"


# Example 8: CLI-specific error handling
from core.errors import CLIError


def cli_command_handler(command: str, args: list[str]) -> None:
    """Handle CLI commands with proper error reporting."""
    valid_commands = ["train", "predict", "evaluate"]
    
    if command not in valid_commands:
        # Find similar commands for suggestions
        similar = [cmd for cmd in valid_commands if cmd.startswith(command[0])]
        
        raise CLIError.invalid_command(
            command,
            similar=similar if similar else valid_commands[:2],
        )
    
    # Check required arguments
    if command == "train" and "--config" not in args:
        raise CLIError.missing_argument("--config", command)


# Example 9: Integration with logging and monitoring
import logging
from core.errors import handle_error, register_type_handler


@register_type_handler(ModelError)
def log_model_errors(error: ModelError) -> None:
    """Custom handler to log model errors to monitoring system."""
    # Log to structured logging
    logging.error(
        "Model error occurred",
        extra={
            "error_code": error.error_code,
            "model_type": error.context.technical_details.get("model_type"),
            "checkpoint": error.context.technical_details.get("checkpoint_path"),
            "suggestions": error.context.suggestions,
        }
    )
    
    # Could also send to monitoring service
    # monitoring.send_event("model_error", error.to_dict())
    
    # Return None to continue with default handling
    return None


# Example 10: Testing with error scenarios
def test_error_handling():
    """Test various error scenarios."""
    # Test configuration error
    try:
        config = load_config(Path("/nonexistent/config.yaml"))
    except ConfigurationError as e:
        print(f"Caught config error: {e.error_code}")
        print(f"Suggestions: {e.context.suggestions}")
    
    # Test recovery
    try:
        # This will retry 3 times
        data = fetch_remote_data("http://unreliable-service.com/data")
    except Exception as e:
        print(f"Failed after retries: {e}")
    
    # Test error group
    try:
        results = process_multiple_files([
            Path("good1.csv"),
            Path("bad1.csv"),
            Path("good2.csv"),
            Path("bad2.csv"),
        ])
    except ErrorGroup as eg:
        print(f"Multiple errors: {len(eg.errors)}")
        for i, error in enumerate(eg.errors):
            print(f"  Error {i+1}: {error.message}")


if __name__ == "__main__":
    # Register custom recovery strategy
    register_recovery_strategy(
        CustomModelRecovery(),
        TrainingError,
    )
    
    # Run examples
    test_error_handling()