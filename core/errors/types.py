"""Specific error types for k-bert modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .base import KBertError


class ConfigurationError(KBertError):
    """Configuration-related errors."""

    def __init__(
        self,
        message: str,
        *,
        config_path: Optional[Path] = None,
        field_path: Optional[str] = None,
        invalid_value: Any = None,
        expected_type: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize configuration error."""
        super().__init__(message, **kwargs)
        
        if config_path:
            self.context.add_technical_detail("config_path", str(config_path))
        if field_path:
            self.context.add_technical_detail("field_path", field_path)
        if invalid_value is not None:
            self.context.add_technical_detail("invalid_value", str(invalid_value))
        if expected_type:
            self.context.add_technical_detail("expected_type", expected_type)

    @classmethod
    def missing_required_field(
        cls,
        field_path: str,
        config_path: Optional[Path] = None,
    ) -> "ConfigurationError":
        """Create error for missing required field."""
        error = cls(
            f"Missing required configuration field: {field_path}",
            field_path=field_path,
            config_path=config_path,
            error_code="CONFIG_MISSING_FIELD",
        )
        error.with_suggestion(f"Add '{field_path}' to your configuration file")
        return error

    @classmethod
    def invalid_value(
        cls,
        field_path: str,
        value: Any,
        expected: str,
        config_path: Optional[Path] = None,
    ) -> "ConfigurationError":
        """Create error for invalid configuration value."""
        error = cls(
            f"Invalid value for {field_path}: got {type(value).__name__}, expected {expected}",
            field_path=field_path,
            invalid_value=value,
            expected_type=expected,
            config_path=config_path,
            error_code="CONFIG_INVALID_VALUE",
        )
        error.with_suggestion(f"Ensure {field_path} is a valid {expected}")
        return error


class ModelError(KBertError):
    """Model creation and loading errors."""

    def __init__(
        self,
        message: str,
        *,
        model_type: Optional[str] = None,
        checkpoint_path: Optional[Path] = None,
        architecture: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize model error."""
        super().__init__(message, **kwargs)
        
        if model_type:
            self.context.add_technical_detail("model_type", model_type)
        if checkpoint_path:
            self.context.add_technical_detail("checkpoint_path", str(checkpoint_path))
        if architecture:
            self.context.add_technical_detail("architecture", architecture)

    @classmethod
    def unsupported_model_type(cls, model_type: str) -> "ModelError":
        """Create error for unsupported model type."""
        error = cls(
            f"Unsupported model type: {model_type}",
            model_type=model_type,
            error_code="MODEL_UNSUPPORTED_TYPE",
        )
        error.with_suggestion("Check available model types with 'k-bert info --models'")
        return error

    @classmethod
    def checkpoint_not_found(cls, path: Path) -> "ModelError":
        """Create error for missing checkpoint."""
        error = cls(
            f"Checkpoint not found: {path}",
            checkpoint_path=path,
            error_code="MODEL_CHECKPOINT_NOT_FOUND",
            recoverable=False,
        )
        error.with_suggestion(f"Ensure the checkpoint exists at {path}")
        error.with_suggestion("Use 'k-bert info --checkpoints' to list available checkpoints")
        return error

    @classmethod
    def incompatible_checkpoint(
        cls,
        checkpoint_path: Path,
        expected_arch: str,
        found_arch: str,
    ) -> "ModelError":
        """Create error for incompatible checkpoint."""
        error = cls(
            f"Checkpoint architecture mismatch: expected {expected_arch}, found {found_arch}",
            checkpoint_path=checkpoint_path,
            architecture=found_arch,
            error_code="MODEL_CHECKPOINT_INCOMPATIBLE",
            recoverable=False,
        )
        error.with_suggestion(f"Use a checkpoint trained with {expected_arch} architecture")
        return error


class DataError(KBertError):
    """Data loading and processing errors."""

    def __init__(
        self,
        message: str,
        *,
        data_path: Optional[Path] = None,
        dataset_name: Optional[str] = None,
        column_names: Optional[List[str]] = None,
        row_count: Optional[int] = None,
        **kwargs: Any,
    ):
        """Initialize data error."""
        super().__init__(message, **kwargs)
        
        if data_path:
            self.context.add_technical_detail("data_path", str(data_path))
        if dataset_name:
            self.context.add_technical_detail("dataset_name", dataset_name)
        if column_names:
            self.context.add_technical_detail("column_names", column_names)
        if row_count is not None:
            self.context.add_technical_detail("row_count", row_count)

    @classmethod
    def file_not_found(cls, path: Path) -> "DataError":
        """Create error for missing data file."""
        error = cls(
            f"Data file not found: {path}",
            data_path=path,
            error_code="DATA_FILE_NOT_FOUND",
            recoverable=False,
        )
        error.with_suggestion(f"Ensure the file exists at {path}")
        error.with_suggestion("Check the data.path configuration setting")
        return error

    @classmethod
    def missing_column(
        cls,
        column: str,
        available_columns: List[str],
        data_path: Optional[Path] = None,
    ) -> "DataError":
        """Create error for missing column."""
        error = cls(
            f"Required column '{column}' not found in data",
            column_names=available_columns,
            data_path=data_path,
            error_code="DATA_MISSING_COLUMN",
        )
        error.with_suggestion(f"Available columns: {', '.join(available_columns)}")
        error.with_suggestion(f"Update your configuration to use one of the available columns")
        return error

    @classmethod
    def invalid_format(
        cls,
        path: Path,
        expected_format: str,
        detected_format: Optional[str] = None,
    ) -> "DataError":
        """Create error for invalid data format."""
        msg = f"Invalid data format for {path}: expected {expected_format}"
        if detected_format:
            msg += f", detected {detected_format}"
        
        error = cls(
            msg,
            data_path=path,
            error_code="DATA_INVALID_FORMAT",
        )
        error.with_suggestion(f"Convert the file to {expected_format} format")
        return error


class TrainingError(KBertError):
    """Training-related errors."""

    def __init__(
        self,
        message: str,
        *,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        loss: Optional[float] = None,
        metric_name: Optional[str] = None,
        metric_value: Optional[float] = None,
        **kwargs: Any,
    ):
        """Initialize training error."""
        super().__init__(message, **kwargs)
        
        if epoch is not None:
            self.context.add_technical_detail("epoch", epoch)
        if step is not None:
            self.context.add_technical_detail("step", step)
        if loss is not None:
            self.context.add_technical_detail("loss", loss)
        if metric_name and metric_value is not None:
            self.context.add_technical_detail(f"metric_{metric_name}", metric_value)

    @classmethod
    def nan_loss(cls, epoch: int, step: int) -> "TrainingError":
        """Create error for NaN loss during training."""
        error = cls(
            "NaN loss detected during training",
            epoch=epoch,
            step=step,
            error_code="TRAINING_NAN_LOSS",
        )
        error.with_suggestion("Reduce learning rate")
        error.with_suggestion("Check for numerical instabilities in data")
        error.with_suggestion("Enable gradient clipping")
        error.with_recovery("Resume from last checkpoint with lower learning rate")
        return error

    @classmethod
    def out_of_memory(cls, batch_size: int, model_size: str) -> "TrainingError":
        """Create error for out of memory."""
        error = cls(
            f"Out of memory with batch size {batch_size}",
            error_code="TRAINING_OOM",
        )
        error.context.add_technical_detail("batch_size", batch_size)
        error.context.add_technical_detail("model_size", model_size)
        error.with_suggestion(f"Reduce batch size (current: {batch_size})")
        error.with_suggestion("Enable gradient accumulation")
        error.with_suggestion("Use LoRA for memory-efficient training")
        error.with_recovery("Retry with halved batch size")
        return error


class ValidationError(KBertError):
    """Input validation errors."""

    def __init__(
        self,
        message: str,
        *,
        field_name: Optional[str] = None,
        value: Any = None,
        constraints: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """Initialize validation error."""
        super().__init__(message, **kwargs)
        
        if field_name:
            self.context.add_technical_detail("field_name", field_name)
        if value is not None:
            self.context.add_technical_detail("value", str(value))
        if constraints:
            self.context.add_technical_detail("constraints", constraints)

    @classmethod
    def invalid_range(
        cls,
        field_name: str,
        value: Union[int, float],
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
    ) -> "ValidationError":
        """Create error for value out of range."""
        constraints = {}
        msg_parts = [f"Value {value} for {field_name} is out of range"]
        
        if min_value is not None:
            constraints["min"] = min_value
            msg_parts.append(f"minimum: {min_value}")
        if max_value is not None:
            constraints["max"] = max_value
            msg_parts.append(f"maximum: {max_value}")
        
        error = cls(
            f"{msg_parts[0]} ({', '.join(msg_parts[1:])})",
            field_name=field_name,
            value=value,
            constraints=constraints,
            error_code="VALIDATION_OUT_OF_RANGE",
        )
        
        if min_value is not None and max_value is not None:
            error.with_suggestion(f"Use a value between {min_value} and {max_value}")
        elif min_value is not None:
            error.with_suggestion(f"Use a value >= {min_value}")
        else:
            error.with_suggestion(f"Use a value <= {max_value}")
        
        return error


class PluginError(KBertError):
    """Plugin-related errors."""

    def __init__(
        self,
        message: str,
        *,
        plugin_name: Optional[str] = None,
        plugin_type: Optional[str] = None,
        plugin_path: Optional[Path] = None,
        **kwargs: Any,
    ):
        """Initialize plugin error."""
        super().__init__(message, **kwargs)
        
        if plugin_name:
            self.context.add_technical_detail("plugin_name", plugin_name)
        if plugin_type:
            self.context.add_technical_detail("plugin_type", plugin_type)
        if plugin_path:
            self.context.add_technical_detail("plugin_path", str(plugin_path))

    @classmethod
    def not_found(cls, plugin_name: str, plugin_type: str) -> "PluginError":
        """Create error for missing plugin."""
        error = cls(
            f"Plugin '{plugin_name}' of type '{plugin_type}' not found",
            plugin_name=plugin_name,
            plugin_type=plugin_type,
            error_code="PLUGIN_NOT_FOUND",
        )
        error.with_suggestion(f"Check available {plugin_type} plugins with 'k-bert info --plugins'")
        error.with_suggestion(f"Ensure the plugin is installed in the project's src/{plugin_type}s/ directory")
        return error

    @classmethod
    def load_failed(
        cls,
        plugin_name: str,
        plugin_path: Path,
        reason: str,
    ) -> "PluginError":
        """Create error for plugin load failure."""
        error = cls(
            f"Failed to load plugin '{plugin_name}': {reason}",
            plugin_name=plugin_name,
            plugin_path=plugin_path,
            error_code="PLUGIN_LOAD_FAILED",
        )
        error.with_suggestion("Check the plugin's imports and dependencies")
        error.with_suggestion("Ensure the plugin follows the expected interface")
        return error


class CLIError(KBertError):
    """CLI-specific errors."""

    def __init__(
        self,
        message: str,
        *,
        command: Optional[str] = None,
        exit_code: int = 1,
        **kwargs: Any,
    ):
        """Initialize CLI error."""
        super().__init__(message, **kwargs)
        self.exit_code = exit_code
        
        if command:
            self.context.add_technical_detail("command", command)
        self.context.add_technical_detail("exit_code", exit_code)

    @classmethod
    def invalid_command(cls, command: str, similar: Optional[List[str]] = None) -> "CLIError":
        """Create error for invalid command."""
        error = cls(
            f"Unknown command: {command}",
            command=command,
            error_code="CLI_INVALID_COMMAND",
            exit_code=2,
        )
        
        if similar:
            error.with_suggestion(f"Did you mean: {', '.join(similar)}?")
        error.with_suggestion("Use 'k-bert --help' to see available commands")
        
        return error

    @classmethod
    def missing_argument(cls, argument: str, command: str) -> "CLIError":
        """Create error for missing required argument."""
        error = cls(
            f"Missing required argument: {argument}",
            command=command,
            error_code="CLI_MISSING_ARGUMENT",
            exit_code=2,
        )
        error.with_suggestion(f"Use 'k-bert {command} --help' for usage information")
        return error