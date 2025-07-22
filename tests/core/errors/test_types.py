"""Tests for specific error types."""

import pytest
from pathlib import Path

from core.errors import (
    ConfigurationError,
    ModelError,
    DataError,
    TrainingError,
    ValidationError,
    PluginError,
    CLIError,
)


class TestConfigurationError:
    """Test ConfigurationError."""

    def test_missing_required_field(self):
        """Test missing required field error."""
        error = ConfigurationError.missing_required_field(
            "models.bert.hidden_size",
            Path("/path/to/config.yaml"),
        )
        
        assert "Missing required configuration field" in error.message
        assert error.error_code == "CONFIG_MISSING_FIELD"
        assert error.context.technical_details["field_path"] == "models.bert.hidden_size"
        assert len(error.context.suggestions) > 0

    def test_invalid_value(self):
        """Test invalid value error."""
        error = ConfigurationError.invalid_value(
            "training.batch_size",
            "invalid",
            "int",
            Path("/path/to/config.yaml"),
        )
        
        assert "Invalid value" in error.message
        assert error.error_code == "CONFIG_INVALID_VALUE"
        assert error.context.technical_details["invalid_value"] == "invalid"
        assert error.context.technical_details["expected_type"] == "int"


class TestModelError:
    """Test ModelError."""

    def test_unsupported_model_type(self):
        """Test unsupported model type error."""
        error = ModelError.unsupported_model_type("unknown_bert")
        
        assert "Unsupported model type" in error.message
        assert error.error_code == "MODEL_UNSUPPORTED_TYPE"
        assert error.context.technical_details["model_type"] == "unknown_bert"

    def test_checkpoint_not_found(self):
        """Test checkpoint not found error."""
        path = Path("/path/to/checkpoint.safetensors")
        error = ModelError.checkpoint_not_found(path)
        
        assert "Checkpoint not found" in error.message
        assert error.error_code == "MODEL_CHECKPOINT_NOT_FOUND"
        assert not error.recoverable

    def test_incompatible_checkpoint(self):
        """Test incompatible checkpoint error."""
        error = ModelError.incompatible_checkpoint(
            Path("/path/to/checkpoint.safetensors"),
            "modernbert",
            "bert",
        )
        
        assert "architecture mismatch" in error.message
        assert error.error_code == "MODEL_CHECKPOINT_INCOMPATIBLE"
        assert not error.recoverable


class TestDataError:
    """Test DataError."""

    def test_file_not_found(self):
        """Test file not found error."""
        path = Path("/path/to/data.csv")
        error = DataError.file_not_found(path)
        
        assert "Data file not found" in error.message
        assert error.error_code == "DATA_FILE_NOT_FOUND"
        assert not error.recoverable

    def test_missing_column(self):
        """Test missing column error."""
        error = DataError.missing_column(
            "target",
            ["feature1", "feature2", "feature3"],
            Path("/path/to/data.csv"),
        )
        
        assert "Required column 'target' not found" in error.message
        assert error.error_code == "DATA_MISSING_COLUMN"
        assert "feature1" in error.context.suggestions[0]

    def test_invalid_format(self):
        """Test invalid format error."""
        error = DataError.invalid_format(
            Path("/path/to/data.txt"),
            "csv",
            "txt",
        )
        
        assert "Invalid data format" in error.message
        assert error.error_code == "DATA_INVALID_FORMAT"


class TestTrainingError:
    """Test TrainingError."""

    def test_nan_loss(self):
        """Test NaN loss error."""
        error = TrainingError.nan_loss(epoch=5, step=100)
        
        assert "NaN loss detected" in error.message
        assert error.error_code == "TRAINING_NAN_LOSS"
        assert error.context.technical_details["epoch"] == 5
        assert error.context.technical_details["step"] == 100
        assert len(error.context.suggestions) > 0
        assert len(error.context.recovery_actions) > 0

    def test_out_of_memory(self):
        """Test out of memory error."""
        error = TrainingError.out_of_memory(batch_size=64, model_size="large")
        
        assert "Out of memory" in error.message
        assert error.error_code == "TRAINING_OOM"
        assert "batch_size" in error.context.technical_details
        assert "Reduce batch size" in error.context.suggestions[0]


class TestValidationError:
    """Test ValidationError."""

    def test_invalid_range(self):
        """Test invalid range error."""
        error = ValidationError.invalid_range(
            "learning_rate",
            2.0,
            min_value=0.0,
            max_value=1.0,
        )
        
        assert "out of range" in error.message
        assert error.error_code == "VALIDATION_OUT_OF_RANGE"
        assert error.context.technical_details["value"] == "2.0"
        assert "between 0.0 and 1.0" in error.context.suggestions[0]


class TestPluginError:
    """Test PluginError."""

    def test_not_found(self):
        """Test plugin not found error."""
        error = PluginError.not_found("custom_head", "heads")
        
        assert "Plugin 'custom_head'" in error.message
        assert error.error_code == "PLUGIN_NOT_FOUND"
        assert error.context.technical_details["plugin_name"] == "custom_head"

    def test_load_failed(self):
        """Test plugin load failure error."""
        error = PluginError.load_failed(
            "custom_head",
            Path("/path/to/plugin.py"),
            "Import error: No module named 'missing'",
        )
        
        assert "Failed to load plugin" in error.message
        assert error.error_code == "PLUGIN_LOAD_FAILED"


class TestCLIError:
    """Test CLIError."""

    def test_invalid_command(self):
        """Test invalid command error."""
        error = CLIError.invalid_command("trian", similar=["train", "trial"])
        
        assert "Unknown command: trian" in error.message
        assert error.error_code == "CLI_INVALID_COMMAND"
        assert error.exit_code == 2
        assert "Did you mean: train, trial?" in error.context.suggestions[0]

    def test_missing_argument(self):
        """Test missing argument error."""
        error = CLIError.missing_argument("--config", "train")
        
        assert "Missing required argument: --config" in error.message
        assert error.error_code == "CLI_MISSING_ARGUMENT"
        assert error.exit_code == 2