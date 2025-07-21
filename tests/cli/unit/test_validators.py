"""Unit tests for CLI validators."""

import re
from unittest.mock import patch

import pytest
import typer

from cli.utils.validators import (
    validate_batch_size,
    validate_epochs,
    validate_kaggle_competition,
    validate_learning_rate,
    validate_model_type,
    validate_output_format,
    validate_path,
    validate_percentage,
    validate_port,
)


class TestValidatePath:
    """Test path validation."""

    def test_valid_existing_file(self, tmp_path):
        """Test validation of existing file."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("content")

        result = validate_path(file_path, must_exist=True, must_be_file=True)
        assert result == file_path

    def test_valid_existing_directory(self, tmp_path):
        """Test validation of existing directory."""
        dir_path = tmp_path / "testdir"
        dir_path.mkdir()

        result = validate_path(dir_path, must_exist=True, must_be_dir=True)
        assert result == dir_path

    def test_nonexistent_path_with_must_exist(self, tmp_path):
        """Test validation fails for nonexistent path when must_exist=True."""
        nonexistent = tmp_path / "nonexistent.txt"

        with pytest.raises(typer.BadParameter, match="Path does not exist"):
            validate_path(nonexistent, must_exist=True)

    def test_nonexistent_path_without_must_exist(self, tmp_path):
        """Test validation passes for nonexistent path when must_exist=False."""
        nonexistent = tmp_path / "nonexistent.txt"

        result = validate_path(nonexistent, must_exist=False)
        assert result == nonexistent

    def test_file_when_expecting_directory(self, tmp_path):
        """Test validation fails when file provided but directory expected."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("content")

        with pytest.raises(typer.BadParameter, match="Path is not a directory"):
            validate_path(file_path, must_exist=True, must_be_dir=True)

    def test_directory_when_expecting_file(self, tmp_path):
        """Test validation fails when directory provided but file expected."""
        dir_path = tmp_path / "testdir"
        dir_path.mkdir()

        with pytest.raises(typer.BadParameter, match="Path is not a file"):
            validate_path(dir_path, must_exist=True, must_be_file=True)

    def test_valid_file_extension(self, tmp_path):
        """Test validation with valid file extension."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("data")

        result = validate_path(
            csv_file, must_exist=True, must_be_file=True, extensions=[".csv", ".txt"]
        )
        assert result == csv_file

    def test_invalid_file_extension(self, tmp_path):
        """Test validation fails with invalid file extension."""
        json_file = tmp_path / "data.json"
        json_file.write_text("{}")

        with pytest.raises(typer.BadParameter, match="Invalid file extension"):
            validate_path(
                json_file,
                must_exist=True,
                must_be_file=True,
                extensions=[".csv", ".txt"],
            )

    def test_extension_check_on_directory(self, tmp_path):
        """Test extension check is skipped for directories."""
        dir_path = tmp_path / "testdir"
        dir_path.mkdir()

        # Should not raise even though directory has no extension
        result = validate_path(
            dir_path, must_exist=True, must_be_dir=True, extensions=[".csv"]
        )
        assert result == dir_path


class TestValidateBatchSize:
    """Test batch size validation."""

    @pytest.mark.parametrize(
        "batch_size", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    )
    def test_valid_batch_sizes_power_of_2(self, batch_size):
        """Test valid batch sizes that are powers of 2."""
        result = validate_batch_size(batch_size)
        assert result == batch_size

    @pytest.mark.parametrize("batch_size", [3, 5, 7, 10, 15, 20, 50, 100])
    def test_valid_batch_sizes_not_power_of_2(self, batch_size):
        """Test valid batch sizes that are not powers of 2 (should warn)."""
        with patch("typer.echo") as mock_echo:
            result = validate_batch_size(batch_size)
            assert result == batch_size

            # Check warning was issued
            mock_echo.assert_called_once()
            warning_msg = mock_echo.call_args[0][0]
            assert "not a power of 2" in warning_msg
            assert str(batch_size) in warning_msg

    @pytest.mark.parametrize("batch_size", [0, -1, -10, -100])
    def test_invalid_batch_sizes_non_positive(self, batch_size):
        """Test validation fails for non-positive batch sizes."""
        with pytest.raises(typer.BadParameter, match="Batch size must be positive"):
            validate_batch_size(batch_size)

    @pytest.mark.parametrize("batch_size", [1025, 2048, 4096, 10000])
    def test_invalid_batch_sizes_too_large(self, batch_size):
        """Test validation fails for excessively large batch sizes."""
        with pytest.raises(typer.BadParameter, match="Batch size > 1024"):
            validate_batch_size(batch_size)

    def test_edge_case_1024(self):
        """Test edge case of exactly 1024 (should pass)."""
        result = validate_batch_size(1024)
        assert result == 1024


class TestValidateModelType:
    """Test model type validation."""

    @pytest.mark.parametrize(
        "model_type",
        [
            "modernbert",
            "modernbert-cnn",
            "mlx-bert",
            "answerdotai/ModernBERT-base",
            "answerdotai/ModernBERT-large",
            "bert-base-uncased",
            "bert-large-uncased",
        ],
    )
    def test_valid_model_types(self, model_type):
        """Test all valid model types."""
        result = validate_model_type(model_type)
        assert result == model_type

    @pytest.mark.parametrize(
        "model_type",
        [
            "gpt",
            "t5",
            "roberta",
            "invalid-model",
            "BERT",  # Wrong case
            "modern-bert",  # Wrong format
            "",
            "bert",  # Incomplete
        ],
    )
    def test_invalid_model_types(self, model_type):
        """Test invalid model types."""
        with pytest.raises(typer.BadParameter, match="Invalid model type"):
            validate_model_type(model_type)


class TestValidateLearningRate:
    """Test learning rate validation."""

    @pytest.mark.parametrize(
        "lr", [1e-8, 1e-7, 1e-6, 1e-5, 2e-5, 5e-5, 1e-4, 1e-3, 0.01, 0.1, 0.5, 1.0]
    )
    def test_valid_learning_rates(self, lr):
        """Test valid learning rates."""
        result = validate_learning_rate(lr)
        assert result == lr

    @pytest.mark.parametrize("lr", [0, -0.001, -1e-5, -1])
    def test_invalid_learning_rates_non_positive(self, lr):
        """Test validation fails for non-positive learning rates."""
        with pytest.raises(typer.BadParameter, match="Learning rate must be positive"):
            validate_learning_rate(lr)

    @pytest.mark.parametrize("lr", [1.1, 2.0, 10.0, 100.0])
    def test_invalid_learning_rates_too_large(self, lr):
        """Test validation fails for excessively large learning rates."""
        with pytest.raises(typer.BadParameter, match="Learning rate > 1.0"):
            validate_learning_rate(lr)

    @pytest.mark.parametrize("lr", [1e-9, 1e-10, 1e-20])
    def test_invalid_learning_rates_too_small(self, lr):
        """Test validation fails for extremely small learning rates."""
        with pytest.raises(typer.BadParameter, match="Learning rate < 1e-8"):
            validate_learning_rate(lr)

    def test_edge_cases(self):
        """Test edge cases."""
        # Exactly 1e-8 (should pass)
        assert validate_learning_rate(1e-8) == 1e-8

        # Exactly 1.0 (should pass)
        assert validate_learning_rate(1.0) == 1.0


class TestValidateEpochs:
    """Test epochs validation."""

    @pytest.mark.parametrize("epochs", [1, 2, 5, 10, 50, 100, 500, 1000])
    def test_valid_epochs(self, epochs):
        """Test valid epoch values."""
        result = validate_epochs(epochs)
        assert result == epochs

    @pytest.mark.parametrize("epochs", [0, -1, -10, -100])
    def test_invalid_epochs_non_positive(self, epochs):
        """Test validation fails for non-positive epochs."""
        with pytest.raises(typer.BadParameter, match="epochs must be positive"):
            validate_epochs(epochs)

    @pytest.mark.parametrize("epochs", [1001, 2000, 10000])
    def test_invalid_epochs_too_many(self, epochs):
        """Test validation fails for excessive epochs."""
        with pytest.raises(typer.BadParameter, match="epochs > 1000 is excessive"):
            validate_epochs(epochs)

    def test_edge_case_1000(self):
        """Test edge case of exactly 1000 epochs (should pass)."""
        result = validate_epochs(1000)
        assert result == 1000


class TestValidatePort:
    """Test port number validation."""

    @pytest.mark.parametrize(
        "port", [1024, 3000, 5000, 8000, 8080, 8888, 9000, 50000, 65535]
    )
    def test_valid_ports_non_privileged(self, port):
        """Test valid non-privileged port numbers."""
        result = validate_port(port)
        assert result == port

    @pytest.mark.parametrize("port", [1, 22, 80, 443, 1023])
    def test_valid_ports_privileged(self, port):
        """Test valid privileged port numbers (should warn)."""
        with patch("typer.echo") as mock_echo:
            result = validate_port(port)
            assert result == port

            # Check warning was issued
            mock_echo.assert_called_once()
            warning_msg = mock_echo.call_args[0][0]
            assert "requires root privileges" in warning_msg
            assert str(port) in warning_msg

    @pytest.mark.parametrize("port", [0, -1, -100, -1000])
    def test_invalid_ports_too_low(self, port):
        """Test validation fails for ports below valid range."""
        with pytest.raises(typer.BadParameter, match="Port must be between"):
            validate_port(port)

    @pytest.mark.parametrize("port", [65536, 70000, 100000])
    def test_invalid_ports_too_high(self, port):
        """Test validation fails for ports above valid range."""
        with pytest.raises(typer.BadParameter, match="Port must be between"):
            validate_port(port)

    def test_edge_cases(self):
        """Test edge cases."""
        # Port 1 (minimum valid, privileged)
        with patch("typer.echo"):
            assert validate_port(1) == 1

        # Port 65535 (maximum valid)
        assert validate_port(65535) == 65535


class TestValidatePercentage:
    """Test percentage validation."""

    @pytest.mark.parametrize("value", [0, 0.1, 0.25, 0.5, 0.75, 0.99, 1.0])
    def test_valid_percentages(self, value):
        """Test valid percentage values."""
        result = validate_percentage(value)
        assert result == value

    @pytest.mark.parametrize("value", [-0.1, -0.5, -1, -10])
    def test_invalid_percentages_negative(self, value):
        """Test validation fails for negative values."""
        with pytest.raises(typer.BadParameter, match="must be between 0 and 1"):
            validate_percentage(value)

    @pytest.mark.parametrize("value", [1.1, 1.5, 2, 10, 100])
    def test_invalid_percentages_too_large(self, value):
        """Test validation fails for values > 1."""
        with pytest.raises(typer.BadParameter, match="must be between 0 and 1"):
            validate_percentage(value)

    def test_custom_name(self):
        """Test custom parameter name in error message."""
        with pytest.raises(typer.BadParameter, match="dropout rate must be between"):
            validate_percentage(1.5, name="dropout rate")

    def test_edge_cases(self):
        """Test edge cases."""
        # Exactly 0 (should pass)
        assert validate_percentage(0) == 0

        # Exactly 1 (should pass)
        assert validate_percentage(1.0) == 1.0


class TestValidateKaggleCompetition:
    """Test Kaggle competition name validation."""

    @pytest.mark.parametrize(
        "name",
        [
            "titanic",
            "house-prices",
            "digit-recognizer",
            "tabular-playground-series-jan-2024",
            "llm-20-questions",
            "competition123",
            "test-123-abc",
        ],
    )
    def test_valid_competition_names(self, name):
        """Test valid Kaggle competition names."""
        result = validate_kaggle_competition(name)
        assert result == name

    @pytest.mark.parametrize(
        "name",
        [
            "Titanic",  # Uppercase
            "House_Prices",  # Underscores
            "competition name",  # Spaces
            "competition@kaggle",  # Special characters
            "competition!",
            "competition#123",
            "",  # Empty
            "CamelCase",
            "snake_case",
            "competition.name",
        ],
    )
    def test_invalid_competition_names(self, name):
        """Test invalid Kaggle competition names."""
        with pytest.raises(typer.BadParameter, match="Invalid competition name"):
            validate_kaggle_competition(name)

    def test_regex_pattern(self):
        """Test the regex pattern explicitly."""
        pattern = r"^[a-z0-9-]+$"

        # Valid names
        assert re.match(pattern, "titanic")
        assert re.match(pattern, "house-prices-2024")
        assert re.match(pattern, "123-competition")

        # Invalid names
        assert not re.match(pattern, "Titanic")
        assert not re.match(pattern, "house_prices")
        assert not re.match(pattern, "competition name")
        assert not re.match(pattern, "")


class TestValidateOutputFormat:
    """Test output format validation."""

    @pytest.mark.parametrize("format", ["human", "json", "csv", "yaml"])
    def test_valid_output_formats(self, format):
        """Test all valid output formats."""
        result = validate_output_format(format)
        assert result == format

    @pytest.mark.parametrize(
        "format",
        [
            "xml",
            "txt",
            "HTML",
            "JSON",  # Wrong case
            "table",
            "markdown",
            "",
            "human-readable",
        ],
    )
    def test_invalid_output_formats(self, format):
        """Test invalid output formats."""
        with pytest.raises(typer.BadParameter, match="Invalid output format"):
            validate_output_format(format)

    def test_error_message_includes_valid_formats(self):
        """Test error message includes list of valid formats."""
        try:
            validate_output_format("invalid")
        except typer.BadParameter as e:
            error_msg = str(e)
            assert "human" in error_msg
            assert "json" in error_msg
            assert "csv" in error_msg
            assert "yaml" in error_msg
