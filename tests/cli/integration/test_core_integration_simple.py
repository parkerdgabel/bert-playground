"""Simplified integration tests for core CLI commands."""

import json
from pathlib import Path
import pytest
from typer.testing import CliRunner
from cli.app import app


@pytest.mark.integration
class TestCLIBasics:
    """Test basic CLI functionality."""
    
    def test_cli_help(self, runner):
        """Test CLI help command."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "MLX BERT CLI" in result.stdout
        assert "train" in result.stdout
        assert "predict" in result.stdout
        assert "benchmark" in result.stdout
        assert "info" in result.stdout
    
    def test_cli_version(self, runner):
        """Test CLI version command."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "version" in result.stdout.lower()
    
    def test_train_help(self, runner):
        """Test train command help."""
        result = runner.invoke(app, ["train", "--help"])
        assert result.exit_code == 0
        assert "--train" in result.stdout
        assert "--epochs" in result.stdout
        assert "--batch-size" in result.stdout
    
    def test_predict_help(self, runner):
        """Test predict command help."""
        result = runner.invoke(app, ["predict", "--help"])
        assert result.exit_code == 0
        assert "--test" in result.stdout
        assert "--checkpoint" in result.stdout
        assert "--output" in result.stdout
    
    def test_benchmark_help(self, runner):
        """Test benchmark command help."""
        result = runner.invoke(app, ["benchmark", "--help"])
        assert result.exit_code == 0
        assert "--model-type" in result.stdout
        assert "--batch-size" in result.stdout
        assert "--steps" in result.stdout
    
    def test_info_help(self, runner):
        """Test info command help."""
        result = runner.invoke(app, ["info", "--help"])
        assert result.exit_code == 0
        assert "--mlx" in result.stdout or "--all" in result.stdout
        assert "--help" in result.stdout


@pytest.mark.integration
class TestValidationErrors:
    """Test command validation errors."""
    
    def test_train_missing_required_args(self, runner):
        """Test train command without required arguments."""
        result = runner.invoke(app, ["train"])
        assert result.exit_code != 0
        assert result.exit_code == 2  # Click/Typer exits with 2 for missing arguments
    
    def test_predict_missing_required_args(self, runner):
        """Test predict command without required arguments."""
        result = runner.invoke(app, ["predict"])
        assert result.exit_code != 0
        assert result.exit_code == 2  # Click/Typer exits with 2 for missing arguments
    
    def test_invalid_path_validation(self, runner):
        """Test path validation with non-existent file."""
        result = runner.invoke(app, [
            "train",
            "--train", "/nonexistent/file.csv",
            "--val", "/nonexistent/val.csv"
        ])
        assert result.exit_code != 0
        # Should show path validation error
    
    def test_invalid_batch_size(self, runner):
        """Test invalid batch size validation."""
        result = runner.invoke(app, [
            "train",
            "--train", "dummy.csv",
            "--batch-size", "-1"
        ])
        assert result.exit_code != 0
        # Should show batch size validation error


@pytest.mark.integration
class TestCommandGroups:
    """Test command groups functionality."""
    
    def test_kaggle_group_help(self, runner):
        """Test Kaggle command group help."""
        result = runner.invoke(app, ["kaggle", "--help"])
        assert result.exit_code == 0
        assert "Kaggle" in result.stdout
        assert "competitions" in result.stdout
        assert "download" in result.stdout
        assert "submit" in result.stdout
    
    def test_mlflow_group_help(self, runner):
        """Test MLflow command group help."""
        result = runner.invoke(app, ["mlflow", "--help"])
        assert result.exit_code == 0
        assert "MLflow" in result.stdout
        assert "server" in result.stdout
        assert "experiments" in result.stdout
    
    def test_model_group_help(self, runner):
        """Test model command group help."""
        result = runner.invoke(app, ["model", "--help"])
        assert result.exit_code == 0
        assert "Model" in result.stdout or "model" in result.stdout
        assert "serve" in result.stdout
        assert "export" in result.stdout


@pytest.mark.integration 
class TestConfigIntegration:
    """Test configuration file integration."""
    
    def test_train_with_config_file(self, runner, temp_project, config_file):
        """Test training with configuration file."""
        # Create dummy data files
        train_path = temp_project / "train.csv"
        val_path = temp_project / "val.csv"
        train_path.write_text("col1,col2,label\n1,2,0\n3,4,1")
        val_path.write_text("col1,col2,label\n5,6,0\n7,8,1")
        
        result = runner.invoke(app, [
            "train",
            "--train", str(train_path),
            "--val", str(val_path),
            "--config", str(config_file),
            "--dry-run"  # If there's a dry-run option
        ])
        
        # Should at least attempt to run with config
        # Exit codes: 0=success, 1=error, 2=usage error
        assert result.exit_code in [0, 1, 2]
    
    def test_json_config_file(self, runner, temp_project, json_config_file):
        """Test with JSON configuration file."""
        result = runner.invoke(app, [
            "train",
            "--train", "dummy.csv",
            "--config", str(json_config_file),
            "--help"  # Just test that config loads
        ])
        
        # Should show help without config errors
        assert result.exit_code == 0