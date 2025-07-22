"""Integration tests for error handling and edge cases."""

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import yaml
from typer.testing import CliRunner

from cli.app import app


class TestErrorHandling:
    """Test error handling and edge cases in CLI."""

    @pytest.fixture
    def runner(self):
        """Create CLI runner."""
        return CliRunner()

    @pytest.fixture
    def invalid_config(self, tmp_path):
        """Create project with invalid configuration."""
        # Invalid YAML syntax
        with open(tmp_path / "k-bert.yaml", "w") as f:
            f.write("invalid yaml: [\n  missing closing bracket")
        return tmp_path

    @pytest.fixture
    def incomplete_config(self, tmp_path):
        """Create project with incomplete configuration."""
        config = {
            "name": "incomplete-project",
            # Missing required fields like models, data
        }
        with open(tmp_path / "k-bert.yaml", "w") as f:
            yaml.dump(config, f)
        return tmp_path

    def test_missing_config_file(self, runner, tmp_path):
        """Test commands when no config file exists."""
        os.chdir(tmp_path)
        
        # Train should fail
        result = runner.invoke(app, ["train"])
        assert result.exit_code == 1
        assert "No k-bert.yaml found" in result.stdout or "configuration" in result.stdout
        
        # Run should fail
        result = runner.invoke(app, ["run"])
        assert result.exit_code == 1
        assert "No k-bert.yaml found" in result.stdout

    def test_invalid_yaml_syntax(self, runner, invalid_config):
        """Test handling of invalid YAML syntax."""
        os.chdir(invalid_config)
        
        result = runner.invoke(app, ["config", "validate"])
        assert result.exit_code == 1
        assert "YAML" in result.stdout or "syntax" in result.stdout

    def test_missing_required_fields(self, runner, incomplete_config):
        """Test validation of incomplete configuration."""
        os.chdir(incomplete_config)
        
        result = runner.invoke(app, ["config", "validate"])
        assert result.exit_code == 1
        assert "Validation errors" in result.stdout

    def test_invalid_model_name(self, runner, tmp_path):
        """Test invalid model name in configuration."""
        os.chdir(tmp_path)
        
        config = {
            "name": "test-project",
            "models": {"default_model": "invalid-model-name-xyz"},
            "data": {"train_path": "data/train.csv"}
        }
        
        with open("k-bert.yaml", "w") as f:
            yaml.dump(config, f)
        
        with patch('cli.commands.core.train._load_data'):
            result = runner.invoke(app, ["train"])
            assert result.exit_code == 1
            assert "model" in result.stdout.lower()

    def test_missing_data_files(self, runner, tmp_path):
        """Test handling of missing data files."""
        os.chdir(tmp_path)
        
        config = {
            "name": "test-project",
            "models": {"default_model": "bert-base-uncased"},
            "data": {
                "train_path": "data/nonexistent.csv",
                "val_path": "data/also_missing.csv"
            }
        }
        
        with open("k-bert.yaml", "w") as f:
            yaml.dump(config, f)
        
        result = runner.invoke(app, ["train"])
        assert result.exit_code == 1
        assert "not found" in result.stdout or "does not exist" in result.stdout

    def test_invalid_experiment_name(self, runner, tmp_path):
        """Test running non-existent experiment."""
        os.chdir(tmp_path)
        
        config = {
            "name": "test-project",
            "models": {"default_model": "bert-base-uncased"},
            "experiments": [
                {"name": "valid_exp", "config": {}}
            ]
        }
        
        with open("k-bert.yaml", "w") as f:
            yaml.dump(config, f)
        
        result = runner.invoke(app, ["run", "--experiment", "nonexistent"])
        assert result.exit_code == 1
        assert "Experiment 'nonexistent' not found" in result.stdout

    def test_kaggle_auth_missing(self, runner):
        """Test Kaggle commands without authentication."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('kaggle.api_client.ApiClient', side_effect=Exception("Authentication required")):
                result = runner.invoke(app, ["competition", "list"])
                assert result.exit_code == 1
                assert "authentication" in result.stdout.lower() or "kaggle" in result.stdout.lower()

    def test_network_error_handling(self, runner):
        """Test handling of network errors."""
        with patch('kaggle.api.competition_list', side_effect=ConnectionError("Network error")):
            result = runner.invoke(app, ["competition", "list"])
            assert result.exit_code == 1
            assert "error" in result.stdout.lower()

    def test_insufficient_permissions(self, runner, tmp_path):
        """Test handling of permission errors."""
        os.chdir(tmp_path)
        
        # Create read-only directory
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)
        
        result = runner.invoke(app, ["project", "init", str(readonly_dir / "new_project")])
        assert result.exit_code == 1
        assert "permission" in result.stdout.lower() or "access" in result.stdout.lower()
        
        # Cleanup
        readonly_dir.chmod(0o755)

    def test_interrupt_handling(self, runner, tmp_path):
        """Test handling of keyboard interrupts."""
        os.chdir(tmp_path)
        
        config = {
            "name": "test-project",
            "models": {"default_model": "bert-base-uncased"},
            "data": {"train_path": "data/train.csv"}
        }
        
        with open("k-bert.yaml", "w") as f:
            yaml.dump(config, f)
        
        with patch('cli.commands.core.train.train_model', side_effect=KeyboardInterrupt()):
            result = runner.invoke(app, ["train"])
            assert result.exit_code != 0
            assert "interrupted" in result.stdout.lower() or "cancelled" in result.stdout.lower()

    def test_out_of_memory_handling(self, runner, tmp_path):
        """Test handling of out of memory errors."""
        os.chdir(tmp_path)
        
        config = {
            "name": "test-project",
            "models": {"default_model": "bert-base-uncased"},
            "data": {"train_path": "data/train.csv", "batch_size": 1024}
        }
        
        with open("k-bert.yaml", "w") as f:
            yaml.dump(config, f)
        
        with patch('models.factory.create_model', side_effect=RuntimeError("Out of memory")):
            result = runner.invoke(app, ["train"])
            assert result.exit_code == 1
            assert "memory" in result.stdout.lower()

    def test_circular_config_includes(self, runner, tmp_path):
        """Test handling of circular configuration includes."""
        os.chdir(tmp_path)
        
        # Create configs that include each other
        config1 = {
            "name": "config1",
            "includes": ["config2.yaml"]
        }
        
        config2 = {
            "name": "config2",
            "includes": ["k-bert.yaml"]  # Circular reference
        }
        
        with open("k-bert.yaml", "w") as f:
            yaml.dump(config1, f)
        
        with open("config2.yaml", "w") as f:
            yaml.dump(config2, f)
        
        result = runner.invoke(app, ["config", "validate"])
        assert result.exit_code == 1
        assert "circular" in result.stdout.lower() or "recursion" in result.stdout.lower()

    def test_invalid_cli_arguments(self, runner):
        """Test invalid CLI argument combinations."""
        # Invalid batch size
        result = runner.invoke(app, ["train", "--batch-size", "-1"])
        assert result.exit_code == 2  # Click validation error
        
        # Invalid learning rate
        result = runner.invoke(app, ["train", "--lr", "invalid"])
        assert result.exit_code == 2
        
        # Conflicting options
        result = runner.invoke(app, ["config", "init", "--interactive", "--no-interactive"])
        assert result.exit_code == 2

    def test_empty_data_files(self, runner, tmp_path):
        """Test handling of empty data files."""
        os.chdir(tmp_path)
        
        config = {
            "name": "test-project",
            "models": {"default_model": "bert-base-uncased"},
            "data": {"train_path": "data/train.csv"}
        }
        
        with open("k-bert.yaml", "w") as f:
            yaml.dump(config, f)
        
        # Create empty data file
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "train.csv").write_text("")
        
        result = runner.invoke(app, ["train"])
        assert result.exit_code == 1
        assert "empty" in result.stdout.lower() or "no data" in result.stdout.lower()

    def test_corrupt_checkpoint_loading(self, runner, tmp_path):
        """Test loading corrupt model checkpoints."""
        os.chdir(tmp_path)
        
        # Create corrupt checkpoint
        checkpoint_dir = tmp_path / "checkpoint"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "config.json").write_text("invalid json {")
        
        result = runner.invoke(app, ["predict", str(checkpoint_dir)])
        assert result.exit_code == 1
        assert "failed to load" in result.stdout.lower() or "corrupt" in result.stdout.lower()

    def test_version_mismatch(self, runner, tmp_path):
        """Test handling of version mismatches."""
        os.chdir(tmp_path)
        
        config = {
            "name": "test-project",
            "version": "99.0",  # Future version
            "min_cli_version": "99.0"
        }
        
        with open("k-bert.yaml", "w") as f:
            yaml.dump(config, f)
        
        result = runner.invoke(app, ["config", "validate"])
        assert result.exit_code == 1
        assert "version" in result.stdout.lower()

    def test_mlflow_connection_error(self, runner, tmp_path):
        """Test MLflow connection errors."""
        os.chdir(tmp_path)
        
        config = {
            "name": "test-project",
            "mlflow": {
                "tracking_uri": "http://nonexistent:5000",
                "auto_log": True
            }
        }
        
        with open("k-bert.yaml", "w") as f:
            yaml.dump(config, f)
        
        with patch('mlflow.set_tracking_uri', side_effect=Exception("Connection refused")):
            result = runner.invoke(app, ["mlflow", "ui"])
            assert result.exit_code == 1
            assert "mlflow" in result.stdout.lower()

    def test_concurrent_access_handling(self, runner, tmp_path):
        """Test handling of concurrent access to resources."""
        os.chdir(tmp_path)
        
        # Simulate locked file
        lock_file = tmp_path / ".k-bert.lock"
        lock_file.write_text("pid: 12345")
        
        # Future: implement lock file checking
        # result = runner.invoke(app, ["train"])
        # assert "already running" in result.stdout or "locked" in result.stdout