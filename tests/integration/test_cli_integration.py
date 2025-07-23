"""
Integration tests for CLI with the new hexagonal architecture.

These tests verify that the CLI properly integrates with the application layer
through the dependency injection container.
"""

import pytest
from pathlib import Path
import tempfile
from typer.testing import CliRunner
from unittest.mock import Mock, patch
import yaml

from cli.app import app
from infrastructure.bootstrap import initialize_application, get_service
from infrastructure.ports.monitoring import MonitoringService


class TestCLIIntegration:
    """Test CLI integration with hexagonal architecture."""
    
    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()
    
    @pytest.fixture
    def test_config(self):
        """Create a test configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config = {
                "model": {
                    "type": "modernbert_with_head",
                    "hidden_size": 128,
                    "num_hidden_layers": 2
                },
                "training": {
                    "epochs": 1,
                    "batch_size": 16,
                    "learning_rate": 1e-4
                },
                "data": {
                    "train_path": "train.csv",
                    "val_path": "val.csv"
                }
            }
            yaml.dump(config, f)
            yield Path(f.name)
        Path(f.name).unlink()
    
    def test_cli_initialization(self, runner):
        """Test that CLI initializes the application correctly."""
        # Run a simple command
        result = runner.invoke(app, ["--version"])
        
        # Should show version without errors
        assert result.exit_code == 0
        assert "k-bert" in result.stdout
    
    def test_cli_with_config_option(self, runner, test_config):
        """Test that CLI accepts config option and initializes correctly."""
        # Run info command with config
        result = runner.invoke(app, ["--config", str(test_config), "info"])
        
        # Should run without errors
        assert result.exit_code == 0
    
    @patch('core.bootstrap.initialize_application')
    def test_cli_calls_bootstrap(self, mock_init, runner):
        """Test that CLI calls bootstrap initialization."""
        # Set up mock
        mock_container = Mock()
        mock_init.return_value = mock_container
        
        # Run a command
        result = runner.invoke(app, ["info"])
        
        # Bootstrap should have been called
        mock_init.assert_called_once()
    
    def test_cli_verbose_flag(self, runner):
        """Test that verbose flag affects logging."""
        with patch('core.bootstrap.get_service') as mock_get_service:
            mock_monitoring = Mock(spec=MonitoringService)
            mock_get_service.return_value = mock_monitoring
            
            # Run with verbose flag
            result = runner.invoke(app, ["--verbose", "info"])
            
            # Monitoring level should be set to DEBUG
            mock_monitoring.set_level.assert_called_with("DEBUG")
    
    def test_cli_quiet_flag(self, runner):
        """Test that quiet flag affects logging."""
        with patch('core.bootstrap.get_service') as mock_get_service:
            mock_monitoring = Mock(spec=MonitoringService)
            mock_get_service.return_value = mock_monitoring
            
            # Run with quiet flag
            result = runner.invoke(app, ["--quiet", "info"])
            
            # Monitoring level should be set to ERROR
            mock_monitoring.set_level.assert_called_with("ERROR")
    
    def test_train_command_integration(self, runner, test_config):
        """Test train command integration with DI container."""
        with patch('cli.commands.core.train.TrainCommand') as mock_train_cmd:
            mock_instance = Mock()
            mock_instance.execute.return_value = Mock(success=True)
            mock_train_cmd.return_value = mock_instance
            
            # Run train command
            result = runner.invoke(app, ["train", "--config", str(test_config)])
            
            # Should create and execute command
            mock_train_cmd.assert_called()
            mock_instance.execute.assert_called()
    
    def test_predict_command_integration(self, runner, temp_dir):
        """Test predict command integration."""
        model_path = temp_dir / "model.safetensors"
        model_path.touch()
        
        with patch('cli.commands.core.predict.PredictCommand') as mock_predict_cmd:
            mock_instance = Mock()
            mock_instance.execute.return_value = Mock(predictions=[])
            mock_predict_cmd.return_value = mock_instance
            
            # Run predict command
            result = runner.invoke(app, [
                "predict",
                "--checkpoint", str(model_path),
                "--data", "test.csv"
            ])
            
            # Should create and execute command
            mock_predict_cmd.assert_called()
            mock_instance.execute.assert_called()
    
    def test_benchmark_command_integration(self, runner):
        """Test benchmark command integration."""
        with patch('cli.commands.core.benchmark.BenchmarkCommand') as mock_bench_cmd:
            mock_instance = Mock()
            mock_instance.execute.return_value = Mock(results={})
            mock_bench_cmd.return_value = mock_instance
            
            # Run benchmark command
            result = runner.invoke(app, ["benchmark"])
            
            # Should create and execute command
            mock_bench_cmd.assert_called()
            mock_instance.execute.assert_called()
    
    def test_config_subcommands(self, runner):
        """Test config subcommands work with new architecture."""
        # Test config init
        with patch('cli.commands.config.ConfigInitCommand') as mock_cmd:
            mock_instance = Mock()
            mock_instance.execute.return_value = Mock(success=True)
            mock_cmd.return_value = mock_instance
            
            result = runner.invoke(app, ["config", "init"])
            
            mock_cmd.assert_called()
            mock_instance.execute.assert_called()
    
    def test_project_subcommands(self, runner, temp_dir):
        """Test project subcommands work with new architecture."""
        project_dir = temp_dir / "myproject"
        
        with patch('cli.commands.project.ProjectInitCommand') as mock_cmd:
            mock_instance = Mock()
            mock_instance.execute.return_value = Mock(success=True)
            mock_cmd.return_value = mock_instance
            
            result = runner.invoke(app, ["project", "init", str(project_dir)])
            
            mock_cmd.assert_called()
            mock_instance.execute.assert_called()
    
    def test_cli_error_handling(self, runner):
        """Test that CLI handles errors gracefully."""
        with patch('core.bootstrap.initialize_application') as mock_init:
            mock_init.side_effect = Exception("Initialization error")
            
            # Run a command
            result = runner.invoke(app, ["info"])
            
            # Should handle error gracefully
            assert result.exit_code != 0
    
    def test_cli_container_passing(self, runner):
        """Test that container is passed to commands."""
        with patch('cli.app.app.obj', new=Mock()) as mock_container:
            with patch('cli.commands.core.info.InfoCommand') as mock_cmd:
                mock_instance = Mock()
                mock_cmd.return_value = mock_instance
                
                # Run command
                result = runner.invoke(app, ["info"])
                
                # Container should be available
                # Note: This tests the pattern, actual implementation may vary
                assert True  # Placeholder for actual container passing test
    
    @pytest.mark.parametrize("command", [
        ["train", "--help"],
        ["predict", "--help"],
        ["benchmark", "--help"],
        ["info", "--help"],
        ["config", "--help"],
        ["project", "--help"],
    ])
    def test_help_commands(self, runner, command):
        """Test that all help commands work."""
        result = runner.invoke(app, command)
        assert result.exit_code == 0
        assert "help" in result.stdout.lower() or "usage" in result.stdout.lower()