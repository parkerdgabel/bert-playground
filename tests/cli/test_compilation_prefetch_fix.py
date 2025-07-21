"""Test the fix for compilation + prefetch conflict."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

def test_compilation_disables_prefetch():
    """Test that prefetch is disabled when compilation is enabled."""
    # Mock the necessary imports
    with patch('cli.commands.core.train.create_dataloader') as mock_create_dataloader, \
         patch('cli.commands.core.train.create_model'), \
         patch('cli.commands.core.train.BaseTrainer'), \
         patch('cli.commands.core.train.AutoTokenizer'), \
         patch('utils.config_loader.ConfigLoader') as mock_config_loader:
        
        # Mock config with compilation enabled
        mock_config_loader.load.return_value = {
            "training": {"use_compilation": True},
            "data": {}  # No explicit mlx_prefetch_size
        }
        
        # Import the train function
        from cli.commands.core.train import train_command
        from typer.testing import CliRunner
        from cli import app
        
        runner = CliRunner()
        
        # Create test data files
        train_file = Path("/tmp/test_train.csv")
        train_file.write_text("text,label\ntest1,0\ntest2,1")
        
        # Run the command with config
        result = runner.invoke(
            app,
            ["train", 
             "--train", str(train_file),
             "--config", "/tmp/test_config.yaml",
             "--epochs", "1",
             "--batch-size", "2"]
        )
        
        # Check that create_dataloader was called with prefetch disabled
        assert mock_create_dataloader.called
        call_kwargs = mock_create_dataloader.call_args[1]
        assert call_kwargs['prefetch_size'] == 0
        assert call_kwargs['mlx_prefetch_size'] == 0


def test_compilation_respects_explicit_prefetch():
    """Test that explicit prefetch setting is respected even with compilation."""
    with patch('cli.commands.core.train.create_dataloader') as mock_create_dataloader, \
         patch('cli.commands.core.train.create_model'), \
         patch('cli.commands.core.train.BaseTrainer'), \
         patch('cli.commands.core.train.AutoTokenizer'), \
         patch('utils.config_loader.ConfigLoader') as mock_config_loader:
        
        # Mock config with compilation enabled AND explicit prefetch
        mock_config_loader.load.return_value = {
            "training": {"use_compilation": True},
            "data": {"mlx_prefetch_size": 4}  # Explicit prefetch size
        }
        
        from typer.testing import CliRunner
        from cli import app
        
        runner = CliRunner()
        
        # Create test data files
        train_file = Path("/tmp/test_train2.csv")
        train_file.write_text("text,label\ntest1,0\ntest2,1")
        
        # Run the command
        result = runner.invoke(
            app,
            ["train", 
             "--train", str(train_file),
             "--config", "/tmp/test_config.yaml",
             "--epochs", "1"]
        )
        
        # Check that explicit prefetch was respected
        assert mock_create_dataloader.called
        call_kwargs = mock_create_dataloader.call_args[1]
        assert call_kwargs['mlx_prefetch_size'] == 4  # Explicit value respected


def test_no_compilation_keeps_prefetch():
    """Test that prefetch is kept when compilation is disabled."""
    with patch('cli.commands.core.train.create_dataloader') as mock_create_dataloader, \
         patch('cli.commands.core.train.create_model'), \
         patch('cli.commands.core.train.BaseTrainer'), \
         patch('cli.commands.core.train.AutoTokenizer'), \
         patch('utils.config_loader.ConfigLoader') as mock_config_loader:
        
        # Mock config with compilation disabled
        mock_config_loader.load.return_value = {
            "training": {"use_compilation": False},
            "data": {}
        }
        
        from typer.testing import CliRunner
        from cli import app
        
        runner = CliRunner()
        
        # Create test data files
        train_file = Path("/tmp/test_train3.csv")
        train_file.write_text("text,label\ntest1,0\ntest2,1")
        
        # Run the command with default prefetch
        result = runner.invoke(
            app,
            ["train", 
             "--train", str(train_file),
             "--config", "/tmp/test_config.yaml",
             "--prefetch", "4",  # CLI default
             "--epochs", "1"]
        )
        
        # Check that prefetch was kept
        assert mock_create_dataloader.called
        call_kwargs = mock_create_dataloader.call_args[1]
        assert call_kwargs['prefetch_size'] == 4  # Default kept