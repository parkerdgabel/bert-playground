"""Test the fix for compilation + prefetch conflict."""

from pathlib import Path
from unittest.mock import MagicMock, patch, ANY
import tempfile
import os


def test_compilation_disables_prefetch():
    """Test that prefetch is disabled when compilation is enabled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        train_file = Path(tmpdir) / "train.csv"
        train_file.write_text("text,label\ntest1,0\ntest2,1")
        
        config_file = Path(tmpdir) / "config.yaml"
        config_file.write_text("""
training:
  use_compilation: true
  epochs: 1
  batch_size: 2
data:
  # no explicit prefetch settings
""")
        
        # Mock imports at module level
        with (
            patch("data.factory.create_dataloader") as mock_create_dataloader,
            patch("data.factory.create_dataset") as mock_create_dataset,
            patch("models.factory.create_model") as mock_create_model,
            patch("training.core.base.BaseTrainer") as mock_trainer_class,
            patch("transformers.AutoTokenizer") as mock_tokenizer_class,
        ):
            # Set up mocks
            mock_dataset = MagicMock()
            mock_create_dataset.return_value = mock_dataset
            
            mock_loader = MagicMock()
            mock_create_dataloader.return_value = mock_loader
            
            mock_model = MagicMock()
            mock_create_model.return_value = mock_model
            
            mock_tokenizer = MagicMock()
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            
            mock_trainer = MagicMock()
            mock_trainer.train.return_value = MagicMock()
            mock_trainer_class.return_value = mock_trainer
            
            # Import and run CLI
            from typer.testing import CliRunner
            from cli import app
            
            runner = CliRunner()
            result = runner.invoke(
                app,
                [
                    "train",
                    "--train", str(train_file),
                    "--config", str(config_file),
                ],
            )
            
            # Check that create_dataloader was called
            assert mock_create_dataloader.called, f"create_dataloader not called. Output: {result.output}"
            
            # Check the arguments - prefetch should be 0
            call_kwargs = mock_create_dataloader.call_args[1]
            assert "prefetch_size" in call_kwargs
            assert call_kwargs["prefetch_size"] == 0, f"Expected prefetch_size=0, got {call_kwargs['prefetch_size']}"
            
            # MLX prefetch should also be 0
            if "mlx_prefetch_size" in call_kwargs:
                assert call_kwargs["mlx_prefetch_size"] == 0


def test_compilation_respects_explicit_prefetch():
    """Test that explicit prefetch setting is respected even with compilation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        train_file = Path(tmpdir) / "train.csv"
        train_file.write_text("text,label\ntest1,0\ntest2,1")
        
        config_file = Path(tmpdir) / "config.yaml"
        config_file.write_text("""
training:
  use_compilation: true
  epochs: 1
  batch_size: 2
data:
  mlx_prefetch_size: 4  # Explicit prefetch
""")
        
        with (
            patch("data.factory.create_dataloader") as mock_create_dataloader,
            patch("data.factory.create_dataset") as mock_create_dataset,
            patch("models.factory.create_model") as mock_create_model,
            patch("training.core.base.BaseTrainer") as mock_trainer_class,
            patch("transformers.AutoTokenizer") as mock_tokenizer_class,
        ):
            # Set up mocks
            mock_dataset = MagicMock()
            mock_create_dataset.return_value = mock_dataset
            
            mock_loader = MagicMock()
            mock_create_dataloader.return_value = mock_loader
            
            mock_model = MagicMock()
            mock_create_model.return_value = mock_model
            
            mock_tokenizer = MagicMock()
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            
            mock_trainer = MagicMock()
            mock_trainer.train.return_value = MagicMock()
            mock_trainer_class.return_value = mock_trainer
            
            # Import and run CLI
            from typer.testing import CliRunner
            from cli import app
            
            runner = CliRunner()
            result = runner.invoke(
                app,
                [
                    "train",
                    "--train", str(train_file),
                    "--config", str(config_file),
                ],
            )
            
            # Check that create_dataloader was called
            assert mock_create_dataloader.called
            
            # Check the arguments - explicit prefetch should be respected
            call_kwargs = mock_create_dataloader.call_args[1]
            if "mlx_prefetch_size" in call_kwargs:
                assert call_kwargs["mlx_prefetch_size"] == 4


def test_no_compilation_keeps_prefetch():
    """Test that prefetch is kept when compilation is disabled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        train_file = Path(tmpdir) / "train.csv"
        train_file.write_text("text,label\ntest1,0\ntest2,1")
        
        config_file = Path(tmpdir) / "config.yaml"
        config_file.write_text("""
training:
  use_compilation: false
  epochs: 1
  batch_size: 2
data:
  # no explicit prefetch settings
""")
        
        with (
            patch("data.factory.create_dataloader") as mock_create_dataloader,
            patch("data.factory.create_dataset") as mock_create_dataset,
            patch("models.factory.create_model") as mock_create_model,
            patch("training.core.base.BaseTrainer") as mock_trainer_class,
            patch("transformers.AutoTokenizer") as mock_tokenizer_class,
        ):
            # Set up mocks
            mock_dataset = MagicMock()
            mock_create_dataset.return_value = mock_dataset
            
            mock_loader = MagicMock()
            mock_create_dataloader.return_value = mock_loader
            
            mock_model = MagicMock()
            mock_create_model.return_value = mock_model
            
            mock_tokenizer = MagicMock()
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            
            mock_trainer = MagicMock()
            mock_trainer.train.return_value = MagicMock()
            mock_trainer_class.return_value = mock_trainer
            
            # Import and run CLI
            from typer.testing import CliRunner
            from cli import app
            
            runner = CliRunner()
            result = runner.invoke(
                app,
                [
                    "train",
                    "--train", str(train_file),
                    "--config", str(config_file),
                    "--prefetch", "4",  # CLI argument
                ],
            )
            
            # Check that create_dataloader was called
            assert mock_create_dataloader.called
            
            # Check the arguments - prefetch should be kept
            call_kwargs = mock_create_dataloader.call_args[1]
            assert "prefetch_size" in call_kwargs
            assert call_kwargs["prefetch_size"] == 4