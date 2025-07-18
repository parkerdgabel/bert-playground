"""Unit tests for core CLI commands (train, predict, info)."""

import pytest
from unittest.mock import patch, MagicMock, call, mock_open
from pathlib import Path
import json
import sys
from typer.testing import CliRunner

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cli.main import app


class TestTrainCommand:
    """Test suite for the train command."""
    
    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()
    
    @pytest.fixture
    def mock_config(self):
        """Create mock training configuration."""
        return {
            "model": {
                "type": "binary",
                "hidden_size": 768,
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
                "max_length": 256
            },
            "training": {
                "batch_size": 32,
                "learning_rate": 2e-5,
                "num_epochs": 5,
                "gradient_accumulation_steps": 1,
                "warmup_steps": 100
            },
            "data": {
                "train_path": "data/titanic/train.csv",
                "val_path": "data/titanic/val.csv",
                "text_column": "text",
                "label_column": "survived"
            },
            "optimization": {
                "weight_decay": 0.01,
                "adam_beta1": 0.9,
                "adam_beta2": 0.999,
                "adam_epsilon": 1e-8
            }
        }
    
    @patch('cli.commands.core.train.ModernBertTrainer')
    @patch('cli.commands.core.train.Path')
    def test_train_basic(self, mock_path, mock_trainer_class, runner, tmp_path):
        """Test basic training functionality."""
        # Create dummy data files
        train_file = tmp_path / "train.csv"
        val_file = tmp_path / "val.csv"
        train_file.write_text("text,survived\n\"passenger info\",1\n")
        val_file.write_text("text,survived\n\"passenger info\",0\n")
        
        # Setup mocks
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {
            'best_accuracy': 0.92,
            'final_loss': 0.25,
            'total_steps': 1000
        }
        mock_trainer_class.return_value = mock_trainer
        
        # Run command
        result = runner.invoke(app, [
            "train",
            "--train", str(train_file),
            "--val", str(val_file)
        ])
        
        # Check success
        assert result.exit_code == 0
        assert "Training" in result.stdout
        assert "Complete" in result.stdout or "Finished" in result.stdout
        
        # Verify trainer was initialized and called
        mock_trainer_class.assert_called_once()
        mock_trainer.train.assert_called_once()
    
    @patch('cli.commands.core.train.ModernBertTrainer')
    @patch('cli.commands.core.train.load_config')
    def test_train_with_config(self, mock_load_config, mock_trainer_class, runner, mock_config):
        """Test training with configuration file."""
        # Setup config mock
        mock_load_config.return_value = mock_config
        
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {'best_accuracy': 0.95}
        mock_trainer_class.return_value = mock_trainer
        
        # Run command
        result = runner.invoke(app, [
            "train",
            "--config", "configs/production.json"
        ])
        
        # Check success
        assert result.exit_code == 0
        
        # Verify config was loaded
        mock_load_config.assert_called_once_with("configs/production.json")
        
        # Verify trainer received config values
        trainer_config = mock_trainer_class.call_args[0][0]
        assert trainer_config.batch_size == 32
        assert trainer_config.learning_rate == 2e-5
    
    @patch('cli.commands.core.train.ModernBertTrainer')
    def test_train_with_mlx_embeddings(self, mock_trainer_class, runner, tmp_path):
        """Test training with MLX embeddings."""
        train_file = tmp_path / "train.csv"
        val_file = tmp_path / "val.csv"
        train_file.write_text("text,label\n\"text\",1\n")
        val_file.write_text("text,label\n\"text\",0\n")
        
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {'best_accuracy': 0.90}
        mock_trainer_class.return_value = mock_trainer
        
        result = runner.invoke(app, [
            "train",
            "--train", str(train_file),
            "--val", str(val_file),
            "--use-mlx-embeddings",
            "--model", "mlx-community/bert-base-4bit"
        ])
        
        assert result.exit_code == 0
        
        # Verify MLX embeddings were enabled
        trainer_config = mock_trainer_class.call_args[0][0]
        assert trainer_config.use_mlx_embeddings is True
        assert trainer_config.model_name_or_path == "mlx-community/bert-base-4bit"
    
    @patch('cli.commands.core.train.ModernBertTrainer')
    def test_train_hyperparameters(self, mock_trainer_class, runner, tmp_path):
        """Test training with custom hyperparameters."""
        train_file = tmp_path / "train.csv"
        val_file = tmp_path / "val.csv"
        train_file.touch()
        val_file.touch()
        
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {'best_accuracy': 0.88}
        mock_trainer_class.return_value = mock_trainer
        
        result = runner.invoke(app, [
            "train",
            "--train", str(train_file),
            "--val", str(val_file),
            "--batch-size", "64",
            "--lr", "3e-5",
            "--epochs", "10",
            "--warmup-steps", "500",
            "--grad-accum", "4"
        ])
        
        assert result.exit_code == 0
        
        # Verify hyperparameters
        trainer_config = mock_trainer_class.call_args[0][0]
        assert trainer_config.batch_size == 64
        assert trainer_config.learning_rate == 3e-5
        assert trainer_config.num_epochs == 10
        assert trainer_config.warmup_steps == 500
        assert trainer_config.gradient_accumulation_steps == 4
    
    @patch('cli.commands.core.train.ModernBertTrainer')
    def test_train_with_experiment_tracking(self, mock_trainer_class, runner, tmp_path):
        """Test training with experiment tracking."""
        train_file = tmp_path / "train.csv"
        val_file = tmp_path / "val.csv"
        train_file.touch()
        val_file.touch()
        
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {'best_accuracy': 0.91}
        mock_trainer_class.return_value = mock_trainer
        
        result = runner.invoke(app, [
            "train",
            "--train", str(train_file),
            "--val", str(val_file),
            "--experiment", "titanic_experiment",
            "--run-name", "improved_model"
        ])
        
        assert result.exit_code == 0
        
        # Verify experiment tracking
        trainer_config = mock_trainer_class.call_args[0][0]
        assert trainer_config.experiment_name == "titanic_experiment"
        assert trainer_config.run_name == "improved_model"
    
    def test_train_missing_data_files(self, runner):
        """Test training with missing data files."""
        result = runner.invoke(app, [
            "train",
            "--train", "nonexistent/train.csv",
            "--val", "nonexistent/val.csv"
        ])
        
        assert result.exit_code != 0
        assert "not found" in result.stdout.lower() or "does not exist" in result.stdout.lower()
    
    def test_train_invalid_batch_size(self, runner):
        """Test training with invalid batch size."""
        result = runner.invoke(app, [
            "train",
            "--train", "data/train.csv",
            "--val", "data/val.csv",
            "--batch-size", "0"
        ])
        
        assert result.exit_code != 0
        assert "Invalid value" in result.stdout or "must be positive" in result.stdout.lower()
    
    @patch('cli.commands.core.train.ModernBertTrainer')
    def test_train_checkpoint_resume(self, mock_trainer_class, runner, tmp_path):
        """Test resuming training from checkpoint."""
        train_file = tmp_path / "train.csv"
        val_file = tmp_path / "val.csv"
        train_file.touch()
        val_file.touch()
        
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {'best_accuracy': 0.94}
        mock_trainer_class.return_value = mock_trainer
        
        result = runner.invoke(app, [
            "train",
            "--train", str(train_file),
            "--val", str(val_file),
            "--resume", "output/run_001/checkpoint_500"
        ])
        
        assert result.exit_code == 0
        
        # Verify resume checkpoint was passed
        trainer_config = mock_trainer_class.call_args[0][0]
        assert trainer_config.resume_from_checkpoint == "output/run_001/checkpoint_500"


class TestPredictCommand:
    """Test suite for the predict command."""
    
    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()
    
    @pytest.fixture
    def mock_predictions(self):
        """Create mock predictions."""
        return {
            'predictions': [0, 1, 1, 0, 1],
            'probabilities': [
                [0.9, 0.1],
                [0.2, 0.8],
                [0.3, 0.7],
                [0.85, 0.15],
                [0.1, 0.9]
            ],
            'ids': ['1', '2', '3', '4', '5']
        }
    
    @patch('cli.commands.core.predict.ModelPredictor')
    @patch('cli.commands.core.predict.Path')
    def test_predict_basic(self, mock_path, mock_predictor_class, runner, tmp_path, mock_predictions):
        """Test basic prediction functionality."""
        # Create test file
        test_file = tmp_path / "test.csv"
        test_file.write_text("id,text\n1,\"passenger info\"\n")
        
        # Setup mocks
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = mock_predictions
        mock_predictor_class.return_value = mock_predictor
        
        checkpoint_path = MagicMock()
        checkpoint_path.exists.return_value = True
        mock_path.return_value = checkpoint_path
        
        # Run command
        result = runner.invoke(app, [
            "predict",
            "--test", str(test_file),
            "--checkpoint", "output/run_001/best_model"
        ])
        
        # Check success
        assert result.exit_code == 0
        assert "Predictions complete" in result.stdout or "Generated predictions" in result.stdout
        
        # Verify predictor was called
        mock_predictor.predict.assert_called_once()
    
    @patch('cli.commands.core.predict.ModelPredictor')
    @patch('cli.commands.core.predict.Path')
    @patch('builtins.open', new_callable=mock_open)
    def test_predict_with_output(self, mock_file, mock_path, mock_predictor_class, runner, tmp_path, mock_predictions):
        """Test prediction with output file."""
        test_file = tmp_path / "test.csv"
        test_file.write_text("id,text\n1,\"text\"\n")
        output_file = tmp_path / "submission.csv"
        
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = mock_predictions
        mock_predictor_class.return_value = mock_predictor
        
        mock_path.return_value.exists.return_value = True
        
        result = runner.invoke(app, [
            "predict",
            "--test", str(test_file),
            "--checkpoint", "output/run_001/best_model",
            "--output", str(output_file)
        ])
        
        assert result.exit_code == 0
        assert str(output_file) in result.stdout or "Saved predictions" in result.stdout
        
        # Verify file was written
        mock_file.assert_called()
    
    @patch('cli.commands.core.predict.ModelPredictor')
    def test_predict_with_probabilities(self, mock_predictor_class, runner, tmp_path, mock_predictions):
        """Test prediction with probability output."""
        test_file = tmp_path / "test.csv"
        test_file.write_text("id,text\n1,\"text\"\n")
        
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = mock_predictions
        mock_predictor_class.return_value = mock_predictor
        
        result = runner.invoke(app, [
            "predict",
            "--test", str(test_file),
            "--checkpoint", "output/run_001/best_model",
            "--output-probs"
        ])
        
        assert result.exit_code == 0
        assert "probabilities" in result.stdout.lower() or "0.9" in result.stdout
    
    def test_predict_missing_checkpoint(self, runner, tmp_path):
        """Test prediction with missing checkpoint."""
        test_file = tmp_path / "test.csv"
        test_file.touch()
        
        result = runner.invoke(app, [
            "predict",
            "--test", str(test_file),
            "--checkpoint", "nonexistent/checkpoint"
        ])
        
        assert result.exit_code != 0
        assert "not found" in result.stdout.lower() or "does not exist" in result.stdout.lower()
    
    @patch('cli.commands.core.predict.ModelPredictor')
    def test_predict_batch_size(self, mock_predictor_class, runner, tmp_path):
        """Test prediction with custom batch size."""
        test_file = tmp_path / "test.csv"
        test_file.touch()
        
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = {'predictions': [0, 1]}
        mock_predictor_class.return_value = mock_predictor
        
        result = runner.invoke(app, [
            "predict",
            "--test", str(test_file),
            "--checkpoint", "output/run_001/best_model",
            "--batch-size", "128"
        ])
        
        assert result.exit_code == 0
        
        # Verify batch size was passed
        predictor_config = mock_predictor_class.call_args[0][1]
        assert predictor_config['batch_size'] == 128


class TestInfoCommand:
    """Test suite for the info command."""
    
    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()
    
    @patch('cli.commands.core.info.platform')
    @patch('cli.commands.core.info.mx')
    @patch('cli.commands.core.info.sys')
    def test_info_basic(self, mock_sys, mock_mx, mock_platform, runner):
        """Test basic system info display."""
        # Setup mocks
        mock_platform.system.return_value = "Darwin"
        mock_platform.machine.return_value = "arm64"
        mock_platform.processor.return_value = "Apple M1"
        mock_platform.python_version.return_value = "3.11.0"
        
        mock_mx.default_device.return_value = MagicMock(
            __repr__=lambda self: "gpu:0"
        )
        mock_mx.__version__ = "0.5.0"
        
        mock_sys.version = "3.11.0 (main, Oct 24 2023, 10:00:00)"
        
        # Run command
        result = runner.invoke(app, ["info"])
        
        # Check success
        assert result.exit_code == 0
        assert "System Information" in result.stdout
        assert "Darwin" in result.stdout
        assert "arm64" in result.stdout
        assert "Python" in result.stdout
        assert "MLX" in result.stdout
        assert "gpu:0" in result.stdout
    
    @patch('cli.commands.core.info.Path')
    def test_info_with_paths(self, mock_path, runner):
        """Test info command showing paths."""
        # Mock paths
        mock_path.cwd.return_value = Path("/Users/user/project")
        mock_path.home.return_value = Path("/Users/user")
        
        result = runner.invoke(app, ["info"])
        
        assert result.exit_code == 0
        assert "Paths" in result.stdout or "/Users/user" in result.stdout
    
    @patch('cli.commands.core.info.importlib')
    def test_info_package_versions(self, mock_importlib, runner):
        """Test showing package versions."""
        # Mock package versions
        packages = {
            'mlx': MagicMock(__version__='0.5.0'),
            'transformers': MagicMock(__version__='4.35.0'),
            'pandas': MagicMock(__version__='2.1.0'),
            'numpy': MagicMock(__version__='1.25.0')
        }
        
        def import_module(name):
            if name in packages:
                return packages[name]
            raise ImportError(f"No module named {name}")
        
        mock_importlib.import_module.side_effect = import_module
        
        result = runner.invoke(app, ["info", "--packages"])
        
        assert result.exit_code == 0
        assert "4.35.0" in result.stdout  # transformers version
        assert "2.1.0" in result.stdout   # pandas version
    
    @patch('cli.commands.core.info.subprocess.run')
    def test_info_git_status(self, mock_subprocess, runner):
        """Test showing git repository status."""
        # Mock git output
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout="branch: main\ncommit: abc123\nstatus: clean"
        )
        
        result = runner.invoke(app, ["info", "--git"])
        
        assert result.exit_code == 0
        assert "Git" in result.stdout or "main" in result.stdout or "abc123" in result.stdout
    
    def test_info_json_output(self, runner):
        """Test JSON output format."""
        result = runner.invoke(app, ["info", "--json"])
        
        assert result.exit_code == 0
        # Output should be valid JSON
        try:
            output = json.loads(result.stdout)
            assert 'system' in output
            assert 'python' in output
        except json.JSONDecodeError:
            # If not pure JSON, check for JSON-like structure
            assert "{" in result.stdout and "}" in result.stdout