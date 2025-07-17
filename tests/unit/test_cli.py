"""Unit tests for CLI commands."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest
from typer.testing import CliRunner

from mlx_bert_cli import app


class TestCLI:
    """Test CLI commands."""
    
    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()
    
    def test_cli_help(self, runner):
        """Test CLI help command."""
        result = runner.invoke(app, ["--help"])
        
        assert result.exit_code == 0
        assert "MLX-based ModernBERT" in result.stdout
        assert "train" in result.stdout
        assert "predict" in result.stdout
        assert "benchmark" in result.stdout
        assert "info" in result.stdout
    
    def test_train_help(self, runner):
        """Test train command help."""
        result = runner.invoke(app, ["train", "--help"])
        
        assert result.exit_code == 0
        assert "--train" in result.stdout
        assert "--epochs" in result.stdout
        assert "--batch-size" in result.stdout
        assert "--lr" in result.stdout
    
    @patch('mlx_bert_cli.UnifiedTitanicDataPipeline')
    @patch('mlx_bert_cli.create_model')
    @patch('mlx_bert_cli.MLXTrainer')
    def test_train_minimal(self, mock_trainer_class, mock_create_model, mock_pipeline, 
                          runner, sample_titanic_data, temp_dir):
        """Test minimal train command."""
        # Setup mocks
        mock_model = MagicMock()
        mock_create_model.return_value = mock_model
        
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {
            "best_metric": 0.95,
            "best_step": 100,
            "total_time": 60.0,
            "final_metrics": {"val_accuracy": 0.95},
        }
        mock_trainer_class.return_value = mock_trainer
        
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.texts = ["text1", "text2"]
        mock_pipeline_instance.__len__.return_value = 10
        mock_pipeline.return_value = mock_pipeline_instance
        
        # Run command
        result = runner.invoke(app, [
            "train",
            "--train", str(sample_titanic_data),
            "--output", str(temp_dir),
            "--epochs", "1",
            "--batch-size", "4",
            "--no-mlflow",
        ])
        
        if result.exit_code != 0:
            print(f"Exit code: {result.exit_code}")
            print(f"Exception: {result.exception}")
            print(f"Output: {result.output}")
        assert result.exit_code == 0
        assert "Training completed" in result.stdout
        
        # Verify trainer was created and called
        mock_trainer_class.assert_called_once()
        mock_trainer.train.assert_called_once()
    
    @patch('mlx_bert_cli.UnifiedTitanicDataPipeline')
    @patch('mlx_bert_cli.create_model')
    @patch('mlx_bert_cli.MLXTrainer')
    def test_train_with_validation(self, mock_trainer_class, mock_create_model, 
                                  mock_pipeline, runner, sample_titanic_data, temp_dir):
        """Test train command with validation set."""
        # Setup mocks
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {
            "best_metric": 0.95,
            "best_step": 100,
            "total_time": 60.0,
            "final_metrics": {},
        }
        mock_trainer_class.return_value = mock_trainer
        
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.texts = ["text1", "text2"]
        mock_pipeline_instance.__len__.return_value = 10
        mock_pipeline.return_value = mock_pipeline_instance
        
        # Run command with validation
        result = runner.invoke(app, [
            "train",
            "--train", str(sample_titanic_data),
            "--val", str(sample_titanic_data),
            "--output", str(temp_dir),
            "--epochs", "1",
            "--no-mlflow",
        ])
        
        assert result.exit_code == 0
        
        # Verify validation loader was created
        assert mock_pipeline.call_count >= 2  # Train and val
    
    def test_train_with_config_file(self, runner, sample_titanic_data, temp_dir):
        """Test train command with config file."""
        # Create config file
        config_path = temp_dir / "config.json"
        config = {
            "learning_rate": 1e-4,
            "num_epochs": 3,
            "batch_size": 16,
        }
        with open(config_path, "w") as f:
            json.dump(config, f)
        
        with patch('mlx_bert_cli.MLXTrainer') as mock_trainer_class:
            mock_trainer = MagicMock()
            mock_trainer.train.return_value = {
                "best_metric": 0.95,
                "best_step": 100,
                "total_time": 60.0,
                "final_metrics": {},
            }
            mock_trainer_class.return_value = mock_trainer
            
            result = runner.invoke(app, [
                "train",
                "--train", str(sample_titanic_data),
                "--config", str(config_path),
                "--output", str(temp_dir),
                "--no-mlflow",
            ])
            
            # Config values should be loaded
            trainer_config = mock_trainer_class.call_args[1]["config"]
            assert trainer_config.learning_rate == 1e-4
            assert trainer_config.num_epochs == 3
    
    @patch('mlx_bert_cli.UnifiedTitanicDataPipeline')
    @patch('mlx_bert_cli.TitanicClassifier')
    @patch('mlx_bert_cli.create_model')
    @patch('pandas.read_csv')
    def test_predict_command(self, mock_read_csv, mock_create_model, mock_classifier,
                            mock_pipeline, runner, sample_test_data, temp_dir):
        """Test predict command."""
        # Setup mocks
        mock_df = MagicMock()
        mock_df.__len__.return_value = 5
        mock_df.__getitem__.return_value = [892, 893, 894, 895, 896]
        mock_read_csv.return_value = mock_df
        
        mock_model_instance = MagicMock()
        mock_classifier.return_value = mock_model_instance
        
        # Mock outputs
        import mlx.core as mx
        mock_outputs = {
            "logits": mx.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.2, 0.8]]),
        }
        mock_model_instance.return_value = mock_outputs
        
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.__len__.return_value = 5
        mock_batch = {
            "input_ids": mx.ones((5, 128)),
            "attention_mask": mx.ones((5, 128)),
        }
        mock_pipeline_instance.get_dataloader.return_value = [mock_batch]
        mock_pipeline.return_value = mock_pipeline_instance
        
        # Create checkpoint directory
        checkpoint_dir = temp_dir / "checkpoint"
        checkpoint_dir.mkdir()
        
        # Run predict
        output_path = temp_dir / "predictions.csv"
        result = runner.invoke(app, [
            "predict",
            "--test", str(sample_test_data),
            "--checkpoint", str(checkpoint_dir),
            "--output", str(output_path),
        ])
        
        assert result.exit_code == 0
        assert "Predictions saved" in result.stdout
    
    @patch('mlx_bert_cli.create_model')
    @patch('mlx_bert_cli.TitanicClassifier')
    @patch('mlx.core.eval')
    @patch('mlx.nn.value_and_grad')
    def test_benchmark_command(self, mock_grad, mock_eval, mock_classifier,
                              mock_create_model, runner):
        """Test benchmark command."""
        # Setup mocks
        import mlx.core as mx
        
        mock_loss = mx.array(0.5)
        mock_grads = {"weight": mx.ones((10, 10))}
        mock_grad.return_value = lambda model: (mock_loss, mock_grads)
        
        mock_model_instance = MagicMock()
        mock_outputs = {"loss": mock_loss}
        mock_model_instance.return_value = mock_outputs
        mock_classifier.return_value = mock_model_instance
        
        # Run benchmark
        result = runner.invoke(app, [
            "benchmark",
            "--batch-size", "8",
            "--seq-length", "128",
            "--steps", "5",
            "--warmup", "2",
        ])
        
        assert result.exit_code == 0
        assert "Benchmark Results" in result.stdout
        assert "Average time per step" in result.stdout
        assert "Throughput" in result.stdout
    
    @patch('mlx_bert_cli.mx.default_device')
    @patch('mlx_bert_cli.mlflow_central.MLflowCentral')
    def test_info_command(self, mock_mlflow_class, mock_device, runner):
        """Test info command."""
        # Setup mocks
        mock_device.return_value = "gpu"
        
        mock_mlflow = MagicMock()
        mock_mlflow.tracking_uri = "sqlite:///test.db"
        mock_mlflow.artifact_root = "./artifacts"
        mock_mlflow_class.return_value = mock_mlflow
        
        # Run info
        result = runner.invoke(app, ["info"])
        
        assert result.exit_code == 0
        assert "System Information" in result.stdout
        assert "MLflow Configuration" in result.stdout
        assert "MLX Device" in result.stdout
    
    @patch('mlx_bert_cli.mlflow_central.MLflowCentral')
    def test_list_experiments_command(self, mock_mlflow_class, runner):
        """Test list-experiments command."""
        # Setup mocks
        mock_mlflow = MagicMock()
        mock_exp1 = MagicMock()
        mock_exp1.experiment_id = "1"
        mock_exp1.name = "experiment1"
        mock_exp1.artifact_location = "/path/to/artifacts"
        mock_exp1.lifecycle_stage = "active"
        
        mock_mlflow.list_experiments.return_value = [mock_exp1]
        mock_mlflow_class.return_value = mock_mlflow
        
        # Run list-experiments
        result = runner.invoke(app, ["list-experiments"])
        
        assert result.exit_code == 0
        assert "MLflow Experiments" in result.stdout
        assert "experiment1" in result.stdout
    
    @patch('shutil.copytree')
    def test_export_command(self, mock_copytree, runner, temp_dir):
        """Test export command."""
        checkpoint_dir = temp_dir / "checkpoint"
        checkpoint_dir.mkdir()
        output_dir = temp_dir / "export"
        
        result = runner.invoke(app, [
            "export",
            "--checkpoint", str(checkpoint_dir),
            "--output", str(output_dir),
            "--format", "mlx",
        ])
        
        assert result.exit_code == 0
        assert "Exported to MLX format" in result.stdout
        mock_copytree.assert_called_once()
    
    def test_export_unsupported_format(self, runner, temp_dir):
        """Test export with unsupported format."""
        checkpoint_dir = temp_dir / "checkpoint"
        checkpoint_dir.mkdir()
        output_dir = temp_dir / "export"
        
        result = runner.invoke(app, [
            "export",
            "--checkpoint", str(checkpoint_dir),
            "--output", str(output_dir),
            "--format", "onnx",
        ])
        
        assert result.exit_code == 0
        assert "not yet supported" in result.stdout


class TestCLIAdvancedOptions:
    """Test advanced CLI options."""
    
    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()
    
    @patch('mlx_bert_cli.create_cnn_hybrid_model')
    @patch('mlx_bert_cli.MLXTrainer')
    def test_train_cnn_hybrid_model(self, mock_trainer_class, mock_create_cnn,
                                   runner, sample_titanic_data, temp_dir):
        """Test training CNN hybrid model."""
        # Setup mocks
        mock_model = MagicMock()
        mock_model.output_hidden_size = 1024
        mock_model.config.hidden_size = 768
        mock_create_cnn.return_value = mock_model
        
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {
            "best_metric": 0.95,
            "best_step": 100,
            "total_time": 60.0,
            "final_metrics": {},
        }
        mock_trainer_class.return_value = mock_trainer
        
        # Run with CNN hybrid
        result = runner.invoke(app, [
            "train",
            "--train", str(sample_titanic_data),
            "--output", str(temp_dir),
            "--model-type", "cnn_hybrid",
            "--cnn-kernels", "2,3,4,5",
            "--cnn-filters", "256",
            "--dilated",
            "--epochs", "1",
            "--no-mlflow",
        ])
        
        assert result.exit_code == 0
        
        # Verify CNN model was created with correct params
        mock_create_cnn.assert_called_once()
        call_args = mock_create_cnn.call_args[1]
        assert call_args["cnn_kernel_sizes"] == [2, 3, 4, 5]
        assert call_args["cnn_num_filters"] == 256
        assert call_args["use_dilated_conv"] == True
    
    @patch('mlx_bert_cli.MLXTrainer')
    def test_train_with_resume(self, mock_trainer_class, runner, 
                              sample_titanic_data, temp_dir):
        """Test resuming training from checkpoint."""
        # Setup mocks
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {
            "best_metric": 0.95,
            "best_step": 100,
            "total_time": 60.0,
            "final_metrics": {},
        }
        mock_trainer_class.return_value = mock_trainer
        
        # Run with resume
        result = runner.invoke(app, [
            "train",
            "--train", str(sample_titanic_data),
            "--output", str(temp_dir),
            "--resume", "checkpoint-100",
            "--epochs", "2",
            "--no-mlflow",
        ])
        
        assert result.exit_code == 0
        
        # Verify train was called with resume parameter
        train_call = mock_trainer.train.call_args[1]
        assert train_call["resume_from_checkpoint"] == "checkpoint-100"
    
    @patch('mlx_bert_cli.MLXTrainer')
    def test_train_all_advanced_options(self, mock_trainer_class, runner,
                                       sample_titanic_data, temp_dir):
        """Test train command with all advanced options."""
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {
            "best_metric": 0.95,
            "best_step": 100,
            "total_time": 60.0,
            "final_metrics": {},
        }
        mock_trainer_class.return_value = mock_trainer
        
        # Run with all options
        result = runner.invoke(app, [
            "train",
            "--train", str(sample_titanic_data),
            "--val", str(sample_titanic_data),
            "--output", str(temp_dir),
            "--model", "bert-base",
            "--model-type", "base",
            "--batch-size", "16",
            "--max-batch-size", "32",
            "--lr", "1e-4",
            "--epochs", "5",
            "--max-length", "512",
            "--warmup-ratio", "0.2",
            "--grad-accum", "2",
            "--workers", "8",
            "--prefetch", "8",
            "--experiment", "test_exp",
            "--run-name", "test_run",
            "--no-augment",
            "--no-dynamic-batch",
            "--early-stopping", "5",
            "--grad-clip", "0.5",
            "--label-smoothing", "0.1",
            "--eval-steps", "50",
            "--save-steps", "100",
            "--no-mlflow",
        ])
        
        assert result.exit_code == 0
        
        # Verify config was created with all options
        config = mock_trainer_class.call_args[1]["config"]
        assert config.base_batch_size == 16
        assert config.max_batch_size == 32
        assert config.learning_rate == 1e-4
        assert config.num_epochs == 5
        assert config.warmup_ratio == 0.2
        assert config.gradient_accumulation_steps == 2
        assert config.num_workers == 8
        assert config.prefetch_size == 8
        assert config.experiment_name == "test_exp"
        assert config.run_name == "test_run"
        assert config.enable_dynamic_batching == False
        assert config.early_stopping_patience == 5
        assert config.gradient_clip_val == 0.5
        assert config.label_smoothing == 0.1
        assert config.eval_steps == 50
        assert config.save_steps == 100
        assert config.enable_mlflow == False