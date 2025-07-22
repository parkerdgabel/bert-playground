"""Integration tests for model and MLflow commands."""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

import pytest
import yaml
from typer.testing import CliRunner

from cli.app import app


class TestModelCommands:
    """Integration tests for model commands."""

    @pytest.fixture
    def runner(self):
        """Create CLI runner."""
        return CliRunner()

    @pytest.fixture
    def model_checkpoint(self, tmp_path):
        """Create mock model checkpoint."""
        checkpoint_dir = tmp_path / "model_checkpoint"
        checkpoint_dir.mkdir()
        
        # Create config file
        config = {
            "model_type": "modernbert",
            "vocab_size": 30522,
            "hidden_size": 768,
            "num_attention_heads": 12,
            "num_hidden_layers": 12
        }
        
        import json
        with open(checkpoint_dir / "config.json", "w") as f:
            json.dump(config, f)
        
        # Create mock weights file
        (checkpoint_dir / "model.safetensors").write_text("mock weights")
        
        return checkpoint_dir

    def test_model_serve(self, runner, model_checkpoint):
        """Test model serving command."""
        with patch('cli.commands.model.serve.ModelServer') as mock_server:
            server_instance = MagicMock()
            mock_server.return_value = server_instance
            
            result = runner.invoke(app, ["model", "serve", str(model_checkpoint)])
            
            assert result.exit_code == 0
            assert "Starting model server" in result.stdout
            mock_server.assert_called_once()
            server_instance.start.assert_called_once()

    def test_model_serve_with_options(self, runner, model_checkpoint):
        """Test model serve with custom options."""
        with patch('cli.commands.model.serve.ModelServer') as mock_server:
            result = runner.invoke(app, [
                "model", "serve", str(model_checkpoint),
                "--port", "8080",
                "--host", "0.0.0.0",
                "--workers", "4"
            ])
            
            assert result.exit_code == 0
            call_kwargs = mock_server.call_args[1]
            assert call_kwargs["port"] == 8080
            assert call_kwargs["host"] == "0.0.0.0"
            assert call_kwargs["workers"] == 4

    def test_model_export(self, runner, model_checkpoint, tmp_path):
        """Test model export command."""
        output_path = tmp_path / "exported_model.onnx"
        
        with patch('cli.commands.model.export.export_to_onnx') as mock_export:
            result = runner.invoke(app, [
                "model", "export", str(model_checkpoint),
                "--format", "onnx",
                "--output", str(output_path)
            ])
            
            assert result.exit_code == 0
            assert "Exported model" in result.stdout
            mock_export.assert_called_once()

    def test_model_export_formats(self, runner, model_checkpoint, tmp_path):
        """Test different export formats."""
        formats = ["onnx", "coreml", "tflite"]
        
        for fmt in formats:
            with patch(f'cli.commands.model.export.export_to_{fmt}') as mock_export:
                result = runner.invoke(app, [
                    "model", "export", str(model_checkpoint),
                    "--format", fmt
                ])
                
                assert result.exit_code == 0

    def test_model_inspect(self, runner, model_checkpoint):
        """Test model inspection command."""
        with patch('cli.commands.model.inspect.load_model_config') as mock_load:
            mock_load.return_value = {
                "architecture": "ModernBERT",
                "parameters": 110000000,
                "layers": 12
            }
            
            result = runner.invoke(app, ["model", "inspect", str(model_checkpoint)])
            
            assert result.exit_code == 0
            assert "Architecture: ModernBERT" in result.stdout
            assert "Parameters: 110000000" in result.stdout

    def test_model_inspect_detailed(self, runner, model_checkpoint):
        """Test detailed model inspection."""
        with patch('cli.commands.model.inspect.get_model_details') as mock_details:
            mock_details.return_value = {
                "layers": ["embeddings", "encoder.0", "encoder.1"],
                "parameter_counts": {"embeddings": 23440896, "encoder": 85054464}
            }
            
            result = runner.invoke(app, ["model", "inspect", str(model_checkpoint), "--detailed"])
            
            assert result.exit_code == 0
            assert "embeddings" in result.stdout
            assert "encoder" in result.stdout

    def test_model_evaluate(self, runner, model_checkpoint, tmp_path):
        """Test model evaluation command."""
        # Create test data
        test_data = tmp_path / "test.csv"
        test_data.write_text("text,label\nTest text,0\nAnother test,1")
        
        with patch('cli.commands.model.evaluate.evaluate_model') as mock_eval:
            mock_eval.return_value = {
                "accuracy": 0.85,
                "f1_score": 0.83,
                "loss": 0.45
            }
            
            result = runner.invoke(app, [
                "model", "evaluate", str(model_checkpoint),
                "--data", str(test_data)
            ])
            
            assert result.exit_code == 0
            assert "Accuracy: 0.85" in result.stdout
            assert "F1 Score: 0.83" in result.stdout

    def test_model_convert(self, runner, model_checkpoint, tmp_path):
        """Test model format conversion."""
        output_path = tmp_path / "converted_model"
        
        with patch('cli.commands.model.convert.convert_model') as mock_convert:
            result = runner.invoke(app, [
                "model", "convert", str(model_checkpoint),
                "--to", "mlx",
                "--output", str(output_path)
            ])
            
            assert result.exit_code == 0
            assert "Converted model" in result.stdout

    def test_model_list(self, runner, tmp_path):
        """Test listing available models."""
        # Create multiple model directories
        for i in range(3):
            model_dir = tmp_path / f"model_{i}"
            model_dir.mkdir()
            (model_dir / "config.json").write_text('{"model_type": "bert"}')
        
        with patch('pathlib.Path.cwd', return_value=tmp_path):
            result = runner.invoke(app, ["model", "list"])
            
            assert result.exit_code == 0
            assert "model_0" in result.stdout
            assert "model_1" in result.stdout
            assert "model_2" in result.stdout


class TestMLflowCommands:
    """Integration tests for MLflow commands."""

    @pytest.fixture
    def runner(self):
        """Create CLI runner."""
        return CliRunner()

    @pytest.fixture
    def mlflow_config(self, tmp_path):
        """Create project with MLflow configuration."""
        config = {
            "name": "mlflow-test",
            "mlflow": {
                "tracking_uri": "http://localhost:5000",
                "experiment_name": "test_experiment",
                "auto_log": True
            }
        }
        
        with open(tmp_path / "k-bert.yaml", "w") as f:
            yaml.dump(config, f)
        
        return tmp_path

    def test_mlflow_ui(self, runner, mlflow_config):
        """Test starting MLflow UI."""
        os.chdir(mlflow_config)
        
        with patch('subprocess.Popen') as mock_popen:
            result = runner.invoke(app, ["mlflow", "ui"])
            
            assert result.exit_code == 0
            assert "Starting MLflow UI" in result.stdout
            mock_popen.assert_called_once()

    def test_mlflow_ui_with_options(self, runner):
        """Test MLflow UI with custom options."""
        with patch('subprocess.Popen') as mock_popen:
            result = runner.invoke(app, [
                "mlflow", "ui",
                "--port", "5001",
                "--host", "0.0.0.0",
                "--backend-store-uri", "./custom_mlruns"
            ])
            
            assert result.exit_code == 0
            call_args = mock_popen.call_args[0][0]
            assert "--port" in call_args
            assert "5001" in call_args
            assert "--host" in call_args
            assert "0.0.0.0" in call_args

    def test_mlflow_health(self, runner, mlflow_config):
        """Test MLflow server health check."""
        os.chdir(mlflow_config)
        
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {"status": "healthy"}
            
            result = runner.invoke(app, ["mlflow", "health"])
            
            assert result.exit_code == 0
            assert "MLflow server is healthy" in result.stdout

    def test_mlflow_health_server_down(self, runner, mlflow_config):
        """Test health check when server is down."""
        os.chdir(mlflow_config)
        
        with patch('requests.get', side_effect=ConnectionError()):
            result = runner.invoke(app, ["mlflow", "health"])
            
            assert result.exit_code == 1
            assert "not responding" in result.stdout or "Connection error" in result.stdout

    def test_mlflow_dashboard(self, runner, mlflow_config):
        """Test opening MLflow dashboard in browser."""
        os.chdir(mlflow_config)
        
        with patch('webbrowser.open') as mock_open:
            result = runner.invoke(app, ["mlflow", "dashboard"])
            
            assert result.exit_code == 0
            assert "Opening MLflow dashboard" in result.stdout
            mock_open.assert_called_with("http://localhost:5000")

    def test_mlflow_compare_experiments(self, runner, mlflow_config):
        """Test comparing MLflow experiments."""
        os.chdir(mlflow_config)
        
        with patch('mlflow.search_runs') as mock_search:
            mock_search.return_value = MagicMock(
                to_dict=lambda: {
                    "experiment_id": ["1", "2"],
                    "metrics.accuracy": [0.85, 0.87],
                    "metrics.loss": [0.45, 0.42]
                }
            )
            
            result = runner.invoke(app, ["mlflow", "compare", "--experiments", "exp1,exp2"])
            
            assert result.exit_code == 0
            assert "Experiment Comparison" in result.stdout

    def test_mlflow_clean_runs(self, runner, mlflow_config):
        """Test cleaning old MLflow runs."""
        os.chdir(mlflow_config)
        
        with patch('mlflow.search_runs') as mock_search:
            with patch('mlflow.delete_run') as mock_delete:
                # Mock old runs
                mock_search.return_value = MagicMock(
                    iterrows=lambda: [
                        (0, {"run_id": "old_run_1"}),
                        (1, {"run_id": "old_run_2"})
                    ]
                )
                
                result = runner.invoke(app, ["mlflow", "clean", "--days", "30", "--confirm"])
                
                assert result.exit_code == 0
                assert "Deleted 2 runs" in result.stdout
                assert mock_delete.call_count == 2

    def test_mlflow_export_runs(self, runner, mlflow_config, tmp_path):
        """Test exporting MLflow runs."""
        os.chdir(mlflow_config)
        output_file = tmp_path / "runs_export.csv"
        
        with patch('mlflow.search_runs') as mock_search:
            mock_df = MagicMock()
            mock_df.to_csv = MagicMock()
            mock_search.return_value = mock_df
            
            result = runner.invoke(app, [
                "mlflow", "export",
                "--output", str(output_file),
                "--format", "csv"
            ])
            
            assert result.exit_code == 0
            assert "Exported runs" in result.stdout
            mock_df.to_csv.assert_called_once()

    def test_mlflow_import_run(self, runner, mlflow_config, tmp_path):
        """Test importing MLflow run."""
        os.chdir(mlflow_config)
        
        # Create mock run export
        run_export = tmp_path / "run_export.zip"
        run_export.write_text("mock export data")
        
        with patch('mlflow.import_run') as mock_import:
            result = runner.invoke(app, ["mlflow", "import", str(run_export)])
            
            assert result.exit_code == 0
            assert "Imported run" in result.stdout

    def test_mlflow_server_start(self, runner, tmp_path):
        """Test starting MLflow tracking server."""
        with patch('subprocess.Popen') as mock_popen:
            result = runner.invoke(app, [
                "mlflow", "server",
                "--backend-store-uri", str(tmp_path / "mlruns"),
                "--default-artifact-root", str(tmp_path / "artifacts")
            ])
            
            assert result.exit_code == 0
            assert "Starting MLflow tracking server" in result.stdout
            
            call_args = mock_popen.call_args[0][0]
            assert "mlflow" in call_args
            assert "server" in call_args

    def test_model_mlflow_integration(self, runner, mlflow_config, model_checkpoint):
        """Test model commands with MLflow integration."""
        os.chdir(mlflow_config)
        
        # Register model in MLflow
        with patch('mlflow.register_model') as mock_register:
            result = runner.invoke(app, [
                "model", "register",
                str(model_checkpoint),
                "--name", "test_model",
                "--stage", "Production"
            ])
            
            assert result.exit_code == 0
            assert "Registered model" in result.stdout