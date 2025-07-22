"""End-to-end tests for complete project workflows."""

import os
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import yaml
from typer.testing import CliRunner

from cli.app import app


class TestProjectWorkflow:
    """Test complete project workflows from init to submission."""

    @pytest.fixture
    def runner(self):
        """Create CLI runner."""
        return CliRunner()

    @pytest.fixture
    def mock_training(self):
        """Mock training to avoid actual model training."""
        with patch('models.factory.create_model') as mock_model:
            with patch('training.trainer.Trainer') as mock_trainer:
                # Mock model
                model_instance = MagicMock()
                model_instance.parameters.return_value = []
                mock_model.return_value = model_instance
                
                # Mock trainer
                trainer_instance = MagicMock()
                trainer_instance.train.return_value = {
                    "best_metric": 0.95,
                    "final_loss": 0.1
                }
                mock_trainer.return_value = trainer_instance
                
                yield mock_model, mock_trainer

    @pytest.fixture
    def mock_kaggle(self):
        """Mock Kaggle API calls."""
        with patch('cli.commands.competition.download.api') as mock_api:
            mock_api.competition_download_files.return_value = None
            mock_api.competition_list.return_value = [
                {"id": "titanic", "title": "Titanic Competition"}
            ]
            mock_api.competition_submit.return_value = {"message": "Submission successful"}
            yield mock_api

    def test_complete_project_workflow(self, runner, tmp_path, mock_training, mock_kaggle):
        """Test complete workflow: init → config → train → predict → submit."""
        project_name = "titanic-solution"
        project_path = tmp_path / project_name
        
        # Switch to temp directory
        original_cwd = Path.cwd()
        os.chdir(tmp_path)
        
        try:
            # Step 1: Initialize user config
            with patch('cli.config.ConfigManager._get_config_dir', return_value=tmp_path / ".k-bert"):
                with patch('rich.prompt.Prompt.ask') as mock_prompt:
                    mock_prompt.side_effect = ["test_user", "test_key", ""]
                    
                    result = runner.invoke(app, ["config", "init", "--no-interactive"])
                    assert result.exit_code == 0
            
            # Step 2: Initialize project with preset
            with patch('cli.commands.project.init._copy_template') as mock_copy:
                with patch('cli.commands.project.init._update_project_config'):
                    result = runner.invoke(app, ["project", "init", project_name, "--template", "kaggle"])
                    assert result.exit_code == 0
            
            # Create project structure manually (since we mocked _copy_template)
            project_path.mkdir()
            os.chdir(project_path)
            
            # Step 3: Create project config with Titanic preset
            project_config = {
                "name": project_name,
                "description": "Titanic survival prediction",
                "competition": "titanic",
                "models": {
                    "default_model": "answerdotai/ModernBERT-base",
                    "head": {"type": "binary_classification"}
                },
                "data": {
                    "train_path": "data/train.csv",
                    "val_path": "data/val.csv",
                    "test_path": "data/test.csv",
                    "batch_size": 32,
                    "max_length": 256
                },
                "training": {
                    "default_epochs": 5,
                    "default_learning_rate": 2e-5,
                    "output_dir": "./outputs"
                },
                "experiments": [
                    {
                        "name": "quick_test",
                        "description": "Quick validation",
                        "config": {"training": {"default_epochs": 1}}
                    },
                    {
                        "name": "full_training",
                        "description": "Complete training",
                        "config": {"training": {"default_epochs": 10}}
                    }
                ]
            }
            
            with open("k-bert.yaml", "w") as f:
                yaml.dump(project_config, f)
            
            # Step 4: Download competition data
            (project_path / "data").mkdir()
            result = runner.invoke(app, ["competition", "download", "titanic"])
            assert result.exit_code == 0
            
            # Create dummy data files
            for filename in ["train.csv", "val.csv", "test.csv"]:
                (project_path / "data" / filename).write_text("PassengerId,Survived,Name\n1,0,Test")
            
            # Step 5: Quick test run
            with patch('cli.commands.core.train._load_data'):
                result = runner.invoke(app, ["run", "--experiment", "quick_test"])
                assert result.exit_code == 0
            
            # Step 6: Full training
            with patch('cli.commands.core.train._load_data'):
                result = runner.invoke(app, ["run", "--experiment", "full_training"])
                assert result.exit_code == 0
            
            # Create mock output directory and model
            output_dir = project_path / "outputs" / "run_001"
            output_dir.mkdir(parents=True)
            (output_dir / "final_model").mkdir()
            (output_dir / "final_model" / "config.json").write_text('{"model_type": "modernbert"}')
            
            # Step 7: Generate predictions
            with patch('cli.commands.core.predict._load_model') as mock_load:
                with patch('cli.commands.core.predict._generate_predictions'):
                    mock_load.return_value = (MagicMock(), MagicMock())
                    
                    result = runner.invoke(app, ["predict", str(output_dir / "final_model")])
                    assert result.exit_code == 0
            
            # Step 8: Submit to Kaggle
            submission_file = project_path / "submission.csv"
            submission_file.write_text("PassengerId,Survived\n1,0")
            
            result = runner.invoke(app, ["competition", "submit", "titanic", str(submission_file), "-m", "Test submission"])
            assert result.exit_code == 0
            
        finally:
            os.chdir(original_cwd)

    def test_experiment_comparison_workflow(self, runner, tmp_path, mock_training):
        """Test workflow for running and comparing multiple experiments."""
        os.chdir(tmp_path)
        
        # Create project with multiple experiments
        config = {
            "name": "experiment-test",
            "models": {"default_model": "bert-base-uncased"},
            "data": {
                "train_path": "data/train.csv",
                "val_path": "data/val.csv",
                "batch_size": 32
            },
            "training": {"default_epochs": 3},
            "experiments": [
                {"name": "baseline", "config": {}},
                {"name": "high_lr", "config": {"training": {"default_learning_rate": 5e-5}}},
                {"name": "large_batch", "config": {"data": {"batch_size": 64}}}
            ]
        }
        
        with open("k-bert.yaml", "w") as f:
            yaml.dump(config, f)
        
        # Create data files
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        for f in ["train.csv", "val.csv"]:
            (data_dir / f).write_text("text,label\ntest,0")
        
        # Run experiments
        with patch('cli.commands.core.train._load_data'):
            for exp_name in ["baseline", "high_lr", "large_batch"]:
                result = runner.invoke(app, ["run", "--experiment", exp_name])
                assert result.exit_code == 0
        
        # Compare experiments (when implemented)
        # result = runner.invoke(app, ["mlflow", "compare", "--experiments", "baseline,high_lr,large_batch"])
        # assert result.exit_code == 0

    def test_plugin_development_workflow(self, runner, tmp_path):
        """Test workflow for developing and using custom plugins."""
        os.chdir(tmp_path)
        
        # Create project
        config = {
            "name": "plugin-test",
            "models": {"default_model": "bert-base-uncased"},
            "data": {"train_path": "data/train.csv"},
            "plugins": {
                "enabled": ["custom_head"],
                "custom_head": {
                    "module": "src.heads.sentiment_head",
                    "config": {"num_classes": 3}
                }
            }
        }
        
        with open("k-bert.yaml", "w") as f:
            yaml.dump(config, f)
        
        # Create plugin structure
        plugin_dir = tmp_path / "src" / "heads"
        plugin_dir.mkdir(parents=True)
        
        # Create custom head plugin
        plugin_code = '''
from k_bert.plugins import HeadPlugin, register_component

@register_component
class SentimentHead(HeadPlugin):
    """Custom sentiment analysis head."""
    
    def __init__(self, config):
        super().__init__(config)
        self.num_classes = config.get("num_classes", 3)
    
    def forward(self, hidden_states, **kwargs):
        # Implementation
        return {"logits": hidden_states}
'''
        (plugin_dir / "sentiment_head.py").write_text(plugin_code)
        (plugin_dir / "__init__.py").write_text("")
        
        # Test plugin loading
        with patch('cli.plugins.load_project_plugins') as mock_load:
            result = runner.invoke(app, ["project", "validate", "."])
            assert result.exit_code == 0
            mock_load.assert_called()

    def test_model_serving_workflow(self, runner, tmp_path):
        """Test workflow for training and serving a model."""
        os.chdir(tmp_path)
        
        # Setup project
        config = {
            "name": "serve-test",
            "models": {"default_model": "bert-base-uncased"},
            "data": {"train_path": "data/train.csv"},
            "training": {"default_epochs": 1}
        }
        
        with open("k-bert.yaml", "w") as f:
            yaml.dump(config, f)
        
        # Create data
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "train.csv").write_text("text,label\ntest,0")
        
        # Train model
        with patch('cli.commands.core.train._load_data'):
            with patch('models.factory.create_model'):
                result = runner.invoke(app, ["train"])
                assert result.exit_code == 0
        
        # Create mock model
        model_dir = tmp_path / "outputs" / "run_001" / "final_model"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text('{"model_type": "bert"}')
        
        # Serve model
        with patch('cli.commands.model.serve.ModelServer') as mock_server:
            server_instance = MagicMock()
            mock_server.return_value = server_instance
            
            # Start server in background
            result = runner.invoke(app, ["model", "serve", str(model_dir), "--port", "8080"])
            assert result.exit_code == 0

    def test_collaboration_workflow(self, runner, tmp_path):
        """Test workflow for collaborative development."""
        os.chdir(tmp_path)
        
        # User A creates project
        config_a = {
            "name": "collab-project",
            "version": "1.0",
            "models": {"default_model": "bert-base-uncased"},
            "data": {"train_path": "data/train.csv"},
            "experiments": [
                {"name": "user_a_exp", "config": {"training": {"default_epochs": 5}}}
            ]
        }
        
        with open("k-bert.yaml", "w") as f:
            yaml.dump(config_a, f)
        
        # User B clones and adds experiment
        config_b = config_a.copy()
        config_b["experiments"].append({
            "name": "user_b_exp",
            "config": {"training": {"default_learning_rate": 1e-5}}
        })
        
        with open("k-bert.yaml", "w") as f:
            yaml.dump(config_b, f)
        
        # Validate merged config
        result = runner.invoke(app, ["config", "validate"])
        assert result.exit_code == 0
        
        # List experiments
        result = runner.invoke(app, ["config", "show"])
        assert "user_a_exp" in result.stdout
        assert "user_b_exp" in result.stdout

    def test_error_recovery_workflow(self, runner, tmp_path):
        """Test workflow with errors and recovery."""
        os.chdir(tmp_path)
        
        # Create config with error
        config = {
            "name": "error-test",
            "models": {"default_model": "invalid-model-name"},
            "data": {"train_path": "missing/train.csv"}
        }
        
        with open("k-bert.yaml", "w") as f:
            yaml.dump(config, f)
        
        # Try to run - should fail
        result = runner.invoke(app, ["run"])
        assert result.exit_code == 1
        
        # Fix config
        config["models"]["default_model"] = "bert-base-uncased"
        config["data"]["train_path"] = "data/train.csv"
        
        with open("k-bert.yaml", "w") as f:
            yaml.dump(config, f)
        
        # Create missing data
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "train.csv").write_text("text,label\ntest,0")
        
        # Validate fixed config
        result = runner.invoke(app, ["config", "validate"])
        assert result.exit_code == 0