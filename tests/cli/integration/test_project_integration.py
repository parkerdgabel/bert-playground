"""Integration tests for project commands."""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

import pytest
import yaml
from typer.testing import CliRunner

from cli.app import app


class TestProjectIntegration:
    """Integration tests for project commands."""

    @pytest.fixture
    def runner(self):
        """Create CLI runner."""
        return CliRunner()

    @pytest.fixture
    def sample_project_config(self):
        """Sample project configuration."""
        return {
            "name": "test-project",
            "description": "Test project",
            "models": {
                "default_model": "answerdotai/ModernBERT-base",
                "head": {"type": "binary_classification"}
            },
            "data": {
                "train_path": "data/train.csv",
                "val_path": "data/val.csv",
                "test_path": "data/test.csv",
                "batch_size": 32
            },
            "training": {
                "default_epochs": 5,
                "default_learning_rate": 2e-5
            },
            "experiments": [
                {
                    "name": "quick_test",
                    "description": "Quick test",
                    "config": {"training": {"default_epochs": 1}}
                },
                {
                    "name": "full_training",
                    "description": "Full training",
                    "config": {"training": {"default_epochs": 10}}
                }
            ]
        }

    @pytest.fixture
    def project_dir(self, tmp_path, sample_project_config):
        """Create a test project directory."""
        # Create project config
        with open(tmp_path / "k-bert.yaml", "w") as f:
            yaml.dump(sample_project_config, f)
        
        # Create data directory
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # Create dummy data files
        (data_dir / "train.csv").write_text("text,label\ntest,0")
        (data_dir / "val.csv").write_text("text,label\ntest,0")
        (data_dir / "test.csv").write_text("text\ntest")
        
        return tmp_path

    def test_project_init_help(self, runner):
        """Test project init help."""
        result = runner.invoke(app, ["project", "init", "--help"])
        assert result.exit_code == 0
        assert "Initialize a new k-bert project" in result.stdout
        assert "--template" in result.stdout
        assert "--force" in result.stdout

    def test_project_init_basic(self, runner, tmp_path):
        """Test basic project initialization."""
        project_path = tmp_path / "new-project"
        
        with patch('cli.commands.project.init._copy_template') as mock_copy:
            with patch('cli.commands.project.init._update_project_config') as mock_update:
                result = runner.invoke(app, ["project", "init", str(project_path)])
                
                assert result.exit_code == 0
                assert "Project created successfully" in result.stdout
                mock_copy.assert_called_once()
                mock_update.assert_called_once()

    def test_project_init_with_template(self, runner, tmp_path):
        """Test project init with specific template."""
        project_path = tmp_path / "new-project"
        
        with patch('cli.commands.project.init._get_available_templates', return_value=["base", "advanced"]):
            with patch('cli.commands.project.init._copy_template') as mock_copy:
                result = runner.invoke(app, ["project", "init", str(project_path), "--template", "advanced"])
                
                assert result.exit_code == 0
                mock_copy.assert_called_with("advanced", project_path)

    def test_project_init_invalid_template(self, runner, tmp_path):
        """Test project init with invalid template."""
        project_path = tmp_path / "new-project"
        
        with patch('cli.commands.project.init._get_available_templates', return_value=["base"]):
            result = runner.invoke(app, ["project", "init", str(project_path), "--template", "invalid"])
            
            assert result.exit_code == 1
            assert "Template 'invalid' not found" in result.stdout

    def test_project_init_existing_directory(self, runner, tmp_path):
        """Test project init in existing directory."""
        project_path = tmp_path / "existing"
        project_path.mkdir()
        (project_path / "file.txt").write_text("existing")
        
        # Without force
        result = runner.invoke(app, ["project", "init", str(project_path)])
        assert result.exit_code == 1
        assert "already exists" in result.stdout
        
        # With force
        with patch('cli.commands.project.init._copy_template'):
            result = runner.invoke(app, ["project", "init", str(project_path), "--force"])
            assert result.exit_code == 0

    def test_project_run_help(self, runner):
        """Test project run help."""
        result = runner.invoke(app, ["project", "run", "--help"])
        assert result.exit_code == 0
        assert "Run project with configuration" in result.stdout
        assert "--experiment" in result.stdout

    def test_project_run_default(self, runner, project_dir):
        """Test running project with default settings."""
        with patch('cli.commands.core.train.train_command') as mock_train:
            mock_train.return_value = None
            
            os.chdir(project_dir)
            result = runner.invoke(app, ["project", "run"])
            
            assert result.exit_code == 0
            mock_train.assert_called_once()

    def test_project_run_with_experiment(self, runner, project_dir):
        """Test running project with specific experiment."""
        with patch('cli.commands.core.train.train_command') as mock_train:
            mock_train.return_value = None
            
            os.chdir(project_dir)
            result = runner.invoke(app, ["project", "run", "--experiment", "quick_test"])
            
            assert result.exit_code == 0
            mock_train.assert_called_once()
            # Check that epochs override was applied
            call_kwargs = mock_train.call_args[1]
            assert call_kwargs.get('epochs') == 1

    def test_project_run_invalid_experiment(self, runner, project_dir):
        """Test running with invalid experiment name."""
        os.chdir(project_dir)
        result = runner.invoke(app, ["project", "run", "--experiment", "nonexistent"])
        
        assert result.exit_code == 1
        assert "Experiment 'nonexistent' not found" in result.stdout

    def test_project_run_no_config(self, runner, tmp_path):
        """Test running without project config."""
        os.chdir(tmp_path)
        result = runner.invoke(app, ["project", "run"])
        
        assert result.exit_code == 1
        assert "No k-bert.yaml found" in result.stdout

    def test_project_template_list(self, runner):
        """Test listing available templates."""
        with patch('cli.commands.project.template._get_available_templates', return_value=["base", "advanced", "kaggle"]):
            result = runner.invoke(app, ["project", "template", "list"])
            
            assert result.exit_code == 0
            assert "Available templates:" in result.stdout
            assert "base" in result.stdout
            assert "advanced" in result.stdout
            assert "kaggle" in result.stdout

    def test_project_template_show(self, runner):
        """Test showing template details."""
        template_info = {
            "name": "base",
            "description": "Basic k-bert project template",
            "files": ["k-bert.yaml", "src/", "data/", "README.md"]
        }
        
        with patch('cli.commands.project.template._get_template_info', return_value=template_info):
            result = runner.invoke(app, ["project", "template", "show", "base"])
            
            assert result.exit_code == 0
            assert "Template: base" in result.stdout
            assert "Basic k-bert project template" in result.stdout
            assert "k-bert.yaml" in result.stdout

    def test_project_template_create(self, runner, tmp_path):
        """Test creating custom template."""
        template_name = "custom"
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "k-bert.yaml").write_text("name: template")
        
        with patch('cli.commands.project.template._save_template') as mock_save:
            result = runner.invoke(app, ["project", "template", "create", template_name, str(source_dir)])
            
            assert result.exit_code == 0
            assert f"Template '{template_name}' created" in result.stdout
            mock_save.assert_called_once()

    def test_project_list(self, runner, tmp_path):
        """Test listing projects."""
        # Create multiple project directories
        for i in range(3):
            project_dir = tmp_path / f"project{i}"
            project_dir.mkdir()
            config = {"name": f"project-{i}", "description": f"Project {i}"}
            with open(project_dir / "k-bert.yaml", "w") as f:
                yaml.dump(config, f)
        
        with patch('pathlib.Path.cwd', return_value=tmp_path):
            result = runner.invoke(app, ["project", "list"])
            
            assert result.exit_code == 0
            assert "project-0" in result.stdout
            assert "project-1" in result.stdout
            assert "project-2" in result.stdout

    def test_run_command_shortcut(self, runner, project_dir):
        """Test the root-level run command shortcut."""
        with patch('cli.commands.core.train.train_command') as mock_train:
            mock_train.return_value = None
            
            os.chdir(project_dir)
            result = runner.invoke(app, ["run"])
            
            assert result.exit_code == 0
            mock_train.assert_called_once()

    def test_run_command_with_overrides(self, runner, project_dir):
        """Test run command with CLI overrides."""
        with patch('cli.commands.core.train.train_command') as mock_train:
            mock_train.return_value = None
            
            os.chdir(project_dir)
            result = runner.invoke(app, ["run", "--epochs", "10", "--batch-size", "64"])
            
            assert result.exit_code == 0
            call_kwargs = mock_train.call_args[1]
            assert call_kwargs.get('epochs') == 10
            assert call_kwargs.get('batch_size') == 64

    def test_project_run_with_plugins(self, runner, project_dir):
        """Test running project with plugins."""
        # Add plugin config to project
        config_path = project_dir / "k-bert.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        config["plugins"] = {
            "enabled": ["custom_head"],
            "custom_head": {"module": "src.heads.custom"}
        }
        
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        # Create plugin file
        plugin_dir = project_dir / "src" / "heads"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "custom.py").write_text("# Custom head plugin")
        
        with patch('cli.plugins.load_project_plugins') as mock_load:
            with patch('cli.commands.core.train.train_command'):
                os.chdir(project_dir)
                result = runner.invoke(app, ["run"])
                
                assert result.exit_code == 0
                mock_load.assert_called_once()

    def test_project_validate(self, runner, project_dir):
        """Test project validation command."""
        result = runner.invoke(app, ["project", "validate", str(project_dir)])
        
        assert result.exit_code == 0
        assert "Project validation passed" in result.stdout

    def test_project_validate_missing_files(self, runner, project_dir):
        """Test validation with missing required files."""
        # Remove data files
        (project_dir / "data" / "train.csv").unlink()
        
        result = runner.invoke(app, ["project", "validate", str(project_dir)])
        
        assert result.exit_code == 1
        assert "Missing file" in result.stdout
        assert "train.csv" in result.stdout