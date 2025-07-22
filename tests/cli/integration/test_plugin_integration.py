"""Integration tests for plugin system with CLI."""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import yaml
from typer.testing import CliRunner

from cli.app import app


class TestPluginIntegration:
    """Integration tests for plugin system."""

    @pytest.fixture
    def runner(self):
        """Create CLI runner."""
        return CliRunner()

    @pytest.fixture
    def plugin_project(self, tmp_path):
        """Create project with plugin configuration."""
        # Create project config
        config = {
            "name": "plugin-test",
            "models": {
                "default_model": "bert-base-uncased",
                "head": {"type": "custom_sentiment"}
            },
            "data": {
                "train_path": "data/train.csv",
                "batch_size": 32
            },
            "plugins": {
                "enabled": ["custom_sentiment", "text_augmenter"],
                "custom_sentiment": {
                    "module": "src.heads.sentiment",
                    "class": "SentimentHead",
                    "config": {
                        "num_classes": 3,
                        "hidden_dim": 256,
                        "dropout": 0.2
                    }
                },
                "text_augmenter": {
                    "module": "src.augmenters.backtranslation",
                    "class": "BackTranslationAugmenter",
                    "config": {
                        "languages": ["de", "fr"],
                        "probability": 0.3
                    }
                }
            }
        }
        
        with open(tmp_path / "k-bert.yaml", "w") as f:
            yaml.dump(config, f)
        
        # Create plugin directories
        heads_dir = tmp_path / "src" / "heads"
        heads_dir.mkdir(parents=True)
        
        augmenters_dir = tmp_path / "src" / "augmenters"
        augmenters_dir.mkdir(parents=True)
        
        # Create sentiment head plugin
        sentiment_code = '''
from cli.plugins import register_component, HeadPlugin

@register_component("custom_sentiment")
class SentimentHead(HeadPlugin):
    """Custom sentiment analysis head."""
    
    def __init__(self, config):
        super().__init__(config)
        self.num_classes = config.get("num_classes", 3)
        self.hidden_dim = config.get("hidden_dim", 256)
        self.dropout = config.get("dropout", 0.1)
        
    def forward(self, hidden_states, **kwargs):
        # Simplified implementation
        batch_size = hidden_states.shape[0]
        logits = [[0.1, 0.2, 0.7]] * batch_size  # Mock output
        return {"logits": logits}
    
    def get_loss(self, logits, labels):
        # Mock loss calculation
        return 0.5
'''
        (heads_dir / "sentiment.py").write_text(sentiment_code)
        (heads_dir / "__init__.py").write_text("")
        
        # Create augmenter plugin
        augmenter_code = '''
from cli.plugins import register_component, DataAugmenterPlugin

@register_component("text_augmenter")
class BackTranslationAugmenter(DataAugmenterPlugin):
    """Back-translation data augmenter."""
    
    def __init__(self, config):
        super().__init__(config)
        self.languages = config.get("languages", ["de"])
        self.probability = config.get("probability", 0.3)
    
    def augment(self, text, label=None):
        # Mock augmentation
        import random
        if random.random() < self.probability:
            return f"[Augmented] {text}"
        return text
    
    def augment_batch(self, texts, labels=None):
        return [self.augment(text) for text in texts]
'''
        (augmenters_dir / "backtranslation.py").write_text(augmenter_code)
        (augmenters_dir / "__init__.py").write_text("")
        
        # Create data directory
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "train.csv").write_text("text,label\nGreat movie!,2\nTerrible film,0\nOkay I guess,1")
        
        return tmp_path

    def test_plugin_loading_on_train(self, runner, plugin_project):
        """Test that plugins are loaded during training."""
        os.chdir(plugin_project)
        
        with patch('cli.plugins.load_project_plugins') as mock_load:
            with patch('cli.commands.core.train._load_data'):
                with patch('models.factory.create_model'):
                    result = runner.invoke(app, ["train", "--epochs", "1"])
                    
                    assert result.exit_code == 0
                    mock_load.assert_called_once()

    def test_plugin_validation_command(self, runner, plugin_project):
        """Test plugin validation command."""
        os.chdir(plugin_project)
        
        # Future command: k-bert plugin validate
        # result = runner.invoke(app, ["plugin", "validate"])
        # assert result.exit_code == 0
        # assert "custom_sentiment: Valid" in result.stdout
        # assert "text_augmenter: Valid" in result.stdout

    def test_plugin_list_command(self, runner, plugin_project):
        """Test listing available plugins."""
        os.chdir(plugin_project)
        
        # Future command: k-bert plugin list
        # result = runner.invoke(app, ["plugin", "list"])
        # assert result.exit_code == 0
        # assert "custom_sentiment" in result.stdout
        # assert "text_augmenter" in result.stdout

    def test_plugin_error_handling(self, runner, plugin_project):
        """Test plugin loading error handling."""
        os.chdir(plugin_project)
        
        # Corrupt the plugin file
        plugin_file = plugin_project / "src" / "heads" / "sentiment.py"
        plugin_file.write_text("import syntax error!")
        
        with patch('cli.commands.core.train._load_data'):
            result = runner.invoke(app, ["train"])
            
            assert result.exit_code == 1
            assert "Failed to load plugin" in result.stdout or "Plugin error" in result.stdout

    def test_plugin_config_override(self, runner, plugin_project):
        """Test overriding plugin config from CLI."""
        os.chdir(plugin_project)
        
        # Modify plugin config via CLI
        result = runner.invoke(app, ["config", "set", "plugins.custom_sentiment.config.num_classes", "5"])
        assert result.exit_code == 0
        
        # Verify config was updated
        with open("k-bert.yaml") as f:
            config = yaml.safe_load(f)
            assert config["plugins"]["custom_sentiment"]["config"]["num_classes"] == "5"

    def test_disabled_plugin_not_loaded(self, runner, plugin_project):
        """Test that disabled plugins are not loaded."""
        os.chdir(plugin_project)
        
        # Disable a plugin
        with open("k-bert.yaml") as f:
            config = yaml.safe_load(f)
        
        config["plugins"]["enabled"] = ["custom_sentiment"]  # Remove text_augmenter
        
        with open("k-bert.yaml", "w") as f:
            yaml.dump(config, f)
        
        with patch('cli.plugins._plugin_manager.load_plugin') as mock_load:
            with patch('cli.commands.core.train._load_data'):
                result = runner.invoke(app, ["train", "--epochs", "1"])
                
                # Should only load custom_sentiment, not text_augmenter
                loaded_modules = [call[0][0] for call in mock_load.call_args_list]
                assert any("sentiment" in m for m in loaded_modules)
                assert not any("backtranslation" in m for m in loaded_modules)

    def test_plugin_with_dependencies(self, runner, tmp_path):
        """Test plugin with dependencies on other plugins."""
        os.chdir(tmp_path)
        
        # Create config with dependent plugins
        config = {
            "name": "dependency-test",
            "models": {"default_model": "bert-base-uncased"},
            "plugins": {
                "enabled": ["base_processor", "advanced_processor"],
                "base_processor": {
                    "module": "src.processors.base",
                    "class": "BaseProcessor"
                },
                "advanced_processor": {
                    "module": "src.processors.advanced",
                    "class": "AdvancedProcessor",
                    "dependencies": ["base_processor"]
                }
            }
        }
        
        with open("k-bert.yaml", "w") as f:
            yaml.dump(config, f)
        
        # Create plugin files
        processors_dir = tmp_path / "src" / "processors"
        processors_dir.mkdir(parents=True)
        
        base_code = '''
from cli.plugins import register_component

@register_component
class BaseProcessor:
    def process(self, data):
        return data.lower()
'''
        (processors_dir / "base.py").write_text(base_code)
        
        advanced_code = '''
from cli.plugins import register_component

@register_component
class AdvancedProcessor:
    def __init__(self, config, base_processor=None):
        self.base = base_processor
    
    def process(self, data):
        if self.base:
            data = self.base.process(data)
        return f"[PROCESSED] {data}"
'''
        (processors_dir / "advanced.py").write_text(advanced_code)
        
        # Test loading with dependencies
        with patch('cli.plugins.load_project_plugins'):
            result = runner.invoke(app, ["project", "validate"])
            assert result.exit_code == 0

    def test_plugin_hot_reload(self, runner, plugin_project):
        """Test plugin hot reloading during development."""
        os.chdir(plugin_project)
        
        # Initial plugin version
        plugin_file = plugin_project / "src" / "heads" / "sentiment.py"
        original_content = plugin_file.read_text()
        
        # Modify plugin
        modified_content = original_content.replace("num_classes = 3", "num_classes = 5")
        plugin_file.write_text(modified_content)
        
        # Future command: k-bert plugin reload
        # result = runner.invoke(app, ["plugin", "reload"])
        # assert result.exit_code == 0
        # assert "Reloaded plugins" in result.stdout

    def test_plugin_create_command(self, runner, tmp_path):
        """Test creating new plugin from template."""
        os.chdir(tmp_path)
        
        # Future command: k-bert plugin create
        # result = runner.invoke(app, ["plugin", "create", "head", "my_custom_head"])
        # assert result.exit_code == 0
        # assert (tmp_path / "src" / "heads" / "my_custom_head.py").exists()

    def test_plugin_test_command(self, runner, plugin_project):
        """Test running plugin tests."""
        os.chdir(plugin_project)
        
        # Create test file for plugin
        test_dir = plugin_project / "tests" / "plugins"
        test_dir.mkdir(parents=True)
        
        test_code = '''
def test_sentiment_head():
    from src.heads.sentiment import SentimentHead
    
    config = {"num_classes": 3}
    head = SentimentHead(config)
    
    # Mock hidden states
    hidden_states = [[0.1] * 768]
    result = head.forward(hidden_states)
    
    assert "logits" in result
    assert len(result["logits"][0]) == 3
'''
        (test_dir / "test_sentiment.py").write_text(test_code)
        
        # Future command: k-bert plugin test
        # result = runner.invoke(app, ["plugin", "test", "custom_sentiment"])
        # assert result.exit_code == 0
        # assert "1 passed" in result.stdout

    def test_plugin_info_command(self, runner, plugin_project):
        """Test getting plugin information."""
        os.chdir(plugin_project)
        
        # Future command: k-bert plugin info
        # result = runner.invoke(app, ["plugin", "info", "custom_sentiment"])
        # assert result.exit_code == 0
        # assert "SentimentHead" in result.stdout
        # assert "num_classes: 3" in result.stdout

    def test_multiple_projects_with_plugins(self, runner, tmp_path):
        """Test managing multiple projects with different plugins."""
        # Create two projects
        for i in range(2):
            project_dir = tmp_path / f"project{i}"
            project_dir.mkdir()
            
            config = {
                "name": f"project-{i}",
                "plugins": {
                    "enabled": [f"plugin_{i}"],
                    f"plugin_{i}": {
                        "module": f"src.custom_plugin_{i}",
                        "class": f"Plugin{i}"
                    }
                }
            }
            
            with open(project_dir / "k-bert.yaml", "w") as f:
                yaml.dump(config, f)
        
        # Plugins should be isolated per project
        with patch('cli.plugins._plugin_manager._registry', {}) as registry:
            os.chdir(tmp_path / "project0")
            # Load project0 plugins
            
            os.chdir(tmp_path / "project1")
            # Load project1 plugins
            
            # Each project should have its own plugins