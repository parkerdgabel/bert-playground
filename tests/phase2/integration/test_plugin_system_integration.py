"""Integration tests for the plugin system with CLI and training components.

This tests the integration of the plugin architecture with existing components
to ensure plugins can be discovered, loaded, and used seamlessly.
"""

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest
import yaml

from cli.plugins.base import BasePlugin, HeadPlugin, AugmenterPlugin
from cli.plugins.loader import PluginLoader
from cli.plugins.registry import PluginRegistry
from cli.commands.core.train import train_command
from data.augmentation.base import BaseAugmenter
from models.heads.base import BaseHead


class CustomHeadPlugin(BasePlugin, HeadPlugin):
    """Example custom head plugin for testing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.num_classes = config.get("num_classes", 2) if config else 2
    
    def create_head(self, input_dim: int, output_dim: int, **kwargs) -> BaseHead:
        """Create a custom classification head."""
        from models.heads.classification import ClassificationHead
        
        # Override with plugin config if available
        if "dropout_rate" in self.config:
            kwargs["dropout_rate"] = self.config["dropout_rate"]
        
        return ClassificationHead(
            input_dim=input_dim,
            num_labels=output_dim,
            **kwargs
        )
    
    def get_metadata(self):
        """Return plugin metadata."""
        from core.protocols.plugins import PluginMetadata
        return PluginMetadata(
            name="CustomHeadPlugin",
            version="1.0.0",
            description="Custom classification head with enhanced features",
            author="Test Author",
            dependencies=[]
        )


class CustomAugmenterPlugin(BasePlugin, AugmenterPlugin):
    """Example custom augmenter plugin for testing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.augmentation_prob = config.get("augmentation_prob", 0.5) if config else 0.5
    
    def create_augmenter(self, **kwargs) -> BaseAugmenter:
        """Create a custom augmenter."""
        class TestAugmenter(BaseAugmenter):
            def __init__(self, prob):
                self.prob = prob
            
            def augment(self, sample: Dict[str, Any]) -> Dict[str, Any]:
                # Simple test augmentation - add prefix to text
                if "text" in sample and mx.random.uniform() < self.prob:
                    sample["text"] = f"[AUGMENTED] {sample['text']}"
                return sample
            
            def get_config(self) -> Dict[str, Any]:
                return {"prob": self.prob}
        
        return TestAugmenter(self.augmentation_prob)
    
    def get_metadata(self):
        """Return plugin metadata."""
        from core.protocols.plugins import PluginMetadata
        return PluginMetadata(
            name="CustomAugmenterPlugin",
            version="1.0.0",
            description="Custom text augmentation plugin",
            requires=["mlx>=0.5.0"]
        )


@pytest.fixture
def plugin_project_dir(tmp_path):
    """Create a project directory with plugins."""
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()
    
    # Create k-bert.yaml config
    config = {
        "project": {
            "name": "test-project",
            "version": "0.1.0"
        },
        "plugins": {
            "enabled": True,
            "directories": ["src/plugins"],
            "auto_discover": True
        },
        "models": {
            "model_type": "modernbert_with_head",
            "head": {
                "plugin": "CustomHeadPlugin",
                "config": {
                    "num_classes": 3,
                    "dropout_rate": 0.2
                }
            }
        },
        "data": {
            "augmentation": {
                "plugin": "CustomAugmenterPlugin",
                "config": {
                    "augmentation_prob": 0.7
                }
            }
        }
    }
    
    with open(project_dir / "k-bert.yaml", "w") as f:
        yaml.dump(config, f)
    
    # Create plugin directory
    plugin_dir = project_dir / "src" / "plugins"
    plugin_dir.mkdir(parents=True)
    
    # Create __init__.py
    (plugin_dir / "__init__.py").write_text("")
    
    # Create head plugin file
    head_plugin_code = '''
from cli.plugins.base import BasePlugin, HeadPlugin
from core.protocols.plugins import PluginMetadata
from models.heads.classification import ClassificationHead

class CustomHeadPlugin(BasePlugin, HeadPlugin):
    """Custom head plugin from project."""
    
    def create_head(self, input_dim: int, output_dim: int, **kwargs):
        # Use config from plugin initialization
        dropout = self.config.get("dropout_rate", 0.1)
        return ClassificationHead(
            input_dim=input_dim,
            num_labels=output_dim,
            dropout_rate=dropout
        )
    
    def get_metadata(self):
        return PluginMetadata(
            name="CustomHeadPlugin",
            version="1.0.0",
            description="Project-specific head plugin"
        )
'''
    (plugin_dir / "custom_head.py").write_text(head_plugin_code)
    
    # Create augmenter plugin file
    augmenter_plugin_code = '''
from cli.plugins.base import BasePlugin, AugmenterPlugin
from core.protocols.plugins import PluginMetadata
import mlx.core as mx

class CustomAugmenterPlugin(BasePlugin, AugmenterPlugin):
    """Custom augmenter plugin from project."""
    
    def create_augmenter(self, **kwargs):
        from data.augmentation.base import BaseAugmenter
        
        class ProjectAugmenter(BaseAugmenter):
            def __init__(self, prob):
                self.prob = prob
            
            def augment(self, sample):
                if "text" in sample and mx.random.uniform() < self.prob:
                    sample["text"] = f"[PROJECT] {sample['text']}"
                return sample
            
            def get_config(self):
                return {"prob": self.prob}
        
        return ProjectAugmenter(self.config.get("augmentation_prob", 0.5))
    
    def get_metadata(self):
        return PluginMetadata(
            name="CustomAugmenterPlugin",
            version="1.0.0",
            description="Project-specific augmenter"
        )
'''
    (plugin_dir / "custom_augmenter.py").write_text(augmenter_plugin_code)
    
    return project_dir


@pytest.fixture
def plugin_registry():
    """Create a fresh plugin registry."""
    return PluginRegistry()


@pytest.fixture
def plugin_loader():
    """Create a plugin loader."""
    return PluginLoader()


class TestPluginSystemIntegration:
    """Test integration of plugin system with various components."""
    
    def test_plugin_discovery(self, plugin_project_dir, plugin_loader):
        """Test that plugins can be discovered from project directory."""
        # Load plugins from project
        plugins = plugin_loader.load_from_directory(plugin_project_dir / "src" / "plugins")
        
        # Should find both plugins
        assert len(plugins) >= 2
        plugin_names = [p.get_metadata().name for p in plugins]
        assert "CustomHeadPlugin" in plugin_names
        assert "CustomAugmenterPlugin" in plugin_names
    
    def test_plugin_registration(self, plugin_registry):
        """Test that plugins can be registered and retrieved."""
        # Create and register plugins
        head_plugin = CustomHeadPlugin({"num_classes": 5})
        augmenter_plugin = CustomAugmenterPlugin({"augmentation_prob": 0.8})
        
        plugin_registry.register("head", "custom_head", head_plugin)
        plugin_registry.register("augmenter", "custom_aug", augmenter_plugin)
        
        # Retrieve plugins
        retrieved_head = plugin_registry.get("head", "custom_head")
        retrieved_aug = plugin_registry.get("augmenter", "custom_aug")
        
        assert retrieved_head is head_plugin
        assert retrieved_aug is augmenter_plugin
        
        # List plugins by type
        head_plugins = plugin_registry.list_by_type("head")
        assert "custom_head" in head_plugins
    
    def test_plugin_config_integration(self, plugin_project_dir, plugin_loader, plugin_registry):
        """Test that plugin configuration is properly loaded and applied."""
        # Load project config
        with open(plugin_project_dir / "k-bert.yaml") as f:
            config = yaml.safe_load(f)
        
        # Load and register plugins
        plugins = plugin_loader.load_from_directory(plugin_project_dir / "src" / "plugins")
        for plugin in plugins:
            metadata = plugin.get_metadata()
            
            # Apply config from project
            if metadata.name == "CustomHeadPlugin":
                plugin.config = config["models"]["head"]["config"]
                plugin_registry.register("head", metadata.name, plugin)
            elif metadata.name == "CustomAugmenterPlugin":
                plugin.config = config["data"]["augmentation"]["config"]
                plugin_registry.register("augmenter", metadata.name, plugin)
        
        # Verify configs were applied
        head_plugin = plugin_registry.get("head", "CustomHeadPlugin")
        assert head_plugin.config["num_classes"] == 3
        assert head_plugin.config["dropout_rate"] == 0.2
        
        aug_plugin = plugin_registry.get("augmenter", "CustomAugmenterPlugin")
        assert aug_plugin.config["augmentation_prob"] == 0.7
    
    def test_plugin_head_creation(self, plugin_registry):
        """Test that head plugins can create model heads."""
        # Register plugin
        head_plugin = CustomHeadPlugin({"num_classes": 10, "dropout_rate": 0.3})
        plugin_registry.register("head", "custom", head_plugin)
        
        # Create head using plugin
        plugin = plugin_registry.get("head", "custom")
        head = plugin.create_head(input_dim=768, output_dim=10)
        
        assert head is not None
        assert hasattr(head, "forward")
        assert head.num_labels == 10
    
    def test_plugin_augmenter_creation(self, plugin_registry):
        """Test that augmenter plugins can create augmenters."""
        # Register plugin
        aug_plugin = CustomAugmenterPlugin({"augmentation_prob": 0.9})
        plugin_registry.register("augmenter", "custom", aug_plugin)
        
        # Create augmenter using plugin
        plugin = plugin_registry.get("augmenter", "custom")
        augmenter = plugin.create_augmenter()
        
        assert augmenter is not None
        assert hasattr(augmenter, "augment")
        
        # Test augmentation
        sample = {"text": "test sample"}
        augmented = augmenter.augment(sample)
        assert "text" in augmented
    
    def test_plugin_cli_integration(self, plugin_project_dir, monkeypatch):
        """Test that plugins integrate with CLI commands."""
        # Change to project directory
        monkeypatch.chdir(plugin_project_dir)
        
        # Mock the actual training to avoid long runs
        with patch("cli.commands.core.train.BaseTrainer") as mock_trainer:
            mock_result = MagicMock()
            mock_result.final_metrics = {"loss": 0.5}
            mock_trainer.return_value.train.return_value = mock_result
            
            # Create mock data files
            data_dir = plugin_project_dir / "data"
            data_dir.mkdir()
            
            # Create minimal CSV files
            (data_dir / "train.csv").write_text("text,label\ntest1,0\ntest2,1")
            (data_dir / "val.csv").write_text("text,label\nval1,0\nval2,1")
            
            # Run train command - plugins should be loaded from config
            from typer.testing import CliRunner
            from cli.app import app
            
            runner = CliRunner()
            result = runner.invoke(app, [
                "train",
                "--train-data", str(data_dir / "train.csv"),
                "--val-data", str(data_dir / "val.csv"),
                "--config", str(plugin_project_dir / "k-bert.yaml")
            ])
            
            # Training should succeed
            assert result.exit_code == 0
    
    def test_plugin_error_handling(self, plugin_registry):
        """Test error handling for plugin operations."""
        # Try to get non-existent plugin
        plugin = plugin_registry.get("head", "non_existent")
        assert plugin is None
        
        # Register plugin with invalid type
        with pytest.raises(ValueError):
            plugin_registry.register("invalid_type", "test", CustomHeadPlugin())
    
    def test_plugin_dependencies(self):
        """Test that plugin dependencies are checked."""
        from core.protocols.plugins import PluginMetadata
        
        class DependentPlugin(BasePlugin):
            def get_metadata(self):
                return PluginMetadata(
                    name="DependentPlugin",
                    version="1.0.0",
                    description="Plugin with dependencies",
                    requires=["non_existent_package>=1.0.0"]
                )
        
        plugin = DependentPlugin()
        metadata = plugin.get_metadata()
        
        # Should have dependencies listed
        assert len(metadata.requires) > 0
        assert "non_existent_package" in metadata.requires[0]
    
    def test_plugin_lifecycle(self, plugin_registry):
        """Test plugin lifecycle management."""
        # Create plugin with lifecycle methods
        class LifecyclePlugin(BasePlugin):
            def __init__(self, config=None):
                super().__init__(config)
                self.initialized = False
                self.cleaned_up = False
            
            def initialize(self):
                self.initialized = True
            
            def cleanup(self):
                self.cleaned_up = True
        
        plugin = LifecyclePlugin()
        plugin_registry.register("test", "lifecycle", plugin)
        
        # Initialize if method exists
        if hasattr(plugin, "initialize"):
            plugin.initialize()
        
        assert plugin.initialized
        
        # Cleanup if method exists
        if hasattr(plugin, "cleanup"):
            plugin.cleanup()
        
        assert plugin.cleaned_up
    
    def test_plugin_compatibility(self):
        """Test backward compatibility with non-plugin components."""
        from models.factory import create_model
        from data.factory import create_dataset
        
        # Should work without plugins
        model = create_model({
            "model_type": "bert_classifier",
            "num_labels": 2
        })
        assert model is not None
        
        # Dataset creation should work without augmenter plugins
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("text,label\ntest1,0\ntest2,1")
            f.flush()
            
            dataset = create_dataset(f.name)
            assert dataset is not None
            
            Path(f.name).unlink()