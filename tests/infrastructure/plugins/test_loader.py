"""Tests for plugin loader and discovery."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from infrastructure.plugins.base import PluginBase, PluginMetadata
from infrastructure.plugins.loader import PluginDiscovery, PluginLoader


class TestPluginDiscovery:
    """Test PluginDiscovery class."""
    
    def test_is_valid_plugin_class(self):
        """Test plugin class validation."""
        discovery = PluginDiscovery()
        
        # Valid plugin class
        class ValidPlugin(PluginBase):
            NAME = "valid"
            def _initialize(self, context):
                pass
        
        assert discovery._is_valid_plugin_class(ValidPlugin) is True
        
        # Invalid classes
        assert discovery._is_valid_plugin_class(PluginBase) is False  # Base class
        assert discovery._is_valid_plugin_class(str) is False  # Not a class
        assert discovery._is_valid_plugin_class(42) is False  # Not a class
    
    def test_implements_plugin_protocol(self):
        """Test protocol implementation checking."""
        discovery = PluginDiscovery()
        
        # Class that implements protocol
        class ProtocolPlugin:
            @property
            def metadata(self):
                return PluginMetadata(name="test")
            
            @property
            def state(self):
                return "running"
            
            def validate(self, context):
                pass
            
            def initialize(self, context):
                pass
            
            def start(self, context):
                pass
            
            def stop(self, context):
                pass
            
            def cleanup(self, context):
                pass
        
        assert discovery._implements_plugin_protocol(ProtocolPlugin) is True
        
        # Class missing methods
        class IncompletePlugin:
            @property
            def metadata(self):
                return PluginMetadata(name="test")
        
        assert discovery._implements_plugin_protocol(IncompletePlugin) is False
    
    def test_discover_from_entry_points(self):
        """Test discovering plugins from entry points."""
        discovery = PluginDiscovery()
        
        # Mock entry points
        mock_entry_point = Mock()
        mock_entry_point.name = "test_plugin"
        
        # Mock plugin class
        class MockPlugin(PluginBase):
            NAME = "test"
            def _initialize(self, context):
                pass
        
        mock_entry_point.load.return_value = MockPlugin
        
        with patch("importlib.metadata.entry_points") as mock_ep:
            # For Python 3.10+, entry_points returns EntryPoints object
            # For older versions, it returns a dict
            import sys
            if sys.version_info >= (3, 10):
                # Mock for Python 3.10+
                mock_ep.return_value = [mock_entry_point]
            else:
                # Mock for older versions
                mock_ep.return_value = {"k_bert.plugins": [mock_entry_point]}
            
            plugins = discovery.discover_from_entry_points()
            
            assert len(plugins) == 1
            assert plugins[0] is MockPlugin
            mock_entry_point.load.assert_called_once()
    
    @pytest.fixture
    def temp_plugin_file(self):
        """Create a temporary plugin file."""
        plugin_code = '''
from infrastructure.plugins.base import PluginBase

class TempPlugin(PluginBase):
    NAME = "temp_plugin"
    VERSION = "1.0.0"
    
    def _initialize(self, context):
        self.initialized = True
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(plugin_code)
            temp_path = Path(f.name)
        
        yield temp_path
        
        # Cleanup
        temp_path.unlink()
    
    def test_discover_from_file(self, temp_plugin_file):
        """Test discovering plugins from a file."""
        discovery = PluginDiscovery()
        
        plugins = discovery._discover_from_file(temp_plugin_file)
        
        assert len(plugins) == 1
        plugin_class = plugins[0]
        assert plugin_class.__name__ == "TempPlugin"
        
        # Test instantiation
        plugin = plugin_class()
        assert plugin.metadata.name == "temp_plugin"
    
    @pytest.fixture
    def temp_plugin_dir(self):
        """Create a temporary directory with plugin files."""
        import os
        
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create plugin files
        plugin1_code = '''
from infrastructure.plugins.base import PluginBase

class Plugin1(PluginBase):
    NAME = "plugin1"
    
    def _initialize(self, context):
        pass
'''
        
        plugin2_code = '''
from infrastructure.plugins.base import PluginBase

class Plugin2(PluginBase):
    NAME = "plugin2"
    
    def _initialize(self, context):
        pass
'''
        
        (temp_dir / "plugin1.py").write_text(plugin1_code)
        (temp_dir / "plugin2.py").write_text(plugin2_code)
        
        # Create subdirectory with plugin
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        (subdir / "plugin3.py").write_text(plugin1_code.replace("Plugin1", "Plugin3").replace("plugin1", "plugin3"))
        
        yield temp_dir
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
    
    def test_discover_from_directory(self, temp_plugin_dir):
        """Test discovering plugins from a directory."""
        discovery = PluginDiscovery()
        
        plugins = discovery._discover_from_directory(temp_plugin_dir)
        
        # Should find all 3 plugins (including subdirectory)
        assert len(plugins) == 3
        plugin_names = {p.__name__ for p in plugins}
        assert plugin_names == {"Plugin1", "Plugin2", "Plugin3"}
    
    @pytest.fixture
    def temp_project(self):
        """Create a temporary project structure."""
        import shutil
        
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create project structure
        src_dir = temp_dir / "src" / "plugins"
        src_dir.mkdir(parents=True)
        
        # Create plugin
        plugin_code = '''
from infrastructure.plugins.base import PluginBase

class ProjectPlugin(PluginBase):
    NAME = "project_plugin"
    
    def _initialize(self, context):
        pass
'''
        (src_dir / "project_plugin.py").write_text(plugin_code)
        
        # Create pyproject.toml with custom paths
        pyproject_content = '''
[tool.k-bert.plugins]
paths = ["custom_plugins"]
'''
        (temp_dir / "pyproject.toml").write_text(pyproject_content)
        
        # Create custom plugin directory
        custom_dir = temp_dir / "custom_plugins"
        custom_dir.mkdir()
        
        custom_plugin_code = '''
from infrastructure.plugins.base import PluginBase

class CustomPlugin(PluginBase):
    NAME = "custom_plugin"
    
    def _initialize(self, context):
        pass
'''
        (custom_dir / "custom_plugin.py").write_text(custom_plugin_code)
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_discover_from_project(self, temp_project):
        """Test discovering plugins from project structure."""
        discovery = PluginDiscovery()
        
        plugins = discovery.discover_from_project(temp_project)
        
        # Should find plugins from both standard and custom directories
        assert len(plugins) >= 2
        plugin_names = {p.__name__ for p in plugins}
        assert "ProjectPlugin" in plugin_names
        assert "CustomPlugin" in plugin_names


class TestPluginLoader:
    """Test PluginLoader class."""
    
    def test_load_plugin(self):
        """Test loading a single plugin."""
        
        class TestPlugin(PluginBase):
            NAME = "test_plugin"
            
            def _initialize(self, context):
                self.initialized = True
        
        loader = PluginLoader()
        # Global config with plugin-specific section
        config = {
            "plugins": {
                "test_plugin": {"test_key": "test_value"}
            }
        }
        
        plugin = loader.load_plugin(TestPlugin, config=config, validate=False)
        
        assert plugin is not None
        assert plugin.metadata.name == "test_plugin"
        assert plugin.config == {"test_key": "test_value"}
        # The plugin doesn't have 'initialized' attribute until _initialize is called
    
    def test_load_plugin_with_validation_failure(self):
        """Test loading plugin with validation failure."""
        
        class BadPlugin(PluginBase):
            NAME = "bad_plugin"
            
            def _initialize(self, context):
                pass
        
        loader = PluginLoader()
        
        # Mock validator to return failure
        from unittest.mock import Mock
        mock_validator = Mock()
        validation_result = Mock()
        validation_result.is_valid = False
        validation_result.errors = ["Validation failed"]
        mock_validator.validate.return_value = validation_result
        loader.validator = mock_validator
        
        # Should raise PluginError on validation failure
        from infrastructure.plugins.base import PluginError
        with pytest.raises(PluginError, match="Failed to load plugin"):
            loader.load_plugin(BadPlugin, validate=True)
    
    def test_get_plugin_config(self):
        """Test getting plugin-specific configuration."""
        
        class TestPlugin(PluginBase):
            NAME = "test_plugin"
            CATEGORY = "test"
        
        loader = PluginLoader()
        
        # Test plugin-specific config
        config = {
            "plugins": {
                "test_plugin": {"plugin_setting": "value1"}
            },
            "test": {
                "test_plugin": {"category_setting": "value2"}
            }
        }
        
        plugin_config = loader._get_plugin_config(TestPlugin, config)
        assert plugin_config == {"plugin_setting": "value1"}
        
        # Test category-specific config when no plugin-specific config
        config = {
            "test": {
                "test_plugin": {"category_setting": "value2"}
            }
        }
        
        plugin_config = loader._get_plugin_config(TestPlugin, config)
        assert plugin_config == {"category_setting": "value2"}
    
    def test_get_loaded_plugin(self):
        """Test getting loaded plugin by name."""
        
        class TestPlugin(PluginBase):
            NAME = "test_plugin"
            
            def _initialize(self, context):
                pass
        
        loader = PluginLoader()
        
        # Load plugin
        plugin = loader.load_plugin(TestPlugin, validate=False)
        assert plugin is not None
        
        # Get loaded plugin
        loaded = loader.get_loaded_plugin("test_plugin")
        assert loaded is plugin
        
        # Test non-existent plugin
        assert loader.get_loaded_plugin("nonexistent") is None
    
    def test_list_loaded_plugins(self):
        """Test listing loaded plugins."""
        
        class Plugin1(PluginBase):
            NAME = "plugin1"
            def _initialize(self, context): pass
        
        class Plugin2(PluginBase):
            NAME = "plugin2"
            def _initialize(self, context): pass
        
        loader = PluginLoader()
        
        # Load plugins
        loader.load_plugin(Plugin1, validate=False)
        loader.load_plugin(Plugin2, validate=False)
        
        # List plugins
        loaded = loader.list_loaded_plugins()
        
        assert len(loaded) == 2
        assert "plugin1" in loaded
        assert "plugin2" in loaded
        assert loaded["plugin1"].name == "plugin1"
        assert loaded["plugin2"].name == "plugin2"