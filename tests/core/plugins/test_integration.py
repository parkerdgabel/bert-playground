"""Tests for plugin system integration."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from core.di import Container
from core.plugins.base import PluginBase
from core.plugins.integration import (
    PluginSystemIntegration,
    setup_plugin_system,
    ensure_plugins_loaded,
)


class TestPluginSystemIntegration:
    """Test PluginSystemIntegration class."""
    
    @pytest.fixture
    def integration(self):
        """Create integration instance for testing."""
        container = Container()
        return PluginSystemIntegration(container=container)
    
    def test_create_wrapper_plugin(self, integration):
        """Test creating wrapper for legacy component."""
        
        # Mock legacy component  
        mock_component = Mock()
        mock_component.__class__.__name__ = "MockComponent"
        mock_component.name = "mock_component"
        # Delete Mock's version attribute so getattr works properly
        if hasattr(mock_component, 'version'):
            del mock_component.version
        
        # Create wrapper
        wrapper = integration._create_wrapper_plugin(mock_component, "test")
        
        assert wrapper is not None
        assert wrapper.metadata.name == "mock_component"
        assert wrapper.metadata.version == "legacy"
        assert wrapper.metadata.category == "test"
        assert "legacy" in wrapper.metadata.tags
        
        # Test getting original component
        assert wrapper.get_component() is mock_component
    
    def test_migrate_from_old_system(self, integration):
        """Test migration from old plugin system."""
        
        # Mock old registry with components
        mock_old_component = Mock()
        mock_old_component.__class__.__name__ = "OldComponent"
        mock_old_component.name = "old_component"
        
        # Initialize old registry and loader with mocks
        integration.old_registry = Mock()
        integration.old_loader = Mock()
        
        integration.old_registry.list_components.return_value = {
            "head": ["test_head"],
            "augmenter": ["test_augmenter"],
        }
        
        integration.old_registry.get.side_effect = lambda comp_type, name: mock_old_component
        integration.old_loader.load_project_plugins.return_value = {}
        
        # Migrate
        migrated = integration.migrate_from_old_system()
        
        # Should create wrapper plugins
        assert len(migrated) == 2
        assert "head_test_head" in migrated
        assert "augmenter_test_augmenter" in migrated
        
        # Check plugins are registered in new system
        new_plugins = integration.new_registry.list_plugins()
        assert "head_test_head" in new_plugins
        assert "augmenter_test_augmenter" in new_plugins
    
    def test_setup_unified_system(self, integration):
        """Test setting up the unified system."""
        
        # Mock loader
        mock_plugin = Mock()
        mock_plugin.metadata.name = "test_plugin"
        
        # Mock the load_and_register method instead
        integration.new_registry.load_and_register = Mock()
        integration.new_registry.load_and_register.return_value = {
            "test_plugin": mock_plugin
        }
        
        # Mock old system
        integration.old_registry = Mock()
        integration.old_loader = Mock()
        integration.old_registry.list_components.return_value = {}
        integration.old_loader.load_project_plugins.return_value = {}
        
        # Setup system
        registry = integration.setup_unified_system(
            project_root="/tmp/project",
            migrate_old=False,
        )
        
        # Should return the registry
        assert registry is integration.new_registry


class TestGlobalFunctions:
    """Test global integration functions."""
    
    def test_setup_plugin_system(self):
        """Test setup_plugin_system function."""
        
        with patch('core.plugins.integration.get_integration') as mock_get_integration:
            mock_integration = Mock()
            mock_registry = Mock()
            mock_integration.setup_unified_system.return_value = mock_registry
            mock_get_integration.return_value = mock_integration
            
            # Call function
            result = setup_plugin_system(
                project_root="/tmp/test",
                config_dict={"test": "config"},
                migrate_old=False,
            )
            
            # Check integration was called correctly
            mock_integration.setup_unified_system.assert_called_once_with(
                project_root="/tmp/test",
                config_file=None,
                config_dict={"test": "config"},
                migrate_old=False,
            )
            
            assert result is mock_registry
    
    def test_ensure_plugins_loaded_already_loaded(self):
        """Test ensure_plugins_loaded when plugins already exist."""
        
        with patch('core.plugins.integration.get_integration') as mock_get_integration:
            mock_integration = Mock()
            mock_registry = Mock()
            mock_registry.list_plugins.return_value = {"existing": Mock()}
            mock_integration.get_integrated_registry.return_value = mock_registry
            mock_get_integration.return_value = mock_integration
            
            # Call function
            ensure_plugins_loaded("/tmp/project")
            
            # Should not call setup since plugins exist
            mock_integration.setup_unified_system.assert_not_called()
    
    def test_ensure_plugins_loaded_no_plugins(self):
        """Test ensure_plugins_loaded when no plugins exist."""
        
        with patch('core.plugins.integration.get_integration') as mock_get_integration, \
             patch('core.plugins.integration.setup_plugin_system') as mock_setup:
            
            mock_integration = Mock()
            mock_registry = Mock()
            mock_registry.list_plugins.return_value = {}  # No plugins
            mock_integration.get_integrated_registry.return_value = mock_registry
            mock_get_integration.return_value = mock_integration
            
            # Call function
            ensure_plugins_loaded("/tmp/project")
            
            # Should call setup
            mock_setup.assert_called_once_with(project_root="/tmp/project")


class TestBackwardsCompatibility:
    """Test backwards compatibility functions."""
    
    def test_get_component_registry_deprecation(self):
        """Test that get_component_registry issues deprecation warning."""
        
        with patch('core.plugins.integration.get_integration') as mock_get_integration:
            mock_integration = Mock()
            mock_get_integration.return_value = mock_integration
            
            with pytest.warns(DeprecationWarning, match="deprecated"):
                from core.plugins.integration import get_component_registry
                registry = get_component_registry()
                
                assert registry is mock_integration.old_registry
    
    def test_load_project_plugins_deprecation(self):
        """Test that load_project_plugins issues deprecation warning."""
        
        with patch('core.plugins.integration.setup_plugin_system') as mock_setup:
            mock_setup.return_value = Mock()
            
            with pytest.warns(DeprecationWarning, match="deprecated"):
                from core.plugins.integration import load_project_plugins
                result = load_project_plugins("/tmp/project")
                
                mock_setup.assert_called_once_with(project_root="/tmp/project")


@pytest.fixture
def temp_project_with_plugins():
    """Create temporary project with plugin files."""
    import shutil
    
    temp_dir = Path(tempfile.mkdtemp())
    
    # Create project structure
    src_dir = temp_dir / "src" / "plugins"
    src_dir.mkdir(parents=True)
    
    # Create new-style plugin
    new_plugin_code = '''
from core.plugins.base import PluginBase

class NewStylePlugin(PluginBase):
    NAME = "new_style_plugin"
    VERSION = "1.0.0"
    CATEGORY = "test"
    
    def _initialize(self, context):
        self.initialized = True
'''
    (src_dir / "new_plugin.py").write_text(new_plugin_code)
    
    # Create k-bert.yaml config
    config_content = '''
plugins:
  auto_initialize: true
  auto_start: false
  configs:
    new_style_plugin:
      test_setting: "test_value"
'''
    (temp_dir / "k-bert.yaml").write_text(config_content)
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


class TestEndToEndIntegration:
    """Test end-to-end integration scenarios."""
    
    def test_full_integration_flow(self, temp_project_with_plugins):
        """Test complete integration flow."""
        
        # Setup plugin system for project
        registry = setup_plugin_system(
            project_root=temp_project_with_plugins,
            migrate_old=False,  # No old plugins in test
        )
        
        # Should have loaded the new plugin
        plugins = registry.list_plugins()
        assert len(plugins) >= 1
        
        # Should have correct categories
        categories = registry.list_categories()
        assert "test" in categories
        
        # Plugin should be initialized (based on config)
        test_plugins = registry.get_by_category("test")
        assert len(test_plugins) >= 1
        
        # Find our specific plugin
        new_plugin = None
        for plugin in test_plugins:
            if plugin.metadata.name == "new_style_plugin":
                new_plugin = plugin
                break
        
        assert new_plugin is not None
        assert new_plugin.metadata.version == "1.0.0"
        # Should be initialized due to auto_initialize: true
        # (depends on actual plugin loading logic)
    
    def test_idempotent_loading(self, temp_project_with_plugins):
        """Test that multiple calls to ensure_plugins_loaded are safe."""
        from core.plugins.integration import get_integration, ensure_plugins_loaded
        
        # First call
        ensure_plugins_loaded(temp_project_with_plugins)
        integration = get_integration()
        plugins1 = integration.get_integrated_registry().list_plugins()
        
        # Second call - should not duplicate plugins
        ensure_plugins_loaded(temp_project_with_plugins)
        plugins2 = integration.get_integrated_registry().list_plugins()
        
        # Should have same plugins
        assert len(plugins1) == len(plugins2)
        for name in plugins1:
            assert name in plugins2