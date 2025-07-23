"""Tests for plugin registry."""

import pytest
from unittest.mock import Mock

from infrastructure.di import Container
from infrastructure.plugins.base import PluginBase, PluginContext, PluginState
from infrastructure.plugins.registry import PluginRegistry


class TestPluginRegistry:
    """Test PluginRegistry class."""
    
    @pytest.fixture
    def registry(self):
        """Create a plugin registry for testing."""
        container = Container()
        return PluginRegistry(container=container)
    
    @pytest.fixture
    def sample_plugin(self):
        """Create a sample plugin for testing."""
        
        class SamplePlugin(PluginBase):
            NAME = "sample_plugin"
            VERSION = "1.0.0"
            CATEGORY = "test"
            
            def _initialize(self, context):
                self.initialized = True
        
        return SamplePlugin()
    
    def test_register_plugin(self, registry, sample_plugin):
        """Test registering a plugin."""
        registry.register(sample_plugin, initialize=False)
        
        # Check plugin is registered
        retrieved = registry.get("sample_plugin")
        assert retrieved is sample_plugin
        
        # Check categories
        categories = registry.list_categories()
        assert "test" in categories
        assert "sample_plugin" in categories["test"]
    
    def test_register_plugin_with_initialization(self, registry, sample_plugin):
        """Test registering and initializing a plugin."""
        sample_plugin._state = PluginState.LOADED
        
        registry.register(sample_plugin, initialize=True)
        
        # Plugin should be initialized
        assert sample_plugin.state == PluginState.INITIALIZED
        assert hasattr(sample_plugin, "initialized")
        assert sample_plugin.initialized is True
    
    def test_register_duplicate_plugin(self, registry, sample_plugin):
        """Test registering duplicate plugin fails."""
        registry.register(sample_plugin, initialize=False)
        
        # Try to register again
        with pytest.raises(Exception):
            registry.register(sample_plugin, initialize=False)
    
    def test_unregister_plugin(self, registry, sample_plugin):
        """Test unregistering a plugin."""
        registry.register(sample_plugin, initialize=False)
        
        # Unregister
        unregistered = registry.unregister("sample_plugin")
        
        assert unregistered is sample_plugin
        assert registry.get("sample_plugin") is None
    
    def test_get_typed_plugin(self, registry):
        """Test getting plugin with type checking."""
        
        class TypedPlugin(PluginBase):
            NAME = "typed_plugin"
            
            def _initialize(self, context):
                pass
        
        plugin = TypedPlugin()
        registry.register(plugin, initialize=False)
        
        # Get with correct type
        retrieved = registry.get_typed("typed_plugin", TypedPlugin)
        assert retrieved is plugin
        
        # Get with incorrect type
        retrieved = registry.get_typed("typed_plugin", str)
        assert retrieved is None
    
    def test_get_by_category(self, registry):
        """Test getting plugins by category."""
        
        class Plugin1(PluginBase):
            NAME = "plugin1"
            CATEGORY = "category_a"
            def _initialize(self, context): pass
        
        class Plugin2(PluginBase):
            NAME = "plugin2"
            CATEGORY = "category_a"
            def _initialize(self, context): pass
        
        class Plugin3(PluginBase):
            NAME = "plugin3"
            CATEGORY = "category_b"
            def _initialize(self, context): pass
        
        # Register plugins
        plugin1, plugin2, plugin3 = Plugin1(), Plugin2(), Plugin3()
        registry.register(plugin1, initialize=False)
        registry.register(plugin2, initialize=False)
        registry.register(plugin3, initialize=False)
        
        # Get by category
        category_a_plugins = registry.get_by_category("category_a")
        assert len(category_a_plugins) == 2
        assert plugin1 in category_a_plugins
        assert plugin2 in category_a_plugins
        
        category_b_plugins = registry.get_by_category("category_b")
        assert len(category_b_plugins) == 1
        assert plugin3 in category_b_plugins
        
        # Non-existent category
        empty_plugins = registry.get_by_category("nonexistent")
        assert empty_plugins == []
    
    def test_start_stop_plugin(self, registry, sample_plugin):
        """Test starting and stopping plugins."""
        sample_plugin._state = PluginState.LOADED
        registry.register(sample_plugin, initialize=False)
        
        # Start plugin
        registry.start_plugin("sample_plugin")
        assert sample_plugin.state == PluginState.RUNNING
        
        # Stop plugin
        registry.stop_plugin("sample_plugin")
        assert sample_plugin.state == PluginState.STOPPED
    
    def test_start_nonexistent_plugin(self, registry):
        """Test starting non-existent plugin fails."""
        with pytest.raises(Exception, match="not found"):
            registry.start_plugin("nonexistent")
    
    def test_start_all_plugins(self, registry):
        """Test starting all plugins."""
        
        class Plugin1(PluginBase):
            NAME = "plugin1"
            def _initialize(self, context): pass
            def _start(self, context): self.started = True
        
        class Plugin2(PluginBase):
            NAME = "plugin2"
            def _initialize(self, context): pass  
            def _start(self, context): self.started = True
        
        plugin1, plugin2 = Plugin1(), Plugin2()
        plugin1._state = PluginState.LOADED
        plugin2._state = PluginState.LOADED
        
        registry.register(plugin1, initialize=False)
        registry.register(plugin2, initialize=False)
        
        # Start all
        registry.start_all()
        
        assert plugin1.state == PluginState.RUNNING
        assert plugin2.state == PluginState.RUNNING
        assert hasattr(plugin1, "started") and plugin1.started
        assert hasattr(plugin2, "started") and plugin2.started
    
    def test_start_all_by_category(self, registry):
        """Test starting all plugins in a category."""
        
        class PluginA(PluginBase):
            NAME = "plugin_a"
            CATEGORY = "category_a"
            def _initialize(self, context): pass
            def _start(self, context): self.started = True
        
        class PluginB(PluginBase):
            NAME = "plugin_b"
            CATEGORY = "category_b"
            def _initialize(self, context): pass
            def _start(self, context): self.started = True
        
        plugin_a, plugin_b = PluginA(), PluginB()
        plugin_a._state = PluginState.LOADED
        plugin_b._state = PluginState.LOADED
        
        registry.register(plugin_a, initialize=False)
        registry.register(plugin_b, initialize=False)
        
        # Start only category_a
        registry.start_all(category="category_a")
        
        assert plugin_a.state == PluginState.RUNNING
        assert plugin_b.state == PluginState.LOADED  # Not started
        assert hasattr(plugin_a, "started") and plugin_a.started
        assert not hasattr(plugin_b, "started")
    
    def test_list_plugins(self, registry):
        """Test listing all plugins."""
        
        class Plugin1(PluginBase):
            NAME = "plugin1"
            VERSION = "1.0.0"
            def _initialize(self, context): pass
        
        class Plugin2(PluginBase):
            NAME = "plugin2"
            VERSION = "2.0.0"
            def _initialize(self, context): pass
        
        plugin1, plugin2 = Plugin1(), Plugin2()
        registry.register(plugin1, initialize=False)
        registry.register(plugin2, initialize=False)
        
        plugins = registry.list_plugins()
        
        assert len(plugins) == 2
        assert "plugin1" in plugins
        assert "plugin2" in plugins
        assert plugins["plugin1"].name == "plugin1"
        assert plugins["plugin1"].version == "1.0.0"
        assert plugins["plugin2"].name == "plugin2"
        assert plugins["plugin2"].version == "2.0.0"
    
    def test_list_categories(self, registry):
        """Test listing categories."""
        
        class PluginA1(PluginBase):
            NAME = "plugin_a1"
            CATEGORY = "category_a"
            def _initialize(self, context): pass
        
        class PluginA2(PluginBase):
            NAME = "plugin_a2"
            CATEGORY = "category_a"
            def _initialize(self, context): pass
        
        class PluginB(PluginBase):
            NAME = "plugin_b"
            CATEGORY = "category_b"
            def _initialize(self, context): pass
        
        # Register plugins
        for plugin_class in [PluginA1, PluginA2, PluginB]:
            plugin = plugin_class()
            registry.register(plugin, initialize=False)
        
        categories = registry.list_categories()
        
        assert len(categories) == 2
        assert "category_a" in categories
        assert "category_b" in categories
        assert set(categories["category_a"]) == {"plugin_a1", "plugin_a2"}
        assert categories["category_b"] == ["plugin_b"]
    
    def test_load_and_register(self, registry):
        """Test loading and registering plugins."""
        
        # Mock the loader
        mock_loader = Mock()
        mock_plugin = Mock()
        mock_plugin.metadata.name = "mock_plugin"
        
        mock_loader.load_plugins.return_value = {"mock_plugin": mock_plugin}
        registry.loader = mock_loader
        
        # Load and register
        result = registry.load_and_register(
            project_root="/tmp/project",
            config={"test": "config"},
            validate=True,
            initialize=False,
        )
        
        # Check loader was called correctly
        mock_loader.load_plugins.assert_called_once_with(
            project_root="/tmp/project",
            additional_paths=None,
            config={"test": "config"},
            validate=True,
        )
        
        # Check plugin was registered
        assert len(result) == 1
        assert "mock_plugin" in result