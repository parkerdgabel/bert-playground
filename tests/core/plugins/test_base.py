"""Tests for plugin base classes and protocols."""

import pytest
from typing import Dict, Any
from unittest.mock import Mock

from core.di import Container
from core.plugins.base import (
    Plugin,
    PluginBase,
    PluginContext,
    PluginError,
    PluginLifecycle,
    PluginMetadata,
    PluginState,
)


class TestPluginMetadata:
    """Test PluginMetadata model."""
    
    def test_minimal_metadata(self):
        """Test creating metadata with minimal fields."""
        metadata = PluginMetadata(name="test_plugin")
        
        assert metadata.name == "test_plugin"
        assert metadata.version == "1.0.0"  # default
        assert metadata.tags == []  # default
        assert metadata.requirements == []  # default
    
    def test_full_metadata(self):
        """Test creating metadata with all fields."""
        metadata = PluginMetadata(
            name="test_plugin",
            version="2.0.0",
            description="A test plugin",
            author="Test Author",
            email="test@example.com",
            tags=["test", "example"],
            requirements=["numpy>=1.20"],
            category="test",
            depends_on=["other_plugin"],
            conflicts_with=["bad_plugin"],
            provides=["test_capability"],
            consumes=["input_capability"],
        )
        
        assert metadata.name == "test_plugin"
        assert metadata.version == "2.0.0"
        assert metadata.description == "A test plugin"
        assert metadata.author == "Test Author"
        assert metadata.tags == ["test", "example"]
        assert metadata.depends_on == ["other_plugin"]
        assert metadata.provides == ["test_capability"]


class TestPluginContext:
    """Test PluginContext class."""
    
    def test_create_context(self):
        """Test creating a plugin context."""
        container = Container()
        config = {"key": "value"}
        
        context = PluginContext(
            container=container,
            config=config,
            state=PluginState.LOADED,
        )
        
        assert context.container is container
        assert context.config == config
        assert context.state == PluginState.LOADED
    
    def test_get_config(self):
        """Test getting configuration values."""
        config = {"key1": "value1", "key2": "value2"}
        context = PluginContext(container=Container(), config=config)
        
        assert context.get_config("key1") == "value1"
        assert context.get_config("key2") == "value2"
        assert context.get_config("missing", "default") == "default"
    
    def test_create_child(self):
        """Test creating child context."""
        parent_config = {"parent_key": "parent_value"}
        parent_context = PluginContext(
            container=Container(),
            config=parent_config,
        )
        
        child_config = {"child_key": "child_value"}
        child_context = parent_context.create_child(config=child_config)
        
        # Child should have both parent and child config
        assert child_context.get_config("parent_key") == "parent_value"
        assert child_context.get_config("child_key") == "child_value"
        assert child_context.parent_context is parent_context


class TestPluginLifecycle:
    """Test PluginLifecycle abstract base class."""
    
    def test_initial_state(self):
        """Test plugin starts in DISCOVERED state."""
        
        class TestPlugin(PluginLifecycle):
            def _create_metadata(self):
                return PluginMetadata(name="test")
            
            def _initialize(self, context):
                pass
        
        plugin = TestPlugin()
        assert plugin.state == PluginState.DISCOVERED
    
    def test_metadata_creation(self):
        """Test metadata is created lazily."""
        
        class TestPlugin(PluginLifecycle):
            def _create_metadata(self):
                return PluginMetadata(name="test", version="1.0.1")
            
            def _initialize(self, context):
                pass
        
        plugin = TestPlugin()
        metadata = plugin.metadata
        
        assert metadata.name == "test"
        assert metadata.version == "1.0.1"
    
    def test_validate_success(self):
        """Test successful validation."""
        
        class TestPlugin(PluginLifecycle):
            def _create_metadata(self):
                return PluginMetadata(name="test")
            
            def _validate(self, context):
                # Custom validation logic
                if not context.get_config("required_setting"):
                    raise PluginError("Missing required setting")
            
            def _initialize(self, context):
                pass
        
        plugin = TestPlugin()
        context = PluginContext(
            container=Container(),
            config={"required_setting": "value"}
        )
        
        plugin.validate(context)
        assert plugin.state == PluginState.VALIDATED
    
    def test_validate_failure(self):
        """Test validation failure."""
        
        class TestPlugin(PluginLifecycle):
            def _create_metadata(self):
                return PluginMetadata(name="test")
            
            def _validate(self, context):
                raise PluginError("Validation failed")
            
            def _initialize(self, context):
                pass
        
        plugin = TestPlugin()
        context = PluginContext(container=Container())
        
        with pytest.raises(PluginError, match="Validation failed"):
            plugin.validate(context)
    
    def test_initialization_success(self):
        """Test successful initialization."""
        
        class TestPlugin(PluginLifecycle):
            def _create_metadata(self):
                return PluginMetadata(name="test")
            
            def _initialize(self, context):
                self.initialized = True
            
            def _start(self, context):
                pass
        
        plugin = TestPlugin()
        plugin._state = PluginState.VALIDATED  # Skip validation
        context = PluginContext(container=Container())
        
        plugin.initialize(context)
        
        assert plugin.state == PluginState.INITIALIZED
        assert plugin.initialized is True
    
    def test_initialization_failure(self):
        """Test initialization failure."""
        
        class TestPlugin(PluginLifecycle):
            def _create_metadata(self):
                return PluginMetadata(name="test")
            
            def _initialize(self, context):
                raise ValueError("Init failed")
        
        plugin = TestPlugin()
        plugin._state = PluginState.VALIDATED
        context = PluginContext(container=Container())
        
        with pytest.raises(PluginError, match="Failed to initialize plugin"):
            plugin.initialize(context)
        
        assert plugin.state == PluginState.FAILED
    
    def test_start_success(self):
        """Test successful start."""
        
        class TestPlugin(PluginLifecycle):
            def _create_metadata(self):
                return PluginMetadata(name="test")
            
            def _initialize(self, context):
                pass
            
            def _start(self, context):
                self.started = True
        
        plugin = TestPlugin()
        plugin._state = PluginState.INITIALIZED
        context = PluginContext(container=Container())
        
        plugin.start(context)
        
        assert plugin.state == PluginState.RUNNING
        assert plugin.started is True
    
    def test_stop_and_cleanup(self):
        """Test stopping and cleanup."""
        
        class TestPlugin(PluginLifecycle):
            def _create_metadata(self):
                return PluginMetadata(name="test")
            
            def _initialize(self, context):
                pass
            
            def _start(self, context):
                pass
            
            def _stop(self, context):
                self.stopped = True
            
            def _cleanup(self, context):
                self.cleaned = True
        
        plugin = TestPlugin()
        plugin._state = PluginState.RUNNING
        context = PluginContext(container=Container())
        
        plugin.stop(context)
        assert plugin.state == PluginState.STOPPED
        assert plugin.stopped is True
        
        plugin.cleanup(context)
        assert plugin.state == PluginState.CLEANED_UP
        assert plugin.cleaned is True


class TestPluginBase:
    """Test PluginBase class."""
    
    def test_create_with_config(self):
        """Test creating plugin with configuration."""
        config = {"setting1": "value1", "setting2": "value2"}
        
        class TestPlugin(PluginBase):
            NAME = "test_plugin"
            VERSION = "1.0.0"
            
            def _initialize(self, context):
                pass
        
        plugin = TestPlugin(config=config)
        
        assert plugin.config == config
        assert plugin.metadata.name == "test_plugin"
        assert plugin.metadata.version == "1.0.0"
    
    def test_metadata_from_attributes(self):
        """Test metadata creation from class attributes."""
        
        class TestPlugin(PluginBase):
            NAME = "custom_plugin"
            VERSION = "2.1.0"
            DESCRIPTION = "A custom plugin"
            AUTHOR = "Test Author"
            EMAIL = "test@example.com"
            TAGS = ["custom", "test"]
            CATEGORY = "custom"
            DEPENDS_ON = ["dep1", "dep2"]
            
            def _initialize(self, context):
                pass
        
        plugin = TestPlugin()
        metadata = plugin.metadata
        
        assert metadata.name == "custom_plugin"
        assert metadata.version == "2.1.0"
        assert metadata.description == "A custom plugin"
        assert metadata.author == "Test Author"
        assert metadata.tags == ["custom", "test"]
        assert metadata.category == "custom"
        assert metadata.depends_on == ["dep1", "dep2"]
    
    def test_default_initialization(self):
        """Test that default initialization doesn't fail."""
        
        class TestPlugin(PluginBase):
            NAME = "test"
        
        plugin = TestPlugin()
        context = PluginContext(container=Container())
        
        plugin._state = PluginState.VALIDATED
        plugin.initialize(context)  # Should not raise
        
        assert plugin.state == PluginState.INITIALIZED


class TestPluginError:
    """Test PluginError exception."""
    
    def test_basic_error(self):
        """Test basic error creation."""
        error = PluginError("Something went wrong")
        
        assert str(error) == "Something went wrong"
        assert error.plugin_name is None
        assert error.cause is None
    
    def test_error_with_plugin_name(self):
        """Test error with plugin name."""
        error = PluginError("Something went wrong", plugin_name="test_plugin")
        
        assert str(error) == "Something went wrong"
        assert error.plugin_name == "test_plugin"
    
    def test_error_with_cause(self):
        """Test error with underlying cause."""
        cause = ValueError("Original error")
        error = PluginError("Wrapper error", cause=cause)
        
        assert str(error) == "Wrapper error"
        assert error.cause is cause