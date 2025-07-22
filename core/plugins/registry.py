"""Plugin registry for managing and accessing loaded plugins.

This module provides:
- Central registry for all plugins
- Plugin categorization and lookup
- Integration with dependency injection
- Plugin lifecycle management
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from loguru import logger

from core.di import Container, get_container
from core.protocols.plugins import (
    AugmenterPlugin,
    DataLoaderPlugin,
    FeatureExtractorPlugin,
    HeadPlugin,
    MetricPlugin,
    ModelPlugin,
)

from .base import Plugin, PluginContext, PluginError, PluginMetadata, PluginState
from .loader import PluginLoader

T = TypeVar("T", bound=Plugin)


class PluginRegistry:
    """Central registry for managing plugins."""
    
    # Category mappings for protocol-based plugins
    PROTOCOL_CATEGORIES = {
        HeadPlugin: "head",
        AugmenterPlugin: "augmenter",
        FeatureExtractorPlugin: "feature_extractor",
        DataLoaderPlugin: "data_loader",
        ModelPlugin: "model",
        MetricPlugin: "metric",
    }
    
    def __init__(self, container: Optional[Container] = None):
        """Initialize plugin registry.
        
        Args:
            container: DI container (uses global if not provided)
        """
        self.container = container or get_container()
        self._plugins: Dict[str, Plugin] = {}
        self._categories: Dict[str, List[str]] = {}
        self._contexts: Dict[str, PluginContext] = {}
        self.loader = PluginLoader()
    
    def register(
        self,
        plugin: Plugin,
        name: Optional[str] = None,
        category: Optional[str] = None,
        context: Optional[PluginContext] = None,
        initialize: bool = True,
    ) -> None:
        """Register a plugin instance.
        
        Args:
            plugin: Plugin instance
            name: Override plugin name
            category: Override plugin category
            context: Plugin context
            initialize: Whether to initialize the plugin
            
        Raises:
            PluginError: If registration fails
        """
        # Get plugin name
        plugin_name = name or plugin.metadata.name
        
        # Check if already registered
        if plugin_name in self._plugins:
            raise PluginError(f"Plugin '{plugin_name}' is already registered")
        
        # Determine category
        if category is None:
            category = self._determine_category(plugin)
        
        # Create context if not provided
        if context is None:
            context = PluginContext(
                container=self.container,
                metadata=plugin.metadata,
                state=plugin.state,
            )
        
        # Store plugin and context
        self._plugins[plugin_name] = plugin
        self._contexts[plugin_name] = context
        
        # Update category index
        if category:
            if category not in self._categories:
                self._categories[category] = []
            self._categories[category].append(plugin_name)
        
        # Register with DI container
        self._register_with_container(plugin, plugin_name, category)
        
        # Initialize if requested
        if initialize and plugin.state == PluginState.LOADED:
            try:
                plugin.validate(context)
                plugin.initialize(context)
                logger.info(f"Registered and initialized plugin: {plugin_name}")
            except Exception as e:
                # Clean up on failure
                self.unregister(plugin_name)
                raise PluginError(
                    f"Failed to initialize plugin '{plugin_name}': {e}",
                    plugin_name=plugin_name,
                    cause=e
                )
        else:
            logger.info(f"Registered plugin: {plugin_name}")
    
    def unregister(self, name: str) -> Optional[Plugin]:
        """Unregister a plugin.
        
        Args:
            name: Plugin name
            
        Returns:
            Unregistered plugin or None
        """
        if name not in self._plugins:
            return None
        
        plugin = self._plugins[name]
        context = self._contexts.get(name)
        
        # Stop and cleanup if needed
        if context and plugin.state == PluginState.RUNNING:
            try:
                plugin.stop(context)
                plugin.cleanup(context)
            except Exception as e:
                logger.error(f"Error during plugin cleanup: {e}")
        
        # Remove from registry
        del self._plugins[name]
        if name in self._contexts:
            del self._contexts[name]
        
        # Remove from categories
        for category_plugins in self._categories.values():
            if name in category_plugins:
                category_plugins.remove(name)
        
        logger.info(f"Unregistered plugin: {name}")
        return plugin
    
    def get(self, name: str) -> Optional[Plugin]:
        """Get a plugin by name.
        
        Args:
            name: Plugin name
            
        Returns:
            Plugin instance or None
        """
        return self._plugins.get(name)
    
    def get_typed(self, name: str, plugin_type: Type[T]) -> Optional[T]:
        """Get a plugin with type checking.
        
        Args:
            name: Plugin name
            plugin_type: Expected plugin type
            
        Returns:
            Typed plugin instance or None
        """
        plugin = self.get(name)
        if plugin and isinstance(plugin, plugin_type):
            return plugin
        return None
    
    def get_by_category(self, category: str) -> List[Plugin]:
        """Get all plugins in a category.
        
        Args:
            category: Plugin category
            
        Returns:
            List of plugins
        """
        plugin_names = self._categories.get(category, [])
        return [self._plugins[name] for name in plugin_names if name in self._plugins]
    
    def get_context(self, name: str) -> Optional[PluginContext]:
        """Get the context for a plugin.
        
        Args:
            name: Plugin name
            
        Returns:
            Plugin context or None
        """
        return self._contexts.get(name)
    
    def list_plugins(self) -> Dict[str, PluginMetadata]:
        """List all registered plugins.
        
        Returns:
            Dictionary of plugin name to metadata
        """
        return {
            name: plugin.metadata
            for name, plugin in self._plugins.items()
        }
    
    def list_categories(self) -> Dict[str, List[str]]:
        """List all categories and their plugins.
        
        Returns:
            Dictionary of category to plugin names
        """
        return {
            category: names.copy()
            for category, names in self._categories.items()
            if names
        }
    
    def start_plugin(self, name: str) -> None:
        """Start a plugin.
        
        Args:
            name: Plugin name
            
        Raises:
            PluginError: If plugin not found or start fails
        """
        plugin = self.get(name)
        if not plugin:
            raise PluginError(f"Plugin '{name}' not found")
        
        context = self.get_context(name)
        if not context:
            raise PluginError(f"No context for plugin '{name}'")
        
        # Initialize if needed
        if plugin.state == PluginState.LOADED:
            plugin.validate(context)
            plugin.initialize(context)
        
        # Start plugin
        plugin.start(context)
        logger.info(f"Started plugin: {name}")
    
    def stop_plugin(self, name: str) -> None:
        """Stop a plugin.
        
        Args:
            name: Plugin name
            
        Raises:
            PluginError: If plugin not found
        """
        plugin = self.get(name)
        if not plugin:
            raise PluginError(f"Plugin '{name}' not found")
        
        context = self.get_context(name)
        if context:
            plugin.stop(context)
            logger.info(f"Stopped plugin: {name}")
    
    def start_all(self, category: Optional[str] = None) -> None:
        """Start all plugins or plugins in a category.
        
        Args:
            category: Optional category filter
        """
        if category:
            plugins = self.get_by_category(category)
            plugin_names = [p.metadata.name for p in plugins]
        else:
            plugin_names = list(self._plugins.keys())
        
        for name in plugin_names:
            try:
                self.start_plugin(name)
            except Exception as e:
                logger.error(f"Failed to start plugin '{name}': {e}")
    
    def stop_all(self, category: Optional[str] = None) -> None:
        """Stop all plugins or plugins in a category.
        
        Args:
            category: Optional category filter
        """
        if category:
            plugins = self.get_by_category(category)
            plugin_names = [p.metadata.name for p in plugins]
        else:
            plugin_names = list(self._plugins.keys())
        
        for name in plugin_names:
            try:
                self.stop_plugin(name)
            except Exception as e:
                logger.error(f"Failed to stop plugin '{name}': {e}")
    
    def load_and_register(
        self,
        project_root: Optional[Union[str, Path]] = None,
        additional_paths: Optional[List[Union[str, Path]]] = None,
        config: Optional[Dict[str, Any]] = None,
        validate: bool = True,
        initialize: bool = True,
    ) -> Dict[str, Plugin]:
        """Load and register plugins from various sources.
        
        Args:
            project_root: Project root directory
            additional_paths: Additional paths to search
            config: Configuration for plugins
            validate: Whether to validate plugins
            initialize: Whether to initialize plugins
            
        Returns:
            Dictionary of loaded plugins
        """
        # Load plugins
        loaded_plugins = self.loader.load_plugins(
            project_root=project_root,
            additional_paths=additional_paths,
            config=config,
            validate=validate,
        )
        
        # Register each plugin
        registered = {}
        for name, plugin in loaded_plugins.items():
            try:
                self.register(
                    plugin=plugin,
                    name=name,
                    initialize=initialize,
                )
                registered[name] = plugin
            except Exception as e:
                logger.error(f"Failed to register plugin '{name}': {e}")
        
        return registered
    
    def _determine_category(self, plugin: Plugin) -> Optional[str]:
        """Determine the category for a plugin.
        
        Args:
            plugin: Plugin instance
            
        Returns:
            Category name or None
        """
        # Check metadata first
        if plugin.metadata.category:
            return plugin.metadata.category
        
        # Check protocol implementations
        for protocol_class, category in self.PROTOCOL_CATEGORIES.items():
            if isinstance(plugin, protocol_class):
                return category
        
        # Check class attributes
        if hasattr(plugin, "CATEGORY"):
            return plugin.CATEGORY
        
        return None
    
    def _register_with_container(
        self,
        plugin: Plugin,
        name: str,
        category: Optional[str]
    ) -> None:
        """Register plugin with DI container.
        
        Args:
            plugin: Plugin instance
            name: Plugin name
            category: Plugin category
        """
        # Register by name
        self.container.register(
            service_type=type(f"plugin_{name}", (), {}),
            implementation=plugin,
            instance=True,
        )
        
        # Register by protocol if applicable
        for protocol_class, protocol_category in self.PROTOCOL_CATEGORIES.items():
            if category == protocol_category and isinstance(plugin, protocol_class):
                # Register with a factory that returns the instance
                self.container.register(
                    service_type=protocol_class,
                    implementation=lambda: plugin,
                    factory=True,
                )
                break


# Global registry instance
_registry: Optional[PluginRegistry] = None


def get_registry() -> PluginRegistry:
    """Get the global plugin registry.
    
    Returns:
        Global PluginRegistry instance
    """
    global _registry
    if _registry is None:
        _registry = PluginRegistry()
    return _registry


def register_plugin(
    plugin: Plugin,
    name: Optional[str] = None,
    category: Optional[str] = None,
    initialize: bool = True,
) -> None:
    """Register a plugin with the global registry.
    
    Args:
        plugin: Plugin instance
        name: Override plugin name
        category: Override plugin category
        initialize: Whether to initialize the plugin
    """
    registry = get_registry()
    registry.register(plugin, name, category, initialize=initialize)


def get_plugin(name: str) -> Optional[Plugin]:
    """Get a plugin from the global registry.
    
    Args:
        name: Plugin name
        
    Returns:
        Plugin instance or None
    """
    registry = get_registry()
    return registry.get(name)


def list_plugins() -> Dict[str, PluginMetadata]:
    """List all plugins in the global registry.
    
    Returns:
        Dictionary of plugin name to metadata
    """
    registry = get_registry()
    return registry.list_plugins()