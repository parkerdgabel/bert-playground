"""Plugin registry adapter implementation."""

from typing import Any, Dict, List, Optional, Type

from infrastructure.plugins.registry import PluginRegistry as InfraRegistry
from infrastructure.plugins.base import PluginBase
from application.ports.secondary.plugins import Plugin, PluginMetadata


class PluginRegistryAdapter:
    """Adapter for plugin registry functionality."""
    
    def __init__(self):
        """Initialize plugin registry adapter."""
        self._registry = InfraRegistry()
        self._type_registry: Dict[Type[Plugin], List[str]] = {}
    
    def register(
        self,
        plugin: Plugin,
        plugin_type: Optional[Type[Plugin]] = None,
        force: bool = False,
    ) -> None:
        """Register a plugin.
        
        Args:
            plugin: Plugin to register
            plugin_type: Optional type classification
            force: Whether to overwrite existing plugin
        """
        # Convert to infrastructure plugin if needed
        if not isinstance(plugin, PluginBase):
            from .loader_adapter import PortPluginWrapper
            infra_plugin = PortPluginWrapper(plugin)
        else:
            infra_plugin = plugin
        
        # Register with infrastructure registry
        self._registry.register(infra_plugin, force=force)
        
        # Track type if specified
        if plugin_type:
            if plugin_type not in self._type_registry:
                self._type_registry[plugin_type] = []
            if plugin.name not in self._type_registry[plugin_type]:
                self._type_registry[plugin_type].append(plugin.name)
    
    def unregister(self, name: str) -> bool:
        """Unregister a plugin.
        
        Args:
            name: Plugin name
            
        Returns:
            True if plugin was unregistered
        """
        # Remove from type registry
        for type_list in self._type_registry.values():
            if name in type_list:
                type_list.remove(name)
        
        # Unregister from infrastructure
        return self._registry.unregister(name)
    
    def get(self, name: str) -> Optional[Plugin]:
        """Get a registered plugin by name.
        
        Args:
            name: Plugin name
            
        Returns:
            Plugin instance or None
        """
        infra_plugin = self._registry.get(name)
        if infra_plugin:
            from .loader_adapter import PluginAdapter
            return PluginAdapter(infra_plugin)
        return None
    
    def get_by_type(self, plugin_type: Type[Plugin]) -> List[Plugin]:
        """Get all plugins of a specific type.
        
        Args:
            plugin_type: Plugin type to filter by
            
        Returns:
            List of plugins of the specified type
        """
        plugins = []
        
        # Get names registered with this type
        if plugin_type in self._type_registry:
            for name in self._type_registry[plugin_type]:
                plugin = self.get(name)
                if plugin:
                    plugins.append(plugin)
        
        # Also check all plugins for type match
        for name, infra_plugin in self._registry.get_all().items():
            from .loader_adapter import PluginAdapter
            plugin = PluginAdapter(infra_plugin)
            if isinstance(plugin, plugin_type) and plugin not in plugins:
                plugins.append(plugin)
        
        return plugins
    
    def get_all(self) -> Dict[str, Plugin]:
        """Get all registered plugins.
        
        Returns:
            Dictionary mapping names to plugins
        """
        plugins = {}
        for name, infra_plugin in self._registry.get_all().items():
            from .loader_adapter import PluginAdapter
            plugins[name] = PluginAdapter(infra_plugin)
        return plugins
    
    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all registered plugins with metadata.
        
        Returns:
            List of plugin information dictionaries
        """
        plugin_list = []
        
        for name, plugin in self.get_all().items():
            metadata = plugin.get_metadata()
            info = {
                "name": name,
                "version": metadata.version,
                "description": metadata.description,
                "author": metadata.author,
                "tags": metadata.tags,
                "active": getattr(plugin, 'is_active', lambda: True)(),
                "type": self._get_plugin_type(plugin),
            }
            plugin_list.append(info)
        
        return plugin_list
    
    def has_plugin(self, name: str) -> bool:
        """Check if a plugin is registered.
        
        Args:
            name: Plugin name
            
        Returns:
            True if plugin is registered
        """
        return self._registry.has_plugin(name)
    
    def clear(self) -> None:
        """Clear all registered plugins."""
        self._registry.clear()
        self._type_registry.clear()
    
    def enable_plugin(self, name: str) -> bool:
        """Enable a plugin.
        
        Args:
            name: Plugin name
            
        Returns:
            True if plugin was enabled
        """
        plugin = self._registry.get(name)
        if plugin:
            plugin.activate()
            return True
        return False
    
    def disable_plugin(self, name: str) -> bool:
        """Disable a plugin.
        
        Args:
            name: Plugin name
            
        Returns:
            True if plugin was disabled
        """
        plugin = self._registry.get(name)
        if plugin:
            plugin.deactivate()
            return True
        return False
    
    def get_plugin_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a plugin.
        
        Args:
            name: Plugin name
            
        Returns:
            Plugin information or None
        """
        plugin = self.get(name)
        if not plugin:
            return None
        
        metadata = plugin.get_metadata()
        return {
            "name": name,
            "version": metadata.version,
            "description": metadata.description,
            "author": metadata.author,
            "tags": metadata.tags,
            "requirements": metadata.requirements,
            "config": plugin.config,
            "type": self._get_plugin_type(plugin),
            "active": getattr(plugin, 'is_active', lambda: True)(),
        }
    
    def _get_plugin_type(self, plugin: Plugin) -> str:
        """Determine plugin type string."""
        from application.ports.secondary.plugins import (
            HeadPlugin, AugmenterPlugin, FeatureExtractorPlugin,
            DataLoaderPlugin, ModelPlugin, MetricPlugin
        )
        
        if isinstance(plugin, HeadPlugin):
            return "head"
        elif isinstance(plugin, AugmenterPlugin):
            return "augmenter"
        elif isinstance(plugin, FeatureExtractorPlugin):
            return "feature_extractor"
        elif isinstance(plugin, DataLoaderPlugin):
            return "data_loader"
        elif isinstance(plugin, ModelPlugin):
            return "model"
        elif isinstance(plugin, MetricPlugin):
            return "metric"
        else:
            return "generic"