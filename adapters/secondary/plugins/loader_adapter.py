"""Plugin loader adapter implementation."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from infrastructure.plugins.loader import PluginDiscovery, PluginLoader
from infrastructure.plugins.registry import PluginRegistry
from infrastructure.plugins.base import PluginBase, PluginMetadata as InfraPluginMetadata
from application.ports.secondary.plugins import (
    Plugin, PluginMetadata, HeadPlugin, AugmenterPlugin,
    FeatureExtractorPlugin, DataLoaderPlugin, ModelPlugin, MetricPlugin
)


class PluginLoaderAdapter:
    """Adapter for plugin loading functionality."""
    
    def __init__(self, registry: Optional[PluginRegistry] = None):
        """Initialize plugin loader adapter.
        
        Args:
            registry: Optional plugin registry to use
        """
        self._discovery = PluginDiscovery()
        self._loader = PluginLoader()
        self._registry = registry or PluginRegistry()
        
        # Map plugin protocol types to their string identifiers
        self._plugin_type_map = {
            HeadPlugin: "head",
            AugmenterPlugin: "augmenter",
            FeatureExtractorPlugin: "feature_extractor",
            DataLoaderPlugin: "data_loader",
            ModelPlugin: "model",
            MetricPlugin: "metric",
        }
    
    def discover_plugins(
        self,
        project_path: Optional[Union[str, Path]] = None,
        extra_paths: Optional[List[Union[str, Path]]] = None,
    ) -> List[Plugin]:
        """Discover all available plugins.
        
        Args:
            project_path: Optional project path to search
            extra_paths: Additional paths to search
            
        Returns:
            List of discovered plugins
        """
        discovered = []
        
        # Discover from project
        if project_path:
            project_plugins = self._discovery.discover_project_plugins(Path(project_path))
            discovered.extend(self._convert_plugins(project_plugins))
        
        # Discover from extra paths
        if extra_paths:
            for path in extra_paths:
                path_plugins = self._discovery.discover_directory_plugins(Path(path))
                discovered.extend(self._convert_plugins(path_plugins))
        
        # Discover from entry points
        entry_plugins = self._discovery.discover_entry_point_plugins()
        discovered.extend(self._convert_plugins(entry_plugins))
        
        return discovered
    
    def load_plugin(
        self,
        plugin_path: Union[str, Path],
        plugin_type: Optional[Type[Plugin]] = None,
    ) -> Plugin:
        """Load a specific plugin.
        
        Args:
            plugin_path: Path to plugin module or file
            plugin_type: Expected plugin type for validation
            
        Returns:
            Loaded plugin instance
        """
        # Load using infrastructure loader
        infra_plugin = self._loader.load_plugin(plugin_path)
        
        # Convert to port plugin
        plugin = self._convert_plugin(infra_plugin)
        
        # Validate type if specified
        if plugin_type and not isinstance(plugin, plugin_type):
            raise TypeError(
                f"Plugin {plugin.name} is not of expected type {plugin_type.__name__}"
            )
        
        return plugin
    
    def load_plugins_by_type(
        self,
        plugin_type: Type[Plugin],
        project_path: Optional[Union[str, Path]] = None,
    ) -> List[Plugin]:
        """Load all plugins of a specific type.
        
        Args:
            plugin_type: Type of plugins to load
            project_path: Optional project path to search
            
        Returns:
            List of loaded plugins of the specified type
        """
        # Get type identifier
        type_id = self._plugin_type_map.get(plugin_type, "plugin")
        
        # Discover all plugins
        all_plugins = self.discover_plugins(project_path)
        
        # Filter by type
        typed_plugins = []
        for plugin in all_plugins:
            if self._matches_type(plugin, plugin_type):
                typed_plugins.append(plugin)
        
        return typed_plugins
    
    def register_plugin(self, plugin: Plugin) -> None:
        """Register a plugin with the registry.
        
        Args:
            plugin: Plugin to register
        """
        # Convert to infrastructure plugin for registration
        infra_plugin = self._to_infra_plugin(plugin)
        self._registry.register(infra_plugin)
    
    def get_registered_plugins(self) -> Dict[str, Plugin]:
        """Get all registered plugins.
        
        Returns:
            Dictionary mapping plugin names to instances
        """
        registered = {}
        for name, infra_plugin in self._registry.get_all().items():
            registered[name] = self._convert_plugin(infra_plugin)
        return registered
    
    def validate_plugin(self, plugin: Plugin) -> List[str]:
        """Validate a plugin.
        
        Args:
            plugin: Plugin to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check basic requirements
        if not hasattr(plugin, 'name') or not plugin.name:
            errors.append("Plugin must have a name")
        
        if not hasattr(plugin, 'get_metadata'):
            errors.append("Plugin must implement get_metadata()")
        
        # Type-specific validation
        if isinstance(plugin, HeadPlugin):
            if not hasattr(plugin, 'forward'):
                errors.append("HeadPlugin must implement forward()")
            if not hasattr(plugin, 'compute_loss'):
                errors.append("HeadPlugin must implement compute_loss()")
        
        elif isinstance(plugin, AugmenterPlugin):
            if not hasattr(plugin, 'augment'):
                errors.append("AugmenterPlugin must implement augment()")
        
        elif isinstance(plugin, FeatureExtractorPlugin):
            if not hasattr(plugin, 'extract_features'):
                errors.append("FeatureExtractorPlugin must implement extract_features()")
        
        elif isinstance(plugin, DataLoaderPlugin):
            if not hasattr(plugin, 'load_data'):
                errors.append("DataLoaderPlugin must implement load_data()")
        
        elif isinstance(plugin, ModelPlugin):
            if not hasattr(plugin, 'create_model'):
                errors.append("ModelPlugin must implement create_model()")
        
        elif isinstance(plugin, MetricPlugin):
            if not hasattr(plugin, 'compute'):
                errors.append("MetricPlugin must implement compute()")
        
        return errors
    
    # Helper methods
    
    def _convert_plugins(self, infra_plugins: List[PluginBase]) -> List[Plugin]:
        """Convert infrastructure plugins to port plugins."""
        return [self._convert_plugin(p) for p in infra_plugins]
    
    def _convert_plugin(self, infra_plugin: PluginBase) -> Plugin:
        """Convert a single infrastructure plugin to port plugin."""
        # Create a wrapper that implements the Plugin protocol
        return PluginAdapter(infra_plugin)
    
    def _to_infra_plugin(self, plugin: Plugin) -> PluginBase:
        """Convert port plugin to infrastructure plugin."""
        if isinstance(plugin, PluginAdapter):
            return plugin._wrapped
        
        # Create infrastructure plugin wrapper
        return PortPluginWrapper(plugin)
    
    def _matches_type(self, plugin: Plugin, plugin_type: Type[Plugin]) -> bool:
        """Check if plugin matches the specified type."""
        return isinstance(plugin, plugin_type)


class PluginAdapter:
    """Adapter that wraps infrastructure plugins to implement port protocols."""
    
    def __init__(self, wrapped: PluginBase):
        """Initialize adapter with wrapped plugin."""
        self._wrapped = wrapped
    
    @property
    def config(self) -> dict[str, Any]:
        """Plugin configuration."""
        return self._wrapped.config
    
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        infra_meta = self._wrapped.get_metadata()
        return PluginMetadata(
            name=infra_meta.name,
            version=infra_meta.version,
            description=infra_meta.description,
            author=infra_meta.author,
            tags=infra_meta.tags,
            requirements=infra_meta.requirements,
        )
    
    @property
    def name(self) -> str:
        """Get plugin name."""
        return self._wrapped.name
    
    @property
    def version(self) -> str:
        """Get plugin version."""
        return self._wrapped.version
    
    def __getattr__(self, name: str) -> Any:
        """Forward unknown attributes to wrapped plugin."""
        return getattr(self._wrapped, name)


class PortPluginWrapper(PluginBase):
    """Wrapper that makes port plugins compatible with infrastructure."""
    
    def __init__(self, plugin: Plugin):
        """Initialize wrapper with port plugin."""
        metadata = plugin.get_metadata()
        super().__init__(
            name=plugin.name,
            version=plugin.version,
            config=plugin.config,
        )
        self._plugin = plugin
        self._metadata = InfraPluginMetadata(
            name=metadata.name,
            version=metadata.version,
            description=metadata.description,
            author=metadata.author,
            tags=metadata.tags,
            requirements=metadata.requirements,
        )
    
    def get_metadata(self) -> InfraPluginMetadata:
        """Get plugin metadata."""
        return self._metadata
    
    def activate(self) -> None:
        """Activate the plugin."""
        if hasattr(self._plugin, 'activate'):
            self._plugin.activate()
    
    def deactivate(self) -> None:
        """Deactivate the plugin."""
        if hasattr(self._plugin, 'deactivate'):
            self._plugin.deactivate()
    
    def validate(self) -> bool:
        """Validate plugin configuration and requirements."""
        if hasattr(self._plugin, 'validate'):
            return self._plugin.validate()
        return True