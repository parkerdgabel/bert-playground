"""Plugin interface for custom augmentation strategies."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
from pathlib import Path
import importlib.util
import inspect

from loguru import logger

from .base import BaseAugmenter, BaseFeatureAugmenter, BaseAugmentationStrategy
from .registry import get_registry


class AugmentationPlugin(ABC):
    """Base class for augmentation plugins."""
    
    def __init__(self, name: str, version: str = "1.0.0"):
        """Initialize plugin.
        
        Args:
            name: Plugin name
            version: Plugin version
        """
        self.name = name
        self.version = version
        self._augmenters: Dict[str, Type[BaseAugmenter]] = {}
        self._strategies: Dict[str, Type[BaseAugmentationStrategy]] = {}
        self._feature_augmenters: Dict[str, Type[BaseFeatureAugmenter]] = {}
    
    @abstractmethod
    def register(self) -> None:
        """Register plugin components with the augmentation system."""
        pass
    
    def add_augmenter(self, name: str, augmenter_class: Type[BaseAugmenter]) -> None:
        """Add an augmenter to the plugin.
        
        Args:
            name: Name for the augmenter
            augmenter_class: Augmenter class
        """
        self._augmenters[name] = augmenter_class
        logger.debug(f"Added augmenter '{name}' to plugin '{self.name}'")
    
    def add_strategy(self, name: str, strategy_class: Type[BaseAugmentationStrategy]) -> None:
        """Add a strategy to the plugin.
        
        Args:
            name: Name for the strategy
            strategy_class: Strategy class
        """
        self._strategies[name] = strategy_class
        logger.debug(f"Added strategy '{name}' to plugin '{self.name}'")
    
    def add_feature_augmenter(self, name: str, augmenter_class: Type[BaseFeatureAugmenter]) -> None:
        """Add a feature augmenter to the plugin.
        
        Args:
            name: Name for the feature augmenter
            augmenter_class: Feature augmenter class
        """
        self._feature_augmenters[name] = augmenter_class
        logger.debug(f"Added feature augmenter '{name}' to plugin '{self.name}'")
    
    def install(self) -> None:
        """Install plugin components into the global registry."""
        registry = get_registry()
        
        # Register augmenters
        for name, augmenter_class in self._augmenters.items():
            full_name = f"{self.name}.{name}"
            registry.register(full_name, augmenter_class)
            logger.info(f"Registered augmenter '{full_name}'")
        
        # Register strategies
        for name, strategy_class in self._strategies.items():
            full_name = f"{self.name}.{name}"
            instance = strategy_class()
            registry.register_strategy(instance)
            logger.info(f"Registered strategy '{full_name}'")
        
        # Register feature augmenters
        for name, augmenter_class in self._feature_augmenters.items():
            full_name = f"{self.name}.{name}"
            registry.register_feature_augmenter(full_name, augmenter_class)
            logger.info(f"Registered feature augmenter '{full_name}'")
        
        logger.info(f"Plugin '{self.name}' v{self.version} installed successfully")
    
    def uninstall(self) -> None:
        """Uninstall plugin components from the global registry."""
        registry = get_registry()
        
        # Unregister augmenters
        for name in self._augmenters:
            full_name = f"{self.name}.{name}"
            try:
                registry.unregister(full_name)
                logger.info(f"Unregistered augmenter '{full_name}'")
            except KeyError:
                pass
        
        # Note: Strategies don't have unregister method in current implementation
        
        logger.info(f"Plugin '{self.name}' uninstalled")
    
    def get_info(self) -> Dict[str, Any]:
        """Get plugin information."""
        return {
            "name": self.name,
            "version": self.version,
            "augmenters": list(self._augmenters.keys()),
            "strategies": list(self._strategies.keys()),
            "feature_augmenters": list(self._feature_augmenters.keys()),
        }


class PluginManager:
    """Manages augmentation plugins."""
    
    def __init__(self):
        """Initialize plugin manager."""
        self._plugins: Dict[str, AugmentationPlugin] = {}
        self._plugin_paths: List[Path] = []
        logger.debug("Initialized plugin manager")
    
    def add_plugin_path(self, path: Path) -> None:
        """Add a path to search for plugins.
        
        Args:
            path: Directory path containing plugins
        """
        path = Path(path)
        if path.exists() and path.is_dir():
            self._plugin_paths.append(path)
            logger.debug(f"Added plugin path: {path}")
        else:
            logger.warning(f"Plugin path does not exist or is not a directory: {path}")
    
    def load_plugin(self, plugin_path: Path) -> Optional[AugmentationPlugin]:
        """Load a plugin from a Python file.
        
        Args:
            plugin_path: Path to plugin file
            
        Returns:
            Loaded plugin instance or None
        """
        if not plugin_path.exists():
            logger.error(f"Plugin file not found: {plugin_path}")
            return None
        
        # Load module
        spec = importlib.util.spec_from_file_location(plugin_path.stem, plugin_path)
        if spec is None or spec.loader is None:
            logger.error(f"Failed to load plugin spec: {plugin_path}")
            return None
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Find plugin class
        plugin_class = None
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                issubclass(obj, AugmentationPlugin) and 
                obj != AugmentationPlugin):
                plugin_class = obj
                break
        
        if plugin_class is None:
            logger.error(f"No AugmentationPlugin subclass found in {plugin_path}")
            return None
        
        # Create instance
        try:
            plugin = plugin_class()
            plugin.register()  # Call register to populate components
            logger.info(f"Loaded plugin '{plugin.name}' from {plugin_path}")
            return plugin
        except Exception as e:
            logger.error(f"Failed to instantiate plugin from {plugin_path}: {e}")
            return None
    
    def discover_plugins(self) -> List[AugmentationPlugin]:
        """Discover plugins in configured paths.
        
        Returns:
            List of discovered plugins
        """
        discovered = []
        
        for path in self._plugin_paths:
            # Look for Python files
            for file_path in path.glob("*.py"):
                if file_path.name.startswith("_"):
                    continue  # Skip private files
                
                plugin = self.load_plugin(file_path)
                if plugin:
                    discovered.append(plugin)
        
        logger.info(f"Discovered {len(discovered)} plugins")
        return discovered
    
    def install_plugin(self, plugin: AugmentationPlugin) -> None:
        """Install a plugin.
        
        Args:
            plugin: Plugin to install
        """
        if plugin.name in self._plugins:
            logger.warning(f"Plugin '{plugin.name}' already installed, replacing")
            self.uninstall_plugin(plugin.name)
        
        plugin.install()
        self._plugins[plugin.name] = plugin
        logger.info(f"Installed plugin '{plugin.name}'")
    
    def uninstall_plugin(self, plugin_name: str) -> None:
        """Uninstall a plugin.
        
        Args:
            plugin_name: Name of plugin to uninstall
        """
        if plugin_name in self._plugins:
            plugin = self._plugins[plugin_name]
            plugin.uninstall()
            del self._plugins[plugin_name]
            logger.info(f"Uninstalled plugin '{plugin_name}'")
        else:
            logger.warning(f"Plugin '{plugin_name}' not found")
    
    def get_plugin(self, name: str) -> Optional[AugmentationPlugin]:
        """Get an installed plugin.
        
        Args:
            name: Plugin name
            
        Returns:
            Plugin instance or None
        """
        return self._plugins.get(name)
    
    def list_plugins(self) -> List[str]:
        """List installed plugin names."""
        return list(self._plugins.keys())
    
    def get_plugin_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a plugin.
        
        Args:
            name: Plugin name
            
        Returns:
            Plugin info or None
        """
        plugin = self.get_plugin(name)
        return plugin.get_info() if plugin else None
    
    def install_all_discovered(self) -> None:
        """Discover and install all plugins in configured paths."""
        plugins = self.discover_plugins()
        for plugin in plugins:
            self.install_plugin(plugin)


# Global plugin manager instance
_plugin_manager = PluginManager()


def get_plugin_manager() -> PluginManager:
    """Get the global plugin manager instance."""
    return _plugin_manager