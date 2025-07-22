"""Plugin loader for discovering and loading k-bert plugins.

This module handles:
- Plugin discovery from multiple sources
- Dynamic loading of plugin classes
- Entry point based plugin loading
- Project-based plugin discovery
"""

import importlib
import importlib.metadata
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, Union

from loguru import logger

from .base import Plugin, PluginBase, PluginError, PluginMetadata, PluginState
from .validators import PluginValidator, validate_plugin


class PluginDiscovery:
    """Handles plugin discovery from various sources."""
    
    # Standard plugin directories in projects
    PROJECT_PLUGIN_DIRS = [
        "src/plugins",
        "src/heads",
        "src/augmenters",
        "src/features",
        "src/models",
        "src/metrics",
        "src/loaders",
        "src/callbacks",
        "src/commands",
        "plugins",
        "components",
    ]
    
    # Entry point group for k-bert plugins
    ENTRY_POINT_GROUP = "k_bert.plugins"
    
    def __init__(self):
        """Initialize plugin discovery."""
        self._discovered_paths: Set[Path] = set()
        self._discovered_modules: Set[str] = set()
    
    def discover_all(
        self,
        project_root: Optional[Union[str, Path]] = None,
        additional_paths: Optional[List[Union[str, Path]]] = None,
    ) -> List[Type[Plugin]]:
        """Discover all plugins from various sources.
        
        Args:
            project_root: Project root directory
            additional_paths: Additional paths to search
            
        Returns:
            List of discovered plugin classes
        """
        plugins = []
        
        # Discover from project
        if project_root:
            plugins.extend(self.discover_from_project(project_root))
        
        # Discover from additional paths
        if additional_paths:
            for path in additional_paths:
                plugins.extend(self.discover_from_path(path))
        
        # Discover from entry points
        plugins.extend(self.discover_from_entry_points())
        
        # Remove duplicates while preserving order
        seen = set()
        unique_plugins = []
        for plugin in plugins:
            plugin_id = f"{plugin.__module__}.{plugin.__name__}"
            if plugin_id not in seen:
                seen.add(plugin_id)
                unique_plugins.append(plugin)
        
        logger.info(f"Discovered {len(unique_plugins)} unique plugins")
        return unique_plugins
    
    def discover_from_project(
        self,
        project_root: Union[str, Path]
    ) -> List[Type[Plugin]]:
        """Discover plugins from a project directory.
        
        Args:
            project_root: Project root directory
            
        Returns:
            List of discovered plugin classes
        """
        project_root = Path(project_root)
        plugins = []
        
        # Add project root to Python path if needed
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        # Search standard plugin directories
        for plugin_dir in self.PROJECT_PLUGIN_DIRS:
            dir_path = project_root / plugin_dir
            if dir_path.exists() and dir_path.is_dir():
                found_plugins = self._discover_from_directory(dir_path)
                if found_plugins:
                    logger.debug(f"Found {len(found_plugins)} plugins in {plugin_dir}")
                    plugins.extend(found_plugins)
        
        # Check for plugin paths in pyproject.toml
        pyproject_path = project_root / "pyproject.toml"
        if pyproject_path.exists():
            custom_paths = self._get_plugin_paths_from_pyproject(pyproject_path)
            for path in custom_paths:
                full_path = project_root / path
                if full_path.exists():
                    found_plugins = self.discover_from_path(full_path)
                    if found_plugins:
                        logger.debug(f"Found {len(found_plugins)} plugins in custom path: {path}")
                        plugins.extend(found_plugins)
        
        return plugins
    
    def discover_from_path(
        self,
        path: Union[str, Path]
    ) -> List[Type[Plugin]]:
        """Discover plugins from a specific path.
        
        Args:
            path: Path to search (file or directory)
            
        Returns:
            List of discovered plugin classes
        """
        path = Path(path)
        
        if not path.exists():
            logger.warning(f"Plugin path does not exist: {path}")
            return []
        
        if path.is_file() and path.suffix == ".py":
            return self._discover_from_file(path)
        elif path.is_dir():
            return self._discover_from_directory(path)
        else:
            logger.warning(f"Invalid plugin path type: {path}")
            return []
    
    def discover_from_entry_points(self) -> List[Type[Plugin]]:
        """Discover plugins from installed packages via entry points.
        
        Returns:
            List of discovered plugin classes
        """
        plugins = []
        
        try:
            # Get all entry points in our group
            if sys.version_info >= (3, 10):
                entry_points = importlib.metadata.entry_points(group=self.ENTRY_POINT_GROUP)
            else:
                entry_points = importlib.metadata.entry_points().get(self.ENTRY_POINT_GROUP, [])
            
            for entry_point in entry_points:
                try:
                    # Load the entry point
                    plugin_class = entry_point.load()
                    
                    # Verify it's a valid plugin
                    if self._is_valid_plugin_class(plugin_class):
                        plugins.append(plugin_class)
                        logger.debug(f"Loaded plugin from entry point: {entry_point.name}")
                    else:
                        logger.warning(f"Invalid plugin from entry point: {entry_point.name}")
                        
                except Exception as e:
                    logger.error(f"Failed to load entry point {entry_point.name}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to discover entry points: {e}")
        
        return plugins
    
    def _discover_from_file(self, file_path: Path) -> List[Type[Plugin]]:
        """Discover plugins from a Python file.
        
        Args:
            file_path: Path to Python file
            
        Returns:
            List of discovered plugin classes
        """
        # Skip if already discovered
        if file_path in self._discovered_paths:
            return []
        
        plugins = []
        
        try:
            # Load module dynamically
            module_name = file_path.stem
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            
            if spec is None or spec.loader is None:
                logger.warning(f"Could not load module spec from {file_path}")
                return []
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            # Find plugin classes
            for name, obj in inspect.getmembers(module):
                if self._is_valid_plugin_class(obj) and obj.__module__ == module_name:
                    plugins.append(obj)
                    logger.debug(f"Found plugin class: {name} in {file_path}")
            
            self._discovered_paths.add(file_path)
            self._discovered_modules.add(module_name)
            
        except Exception as e:
            logger.error(f"Failed to load plugins from {file_path}: {e}")
        
        return plugins
    
    def _discover_from_directory(self, directory: Path) -> List[Type[Plugin]]:
        """Discover plugins from a directory.
        
        Args:
            directory: Directory to search
            
        Returns:
            List of discovered plugin classes
        """
        plugins = []
        
        # Add parent to Python path if needed
        parent_dir = str(directory.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        # Search for Python files
        for py_file in directory.rglob("*.py"):
            # Skip private files and __pycache__
            if py_file.name.startswith("_") or "__pycache__" in str(py_file):
                continue
            
            found_plugins = self._discover_from_file(py_file)
            plugins.extend(found_plugins)
        
        return plugins
    
    def _is_valid_plugin_class(self, obj: Any) -> bool:
        """Check if an object is a valid plugin class.
        
        Args:
            obj: Object to check
            
        Returns:
            True if valid plugin class
        """
        return (
            inspect.isclass(obj) and
            (issubclass(obj, PluginBase) or self._implements_plugin_protocol(obj)) and
            obj not in (Plugin, PluginBase) and
            not inspect.isabstract(obj)
        )
    
    def _implements_plugin_protocol(self, cls: Type) -> bool:
        """Check if a class implements the Plugin protocol.
        
        Args:
            cls: Class to check
            
        Returns:
            True if implements Plugin protocol
        """
        required_methods = ["metadata", "state", "validate", "initialize", "start", "stop", "cleanup"]
        return all(hasattr(cls, method) for method in required_methods)
    
    def _get_plugin_paths_from_pyproject(self, pyproject_path: Path) -> List[str]:
        """Extract plugin paths from pyproject.toml.
        
        Args:
            pyproject_path: Path to pyproject.toml
            
        Returns:
            List of plugin paths
        """
        try:
            import toml
            
            with open(pyproject_path) as f:
                data = toml.load(f)
            
            # Look for k-bert plugin configuration
            k_bert_config = data.get("tool", {}).get("k-bert", {})
            plugin_paths = k_bert_config.get("plugins", {}).get("paths", [])
            
            if isinstance(plugin_paths, str):
                plugin_paths = [plugin_paths]
            
            return plugin_paths
            
        except Exception as e:
            logger.debug(f"Could not load plugin paths from pyproject.toml: {e}")
            return []


class PluginLoader:
    """Handles plugin loading and instantiation."""
    
    def __init__(self, validator: Optional[PluginValidator] = None):
        """Initialize plugin loader.
        
        Args:
            validator: Plugin validator (creates default if not provided)
        """
        self.validator = validator or PluginValidator()
        self.discovery = PluginDiscovery()
        self._loaded_plugins: Dict[str, Plugin] = {}
    
    def load_plugins(
        self,
        project_root: Optional[Union[str, Path]] = None,
        additional_paths: Optional[List[Union[str, Path]]] = None,
        config: Optional[Dict[str, Any]] = None,
        validate: bool = True,
    ) -> Dict[str, Plugin]:
        """Load all plugins from various sources.
        
        Args:
            project_root: Project root directory
            additional_paths: Additional paths to search
            config: Configuration for plugins
            validate: Whether to validate plugins
            
        Returns:
            Dictionary of plugin name to instance
        """
        # Discover plugin classes
        plugin_classes = self.discovery.discover_all(project_root, additional_paths)
        
        # Load and instantiate plugins
        loaded = {}
        for plugin_class in plugin_classes:
            try:
                plugin = self._load_plugin_class(plugin_class, config, validate)
                if plugin:
                    loaded[plugin.metadata.name] = plugin
                    self._loaded_plugins[plugin.metadata.name] = plugin
            except Exception as e:
                logger.error(f"Failed to load plugin {plugin_class.__name__}: {e}")
        
        logger.info(f"Successfully loaded {len(loaded)} plugins")
        return loaded
    
    def load_plugin(
        self,
        plugin_class: Type[Plugin],
        config: Optional[Dict[str, Any]] = None,
        validate: bool = True,
    ) -> Plugin:
        """Load a single plugin.
        
        Args:
            plugin_class: Plugin class to load
            config: Plugin configuration
            validate: Whether to validate plugin
            
        Returns:
            Plugin instance
            
        Raises:
            PluginError: If loading fails
        """
        plugin = self._load_plugin_class(plugin_class, config, validate)
        if plugin:
            self._loaded_plugins[plugin.metadata.name] = plugin
            return plugin
        
        raise PluginError(f"Failed to load plugin {plugin_class.__name__}")
    
    def get_loaded_plugin(self, name: str) -> Optional[Plugin]:
        """Get a loaded plugin by name.
        
        Args:
            name: Plugin name
            
        Returns:
            Plugin instance or None
        """
        return self._loaded_plugins.get(name)
    
    def list_loaded_plugins(self) -> Dict[str, PluginMetadata]:
        """List all loaded plugins.
        
        Returns:
            Dictionary of plugin name to metadata
        """
        return {
            name: plugin.metadata
            for name, plugin in self._loaded_plugins.items()
        }
    
    def _load_plugin_class(
        self,
        plugin_class: Type[Plugin],
        config: Optional[Dict[str, Any]] = None,
        validate: bool = True,
    ) -> Optional[Plugin]:
        """Load and instantiate a plugin class.
        
        Args:
            plugin_class: Plugin class
            config: Plugin configuration
            validate: Whether to validate
            
        Returns:
            Plugin instance or None if failed
        """
        try:
            # Instantiate plugin
            if issubclass(plugin_class, PluginBase):
                # Use config in constructor for PluginBase subclasses
                plugin_config = self._get_plugin_config(plugin_class, config)
                plugin = plugin_class(config=plugin_config)
            else:
                # For protocol implementations, try default constructor
                plugin = plugin_class()
            
            # Mark as loaded
            if hasattr(plugin, "_state"):
                plugin._state = PluginState.LOADED
            
            # Validate if requested
            if validate:
                validation_result = self.validator.validate(plugin)
                if not validation_result.is_valid:
                    logger.error(
                        f"Plugin {plugin.metadata.name} validation failed: "
                        f"{', '.join(validation_result.errors)}"
                    )
                    return None
            
            logger.debug(f"Successfully loaded plugin: {plugin.metadata.name}")
            return plugin
            
        except Exception as e:
            logger.error(f"Failed to instantiate plugin {plugin_class.__name__}: {e}")
            return None
    
    def _get_plugin_config(
        self,
        plugin_class: Type[Plugin],
        config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Get configuration for a specific plugin.
        
        Args:
            plugin_class: Plugin class
            config: Global configuration
            
        Returns:
            Plugin-specific configuration
        """
        if not config:
            return {}
        
        # Try to get plugin-specific config
        plugin_name = getattr(plugin_class, "NAME", plugin_class.__name__)
        
        # Check for plugin-specific config
        if "plugins" in config and plugin_name in config["plugins"]:
            return config["plugins"][plugin_name]
        
        # Check for category-specific config
        category = getattr(plugin_class, "CATEGORY", None)
        if category and category in config:
            return config[category].get(plugin_name, {})
        
        return {}


# Convenience functions
def load_plugins(
    project_root: Optional[Union[str, Path]] = None,
    additional_paths: Optional[List[Union[str, Path]]] = None,
    config: Optional[Dict[str, Any]] = None,
    validate: bool = True,
) -> Dict[str, Plugin]:
    """Load all plugins from various sources.
    
    Args:
        project_root: Project root directory
        additional_paths: Additional paths to search
        config: Configuration for plugins
        validate: Whether to validate plugins
        
    Returns:
        Dictionary of plugin name to instance
    """
    loader = PluginLoader()
    return loader.load_plugins(project_root, additional_paths, config, validate)


def discover_plugins(
    project_root: Optional[Union[str, Path]] = None,
    additional_paths: Optional[List[Union[str, Path]]] = None,
) -> List[Type[Plugin]]:
    """Discover all available plugin classes.
    
    Args:
        project_root: Project root directory
        additional_paths: Additional paths to search
        
    Returns:
        List of plugin classes
    """
    discovery = PluginDiscovery()
    return discovery.discover_all(project_root, additional_paths)