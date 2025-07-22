"""Component registry for k-bert plugins.

This module provides a central registry for all custom components,
allowing dynamic discovery and loading of user-defined plugins.
"""

from typing import Any, Dict, List, Optional, Type, Union
from pathlib import Path
import importlib.util
import inspect

from loguru import logger

from .base import (
    BasePlugin,
    HeadPlugin,
    AugmenterPlugin,
    FeatureExtractorPlugin,
    DataLoaderPlugin,
    ModelPlugin,
    MetricPlugin,
)


class ComponentRegistry:
    """Central registry for k-bert components."""
    
    def __init__(self):
        """Initialize component registry."""
        self._components: Dict[str, Dict[str, Type[BasePlugin]]] = {
            "head": {},
            "augmenter": {},
            "feature_extractor": {},
            "data_loader": {},
            "model": {},
            "metric": {},
        }
        
        # Type mapping
        self._type_map = {
            HeadPlugin: "head",
            AugmenterPlugin: "augmenter",
            FeatureExtractorPlugin: "feature_extractor",
            DataLoaderPlugin: "data_loader",
            ModelPlugin: "model",
            MetricPlugin: "metric",
        }
        
        # Reverse type map
        self._class_map = {v: k for k, v in self._type_map.items()}
    
    def register(
        self,
        component_class: Type[BasePlugin],
        name: Optional[str] = None,
        component_type: Optional[str] = None,
        override: bool = False,
    ) -> None:
        """Register a component class.
        
        Args:
            component_class: Component class to register
            name: Component name (defaults to class name)
            component_type: Component type (auto-detected if not provided)
            override: Whether to override existing component
        """
        # Auto-detect component type
        if component_type is None:
            for base_class, type_name in self._type_map.items():
                if issubclass(component_class, base_class):
                    component_type = type_name
                    break
        
        if component_type is None:
            raise ValueError(
                f"Could not determine component type for {component_class.__name__}. "
                "Please specify component_type explicitly."
            )
        
        if component_type not in self._components:
            raise ValueError(f"Unknown component type: {component_type}")
        
        # Use class name if no name provided
        if name is None:
            name = component_class.__name__
        
        # Check if already registered
        if name in self._components[component_type] and not override:
            raise ValueError(
                f"Component '{name}' already registered for type '{component_type}'. "
                "Use override=True to replace."
            )
        
        # Register component
        self._components[component_type][name] = component_class
        logger.debug(f"Registered {component_type} component: {name}")
    
    def get(
        self,
        component_type: str,
        name: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> BasePlugin:
        """Get an instance of a registered component.
        
        Args:
            component_type: Type of component
            name: Component name
            config: Configuration for component
            
        Returns:
            Component instance
        """
        if component_type not in self._components:
            raise ValueError(f"Unknown component type: {component_type}")
        
        if name not in self._components[component_type]:
            available = list(self._components[component_type].keys())
            raise ValueError(
                f"Component '{name}' not found for type '{component_type}'. "
                f"Available: {available}"
            )
        
        component_class = self._components[component_type][name]
        return component_class(config=config)
    
    def list_components(
        self,
        component_type: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """List all registered components.
        
        Args:
            component_type: Optional type filter
            
        Returns:
            Dictionary of component type to list of names
        """
        if component_type:
            if component_type not in self._components:
                return {}
            return {component_type: list(self._components[component_type].keys())}
        
        return {
            ctype: list(components.keys())
            for ctype, components in self._components.items()
            if components
        }
    
    def load_from_module(
        self,
        module_path: Union[str, Path],
        override: bool = False,
    ) -> int:
        """Load components from a Python module.
        
        Args:
            module_path: Path to Python module
            override: Whether to override existing components
            
        Returns:
            Number of components loaded
        """
        module_path = Path(module_path)
        
        if not module_path.exists():
            raise FileNotFoundError(f"Module not found: {module_path}")
        
        # Load module
        spec = importlib.util.spec_from_file_location(
            module_path.stem,
            module_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module: {module_path}")
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Find and register all plugin classes
        count = 0
        for name, obj in inspect.getmembers(module):
            if (
                inspect.isclass(obj) and
                issubclass(obj, BasePlugin) and
                obj is not BasePlugin and
                obj not in self._type_map
            ):
                try:
                    self.register(obj, override=override)
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to register {name}: {e}")
        
        return count
    
    def load_from_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = True,
        override: bool = False,
    ) -> int:
        """Load all components from a directory.
        
        Args:
            directory: Directory containing Python modules
            recursive: Whether to search recursively
            override: Whether to override existing components
            
        Returns:
            Total number of components loaded
        """
        directory = Path(directory)
        
        if not directory.exists() or not directory.is_dir():
            return 0
        
        # Find all Python files
        if recursive:
            pattern = "**/*.py"
        else:
            pattern = "*.py"
        
        total_count = 0
        for py_file in directory.glob(pattern):
            # Skip __init__.py and private files
            if py_file.name.startswith("_"):
                continue
            
            try:
                count = self.load_from_module(py_file, override=override)
                if count > 0:
                    logger.info(f"Loaded {count} components from {py_file}")
                total_count += count
            except Exception as e:
                logger.warning(f"Failed to load components from {py_file}: {e}")
        
        return total_count
    
    def clear(self, component_type: Optional[str] = None) -> None:
        """Clear registered components.
        
        Args:
            component_type: Optional type to clear (clears all if None)
        """
        if component_type:
            if component_type in self._components:
                self._components[component_type].clear()
        else:
            for components in self._components.values():
                components.clear()


# Global registry instance
_registry = ComponentRegistry()


def register_component(
    component_class: Type[BasePlugin] = None,
    name: Optional[str] = None,
    component_type: Optional[str] = None,
    override: bool = False,
):
    """Decorator to register a component class.
    
    Can be used as:
        @register_component
        class MyHead(HeadPlugin):
            ...
    
    Or with parameters:
        @register_component(name="custom_head", override=True)
        class MyHead(HeadPlugin):
            ...
    """
    def decorator(cls):
        _registry.register(cls, name=name, component_type=component_type, override=override)
        return cls
    
    if component_class is not None:
        # Used without parameters
        return decorator(component_class)
    
    # Used with parameters
    return decorator


def get_component(
    component_type: str,
    name: str,
    config: Optional[Dict[str, Any]] = None,
) -> BasePlugin:
    """Get a component instance from the registry.
    
    Args:
        component_type: Type of component
        name: Component name
        config: Configuration for component
        
    Returns:
        Component instance
    """
    return _registry.get(component_type, name, config)


def list_components(component_type: Optional[str] = None) -> Dict[str, List[str]]:
    """List all registered components.
    
    Args:
        component_type: Optional type filter
        
    Returns:
        Dictionary of component type to list of names
    """
    return _registry.list_components(component_type)


def load_components_from_directory(
    directory: Union[str, Path],
    recursive: bool = True,
    override: bool = False,
) -> int:
    """Load all components from a directory.
    
    Args:
        directory: Directory containing Python modules
        recursive: Whether to search recursively
        override: Whether to override existing components
        
    Returns:
        Total number of components loaded
    """
    return _registry.load_from_directory(directory, recursive, override)


def get_registry() -> ComponentRegistry:
    """Get the global component registry.
    
    Returns:
        Global ComponentRegistry instance
    """
    return _registry