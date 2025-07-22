"""Plugin system for k-bert custom components using hexagonal architecture.

This module provides a plugin architecture that integrates with the dependency injection
container and allows users to extend k-bert with custom components.
"""

# Import from unified plugin system
from core.plugins.integration import (
    get_component_registry as get_registry,
    load_project_plugins,
    setup_plugin_system,
    ensure_plugins_loaded,
)

# Plugin base classes and components
from .registry import ComponentRegistry, register_component, get_component
from .loader import PluginLoader
from .base import (
    BasePlugin,
    HeadPlugin,
    AugmenterPlugin,
    FeatureExtractorPlugin,
    DataLoaderPlugin,
)


__all__ = [
    # Registry
    "ComponentRegistry",
    "register_component", 
    "get_component",
    "get_registry",
    # Loader
    "PluginLoader",
    "load_project_plugins",
    # Base classes
    "BasePlugin",
    "HeadPlugin",
    "AugmenterPlugin",
    "FeatureExtractorPlugin",
    "DataLoaderPlugin",
    # Unified system
    "setup_plugin_system",
    "ensure_plugins_loaded",
]