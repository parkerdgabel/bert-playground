"""Plugin system for k-bert custom components.

This module provides a plugin architecture that allows users to extend k-bert
with custom BERT heads, data augmenters, feature extractors, and other components.

DEPRECATED: This module is deprecated in favor of the unified plugin system
in core.plugins. Use core.plugins.integration for backwards compatibility.
"""

import warnings

# Import from unified system with backwards compatibility
from core.plugins.integration import (
    get_component_registry as get_registry,
    load_project_plugins,
    setup_plugin_system,
    ensure_plugins_loaded,
)

# Legacy imports for backwards compatibility
from .registry import ComponentRegistry, register_component, get_component
from .loader import PluginLoader
from .base import (
    BasePlugin,
    HeadPlugin,
    AugmenterPlugin,
    FeatureExtractorPlugin,
    DataLoaderPlugin,
)

# Issue deprecation warning
warnings.warn(
    "cli.plugins is deprecated. Use core.plugins and core.plugins.integration instead.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    # Registry (legacy)
    "ComponentRegistry",
    "register_component", 
    "get_component",
    "get_registry",
    # Loader (legacy)
    "PluginLoader",
    "load_project_plugins",
    # Base classes (legacy)
    "BasePlugin",
    "HeadPlugin",
    "AugmenterPlugin",
    "FeatureExtractorPlugin",
    "DataLoaderPlugin",
    # New unified system
    "setup_plugin_system",
    "ensure_plugins_loaded",
]