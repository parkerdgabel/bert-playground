"""Plugin system for k-bert custom components.

This module provides a plugin architecture that allows users to extend k-bert
with custom BERT heads, data augmenters, feature extractors, and other components.
"""

from .registry import ComponentRegistry, register_component, get_component, get_registry
from .loader import PluginLoader, load_project_plugins
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
]