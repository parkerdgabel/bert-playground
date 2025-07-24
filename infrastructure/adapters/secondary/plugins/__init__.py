"""Plugin adapters for extensibility management."""

from .loader_adapter import PluginLoaderAdapter
from .registry_adapter import PluginRegistryAdapter

__all__ = ["PluginLoaderAdapter", "PluginRegistryAdapter"]