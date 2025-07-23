"""Unified plugin system for k-bert.

This module provides a comprehensive plugin architecture that allows extending
k-bert with custom components like models, data processors, training callbacks,
and CLI commands.

Key Features:
- Protocol-based plugin contracts
- Automatic discovery from multiple sources
- Dependency injection integration
- Lifecycle management
- Configuration support
- Validation and error handling
"""

from .base import (
    Plugin,
    PluginBase,
    PluginMetadata,
    PluginLifecycle,
    PluginState,
    PluginContext,
    PluginError,
)
from .loader import (
    PluginLoader,
    PluginDiscovery,
    load_plugins,
    discover_plugins,
)
from .integration import (
    setup_plugin_system,
    ensure_plugins_loaded,
)
from .registry import (
    PluginRegistry,
    get_registry,
    register_plugin,
    get_plugin,
    list_plugins,
)
from .validators import (
    PluginValidator,
    ValidationResult,
    validate_plugin,
)
from .config import (
    PluginConfig,
    PluginConfigSchema,
    load_plugin_config,
)

__all__ = [
    # Base classes and types
    "Plugin",
    "PluginBase",
    "PluginMetadata",
    "PluginLifecycle",
    "PluginState",
    "PluginContext",
    "PluginError",
    # Loader
    "PluginLoader",
    "PluginDiscovery",
    "load_plugins",
    "discover_plugins",
    # Integration
    "setup_plugin_system",
    "ensure_plugins_loaded",
    # Registry
    "PluginRegistry",
    "get_registry",
    "register_plugin",
    "get_plugin",
    "list_plugins",
    # Validators
    "PluginValidator",
    "ValidationResult",
    "validate_plugin",
    # Config
    "PluginConfig",
    "PluginConfigSchema",
    "load_plugin_config",
]