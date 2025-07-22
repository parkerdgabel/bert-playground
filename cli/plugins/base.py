"""Base classes for k-bert plugins - re-exported from core.protocols for backward compatibility.

This module provides backward compatibility for the old plugin base classes
by re-exporting the new protocol-based interfaces.
"""

# Re-export plugin protocols from the centralized location
from core.protocols.plugins import (
    AugmenterPlugin,
    DataLoaderPlugin,
    FeatureExtractorPlugin,
    HeadPlugin,
    MetricPlugin,
    ModelPlugin,
    Plugin,
    PluginMetadata,
)

# For backward compatibility, create base classes that implement the protocols
from typing import Any, Dict, List, Optional

import mlx.core as mx


class BasePlugin:
    """Base class for all k-bert plugins (backward compatibility wrapper)."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize plugin with configuration.
        
        Args:
            config: Plugin-specific configuration
        """
        self.config = config or {}
        self._metadata = self.get_metadata()
    
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name=self.__class__.__name__,
            version="1.0.0",
            description=self.__class__.__doc__,
        )
    
    @property
    def name(self) -> str:
        """Get plugin name."""
        return self._metadata.name
    
    @property
    def version(self) -> str:
        """Get plugin version."""
        return self._metadata.version


# Re-export all for convenience
__all__ = [
    "PluginMetadata",
    "Plugin",
    "BasePlugin",
    "HeadPlugin",
    "AugmenterPlugin",
    "FeatureExtractorPlugin",
    "DataLoaderPlugin",
    "ModelPlugin",
    "MetricPlugin",
]