"""
Migration utilities for Phase 2 transitions.

This module provides tools and wrappers to help migrate from Phase 1 to Phase 2
architecture while maintaining backward compatibility.
"""

from .compatibility import (
    LegacyTrainerWrapper,
    PluginMigrationWrapper, 
    MLXCompatibilityLayer,
    DataProcessingCompat
)
from .helpers import (
    migrate_trainer_code,
    migrate_plugin_code,
    migrate_config_format,
    validate_migration
)
from .deprecation import (
    deprecated,
    warn_deprecated,
    DEPRECATION_TIMELINE
)

__all__ = [
    "LegacyTrainerWrapper",
    "PluginMigrationWrapper",
    "MLXCompatibilityLayer", 
    "DataProcessingCompat",
    "migrate_trainer_code",
    "migrate_plugin_code", 
    "migrate_config_format",
    "validate_migration",
    "deprecated",
    "warn_deprecated",
    "DEPRECATION_TIMELINE"
]