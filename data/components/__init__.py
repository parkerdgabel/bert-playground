"""Data components for composition-based dataset architecture.

This module contains focused, single-responsibility components that can be
composed together to build complex dataset functionality.
"""

from .cache import CacheManager
from .metadata import MetadataManager
from .reader import DataReader
from .selector import ColumnSelector
from .transformer import DataTransformer
from .validator import DataValidator

__all__ = [
    "DataReader",
    "DataValidator",
    "DataTransformer",
    "MetadataManager",
    "CacheManager",
    "ColumnSelector",
]