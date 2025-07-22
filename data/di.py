"""Dependency injection setup for data module.

This module provides DI configuration for creating and managing
data components and their dependencies.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Union

from .components import (
    CacheManager,
    ColumnSelector,
    DataReader,
    DataTransformer,
    DataValidator,
    MetadataManager,
)
from .composed_dataset import ComposedDataset
from .core.base import DatasetSpec


class DataContainer:
    """Dependency injection container for data components."""

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        """Initialize the data container.

        Args:
            cache_dir: Default cache directory for all components
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._instances: dict[str, Any] = {}

    @lru_cache(maxsize=None)
    def get_reader(self) -> DataReader:
        """Get or create a DataReader instance."""
        if "reader" not in self._instances:
            self._instances["reader"] = DataReader()
        return self._instances["reader"]

    def get_validator(self, spec: DatasetSpec) -> DataValidator:
        """Get or create a DataValidator instance.

        Args:
            spec: Dataset specification for validation

        Returns:
            DataValidator instance
        """
        key = f"validator_{id(spec)}"
        if key not in self._instances:
            self._instances[key] = DataValidator(spec)
        return self._instances[key]

    def get_transformer(self, spec: DatasetSpec) -> DataTransformer:
        """Get or create a DataTransformer instance.

        Args:
            spec: Dataset specification for transformation

        Returns:
            DataTransformer instance
        """
        key = f"transformer_{id(spec)}"
        if key not in self._instances:
            self._instances[key] = DataTransformer(spec)
        return self._instances[key]

    def get_metadata_manager(self, spec: DatasetSpec) -> MetadataManager:
        """Get or create a MetadataManager instance.

        Args:
            spec: Dataset specification

        Returns:
            MetadataManager instance
        """
        key = f"metadata_{id(spec)}"
        if key not in self._instances:
            self._instances[key] = MetadataManager(spec)
        return self._instances[key]

    def get_cache_manager(
        self, cache_dir: Optional[Union[str, Path]] = None
    ) -> CacheManager:
        """Get or create a CacheManager instance.

        Args:
            cache_dir: Cache directory (uses container default if not provided)

        Returns:
            CacheManager instance
        """
        effective_cache_dir = cache_dir or self.cache_dir
        key = f"cache_{effective_cache_dir}"
        
        if key not in self._instances:
            self._instances[key] = CacheManager(effective_cache_dir)
        return self._instances[key]

    @lru_cache(maxsize=None)
    def get_column_selector(self) -> ColumnSelector:
        """Get or create a ColumnSelector instance."""
        if "selector" not in self._instances:
            self._instances["selector"] = ColumnSelector()
        return self._instances["selector"]

    def create_dataset(
        self,
        spec: DatasetSpec,
        split: str = "train",
        transform: Optional[Any] = None,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> ComposedDataset:
        """Create a new ComposedDataset with injected dependencies.

        Args:
            spec: Dataset specification
            split: Dataset split
            transform: Optional transform function
            cache_dir: Cache directory

        Returns:
            ComposedDataset instance
        """
        # Create dataset with shared components
        dataset = ComposedDataset(
            spec=spec,
            split=split,
            transform=transform,
            cache_dir=cache_dir or self.cache_dir,
        )

        # Inject shared components if needed
        # (ComposedDataset creates its own instances by default,
        # but we could override them here if needed)

        return dataset

    def clear_cache(self) -> None:
        """Clear all cached instances."""
        self._instances.clear()
        # Clear LRU caches
        self.get_reader.cache_clear()
        self.get_column_selector.cache_clear()


# Global container instance
_default_container = None


def get_default_container() -> DataContainer:
    """Get the default global data container.

    Returns:
        Default DataContainer instance
    """
    global _default_container
    if _default_container is None:
        _default_container = DataContainer()
    return _default_container


def set_default_container(container: DataContainer) -> None:
    """Set the default global data container.

    Args:
        container: New default container
    """
    global _default_container
    _default_container = container


def create_dataset(
    spec: DatasetSpec,
    split: str = "train",
    transform: Optional[Any] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    container: Optional[DataContainer] = None,
) -> ComposedDataset:
    """Create a dataset using dependency injection.

    Args:
        spec: Dataset specification
        split: Dataset split
        transform: Optional transform function
        cache_dir: Cache directory
        container: DI container (uses default if not provided)

    Returns:
        ComposedDataset instance
    """
    if container is None:
        container = get_default_container()
    
    return container.create_dataset(
        spec=spec,
        split=split,
        transform=transform,
        cache_dir=cache_dir,
    )


class DataComponentBuilder:
    """Builder for creating configured data components."""

    def __init__(self):
        """Initialize the builder."""
        self._cache_dir: Optional[Path] = None
        self._spec: Optional[DatasetSpec] = None

    def with_cache_dir(self, cache_dir: Union[str, Path]) -> "DataComponentBuilder":
        """Set the cache directory.

        Args:
            cache_dir: Cache directory path

        Returns:
            Self for chaining
        """
        self._cache_dir = Path(cache_dir)
        return self

    def with_spec(self, spec: DatasetSpec) -> "DataComponentBuilder":
        """Set the dataset specification.

        Args:
            spec: Dataset specification

        Returns:
            Self for chaining
        """
        self._spec = spec
        return self

    def build_reader(self) -> DataReader:
        """Build a DataReader instance."""
        return DataReader()

    def build_validator(self) -> DataValidator:
        """Build a DataValidator instance."""
        if self._spec is None:
            raise ValueError("DatasetSpec required for validator")
        return DataValidator(self._spec)

    def build_transformer(self) -> DataTransformer:
        """Build a DataTransformer instance."""
        if self._spec is None:
            raise ValueError("DatasetSpec required for transformer")
        return DataTransformer(self._spec)

    def build_metadata_manager(self) -> MetadataManager:
        """Build a MetadataManager instance."""
        if self._spec is None:
            raise ValueError("DatasetSpec required for metadata manager")
        return MetadataManager(self._spec)

    def build_cache_manager(self) -> CacheManager:
        """Build a CacheManager instance."""
        return CacheManager(self._cache_dir)

    def build_column_selector(self) -> ColumnSelector:
        """Build a ColumnSelector instance."""
        return ColumnSelector()

    def build_all(self) -> dict[str, Any]:
        """Build all components.

        Returns:
            Dictionary of component name to instance
        """
        if self._spec is None:
            raise ValueError("DatasetSpec required to build all components")

        return {
            "reader": self.build_reader(),
            "validator": self.build_validator(),
            "transformer": self.build_transformer(),
            "metadata_manager": self.build_metadata_manager(),
            "cache_manager": self.build_cache_manager(),
            "column_selector": self.build_column_selector(),
        }