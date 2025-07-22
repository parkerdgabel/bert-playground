"""Base protocols shared across the k-bert codebase.

These protocols define common interfaces that are used by multiple modules.
"""

from pathlib import Path
from typing import Any, Protocol


class Configurable(Protocol):
    """Protocol for objects that can be configured from dictionaries."""

    def get_config(self) -> dict[str, Any]:
        """Get configuration as a dictionary.
        
        Returns:
            Dictionary containing configuration parameters
        """
        ...

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Configurable":
        """Create instance from configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            New instance configured according to the dictionary
        """
        ...

    def validate_config(self) -> list[str]:
        """Validate configuration and return list of errors.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        ...


class Serializable(Protocol):
    """Protocol for objects that can be serialized to/from disk."""

    def save(self, path: Path) -> None:
        """Save object to disk.
        
        Args:
            path: Path to save to
        """
        ...

    @classmethod
    def load(cls, path: Path) -> "Serializable":
        """Load object from disk.
        
        Args:
            path: Path to load from
            
        Returns:
            Loaded object
        """
        ...


class Comparable(Protocol):
    """Protocol for objects that can be compared for better/worse performance."""

    def is_better_than(self, other: "Comparable", metric: str = "default") -> bool:
        """Check if this object is better than another according to a metric.
        
        Args:
            other: Object to compare against
            metric: Metric to use for comparison
            
        Returns:
            True if this object is better than the other
        """
        ...