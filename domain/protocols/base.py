"""Base domain protocols.

These protocols define fundamental contracts that many domain objects implement.
"""

from pathlib import Path
from typing import Any, Protocol


class Component(Protocol):
    """Base protocol for all components."""
    
    @property
    def name(self) -> str:
        """Get component name."""
        ...


class Configurable(Protocol):
    """Protocol for configurable components."""
    
    def configure(self, config: dict[str, Any]) -> None:
        """Configure the component."""
        ...
    
    def get_config(self) -> dict[str, Any]:
        """Get current configuration."""
        ...


class Stateful(Protocol):
    """Protocol for stateful components."""
    
    def save_state(self, path: Path) -> None:
        """Save component state."""
        ...
    
    def load_state(self, path: Path) -> None:
        """Load component state."""
        ...