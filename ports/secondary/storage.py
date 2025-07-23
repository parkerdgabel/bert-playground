"""Secondary storage port - Storage services that the application depends on.

This port defines the storage interface that the application core uses
to persist models, checkpoints, and data. It's a driven port implemented
by adapters for different storage backends (filesystem, cloud, database).
"""

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from typing_extensions import TypeAlias

# Type aliases for clarity
StorageKey: TypeAlias = str | Path
StorageValue: TypeAlias = Any
Metadata: TypeAlias = dict[str, Any]
StorageMetadata: TypeAlias = dict[str, Any]
ModelMetadata: TypeAlias = dict[str, Any]

# Data classes for structured types
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ModelCheckpoint:
    """Represents a saved model checkpoint."""
    path: Path
    step: int
    epoch: int
    train_loss: float
    val_loss: float | None
    metrics: dict[str, float]
    created_at: datetime
    size_bytes: int
    is_best: bool = False


@runtime_checkable
class StorageService(Protocol):
    """Secondary port for general storage operations.
    
    This interface is implemented by adapters for specific storage backends.
    The application core depends on this for persisting any kind of data.
    """

    def save(
        self, 
        key: StorageKey, 
        value: StorageValue, 
        metadata: Metadata | None = None
    ) -> None:
        """Save a value to storage.
        
        Args:
            key: Storage key/path
            value: Value to store
            metadata: Optional metadata about the stored value
        """
        ...

    def load(
        self, 
        key: StorageKey,
        expected_type: type[Any] | None = None
    ) -> StorageValue:
        """Load a value from storage.
        
        Args:
            key: Storage key/path
            expected_type: Optional type to validate loaded value
            
        Returns:
            Loaded value
            
        Raises:
            KeyError: If key does not exist
            TypeError: If loaded value doesn't match expected_type
        """
        ...

    def exists(self, key: StorageKey) -> bool:
        """Check if a key exists in storage.
        
        Args:
            key: Storage key/path
            
        Returns:
            True if key exists
        """
        ...

    def delete(self, key: StorageKey) -> None:
        """Delete a value from storage.
        
        Args:
            key: Storage key/path
            
        Raises:
            KeyError: If key does not exist
        """
        ...

    def list_keys(
        self, 
        prefix: StorageKey | None = None,
        pattern: str | None = None
    ) -> list[StorageKey]:
        """List all keys in storage.
        
        Args:
            prefix: Optional prefix to filter keys
            pattern: Optional glob pattern to match keys
            
        Returns:
            List of matching keys
        """
        ...

    def get_metadata(self, key: StorageKey) -> Metadata | None:
        """Get metadata for a stored value.
        
        Args:
            key: Storage key/path
            
        Returns:
            Metadata if available, None otherwise
            
        Raises:
            KeyError: If key does not exist
        """
        ...

    def move(self, source: StorageKey, destination: StorageKey) -> None:
        """Move a value from one key to another.
        
        Args:
            source: Source key
            destination: Destination key
            
        Raises:
            KeyError: If source does not exist
        """
        ...

    def copy(self, source: StorageKey, destination: StorageKey) -> None:
        """Copy a value from one key to another.
        
        Args:
            source: Source key
            destination: Destination key
            
        Raises:
            KeyError: If source does not exist
        """
        ...


@runtime_checkable
class ModelStorageService(Protocol):
    """Specialized storage port for ML models.
    
    This provides model-specific storage operations that handle
    the complexities of saving and loading models with their state.
    """

    def save_model(
        self,
        model: Any,
        path: Path,
        include_optimizer: bool = True,
        include_metrics: bool = True,
        metadata: Metadata | None = None
    ) -> None:
        """Save a model with its associated state.
        
        Args:
            model: Model to save
            path: Save path
            include_optimizer: Whether to save optimizer state
            include_metrics: Whether to save training metrics
            metadata: Additional metadata
        """
        ...

    def load_model(
        self,
        path: Path,
        model_class: type[Any] | None = None,
        load_optimizer: bool = True,
        load_metrics: bool = True
    ) -> tuple[Any, Metadata | None]:
        """Load a model with its associated state.
        
        Args:
            path: Model path
            model_class: Expected model class for validation
            load_optimizer: Whether to load optimizer state
            load_metrics: Whether to load training metrics
            
        Returns:
            Tuple of (model, metadata)
        """
        ...

    def save_checkpoint(
        self,
        checkpoint_data: dict[str, Any],
        path: Path,
        keep_last_n: int | None = None
    ) -> None:
        """Save a training checkpoint.
        
        Args:
            checkpoint_data: Checkpoint data to save
            path: Checkpoint path
            keep_last_n: Number of recent checkpoints to keep
        """
        ...

    def load_checkpoint(
        self,
        path: Path,
        strict: bool = True
    ) -> dict[str, Any]:
        """Load a training checkpoint.
        
        Args:
            path: Checkpoint path
            strict: Whether to enforce strict loading
            
        Returns:
            Checkpoint data
        """
        ...

    def list_checkpoints(
        self,
        directory: Path,
        pattern: str = "checkpoint_*"
    ) -> list[Path]:
        """List available checkpoints.
        
        Args:
            directory: Directory to search
            pattern: Checkpoint filename pattern
            
        Returns:
            List of checkpoint paths, sorted by creation time
        """
        ...

    def get_model_size(self, path: Path) -> int:
        """Get the size of a saved model in bytes.
        
        Args:
            path: Model path
            
        Returns:
            Size in bytes
        """
        ...

    def verify_model(self, path: Path) -> bool:
        """Verify that a saved model is valid and loadable.
        
        Args:
            path: Model path
            
        Returns:
            True if model is valid
        """
        ...

    def export_weights(
        self,
        model: Any,
        path: Path,
        format: str = "safetensors"
    ) -> None:
        """Export just the model weights.
        
        Args:
            model: Model to export weights from
            path: Export path
            format: Weight format (safetensors, npz, etc.)
        """
        ...

    def import_weights(
        self,
        model: Any,
        path: Path,
        strict: bool = True
    ) -> None:
        """Import weights into a model.
        
        Args:
            model: Model to import weights into
            path: Weights path
            strict: Whether to enforce strict loading
        """
        ...