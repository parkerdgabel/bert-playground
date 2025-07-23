"""Secondary checkpointing port - Checkpoint management that the application depends on.

This port defines the checkpointing interface that the application core uses
for saving and loading training checkpoints. It's a driven port implemented
by adapters for different storage backends.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from infrastructure.protocols.models import Model
from .optimization import Optimizer


@dataclass
class CheckpointInfo:
    """Information about a checkpoint."""
    
    path: Path
    step: int
    epoch: int
    train_loss: float
    val_loss: float | None
    metrics: dict[str, float]
    created_at: str
    size_bytes: int
    is_best: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": str(self.path),
            "step": self.step,
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "metrics": self.metrics,
            "created_at": self.created_at,
            "size_bytes": self.size_bytes,
            "is_best": self.is_best,
        }


@runtime_checkable
class CheckpointManager(Protocol):
    """Secondary port for checkpoint management.
    
    This interface is implemented by adapters that handle checkpoint
    saving and loading. The application core depends on this for
    training persistence.
    """

    @property
    def checkpoint_dir(self) -> Path:
        """Get checkpoint directory."""
        ...

    def save_checkpoint(
        self,
        model: Model,
        optimizer: Optimizer,
        state: dict[str, Any],
        metrics: dict[str, float],
        is_best: bool = False,
    ) -> Path:
        """Save a training checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            state: Training state dictionary
            metrics: Current metrics
            is_best: Whether this is the best checkpoint
            
        Returns:
            Path to saved checkpoint
        """
        ...

    def load_checkpoint(
        self,
        path: Path,
        model: Model,
        optimizer: Optimizer | None = None,
        strict: bool = True,
    ) -> dict[str, Any]:
        """Load a training checkpoint.
        
        Args:
            path: Checkpoint path
            model: Model to load into
            optimizer: Optional optimizer to load into
            strict: Whether to enforce strict loading
            
        Returns:
            Training state dictionary
        """
        ...

    def get_best_checkpoint(self) -> CheckpointInfo | None:
        """Get information about the best checkpoint.
        
        Returns:
            Best checkpoint info or None if no checkpoints
        """
        ...

    def get_latest_checkpoint(self) -> CheckpointInfo | None:
        """Get information about the latest checkpoint.
        
        Returns:
            Latest checkpoint info or None if no checkpoints
        """
        ...

    def list_checkpoints(self) -> list[CheckpointInfo]:
        """List all available checkpoints.
        
        Returns:
            List of checkpoint information, sorted by creation time
        """
        ...

    def cleanup_old_checkpoints(
        self,
        keep_best: int = 1,
        keep_last: int = 1,
        keep_every_n_epochs: int | None = None,
    ) -> list[Path]:
        """Remove old checkpoints according to retention policy.
        
        Args:
            keep_best: Number of best checkpoints to keep
            keep_last: Number of recent checkpoints to keep
            keep_every_n_epochs: Keep checkpoint every N epochs
            
        Returns:
            List of removed checkpoint paths
        """
        ...

    def delete_checkpoint(self, path: Path) -> bool:
        """Delete a specific checkpoint.
        
        Args:
            path: Checkpoint path
            
        Returns:
            True if deleted successfully
        """
        ...

    def verify_checkpoint(self, path: Path) -> bool:
        """Verify that a checkpoint is valid and loadable.
        
        Args:
            path: Checkpoint path
            
        Returns:
            True if checkpoint is valid
        """
        ...

    def get_checkpoint_info(self, path: Path) -> CheckpointInfo | None:
        """Get information about a specific checkpoint.
        
        Args:
            path: Checkpoint path
            
        Returns:
            Checkpoint info or None if not found
        """
        ...

    def save_partial_checkpoint(
        self,
        data: dict[str, Any],
        name: str,
    ) -> Path:
        """Save a partial checkpoint (e.g., just model or optimizer).
        
        Args:
            data: Data to save
            name: Checkpoint name
            
        Returns:
            Path to saved checkpoint
        """
        ...

    def load_partial_checkpoint(
        self,
        path: Path,
        keys: list[str] | None = None,
    ) -> dict[str, Any]:
        """Load specific parts of a checkpoint.
        
        Args:
            path: Checkpoint path
            keys: Optional list of keys to load
            
        Returns:
            Loaded data
        """
        ...

    def create_checkpoint_index(self) -> dict[str, Any]:
        """Create an index of all checkpoints with metadata.
        
        Returns:
            Dictionary with checkpoint metadata
        """
        ...

    def export_checkpoint(
        self,
        checkpoint_path: Path,
        export_path: Path,
        format: str = "safetensors",
        include_optimizer: bool = False,
    ) -> Path:
        """Export checkpoint to a different format.
        
        Args:
            checkpoint_path: Source checkpoint
            export_path: Export destination
            format: Export format
            include_optimizer: Whether to include optimizer state
            
        Returns:
            Path to exported checkpoint
        """
        ...

    def compare_checkpoints(
        self,
        checkpoint1: Path,
        checkpoint2: Path,
    ) -> dict[str, Any]:
        """Compare two checkpoints.
        
        Args:
            checkpoint1: First checkpoint
            checkpoint2: Second checkpoint
            
        Returns:
            Comparison results
        """
        ...