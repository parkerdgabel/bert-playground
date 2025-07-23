"""Checkpointing adapters for saving and loading training state."""

from .filesystem_adapter import FilesystemCheckpointManager

__all__ = ["FilesystemCheckpointManager"]