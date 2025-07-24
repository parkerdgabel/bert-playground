"""Storage adapters for persistence operations."""

from .base import BaseStorageAdapter, BaseCheckpointAdapter
from .filesystem import FilesystemStorageAdapter, FilesystemCheckpointAdapter
from .file_storage import FileStorageAdapter, ModelFileStorageAdapter
from .cloud import (
    S3StorageAdapter,
    S3CheckpointAdapter,
    GCSStorageAdapter,
    GCSCheckpointAdapter,
)

__all__ = [
    # Base classes
    "BaseStorageAdapter",
    "BaseCheckpointAdapter",
    # Filesystem adapters
    "FilesystemStorageAdapter",
    "FilesystemCheckpointAdapter",
    "FileStorageAdapter",
    "ModelFileStorageAdapter",
    # Cloud adapters
    "S3StorageAdapter",
    "S3CheckpointAdapter",
    "GCSStorageAdapter",
    "GCSCheckpointAdapter",
]