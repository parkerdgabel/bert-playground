"""
Storage backend adapters for persisting data and models.

This package contains implementations of storage backends:
- FileSystem: Local file system storage
- S3: Amazon S3 storage (future)
- GCS: Google Cloud Storage (future)
"""

from .filesystem import FileStorageAdapter, ModelFileStorageAdapter

__all__ = ["FileStorageAdapter", "ModelFileStorageAdapter"]