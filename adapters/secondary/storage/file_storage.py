"""File system storage adapter implementation.

This module provides file system implementations of the storage ports,
enabling persistent storage of data, models, and checkpoints on local disk.
"""

import json
import pickle
import shutil
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from safetensors import safe_open
from safetensors.mlx import save_file

from ports.secondary.storage import (
    ModelCheckpoint,
    ModelMetadata,
    ModelStorageService,
    StorageMetadata,
    StorageService,
)


class FileStorageAdapter(StorageService):
    """File system implementation of the StoragePort."""

    def __init__(self, base_path: Path | str | None = None):
        """Initialize with optional base path."""
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, key: str) -> Path:
        """Resolve a storage key to an absolute path."""
        path = Path(key)
        
        if not path.is_absolute():
            path = self.base_path / path
        
        return path

    def save(
        self, 
        key: str, 
        value: Any, 
        metadata: StorageMetadata | None = None
    ) -> None:
        """Save a value to file system."""
        path = self._resolve_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine format based on file extension or value type
        if path.suffix == ".json" or isinstance(value, (dict, list)):
            with open(path, "w") as f:
                json.dump(value, f, indent=2)
        elif path.suffix == ".pkl":
            with open(path, "wb") as f:
                pickle.dump(value, f)
        elif path.suffix == ".txt" or isinstance(value, str):
            with open(path, "w") as f:
                f.write(value)
        else:
            # Default to pickle for unknown types
            with open(path, "wb") as f:
                pickle.dump(value, f)
        
        # Save metadata if provided
        if metadata:
            meta_path = path.with_suffix(path.suffix + ".meta")
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)

    def load(
        self, 
        key: str,
        expected_type: type[Any] | None = None
    ) -> Any:
        """Load a value from file system."""
        path = self._resolve_path(key)
        
        if not path.exists():
            raise KeyError(f"Storage key not found: {key}")
        
        # Load based on file extension
        if path.suffix == ".json":
            with open(path, "r") as f:
                value = json.load(f)
        elif path.suffix == ".pkl":
            with open(path, "rb") as f:
                value = pickle.load(f)
        elif path.suffix == ".txt":
            with open(path, "r") as f:
                value = f.read()
        else:
            # Try to load as text first, then pickle
            try:
                with open(path, "r") as f:
                    value = f.read()
            except UnicodeDecodeError:
                with open(path, "rb") as f:
                    value = pickle.load(f)
        
        # Validate type if specified
        if expected_type and not isinstance(value, expected_type):
            raise TypeError(
                f"Expected type {expected_type}, got {type(value)}"
            )
        
        return value

    def exists(self, key: str) -> bool:
        """Check if a key exists in storage."""
        path = self._resolve_path(key)
        return path.exists()

    def delete(self, key: str) -> None:
        """Delete a value from storage."""
        path = self._resolve_path(key)
        
        if not path.exists():
            raise KeyError(f"Storage key not found: {key}")
        
        if path.is_file():
            path.unlink()
            # Also delete metadata if it exists
            meta_path = path.with_suffix(path.suffix + ".meta")
            if meta_path.exists():
                meta_path.unlink()
        else:
            shutil.rmtree(path)

    def list_keys(
        self, 
        prefix: str | None = None,
        pattern: str | None = None
    ) -> list[str]:
        """List all keys in storage."""
        search_path = self._resolve_path(prefix) if prefix else self.base_path
        
        if not search_path.exists():
            return []
        
        if pattern:
            # Use glob pattern matching
            if search_path.is_dir():
                paths = list(search_path.glob(pattern))
            else:
                paths = list(search_path.parent.glob(pattern))
        else:
            # List all files recursively
            if search_path.is_dir():
                paths = [p for p in search_path.rglob("*") if p.is_file()]
            else:
                paths = [search_path] if search_path.is_file() else []
        
        # Filter out metadata files
        paths = [p for p in paths if not p.name.endswith(".meta")]
        
        # Convert to relative paths if within base_path
        keys = []
        for path in paths:
            try:
                rel_path = path.relative_to(self.base_path)
                keys.append(str(rel_path))
            except ValueError:
                keys.append(str(path))
        
        return sorted(keys)

    def get_metadata(self, key: str) -> StorageMetadata | None:
        """Get metadata for a stored value."""
        path = self._resolve_path(key)
        
        if not path.exists():
            raise KeyError(f"Storage key not found: {key}")
        
        meta_path = path.with_suffix(path.suffix + ".meta")
        if meta_path.exists():
            with open(meta_path, "r") as f:
                return json.load(f)
        
        return None


class ModelFileStorageAdapter(ModelStorageService):
    """File system implementation of the ModelStoragePort."""

    def __init__(self, base_path: Path | str | None = None):
        """Initialize with optional base path."""
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.storage = FileStorageAdapter(base_path)

    def save_model(
        self,
        model: Any,
        path: str,
        metadata: ModelMetadata | None = None
    ) -> None:
        """Save an MLX model with its associated state."""
        model_path = self.base_path / path
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save model weights using safetensors
        if hasattr(model, "parameters"):
            weights = dict(model.parameters())
            save_file(weights, model_path / "model.safetensors")
        
        # Save model config if available
        if hasattr(model, "config"):
            config = model.config
            if hasattr(config, "to_dict"):
                config = config.to_dict()
            self.storage.save(str(model_path / "config.json"), config)
        
        # Save metadata
        if metadata:
            metadata_dict = {
                "model_type": metadata.model_type,
                "framework": metadata.framework,
                "version": metadata.version,
                "training_config": metadata.training_config,
                "performance_metrics": metadata.performance_metrics,
                "created_at": metadata.created_at.isoformat() if metadata.created_at else None,
                "tags": metadata.tags,
            }
            self.storage.save(str(model_path / "metadata.json"), metadata_dict)

    def load_model(
        self,
        path: str,
        model_class: type[Any] | None = None,
    ) -> tuple[Any, ModelMetadata | None]:
        """Load an MLX model with its associated state."""
        model_path = self.base_path / path
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
        # Load metadata
        metadata = None
        if (model_path / "metadata.json").exists():
            metadata_dict = self.storage.load(str(model_path / "metadata.json"))
            if metadata_dict:
                from datetime import datetime
                metadata = ModelMetadata(
                    model_type=metadata_dict.get("model_type", ""),
                    framework=metadata_dict.get("framework", "mlx"),
                    version=metadata_dict.get("version", ""),
                    training_config=metadata_dict.get("training_config"),
                    performance_metrics=metadata_dict.get("performance_metrics"),
                    created_at=datetime.fromisoformat(metadata_dict["created_at"]) if metadata_dict.get("created_at") else None,
                    tags=metadata_dict.get("tags", {}),
                )
        
        # Load config
        config = None
        if (model_path / "config.json").exists():
            config = self.storage.load(str(model_path / "config.json"))
        
        # Load weights
        weights = {}
        if (model_path / "model.safetensors").exists():
            with safe_open(model_path / "model.safetensors", framework="mlx") as f:
                for key in f.keys():
                    weights[key] = f.get_tensor(key)
        
        # Create model instance if class provided
        if model_class:
            if config:
                model = model_class(config)
            else:
                model = model_class()
            
            # Load weights into model
            if hasattr(model, "load_weights"):
                model.load_weights(weights)
            elif hasattr(model, "update"):
                model.update(weights)
        else:
            # Return raw weights if no class provided
            model = weights
        
        return model, metadata

    def save_checkpoint(
        self,
        checkpoint: ModelCheckpoint,
        path: str,
    ) -> None:
        """Save a training checkpoint."""
        checkpoint_path = self.base_path / path
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert checkpoint to dict
        checkpoint_dict = {
            "epoch": checkpoint.epoch,
            "step": checkpoint.step,
            "model_state": checkpoint.model_state,
            "optimizer_state": checkpoint.optimizer_state,
            "metrics": checkpoint.metrics,
            "best_metric": checkpoint.best_metric,
            "config": checkpoint.config,
        }
        
        # Save checkpoint data
        self.storage.save(str(checkpoint_path), checkpoint_dict)

    def load_checkpoint(
        self,
        path: str,
    ) -> ModelCheckpoint:
        """Load a training checkpoint."""
        checkpoint_path = self.base_path / path
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint_dict = self.storage.load(str(checkpoint_path))
        
        return ModelCheckpoint(
            epoch=checkpoint_dict["epoch"],
            step=checkpoint_dict["step"],
            model_state=checkpoint_dict["model_state"],
            optimizer_state=checkpoint_dict.get("optimizer_state"),
            metrics=checkpoint_dict.get("metrics", {}),
            best_metric=checkpoint_dict.get("best_metric"),
            config=checkpoint_dict.get("config"),
        )

    def list_checkpoints(
        self,
        directory: str,
        pattern: str = "checkpoint_*"
    ) -> list[str]:
        """List available checkpoints."""
        dir_path = self.base_path / directory
        
        if not dir_path.exists():
            return []
        
        checkpoints = list(dir_path.glob(pattern))
        
        # Convert to relative paths
        checkpoint_paths = []
        for cp in checkpoints:
            try:
                rel_path = cp.relative_to(self.base_path)
                checkpoint_paths.append(str(rel_path))
            except ValueError:
                checkpoint_paths.append(str(cp))
        
        # Sort by modification time
        checkpoint_paths.sort(
            key=lambda p: (self.base_path / p).stat().st_mtime
        )
        
        return checkpoint_paths

    def delete_checkpoint(self, path: str) -> None:
        """Delete a checkpoint."""
        self.storage.delete(path)