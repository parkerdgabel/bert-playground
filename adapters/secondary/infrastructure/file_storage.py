"""File system storage adapter implementation."""

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
    Metadata,
    ModelStorageService,
    StorageKey,
    StorageService,
    StorageValue,
)


class FileStorageAdapter:
    """File system implementation of the StorageService port."""

    def __init__(self, base_path: Path | None = None):
        """Initialize with optional base path."""
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, key: StorageKey) -> Path:
        """Resolve a storage key to an absolute path."""
        if isinstance(key, Path):
            path = key
        else:
            path = Path(key)
        
        if not path.is_absolute():
            path = self.base_path / path
        
        return path

    def save(
        self, 
        key: StorageKey, 
        value: StorageValue, 
        metadata: Metadata | None = None
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
        key: StorageKey,
        expected_type: type[Any] | None = None
    ) -> StorageValue:
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

    def exists(self, key: StorageKey) -> bool:
        """Check if a key exists in storage."""
        path = self._resolve_path(key)
        return path.exists()

    def delete(self, key: StorageKey) -> None:
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
        prefix: StorageKey | None = None,
        pattern: str | None = None
    ) -> list[StorageKey]:
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

    def get_metadata(self, key: StorageKey) -> Metadata | None:
        """Get metadata for a stored value."""
        path = self._resolve_path(key)
        
        if not path.exists():
            raise KeyError(f"Storage key not found: {key}")
        
        meta_path = path.with_suffix(path.suffix + ".meta")
        if meta_path.exists():
            with open(meta_path, "r") as f:
                return json.load(f)
        
        return None


class ModelFileStorageAdapter:
    """File system implementation of the ModelStorageService port."""

    def __init__(self, base_path: Path | None = None):
        """Initialize with optional base path."""
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.storage = FileStorageAdapter(base_path)

    def save_model(
        self,
        model: Any,
        path: Path,
        include_optimizer: bool = True,
        include_metrics: bool = True,
        metadata: Metadata | None = None
    ) -> None:
        """Save an MLX model with its associated state."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model weights using safetensors
        if hasattr(model, "parameters"):
            weights = dict(model.parameters())
            save_file(weights, path / "model.safetensors")
        
        # Save model config if available
        if hasattr(model, "config"):
            config = model.config
            if hasattr(config, "to_dict"):
                config = config.to_dict()
            self.storage.save(path / "config.json", config)
        
        # Save metadata
        full_metadata = metadata or {}
        full_metadata.update({
            "include_optimizer": include_optimizer,
            "include_metrics": include_metrics,
            "model_class": model.__class__.__name__,
            "model_module": model.__class__.__module__,
        })
        self.storage.save(path / "metadata.json", full_metadata)

    def load_model(
        self,
        path: Path,
        model_class: type[Any] | None = None,
        load_optimizer: bool = True,
        load_metrics: bool = True
    ) -> tuple[Any, Metadata | None]:
        """Load an MLX model with its associated state."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Model path not found: {path}")
        
        # Load metadata
        metadata = None
        if (path / "metadata.json").exists():
            metadata = self.storage.load(path / "metadata.json")
        
        # Load config
        config = None
        if (path / "config.json").exists():
            config = self.storage.load(path / "config.json")
        
        # Load weights
        weights = {}
        if (path / "model.safetensors").exists():
            with safe_open(path / "model.safetensors", framework="mlx") as f:
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
        checkpoint_data: dict[str, Any],
        path: Path,
        keep_last_n: int | None = None
    ) -> None:
        """Save a training checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save checkpoint data
        self.storage.save(path, checkpoint_data)
        
        # Clean up old checkpoints if requested
        if keep_last_n is not None and keep_last_n > 0:
            checkpoint_dir = path.parent
            checkpoints = sorted(
                checkpoint_dir.glob("checkpoint_*"),
                key=lambda p: p.stat().st_mtime
            )
            
            # Remove old checkpoints
            for old_checkpoint in checkpoints[:-keep_last_n]:
                old_checkpoint.unlink()

    def load_checkpoint(
        self,
        path: Path,
        strict: bool = True
    ) -> dict[str, Any]:
        """Load a training checkpoint."""
        path = Path(path)
        
        if not path.exists():
            if strict:
                raise FileNotFoundError(f"Checkpoint not found: {path}")
            else:
                return {}
        
        return self.storage.load(path)

    def list_checkpoints(
        self,
        directory: Path,
        pattern: str = "checkpoint_*"
    ) -> list[Path]:
        """List available checkpoints."""
        directory = Path(directory)
        
        if not directory.exists():
            return []
        
        checkpoints = list(directory.glob(pattern))
        # Sort by modification time
        return sorted(checkpoints, key=lambda p: p.stat().st_mtime)