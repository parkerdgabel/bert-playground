"""Filesystem implementation of CheckpointManager port."""

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import re

from infrastructure.di import adapter, Scope
from ports.secondary.checkpointing import CheckpointManager, CheckpointInfo
from ports.secondary.optimization import Optimizer
from domain.protocols.models import Model


@adapter(CheckpointManager, scope=Scope.SINGLETON)
class FilesystemCheckpointManager:
    """Filesystem-based checkpoint manager implementation."""
    
    def __init__(self, checkpoint_dir: Path | str):
        """Initialize filesystem checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for storing checkpoints
        """
        self._checkpoint_dir = Path(checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_file = self._checkpoint_dir / "checkpoints.json"
        self._metadata = self._load_metadata()
    
    @property
    def checkpoint_dir(self) -> Path:
        """Get checkpoint directory."""
        return self._checkpoint_dir
    
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
        # Generate checkpoint name
        step = state.get("global_step", 0)
        epoch = state.get("current_epoch", 0)
        checkpoint_name = f"checkpoint_epoch{epoch}_step{step}"
        checkpoint_path = self._checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        model_path = checkpoint_path / "model.safetensors"
        self._save_model_weights(model, model_path)
        
        # Save optimizer state
        optimizer_path = checkpoint_path / "optimizer.pkl"
        self._save_optimizer_state(optimizer, optimizer_path)
        
        # Save training state
        state_path = checkpoint_path / "state.json"
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        # Create checkpoint info
        info = CheckpointInfo(
            path=checkpoint_path,
            step=step,
            epoch=epoch,
            train_loss=metrics.get("train_loss", 0.0),
            val_loss=metrics.get("val_loss"),
            metrics=metrics,
            created_at=datetime.now().isoformat(),
            size_bytes=self._get_directory_size(checkpoint_path),
            is_best=is_best,
        )
        
        # Update metadata
        self._metadata[checkpoint_name] = info.to_dict()
        if is_best:
            # Update all other checkpoints to not be best
            for name, data in self._metadata.items():
                if name != checkpoint_name:
                    data["is_best"] = False
            # Create symlink to best checkpoint
            best_link = self._checkpoint_dir / "best"
            if best_link.exists():
                best_link.unlink()
            best_link.symlink_to(checkpoint_path.name)
        
        # Create latest symlink
        latest_link = self._checkpoint_dir / "latest"
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(checkpoint_path.name)
        
        self._save_metadata()
        
        return checkpoint_path
    
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
        checkpoint_path = path if path.is_absolute() else self._checkpoint_dir / path
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load model weights
        model_path = checkpoint_path / "model.safetensors"
        self._load_model_weights(model, model_path, strict=strict)
        
        # Load optimizer state if provided
        if optimizer is not None:
            optimizer_path = checkpoint_path / "optimizer.pkl"
            if optimizer_path.exists():
                self._load_optimizer_state(optimizer, optimizer_path)
        
        # Load training state
        state_path = checkpoint_path / "state.json"
        with open(state_path, 'r') as f:
            state = json.load(f)
        
        return state
    
    def get_best_checkpoint(self) -> CheckpointInfo | None:
        """Get information about the best checkpoint.
        
        Returns:
            Best checkpoint info or None if no checkpoints
        """
        for name, data in self._metadata.items():
            if data.get("is_best", False):
                return self._dict_to_checkpoint_info(name, data)
        return None
    
    def get_latest_checkpoint(self) -> CheckpointInfo | None:
        """Get information about the latest checkpoint.
        
        Returns:
            Latest checkpoint info or None if no checkpoints
        """
        if not self._metadata:
            return None
        
        # Sort by creation time
        sorted_checkpoints = sorted(
            self._metadata.items(),
            key=lambda x: x[1].get("created_at", ""),
            reverse=True
        )
        
        if sorted_checkpoints:
            name, data = sorted_checkpoints[0]
            return self._dict_to_checkpoint_info(name, data)
        
        return None
    
    def list_checkpoints(self) -> list[CheckpointInfo]:
        """List all available checkpoints.
        
        Returns:
            List of checkpoint information, sorted by creation time
        """
        checkpoints = []
        
        for name, data in self._metadata.items():
            info = self._dict_to_checkpoint_info(name, data)
            if info:
                checkpoints.append(info)
        
        # Sort by creation time
        checkpoints.sort(key=lambda x: x.created_at)
        
        return checkpoints
    
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
        removed_paths = []
        
        # Get all checkpoints sorted by creation time
        all_checkpoints = sorted(
            [(name, data) for name, data in self._metadata.items()],
            key=lambda x: x[1].get("created_at", ""),
            reverse=True
        )
        
        if not all_checkpoints:
            return removed_paths
        
        # Determine which checkpoints to keep
        keep_names = set()
        
        # Keep best checkpoints
        best_checkpoints = [
            name for name, data in all_checkpoints
            if data.get("is_best", False)
        ][:keep_best]
        keep_names.update(best_checkpoints)
        
        # Keep latest checkpoints
        latest_checkpoints = [name for name, _ in all_checkpoints[:keep_last]]
        keep_names.update(latest_checkpoints)
        
        # Keep every N epochs if specified
        if keep_every_n_epochs:
            for name, data in all_checkpoints:
                epoch = data.get("epoch", 0)
                if epoch % keep_every_n_epochs == 0:
                    keep_names.add(name)
        
        # Remove checkpoints not in keep list
        for name, data in all_checkpoints:
            if name not in keep_names:
                checkpoint_path = self._checkpoint_dir / name
                if checkpoint_path.exists():
                    shutil.rmtree(checkpoint_path)
                    removed_paths.append(checkpoint_path)
                    del self._metadata[name]
        
        self._save_metadata()
        
        return removed_paths
    
    def delete_checkpoint(self, path: Path) -> bool:
        """Delete a specific checkpoint.
        
        Args:
            path: Checkpoint path
            
        Returns:
            True if deleted successfully
        """
        checkpoint_path = path if path.is_absolute() else self._checkpoint_dir / path
        
        if not checkpoint_path.exists():
            return False
        
        # Remove from metadata
        checkpoint_name = checkpoint_path.name
        if checkpoint_name in self._metadata:
            del self._metadata[checkpoint_name]
            self._save_metadata()
        
        # Remove directory
        shutil.rmtree(checkpoint_path)
        
        return True
    
    def verify_checkpoint(self, path: Path) -> bool:
        """Verify that a checkpoint is valid and loadable.
        
        Args:
            path: Checkpoint path
            
        Returns:
            True if checkpoint is valid
        """
        checkpoint_path = path if path.is_absolute() else self._checkpoint_dir / path
        
        if not checkpoint_path.exists():
            return False
        
        # Check required files exist
        required_files = [
            "model.safetensors",
            "optimizer.pkl",
            "state.json",
        ]
        
        for file_name in required_files:
            if not (checkpoint_path / file_name).exists():
                return False
        
        # Try to load state file
        try:
            state_path = checkpoint_path / "state.json"
            with open(state_path, 'r') as f:
                json.load(f)
            return True
        except Exception:
            return False
    
    def get_checkpoint_info(self, path: Path) -> CheckpointInfo | None:
        """Get information about a specific checkpoint.
        
        Args:
            path: Checkpoint path
            
        Returns:
            Checkpoint info or None if not found
        """
        checkpoint_path = path if path.is_absolute() else self._checkpoint_dir / path
        checkpoint_name = checkpoint_path.name
        
        if checkpoint_name in self._metadata:
            return self._dict_to_checkpoint_info(
                checkpoint_name,
                self._metadata[checkpoint_name]
            )
        
        return None
    
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
        checkpoint_path = self._checkpoint_dir / f"partial_{name}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save data as JSON or pickle depending on content
        data_path = checkpoint_path / "data.json"
        try:
            with open(data_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except TypeError:
            # Fall back to pickle for non-JSON serializable data
            import pickle
            data_path = checkpoint_path / "data.pkl"
            with open(data_path, 'wb') as f:
                pickle.dump(data, f)
        
        return checkpoint_path
    
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
        checkpoint_path = path if path.is_absolute() else self._checkpoint_dir / path
        
        # Try JSON first
        json_path = checkpoint_path / "data.json"
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
        else:
            # Try pickle
            import pickle
            pkl_path = checkpoint_path / "data.pkl"
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
        
        # Filter by keys if specified
        if keys:
            data = {k: v for k, v in data.items() if k in keys}
        
        return data
    
    def create_checkpoint_index(self) -> dict[str, Any]:
        """Create an index of all checkpoints with metadata.
        
        Returns:
            Dictionary with checkpoint metadata
        """
        return {
            "checkpoint_dir": str(self._checkpoint_dir),
            "total_checkpoints": len(self._metadata),
            "checkpoints": self._metadata,
            "best_checkpoint": self.get_best_checkpoint().to_dict() if self.get_best_checkpoint() else None,
            "latest_checkpoint": self.get_latest_checkpoint().to_dict() if self.get_latest_checkpoint() else None,
        }
    
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
        source_path = checkpoint_path if checkpoint_path.is_absolute() else self._checkpoint_dir / checkpoint_path
        
        if not source_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {source_path}")
        
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "safetensors":
            # Copy model weights
            shutil.copy2(source_path / "model.safetensors", export_path)
            
            if include_optimizer:
                # Create directory for full export
                export_dir = export_path.with_suffix("")
                export_dir.mkdir(exist_ok=True)
                shutil.copy2(source_path / "model.safetensors", export_dir / "model.safetensors")
                shutil.copy2(source_path / "optimizer.pkl", export_dir / "optimizer.pkl")
                return export_dir
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        return export_path
    
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
        path1 = checkpoint1 if checkpoint1.is_absolute() else self._checkpoint_dir / checkpoint1
        path2 = checkpoint2 if checkpoint2.is_absolute() else self._checkpoint_dir / checkpoint2
        
        info1 = self.get_checkpoint_info(path1)
        info2 = self.get_checkpoint_info(path2)
        
        if not info1 or not info2:
            raise ValueError("One or both checkpoints not found")
        
        comparison = {
            "checkpoint1": info1.to_dict(),
            "checkpoint2": info2.to_dict(),
            "differences": {
                "epochs": info2.epoch - info1.epoch,
                "steps": info2.step - info1.step,
                "train_loss_diff": info2.train_loss - info1.train_loss if info1.train_loss and info2.train_loss else None,
                "val_loss_diff": info2.val_loss - info1.val_loss if info1.val_loss and info2.val_loss else None,
            },
            "metrics_comparison": {},
        }
        
        # Compare metrics
        for metric in set(info1.metrics.keys()) | set(info2.metrics.keys()):
            val1 = info1.metrics.get(metric)
            val2 = info2.metrics.get(metric)
            if val1 is not None and val2 is not None:
                comparison["metrics_comparison"][metric] = {
                    "checkpoint1": val1,
                    "checkpoint2": val2,
                    "difference": val2 - val1,
                }
        
        return comparison
    
    # Private helper methods
    
    def _load_metadata(self) -> dict[str, Any]:
        """Load checkpoint metadata from disk."""
        if self._metadata_file.exists():
            with open(self._metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self) -> None:
        """Save checkpoint metadata to disk."""
        with open(self._metadata_file, 'w') as f:
            json.dump(self._metadata, f, indent=2)
    
    def _dict_to_checkpoint_info(self, name: str, data: dict[str, Any]) -> CheckpointInfo | None:
        """Convert dictionary to CheckpointInfo."""
        checkpoint_path = self._checkpoint_dir / name
        if not checkpoint_path.exists():
            return None
        
        return CheckpointInfo(
            path=checkpoint_path,
            step=data.get("step", 0),
            epoch=data.get("epoch", 0),
            train_loss=data.get("train_loss", 0.0),
            val_loss=data.get("val_loss"),
            metrics=data.get("metrics", {}),
            created_at=data.get("created_at", ""),
            size_bytes=data.get("size_bytes", 0),
            is_best=data.get("is_best", False),
        )
    
    def _get_directory_size(self, path: Path) -> int:
        """Get total size of directory in bytes."""
        total_size = 0
        for file_path in path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size
    
    def _save_model_weights(self, model: Model, path: Path) -> None:
        """Save model weights to file."""
        # This would use the actual model's save method
        # For now, we'll use a placeholder implementation
        import pickle
        
        # Extract model state
        if hasattr(model, 'state_dict'):
            state = model.state_dict()
        elif hasattr(model, 'get_weights'):
            state = model.get_weights()
        else:
            state = {'model': model}
        
        # For safetensors format, we'd use the safetensors library
        # For now, use pickle as fallback
        with open(path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(state, f)
    
    def _load_model_weights(self, model: Model, path: Path, strict: bool = True) -> None:
        """Load model weights from file."""
        import pickle
        
        # Load state
        pkl_path = path.with_suffix('.pkl')
        if pkl_path.exists():
            with open(pkl_path, 'rb') as f:
                state = pickle.load(f)
        else:
            raise FileNotFoundError(f"Model weights not found: {path}")
        
        # Apply to model
        if hasattr(model, 'load_state_dict'):
            model.load_state_dict(state, strict=strict)
        elif hasattr(model, 'set_weights'):
            model.set_weights(state)
    
    def _save_optimizer_state(self, optimizer: Optimizer, path: Path) -> None:
        """Save optimizer state to file."""
        import pickle
        
        # Extract optimizer state
        if hasattr(optimizer, 'state_dict'):
            state = optimizer.state_dict()
        elif hasattr(optimizer, 'get_state'):
            state = optimizer.get_state()
        else:
            state = {'optimizer': optimizer}
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    def _load_optimizer_state(self, optimizer: Optimizer, path: Path) -> None:
        """Load optimizer state from file."""
        import pickle
        
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        # Apply to optimizer
        if hasattr(optimizer, 'load_state_dict'):
            optimizer.load_state_dict(state)
        elif hasattr(optimizer, 'set_state'):
            optimizer.set_state(state)