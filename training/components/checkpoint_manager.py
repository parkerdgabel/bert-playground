"""Checkpoint manager component for handling all checkpoint operations.

This component is responsible for:
- Saving model checkpoints
- Loading checkpoints
- Managing checkpoint retention policies
- Tracking best checkpoints
"""

import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
import mlx.core as mx
from loguru import logger

from core.protocols.training import TrainingState, Optimizer, CheckpointManager as CheckpointManagerProtocol
from core.protocols.models import Model


class CheckpointManager(CheckpointManagerProtocol):
    """Manages checkpoint saving, loading, and retention.
    
    This component handles:
    - Checkpoint serialization/deserialization
    - Best model tracking
    - Checkpoint cleanup policies
    - Metadata management
    """
    
    def __init__(
        self,
        checkpoint_dir: Path,
        save_total_limit: int = 5,
        keep_best_only: bool = False,
    ):
        """Initialize the checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            save_total_limit: Maximum number of checkpoints to keep
            keep_best_only: Whether to only keep the best checkpoint
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_total_limit = save_total_limit
        self.keep_best_only = keep_best_only
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Track checkpoints
        self._checkpoints: list[tuple[int, Path]] = []  # (step, path) tuples
        self._best_checkpoint: Path | None = None
        
        # Load existing checkpoint metadata if available
        self._load_checkpoint_metadata()
        
        logger.debug(f"Initialized CheckpointManager at {checkpoint_dir}")
        
    def save_checkpoint(
        self,
        model: Model,
        optimizer: Optimizer,
        state: TrainingState,
        metrics: dict[str, float],
        is_best: bool = False,
        name: str | None = None,
    ) -> Path:
        """Save a training checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            state: Training state to save
            metrics: Current metrics
            is_best: Whether this is the best checkpoint
            name: Optional checkpoint name
            
        Returns:
            Path to the saved checkpoint
        """
        # Determine checkpoint name
        if name is None:
            name = f"checkpoint-{state.global_step}"
            
        checkpoint_path = self.checkpoint_dir / name
        checkpoint_path.mkdir(exist_ok=True)
        
        # Save model weights
        model_path = checkpoint_path / "model.safetensors"
        mx.save_safetensors(str(model_path), dict(model.parameters()))
        
        # Save optimizer state
        optimizer_path = checkpoint_path / "optimizer.safetensors"
        mx.save_safetensors(str(optimizer_path), optimizer.state)
        
        # Save training state
        state_path = checkpoint_path / "state.json"
        with open(state_path, "w") as f:
            json.dump(state.to_dict(), f, indent=2)
            
        # Save metrics
        metrics_path = checkpoint_path / "metrics.json"
        with open(metrics_path, "w") as f:
            # Convert metrics to JSON-serializable format
            json_metrics = self._make_json_serializable(metrics)
            json.dump(json_metrics, f, indent=2)
            
        # Save metadata
        metadata = {
            "step": state.global_step,
            "epoch": state.epoch,
            "is_best": is_best,
            "checkpoint_name": name,
        }
        metadata_path = checkpoint_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
            
        # Update checkpoint tracking
        if name not in ["best", "final"]:
            self._checkpoints.append((state.global_step, checkpoint_path))
            self._checkpoints.sort(key=lambda x: x[0])
            
        if is_best:
            self._best_checkpoint = checkpoint_path
            # Also save as "best" checkpoint
            best_path = self.checkpoint_dir / "best"
            if best_path.exists():
                shutil.rmtree(best_path)
            shutil.copytree(checkpoint_path, best_path)
            
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
        
        # Save checkpoint metadata
        self._save_checkpoint_metadata()
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        return checkpoint_path
        
    def load_checkpoint(
        self,
        path: Path,
        model: Model,
        optimizer: Optimizer,
    ) -> TrainingState:
        """Load a training checkpoint.
        
        Args:
            path: Path to checkpoint
            model: Model to load weights into
            optimizer: Optimizer to load state into
            
        Returns:
            Loaded training state
        """
        checkpoint_path = Path(path)
        
        # Load model weights
        model_path = checkpoint_path / "model.safetensors"
        if model_path.exists():
            weights = mx.load(str(model_path))
            model.load_weights(list(weights.items()))
        else:
            raise FileNotFoundError(f"Model weights not found: {model_path}")
            
        # Load optimizer state
        optimizer_path = checkpoint_path / "optimizer.safetensors"
        if optimizer_path.exists():
            optimizer.state = mx.load(str(optimizer_path))
            
        # Load training state
        state_path = checkpoint_path / "state.json"
        if state_path.exists():
            with open(state_path, "r") as f:
                state_dict = json.load(f)
            state = TrainingState.from_dict(state_dict)
        else:
            state = TrainingState()
            
        logger.info(f"Loaded checkpoint from: {checkpoint_path}")
        return state
        
    def get_best_checkpoint(self) -> Path | None:
        """Get path to best checkpoint.
        
        Returns:
            Path to best checkpoint or None
        """
        return self._best_checkpoint
        
    def get_latest_checkpoint(self) -> Path | None:
        """Get path to latest checkpoint.
        
        Returns:
            Path to latest checkpoint or None
        """
        if self._checkpoints:
            return self._checkpoints[-1][1]
        return None
        
    def cleanup_old_checkpoints(self, keep_best: int = 1, keep_last: int = 1) -> None:
        """Remove old checkpoints.
        
        Args:
            keep_best: Number of best checkpoints to keep
            keep_last: Number of recent checkpoints to keep
        """
        # This is handled by _cleanup_checkpoints() which is called after each save
        self._cleanup_checkpoints()
        
    @property
    def checkpoint_dir(self) -> Path:
        """Get checkpoint directory.
        
        Returns:
            Checkpoint directory path
        """
        return self._checkpoint_dir
        
    @checkpoint_dir.setter
    def checkpoint_dir(self, value: Path) -> None:
        """Set checkpoint directory."""
        self._checkpoint_dir = Path(value)
        
    def _cleanup_checkpoints(self) -> None:
        """Clean up old checkpoints based on retention policy."""
        if self.keep_best_only:
            # Keep only the best checkpoint
            for step, path in self._checkpoints:
                if path != self._best_checkpoint and path.name not in ["best", "final"]:
                    if path.exists():
                        shutil.rmtree(path)
                        logger.debug(f"Removed checkpoint: {path}")
            self._checkpoints = [(s, p) for s, p in self._checkpoints if p == self._best_checkpoint]
            
        elif self.save_total_limit > 0:
            # Keep only the most recent checkpoints
            while len(self._checkpoints) > self.save_total_limit:
                step, path = self._checkpoints.pop(0)
                if path.exists() and path.name not in ["best", "final"]:
                    shutil.rmtree(path)
                    logger.debug(f"Removed old checkpoint: {path}")
                    
    def _save_checkpoint_metadata(self) -> None:
        """Save checkpoint metadata to track checkpoints."""
        metadata = {
            "checkpoints": [(step, str(path)) for step, path in self._checkpoints],
            "best_checkpoint": str(self._best_checkpoint) if self._best_checkpoint else None,
        }
        metadata_path = self.checkpoint_dir / "checkpoint_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
            
    def _load_checkpoint_metadata(self) -> None:
        """Load checkpoint metadata if available."""
        metadata_path = self.checkpoint_dir / "checkpoint_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self._checkpoints = [(step, Path(path)) for step, path in metadata.get("checkpoints", [])]
            best_path = metadata.get("best_checkpoint")
            self._best_checkpoint = Path(best_path) if best_path else None
            
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert MLX arrays and other non-serializable objects to JSON format."""
        if isinstance(obj, mx.array):
            return obj.item() if obj.size == 1 else obj.tolist()
        elif hasattr(obj, "__module__") and obj.__module__ and "mlx" in obj.__module__:
            if hasattr(obj, "item") and hasattr(obj, "size"):
                return obj.item() if obj.size == 1 else obj.tolist()
            else:
                return str(obj)
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj