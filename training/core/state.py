"""
State management for training including checkpoints and training state persistence.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

import mlx.core as mx
import mlx.nn as nn
from safetensors.mlx import save_model, load_model
from loguru import logger

from .protocols import Model, Optimizer, TrainingState


class TrainingStateManager:
    """Manages training state persistence and recovery."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize state manager.
        
        Args:
            output_dir: Directory for saving state files
        """
        self.output_dir = Path(output_dir)
        self.state_dir = self.output_dir / "training_state"
        self.state_dir.mkdir(parents=True, exist_ok=True)
    
    def save_state(self, state: TrainingState, name: str = "latest") -> Path:
        """
        Save training state to disk.
        
        Args:
            state: Training state to save
            name: Name for the state file
            
        Returns:
            Path to saved state file
        """
        state_path = self.state_dir / f"{name}_state.json"
        
        with open(state_path, "w") as f:
            json.dump(state.to_dict(), f, indent=2)
        
        logger.debug(f"Saved training state to {state_path}")
        return state_path
    
    def load_state(self, name: str = "latest") -> Optional[TrainingState]:
        """
        Load training state from disk.
        
        Args:
            name: Name of the state file to load
            
        Returns:
            Loaded training state or None if not found
        """
        state_path = self.state_dir / f"{name}_state.json"
        
        if not state_path.exists():
            logger.warning(f"State file not found: {state_path}")
            return None
        
        with open(state_path) as f:
            state_dict = json.load(f)
        
        state = TrainingState.from_dict(state_dict)
        logger.debug(f"Loaded training state from {state_path}")
        
        return state
    
    def list_states(self) -> List[str]:
        """List available state files."""
        state_files = list(self.state_dir.glob("*_state.json"))
        return [f.stem.replace("_state", "") for f in state_files]
    
    def cleanup_old_states(self, keep: int = 3) -> None:
        """
        Remove old state files, keeping the most recent ones.
        
        Args:
            keep: Number of recent states to keep
        """
        state_files = sorted(
            self.state_dir.glob("*_state.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        for state_file in state_files[keep:]:
            state_file.unlink()
            logger.debug(f"Removed old state file: {state_file}")


class CheckpointManager:
    """Manages model checkpoints with automatic cleanup and best model tracking."""
    
    def __init__(
        self,
        checkpoint_dir: Path,
        save_total_limit: Optional[int] = None,
        keep_best: bool = True,
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for saving checkpoints
            save_total_limit: Maximum number of checkpoints to keep
            keep_best: Whether to always keep the best checkpoint
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_total_limit = save_total_limit
        self.keep_best = keep_best
        
        # Track best checkpoint
        self.best_checkpoint_path: Optional[Path] = None
        self.checkpoint_metadata: Dict[str, Any] = {}
        
        # Load existing metadata
        self._load_metadata()
    
    def save_checkpoint(
        self,
        model: Model,
        optimizer: Optimizer,
        state: TrainingState,
        metrics: Dict[str, float],
        is_best: bool = False,
        name: Optional[str] = None,
    ) -> Path:
        """
        Save a training checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            state: Current training state
            metrics: Current metrics
            is_best: Whether this is the best checkpoint
            name: Optional checkpoint name
            
        Returns:
            Path to saved checkpoint
        """
        # Create checkpoint name
        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"checkpoint_{state.global_step}_{timestamp}"
        
        # Create checkpoint directory
        checkpoint_path = self.checkpoint_dir / name
        checkpoint_path.mkdir(exist_ok=True)
        
        # Save model weights
        model_path = checkpoint_path / "model.safetensors"
        save_model(model, str(model_path))
        
        # Save model config if available
        if hasattr(model, "config") and model.config is not None:
            config_path = checkpoint_path / "config.json"
            if hasattr(model.config, "to_dict"):
                config_dict = model.config.to_dict()
            else:
                config_dict = model.config.__dict__
            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=2)
        
        # Save optimizer state
        optimizer_path = checkpoint_path / "optimizer.json"
        optimizer_state = {
            "learning_rate": optimizer.learning_rate,
            "state": optimizer.state if hasattr(optimizer, "state") else {},
        }
        with open(optimizer_path, "w") as f:
            json.dump(optimizer_state, f, indent=2)
        
        # Save training state
        state_path = checkpoint_path / "training_state.json"
        with open(state_path, "w") as f:
            json.dump(state.to_dict(), f, indent=2)
        
        # Save metrics
        metrics_path = checkpoint_path / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Save checkpoint metadata
        metadata = {
            "name": name,
            "path": str(checkpoint_path),
            "step": state.global_step,
            "epoch": state.epoch,
            "metrics": metrics,
            "is_best": is_best,
            "timestamp": datetime.now().isoformat(),
        }
        
        metadata_path = checkpoint_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Update checkpoint tracking
        self.checkpoint_metadata[name] = metadata
        self._save_metadata()
        
        # Update best checkpoint if needed
        if is_best:
            self.best_checkpoint_path = checkpoint_path
            # Create symlink to best checkpoint
            best_link = self.checkpoint_dir / "best"
            if best_link.exists():
                best_link.unlink()
            best_link.symlink_to(checkpoint_path.name)
        
        # Cleanup old checkpoints
        if self.save_total_limit is not None:
            self._cleanup_old_checkpoints()
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        return checkpoint_path
    
    def load_checkpoint(
        self,
        path: Path,
        model: Model,
        optimizer: Optimizer,
    ) -> TrainingState:
        """
        Load a training checkpoint.
        
        Args:
            path: Path to checkpoint directory
            model: Model to load weights into
            optimizer: Optimizer to load state into
            
        Returns:
            Loaded training state
        """
        checkpoint_path = Path(path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load model weights
        model_path = checkpoint_path / "model.safetensors"
        if model_path.exists():
            load_model(model, str(model_path))
            logger.info(f"Loaded model weights from {model_path}")
        else:
            logger.warning(f"Model weights not found at {model_path}")
        
        # Load optimizer state
        optimizer_path = checkpoint_path / "optimizer.json"
        if optimizer_path.exists():
            with open(optimizer_path) as f:
                optimizer_state = json.load(f)
            
            optimizer.learning_rate = optimizer_state.get("learning_rate", optimizer.learning_rate)
            if hasattr(optimizer, "state") and "state" in optimizer_state:
                optimizer.state = optimizer_state["state"]
            
            logger.info(f"Loaded optimizer state from {optimizer_path}")
        
        # Load training state
        state_path = checkpoint_path / "training_state.json"
        if state_path.exists():
            with open(state_path) as f:
                state_dict = json.load(f)
            state = TrainingState.from_dict(state_dict)
            logger.info(f"Loaded training state from {state_path}")
        else:
            state = TrainingState()
            logger.warning(f"Training state not found at {state_path}, using default")
        
        return state
    
    def get_best_checkpoint(self) -> Optional[Path]:
        """Get path to best checkpoint."""
        if self.best_checkpoint_path and self.best_checkpoint_path.exists():
            return self.best_checkpoint_path
        
        # Try to find best checkpoint from metadata
        best_checkpoint = None
        for name, metadata in self.checkpoint_metadata.items():
            if metadata.get("is_best", False):
                best_checkpoint = Path(metadata["path"])
                if best_checkpoint.exists():
                    self.best_checkpoint_path = best_checkpoint
                    return best_checkpoint
        
        # Check for best symlink
        best_link = self.checkpoint_dir / "best"
        if best_link.exists() and best_link.is_symlink():
            return best_link.resolve()
        
        return None
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to latest checkpoint."""
        checkpoints = self._list_checkpoints()
        if not checkpoints:
            return None
        
        # Sort by step number
        checkpoints.sort(key=lambda c: self.checkpoint_metadata.get(c.name, {}).get("step", 0))
        return checkpoints[-1]
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints with metadata."""
        checkpoints = []
        
        for checkpoint_dir in self._list_checkpoints():
            metadata_path = checkpoint_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                checkpoints.append(metadata)
            else:
                # Create basic metadata
                checkpoints.append({
                    "name": checkpoint_dir.name,
                    "path": str(checkpoint_dir),
                })
        
        return checkpoints
    
    def cleanup_old_checkpoints(self, keep_best: int = 1, keep_last: int = 1) -> None:
        """
        Remove old checkpoints.
        
        Args:
            keep_best: Number of best checkpoints to keep
            keep_last: Number of recent checkpoints to keep
        """
        self._cleanup_old_checkpoints(keep_best=keep_best, keep_last=keep_last)
    
    def _list_checkpoints(self) -> List[Path]:
        """List checkpoint directories."""
        checkpoints = []
        
        for item in self.checkpoint_dir.iterdir():
            if item.is_dir() and (item / "model.safetensors").exists():
                checkpoints.append(item)
        
        return checkpoints
    
    def _cleanup_old_checkpoints(
        self,
        keep_best: Optional[int] = None,
        keep_last: Optional[int] = None,
    ) -> None:
        """Internal cleanup method."""
        if keep_best is None:
            keep_best = 1 if self.keep_best else 0
        if keep_last is None:
            keep_last = self.save_total_limit or 3
        
        checkpoints = self._list_checkpoints()
        
        # Identify checkpoints to keep
        keep_paths = set()
        
        # Keep best checkpoints
        best_checkpoints = [
            c for c in checkpoints
            if self.checkpoint_metadata.get(c.name, {}).get("is_best", False)
        ]
        for checkpoint in best_checkpoints[:keep_best]:
            keep_paths.add(checkpoint)
        
        # Keep recent checkpoints
        recent_checkpoints = sorted(
            checkpoints,
            key=lambda c: c.stat().st_mtime,
            reverse=True
        )
        for checkpoint in recent_checkpoints[:keep_last]:
            keep_paths.add(checkpoint)
        
        # Remove others
        for checkpoint in checkpoints:
            if checkpoint not in keep_paths:
                shutil.rmtree(checkpoint)
                # Remove from metadata
                if checkpoint.name in self.checkpoint_metadata:
                    del self.checkpoint_metadata[checkpoint.name]
                logger.debug(f"Removed old checkpoint: {checkpoint}")
        
        # Save updated metadata
        self._save_metadata()
    
    def _save_metadata(self) -> None:
        """Save checkpoint metadata."""
        metadata_path = self.checkpoint_dir / "checkpoints_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.checkpoint_metadata, f, indent=2)
    
    def _load_metadata(self) -> None:
        """Load checkpoint metadata."""
        metadata_path = self.checkpoint_dir / "checkpoints_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.checkpoint_metadata = json.load(f)
        
        # Verify checkpoints still exist
        valid_metadata = {}
        for name, metadata in self.checkpoint_metadata.items():
            checkpoint_path = Path(metadata["path"])
            if checkpoint_path.exists():
                valid_metadata[name] = metadata
        
        self.checkpoint_metadata = valid_metadata
    
    @property
    def checkpoint_dir(self) -> Path:
        """Get checkpoint directory."""
        return self._checkpoint_dir
    
    @checkpoint_dir.setter
    def checkpoint_dir(self, value: Path):
        """Set checkpoint directory."""
        self._checkpoint_dir = Path(value)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)