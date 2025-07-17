"""Comprehensive checkpoint management system for production MLX training.

This module provides robust checkpointing with automatic recovery, versioning,
and distributed training support.
"""

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from loguru import logger


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint."""
    
    checkpoint_id: str
    global_step: int
    epoch: int
    timestamp: float
    metrics: Dict[str, float] = field(default_factory=dict)
    is_best: bool = False
    validation_metric: Optional[float] = None
    training_time_seconds: float = 0.0
    model_hash: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "global_step": self.global_step,
            "epoch": self.epoch,
            "timestamp": self.timestamp,
            "metrics": self.metrics,
            "is_best": self.is_best,
            "validation_metric": self.validation_metric,
            "training_time_seconds": self.training_time_seconds,
            "model_hash": self.model_hash,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointMetadata":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint management."""
    
    # Basic settings
    checkpoint_dir: str = "./checkpoints"
    save_frequency: int = 1000  # Steps between checkpoints
    keep_last_n: int = 5  # Number of recent checkpoints to keep
    keep_best_n: int = 3  # Number of best checkpoints to keep
    
    # Advanced settings
    save_optimizer_state: bool = True
    save_random_state: bool = True
    save_training_history: bool = True
    atomic_saves: bool = True  # Use atomic writes for safety
    compression: bool = False  # Compress checkpoints
    
    # Auto-recovery settings
    auto_recover: bool = True
    recovery_check_interval: int = 100  # Steps between recovery checks
    max_recovery_attempts: int = 3
    
    # Validation settings
    validate_checkpoint_on_save: bool = True
    validate_checkpoint_on_load: bool = True
    
    # Performance settings
    async_save: bool = False  # Save checkpoints asynchronously
    save_timeout_seconds: float = 300.0  # Timeout for checkpoint saves


class CheckpointManager:
    """Manages checkpoints for MLX training with production features."""
    
    def __init__(self, config: CheckpointConfig):
        """Initialize checkpoint manager.
        
        Args:
            config: Checkpoint configuration
        """
        self.config = config
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Checkpoint tracking
        self.checkpoints: List[CheckpointMetadata] = []
        self.best_checkpoints: List[CheckpointMetadata] = []
        self.last_checkpoint_step = -1
        
        # Recovery state
        self.recovery_attempts = 0
        self.last_recovery_check = 0
        
        # Load existing checkpoint metadata
        self._load_checkpoint_metadata()
        
        logger.info(
            f"Checkpoint Manager initialized:\n"
            f"  Directory: {self.checkpoint_dir}\n"
            f"  Save frequency: {config.save_frequency} steps\n"
            f"  Keep last: {config.keep_last_n}\n"
            f"  Keep best: {config.keep_best_n}\n"
            f"  Auto-recovery: {config.auto_recover}"
        )
    
    def _load_checkpoint_metadata(self) -> None:
        """Load metadata for existing checkpoints."""
        metadata_file = self.checkpoint_dir / "checkpoint_metadata.json"
        
        if metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    data = json.load(f)
                
                self.checkpoints = [
                    CheckpointMetadata.from_dict(cp) for cp in data.get("checkpoints", [])
                ]
                self.best_checkpoints = [
                    CheckpointMetadata.from_dict(cp) for cp in data.get("best_checkpoints", [])
                ]
                
                logger.info(
                    f"Loaded {len(self.checkpoints)} checkpoints, "
                    f"{len(self.best_checkpoints)} best checkpoints"
                )
            except Exception as e:
                logger.warning(f"Failed to load checkpoint metadata: {e}")
    
    def _save_checkpoint_metadata(self) -> None:
        """Save checkpoint metadata."""
        metadata_file = self.checkpoint_dir / "checkpoint_metadata.json"
        
        data = {
            "checkpoints": [cp.to_dict() for cp in self.checkpoints],
            "best_checkpoints": [cp.to_dict() for cp in self.best_checkpoints],
            "last_updated": time.time(),
        }
        
        try:
            with open(metadata_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save checkpoint metadata: {e}")
    
    def should_save_checkpoint(self, global_step: int) -> bool:
        """Check if checkpoint should be saved at this step.
        
        Args:
            global_step: Current global step
            
        Returns:
            True if checkpoint should be saved
        """
        if global_step <= self.last_checkpoint_step:
            return False
            
        return (global_step - self.last_checkpoint_step) >= self.config.save_frequency
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Any,
        global_step: int,
        epoch: int,
        metrics: Dict[str, float],
        training_state: Dict[str, Any],
        is_best: bool = False,
        checkpoint_name: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Save a checkpoint with all training state.
        
        Args:
            model: Model to save
            optimizer: Optimizer with state to save
            global_step: Current global step
            epoch: Current epoch
            metrics: Current metrics
            training_state: Additional training state
            is_best: Whether this is the best checkpoint
            checkpoint_name: Optional custom checkpoint name
            
        Returns:
            (success, checkpoint_path)
        """
        # Generate checkpoint ID
        if checkpoint_name:
            checkpoint_id = checkpoint_name
        else:
            checkpoint_id = f"checkpoint_step_{global_step:08d}"
        
        checkpoint_path = self.checkpoint_dir / checkpoint_id
        
        # Use atomic saves if configured
        if self.config.atomic_saves:
            temp_path = self.checkpoint_dir / f".tmp_{checkpoint_id}"
            save_path = temp_path
        else:
            save_path = checkpoint_path
        
        try:
            # Create checkpoint directory
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save model weights
            model_path = save_path / "model.safetensors"
            model.save_pretrained(str(save_path))
            
            # Save optimizer state if configured
            if self.config.save_optimizer_state and optimizer:
                optimizer_path = save_path / "optimizer.safetensors"
                try:
                    # Flatten optimizer state for saving
                    from mlx.utils import tree_flatten
                    
                    state_flat = dict(tree_flatten(optimizer.state))
                    mx.save_safetensors(str(optimizer_path), state_flat)
                except Exception as e:
                    logger.warning(f"Failed to save optimizer state: {e}")
            
            # Save training state
            state_data = {
                "global_step": global_step,
                "epoch": epoch,
                "metrics": metrics,
                "training_state": training_state,
                "timestamp": time.time(),
            }
            
            # Add random state if configured
            if self.config.save_random_state:
                state_data["random_state"] = {
                    "numpy_state": np.random.get_state(),
                    "python_random_state": self._get_python_random_state(),
                }
            
            state_path = save_path / "trainer_state.json"
            with open(state_path, "w") as f:
                json.dump(state_data, f, indent=2, default=str)
            
            # Save training history if configured
            if self.config.save_training_history and "history" in training_state:
                history_path = save_path / "training_history.json"
                with open(history_path, "w") as f:
                    json.dump(training_state["history"], f, indent=2)
            
            # Validate checkpoint if configured
            if self.config.validate_checkpoint_on_save:
                if not self._validate_checkpoint(save_path):
                    raise ValueError("Checkpoint validation failed")
            
            # Move from temp to final location if using atomic saves
            if self.config.atomic_saves:
                shutil.move(str(temp_path), str(checkpoint_path))
            
            # Create metadata
            metadata = CheckpointMetadata(
                checkpoint_id=checkpoint_id,
                global_step=global_step,
                epoch=epoch,
                timestamp=time.time(),
                metrics=metrics,
                is_best=is_best,
                validation_metric=metrics.get("val_loss"),
                training_time_seconds=training_state.get("total_time", 0.0),
            )
            
            # Update checkpoint tracking
            self.checkpoints.append(metadata)
            if is_best:
                self.best_checkpoints.append(metadata)
            
            self.last_checkpoint_step = global_step
            
            # Clean up old checkpoints
            self._cleanup_old_checkpoints()
            
            # Save metadata
            self._save_checkpoint_metadata()
            
            logger.info(
                f"Checkpoint saved: {checkpoint_path}\n"
                f"  Step: {global_step}, Epoch: {epoch}\n"
                f"  Metrics: {metrics}\n"
                f"  Is best: {is_best}"
            )
            
            return True, str(checkpoint_path)
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            
            # Clean up temp directory if it exists
            if self.config.atomic_saves and temp_path.exists():
                shutil.rmtree(temp_path, ignore_errors=True)
            
            return False, ""
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[Any] = None,
        strict: bool = True,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Load checkpoint and restore training state.
        
        Args:
            checkpoint_path: Path to checkpoint
            model: Model to load weights into
            optimizer: Optional optimizer to restore state
            strict: Whether to enforce strict weight loading
            
        Returns:
            (success, training_state)
        """
        checkpoint_dir = Path(checkpoint_path)
        
        if not checkpoint_dir.exists():
            logger.error(f"Checkpoint not found: {checkpoint_dir}")
            return False, {}
        
        try:
            # Validate checkpoint if configured
            if self.config.validate_checkpoint_on_load:
                if not self._validate_checkpoint(checkpoint_dir):
                    raise ValueError("Checkpoint validation failed")
            
            # Load model weights
            model_path = checkpoint_dir / "model.safetensors"
            if model_path.exists():
                # Load model using MLX model loading
                model.load_pretrained(str(checkpoint_dir))
            else:
                logger.warning("Model weights not found in checkpoint")
            
            # Load optimizer state if requested
            if optimizer and self.config.save_optimizer_state:
                optimizer_path = checkpoint_dir / "optimizer.safetensors"
                if optimizer_path.exists():
                    try:
                        from mlx.utils import tree_unflatten
                        
                        state_flat = mx.load(str(optimizer_path))
                        optimizer.state = tree_unflatten(list(state_flat.items()))
                    except Exception as e:
                        logger.warning(f"Failed to load optimizer state: {e}")
            
            # Load training state
            state_path = checkpoint_dir / "trainer_state.json"
            training_state = {}
            
            if state_path.exists():
                with open(state_path) as f:
                    state_data = json.load(f)
                
                training_state = {
                    "global_step": state_data.get("global_step", 0),
                    "epoch": state_data.get("epoch", 0),
                    "metrics": state_data.get("metrics", {}),
                    "training_state": state_data.get("training_state", {}),
                }
                
                # Restore random state if available
                if "random_state" in state_data and self.config.save_random_state:
                    try:
                        np.random.set_state(state_data["random_state"]["numpy_state"])
                        self._set_python_random_state(
                            state_data["random_state"].get("python_random_state")
                        )
                    except Exception as e:
                        logger.warning(f"Failed to restore random state: {e}")
            
            # Load training history if available
            history_path = checkpoint_dir / "training_history.json"
            if history_path.exists():
                with open(history_path) as f:
                    training_state["history"] = json.load(f)
            
            logger.info(
                f"Checkpoint loaded from: {checkpoint_dir}\n"
                f"  Global step: {training_state.get('global_step', 0)}\n"
                f"  Epoch: {training_state.get('epoch', 0)}"
            )
            
            return True, training_state
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False, {}
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to the latest checkpoint.
        
        Returns:
            Path to latest checkpoint or None
        """
        if not self.checkpoints:
            return None
        
        # Sort by global step
        sorted_checkpoints = sorted(
            self.checkpoints, key=lambda cp: cp.global_step, reverse=True
        )
        
        # Find the first existing checkpoint
        for checkpoint in sorted_checkpoints:
            checkpoint_path = self.checkpoint_dir / checkpoint.checkpoint_id
            if checkpoint_path.exists():
                return str(checkpoint_path)
        
        return None
    
    def get_best_checkpoint(self) -> Optional[str]:
        """Get path to the best checkpoint.
        
        Returns:
            Path to best checkpoint or None
        """
        if not self.best_checkpoints:
            return None
        
        # Find the first existing best checkpoint
        for checkpoint in reversed(self.best_checkpoints):
            checkpoint_path = self.checkpoint_dir / checkpoint.checkpoint_id
            if checkpoint_path.exists():
                return str(checkpoint_path)
        
        return None
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints.
        
        Returns:
            List of checkpoint information
        """
        checkpoints_info = []
        
        for checkpoint in self.checkpoints:
            checkpoint_path = self.checkpoint_dir / checkpoint.checkpoint_id
            if checkpoint_path.exists():
                info = checkpoint.to_dict()
                info["path"] = str(checkpoint_path)
                info["size_mb"] = self._get_checkpoint_size(checkpoint_path)
                checkpoints_info.append(info)
        
        return checkpoints_info
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints based on retention policy."""
        # Clean up regular checkpoints
        if len(self.checkpoints) > self.config.keep_last_n:
            # Sort by step
            sorted_checkpoints = sorted(
                self.checkpoints, key=lambda cp: cp.global_step
            )
            
            # Remove oldest checkpoints
            to_remove = sorted_checkpoints[: -self.config.keep_last_n]
            
            for checkpoint in to_remove:
                # Don't remove if it's a best checkpoint
                if not any(
                    bp.checkpoint_id == checkpoint.checkpoint_id
                    for bp in self.best_checkpoints
                ):
                    self._remove_checkpoint(checkpoint)
                    self.checkpoints.remove(checkpoint)
        
        # Clean up best checkpoints
        if len(self.best_checkpoints) > self.config.keep_best_n:
            # Sort by validation metric (assuming lower is better)
            sorted_best = sorted(
                self.best_checkpoints,
                key=lambda cp: cp.validation_metric or float("inf")
            )
            
            # Remove worst best checkpoints
            to_remove = sorted_best[self.config.keep_best_n:]
            
            for checkpoint in to_remove:
                # Only remove if it's not also a recent checkpoint
                if checkpoint not in self.checkpoints[-self.config.keep_last_n:]:
                    self._remove_checkpoint(checkpoint)
                self.best_checkpoints.remove(checkpoint)
    
    def _remove_checkpoint(self, checkpoint: CheckpointMetadata) -> None:
        """Remove a checkpoint from disk.
        
        Args:
            checkpoint: Checkpoint metadata
        """
        checkpoint_path = self.checkpoint_dir / checkpoint.checkpoint_id
        
        if checkpoint_path.exists():
            try:
                shutil.rmtree(checkpoint_path)
                logger.debug(f"Removed checkpoint: {checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {checkpoint_path}: {e}")
    
    def _validate_checkpoint(self, checkpoint_path: Path) -> bool:
        """Validate checkpoint integrity.
        
        Args:
            checkpoint_path: Path to checkpoint
            
        Returns:
            True if checkpoint is valid
        """
        required_files = ["model.safetensors", "trainer_state.json"]
        
        for file_name in required_files:
            file_path = checkpoint_path / file_name
            if not file_path.exists():
                logger.warning(f"Missing required file in checkpoint: {file_name}")
                return False
            
            # Check file size
            if file_path.stat().st_size == 0:
                logger.warning(f"Empty file in checkpoint: {file_name}")
                return False
        
        # Try to load state file
        try:
            with open(checkpoint_path / "trainer_state.json") as f:
                json.load(f)
        except Exception as e:
            logger.warning(f"Invalid trainer state file: {e}")
            return False
        
        return True
    
    def _get_checkpoint_size(self, checkpoint_path: Path) -> float:
        """Get checkpoint size in MB.
        
        Args:
            checkpoint_path: Path to checkpoint
            
        Returns:
            Size in MB
        """
        total_size = 0
        
        try:
            for file_path in checkpoint_path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception:
            pass
        
        return total_size / (1024 * 1024)
    
    def _get_python_random_state(self) -> Any:
        """Get Python random state."""
        try:
            import random
            return random.getstate()
        except Exception:
            return None
    
    def _set_python_random_state(self, state: Any) -> None:
        """Set Python random state."""
        if state is None:
            return
            
        try:
            import random
            random.setstate(state)
        except Exception:
            pass


class AutoRecoveryManager:
    """Manages automatic recovery from training failures."""
    
    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        max_attempts: int = 3,
        recovery_window_seconds: float = 300.0,
    ):
        """Initialize auto-recovery manager.
        
        Args:
            checkpoint_manager: Checkpoint manager instance
            max_attempts: Maximum recovery attempts
            recovery_window_seconds: Time window for recovery attempts
        """
        self.checkpoint_manager = checkpoint_manager
        self.max_attempts = max_attempts
        self.recovery_window_seconds = recovery_window_seconds
        
        # Recovery tracking
        self.recovery_attempts = 0
        self.last_recovery_time = 0.0
        self.recovery_history: List[Dict[str, Any]] = []
    
    def check_and_recover(
        self,
        model: nn.Module,
        optimizer: Any,
        current_step: int,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check if recovery is needed and attempt recovery.
        
        Args:
            model: Model to recover
            optimizer: Optimizer to recover
            current_step: Current training step
            
        Returns:
            (recovered, recovery_state)
        """
        # Check if we're within recovery window
        current_time = time.time()
        if (
            self.last_recovery_time > 0
            and (current_time - self.last_recovery_time) > self.recovery_window_seconds
        ):
            # Reset recovery attempts after window expires
            self.recovery_attempts = 0
        
        # Check if we've exceeded max attempts
        if self.recovery_attempts >= self.max_attempts:
            logger.warning(
                f"Exceeded maximum recovery attempts ({self.max_attempts})"
            )
            return False, {}
        
        # Try to find a checkpoint to recover from
        latest_checkpoint = self.checkpoint_manager.get_latest_checkpoint()
        
        if not latest_checkpoint:
            logger.warning("No checkpoint available for recovery")
            return False, {}
        
        # Attempt recovery
        logger.info(
            f"Attempting recovery from checkpoint: {latest_checkpoint}\n"
            f"  Recovery attempt: {self.recovery_attempts + 1}/{self.max_attempts}"
        )
        
        success, training_state = self.checkpoint_manager.load_checkpoint(
            latest_checkpoint, model, optimizer
        )
        
        if success:
            self.recovery_attempts += 1
            self.last_recovery_time = current_time
            
            # Record recovery
            self.recovery_history.append({
                "timestamp": current_time,
                "checkpoint": latest_checkpoint,
                "recovered_step": training_state.get("global_step", 0),
                "current_step": current_step,
                "attempt": self.recovery_attempts,
            })
            
            logger.info(
                f"Recovery successful!\n"
                f"  Recovered to step: {training_state.get('global_step', 0)}\n"
                f"  Current step was: {current_step}"
            )
            
            return True, training_state
        else:
            logger.error("Recovery failed")
            return False, {}
    
    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get recovery statistics.
        
        Returns:
            Recovery statistics
        """
        return {
            "total_recoveries": len(self.recovery_history),
            "recovery_attempts": self.recovery_attempts,
            "last_recovery_time": self.last_recovery_time,
            "recovery_history": self.recovery_history,
        }