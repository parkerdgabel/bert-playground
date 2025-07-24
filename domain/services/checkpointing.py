"""Checkpointing service - pure business logic for model persistence.

This service contains only the business logic for checkpointing,
without any dependencies on specific storage implementations.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
from domain.entities.model import BertModel
from domain.entities.training import TrainingSession, TrainingState
from domain.exceptions import CheckpointError
from infrastructure.di import service


@service
class CheckpointingService:
    """Service for checkpoint management business logic.
    
    This service defines the business rules for checkpointing
    without depending on specific storage implementations.
    """
    
    def create_checkpoint_metadata(
        self,
        model: BertModel,
        session: TrainingSession,
        checkpoint_path: Path,
    ) -> Dict[str, Any]:
        """Create metadata for a checkpoint.
        
        Args:
            model: Model being checkpointed
            session: Current training session
            checkpoint_path: Where checkpoint will be saved
            
        Returns:
            Checkpoint metadata dictionary
        """
        return {
            "checkpoint_path": str(checkpoint_path),
            "model_id": model.id,
            "model_name": model.name,
            "architecture": model.architecture.model_type.value,
            "session_id": session.session_id,
            "epoch": session.state.epoch,
            "global_step": session.state.global_step,
            "best_metric": session.state.best_val_metric,
            "timestamp": datetime.now().isoformat(),
            "training_time": session.state.accumulated_time.total_seconds() if session.state.accumulated_time else 0,
            "config": {
                "num_epochs": session.config.num_epochs,
                "batch_size": session.config.batch_size,
                "learning_rate": session.config.learning_rate,
            }
        }
    
    def should_save_checkpoint(
        self,
        state: TrainingState,
        save_strategy: str,
        save_steps: Optional[int] = None,
        save_epochs: Optional[int] = None,
        steps_per_epoch: Optional[int] = None,
    ) -> bool:
        """Determine if checkpoint should be saved.
        
        Args:
            state: Current training state
            save_strategy: Strategy for saving ("steps", "epoch", "best")
            save_steps: Save every N steps
            save_epochs: Save every N epochs
            steps_per_epoch: Steps per epoch for epoch-based saving
            
        Returns:
            True if checkpoint should be saved
        """
        if save_strategy == "steps" and save_steps:
            return state.global_step % save_steps == 0
        
        elif save_strategy == "epoch" and save_epochs and steps_per_epoch:
            # Check if we're at the start of an epoch that should be saved
            return state.epoch % save_epochs == 0 and state.global_step % steps_per_epoch == 0
        
        elif save_strategy == "best":
            # This would be determined by caller comparing metrics
            return False  # Caller should explicitly save for best
        
        return False
    
    def get_checkpoint_name(
        self,
        base_name: str,
        state: TrainingState,
        is_best: bool = False,
    ) -> str:
        """Generate checkpoint filename.
        
        Args:
            base_name: Base name for checkpoint
            state: Current training state
            is_best: Whether this is the best checkpoint
            
        Returns:
            Checkpoint filename
        """
        if is_best:
            return f"{base_name}_best"
        else:
            return f"{base_name}_epoch{state.epoch}_step{state.global_step}"
    
    def validate_checkpoint_data(
        self,
        checkpoint_data: Dict[str, Any],
    ) -> List[str]:
        """Validate checkpoint data completeness.
        
        Args:
            checkpoint_data: Checkpoint data to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Required fields
        required = ["model_id", "model_config", "training_state"]
        for field in required:
            if field not in checkpoint_data:
                errors.append(f"Missing required field: {field}")
        
        # Validate model config
        if "model_config" in checkpoint_data:
            model_config = checkpoint_data["model_config"]
            if "architecture" not in model_config:
                errors.append("Model config missing architecture")
        
        # Validate training state
        if "training_state" in checkpoint_data:
            state = checkpoint_data["training_state"]
            if "epoch" not in state or "global_step" not in state:
                errors.append("Training state missing epoch or global_step")
        
        return errors
    
    def get_latest_checkpoint(
        self,
        checkpoints: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Get the most recent checkpoint from a list.
        
        Args:
            checkpoints: List of checkpoint metadata
            
        Returns:
            Latest checkpoint or None
        """
        if not checkpoints:
            return None
        
        # Sort by global step (most training progress)
        sorted_checkpoints = sorted(
            checkpoints,
            key=lambda c: c.get("global_step", 0),
            reverse=True
        )
        
        return sorted_checkpoints[0]
    
    def get_best_checkpoint(
        self,
        checkpoints: List[Dict[str, Any]],
        metric_name: str = "best_metric",
        greater_is_better: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Get the best checkpoint based on a metric.
        
        Args:
            checkpoints: List of checkpoint metadata
            metric_name: Name of metric to compare
            greater_is_better: Whether higher metric is better
            
        Returns:
            Best checkpoint or None
        """
        if not checkpoints:
            return None
        
        # Filter checkpoints that have the metric
        valid_checkpoints = [
            c for c in checkpoints 
            if metric_name in c and c[metric_name] is not None
        ]
        
        if not valid_checkpoints:
            return None
        
        # Sort by metric
        sorted_checkpoints = sorted(
            valid_checkpoints,
            key=lambda c: c[metric_name],
            reverse=greater_is_better
        )
        
        return sorted_checkpoints[0]
    
    def cleanup_old_checkpoints(
        self,
        checkpoints: List[Dict[str, Any]],
        keep_total: int = 5,
        keep_best: bool = True,
    ) -> List[str]:
        """Determine which checkpoints to delete.
        
        Args:
            checkpoints: List of checkpoint metadata
            keep_total: Total number to keep
            keep_best: Whether to always keep the best
            
        Returns:
            List of checkpoint paths to delete
        """
        if len(checkpoints) <= keep_total:
            return []
        
        # Sort by global step (newest first)
        sorted_checkpoints = sorted(
            checkpoints,
            key=lambda c: c.get("global_step", 0),
            reverse=True
        )
        
        to_keep = set()
        to_delete = []
        
        # Always keep the best if requested
        if keep_best:
            best = self.get_best_checkpoint(checkpoints)
            if best:
                to_keep.add(best["checkpoint_path"])
        
        # Keep the most recent ones
        for checkpoint in sorted_checkpoints[:keep_total]:
            to_keep.add(checkpoint["checkpoint_path"])
        
        # Mark others for deletion
        for checkpoint in checkpoints:
            if checkpoint["checkpoint_path"] not in to_keep:
                to_delete.append(checkpoint["checkpoint_path"])
        
        return to_delete
    
    def prepare_checkpoint_data(
        self,
        model: BertModel,
        session: TrainingSession,
        optimizer_state: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Prepare complete checkpoint data.
        
        Args:
            model: Current model
            session: Training session
            optimizer_state: Optional optimizer state
            metrics: Optional current metrics
            
        Returns:
            Checkpoint data dictionary
        """
        checkpoint_data = {
            "model_id": model.id,
            "model_name": model.name,
            "model_config": model.get_config(),
            "session_id": session.session_id,
            "training_state": {
                "epoch": session.state.epoch,
                "global_step": session.state.global_step,
                "best_metric": session.state.best_val_metric,
                "no_improvement_count": session.state.no_improvement_count,
                "train_history": session.state.train_history,
                "val_history": session.state.val_history,
            },
            "training_config": session.config.to_dict() if hasattr(session.config, 'to_dict') else {},
            "timestamp": datetime.now().isoformat(),
        }
        
        if optimizer_state:
            checkpoint_data["optimizer_state"] = optimizer_state
            
        if metrics:
            checkpoint_data["metrics"] = metrics
        
        return checkpoint_data