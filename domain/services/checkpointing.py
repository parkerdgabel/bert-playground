"""Checkpointing service for model persistence."""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path
from domain.entities.model import BertModel
from domain.entities.training import TrainingSession, TrainingState
from domain.ports.storage import CheckpointPort, StoragePort
from domain.ports.monitoring import MonitoringPort


@dataclass
class CheckpointingService:
    """Service for managing model checkpoints."""
    checkpoint_port: CheckpointPort
    storage_port: StoragePort
    monitoring_port: MonitoringPort
    
    def save_checkpoint(
        self,
        model: BertModel,
        training_session: TrainingSession,
        optimizer_state: Dict[str, Any],
        checkpoint_dir: str,
        keep_last_n: int = 3,
        is_best: bool = False,
    ) -> str:
        """Save a training checkpoint.
        
        Args:
            model: Model to save
            training_session: Current training session
            optimizer_state: Optimizer state
            checkpoint_dir: Directory for checkpoints
            keep_last_n: Number of recent checkpoints to keep
            is_best: Whether this is the best checkpoint
            
        Returns:
            Path to saved checkpoint
        """
        state = training_session.state
        
        # Generate checkpoint name
        checkpoint_name = f"checkpoint_epoch_{state.epoch}_step_{state.global_step}"
        if is_best:
            checkpoint_name = "best_" + checkpoint_name
        
        checkpoint_path = f"{checkpoint_dir}/{checkpoint_name}"
        
        # Save checkpoint
        self.checkpoint_port.save_checkpoint(
            model=model,
            training_state=state,
            optimizer_state=optimizer_state,
            path=checkpoint_path,
            metadata={
                'session_id': training_session.session_id,
                'epoch': state.epoch,
                'global_step': state.global_step,
                'best_metric': state.best_metric,
                'is_best': is_best,
                'config': training_session.config.__dict__,
            }
        )
        
        # Log artifact
        self.monitoring_port.log_artifact(
            path=checkpoint_path,
            artifact_type='checkpoint',
            metadata={'is_best': is_best}
        )
        
        # Clean up old checkpoints
        if keep_last_n > 0:
            self._cleanup_old_checkpoints(checkpoint_dir, keep_last_n, keep_best=True)
        
        training_session.add_checkpoint(checkpoint_path)
        
        return checkpoint_path
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
    ) -> Dict[str, Any]:
        """Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            
        Returns:
            Dictionary with checkpoint contents
        """
        if not self.storage_port.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = self.checkpoint_port.load_checkpoint(checkpoint_path)
        
        self.monitoring_port.log_message(
            f"Loaded checkpoint from {checkpoint_path}",
            level="INFO",
            context={
                'epoch': checkpoint.get('training_state', {}).epoch,
                'step': checkpoint.get('training_state', {}).global_step,
            }
        )
        
        return checkpoint
    
    def resume_training(
        self,
        checkpoint_path: str,
        training_session: TrainingSession,
    ) -> Dict[str, Any]:
        """Resume training from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            training_session: Training session to update
            
        Returns:
            Checkpoint data including model and optimizer state
        """
        checkpoint = self.load_checkpoint(checkpoint_path)
        
        # Update training session state
        training_session.state = checkpoint['training_state']
        
        # Log resumption
        self.monitoring_port.log_message(
            f"Resuming training from epoch {training_session.state.epoch}, "
            f"step {training_session.state.global_step}",
            level="INFO"
        )
        
        return checkpoint
    
    def save_for_inference(
        self,
        model: BertModel,
        save_path: str,
        include_config: bool = True,
        optimize: bool = False,
    ) -> None:
        """Save model for inference.
        
        Args:
            model: Model to save
            save_path: Save path
            include_config: Whether to save configuration
            optimize: Whether to optimize for inference
        """
        # Save model
        self.checkpoint_port.save_model(
            model=model,
            path=save_path,
            save_config=include_config,
        )
        
        # Optionally export to optimized format
        if optimize:
            export_path = save_path + "_optimized"
            self.checkpoint_port.export_model(
                model=model,
                path=export_path,
                format="onnx",
                optimize=True,
            )
            
            self.monitoring_port.log_artifact(
                path=export_path,
                artifact_type='optimized_model',
            )
        
        self.monitoring_port.log_artifact(
            path=save_path,
            artifact_type='model',
            metadata={'optimized': optimize}
        )
    
    def _cleanup_old_checkpoints(
        self,
        checkpoint_dir: str,
        keep_last_n: int,
        keep_best: bool,
    ) -> None:
        """Clean up old checkpoints.
        
        Args:
            checkpoint_dir: Checkpoint directory
            keep_last_n: Number of recent checkpoints to keep
            keep_best: Whether to keep best checkpoint
        """
        deleted = self.checkpoint_port.cleanup_checkpoints(
            directory=checkpoint_dir,
            keep_last=keep_last_n,
            keep_best=keep_best,
        )
        
        if deleted:
            self.monitoring_port.log_message(
                f"Cleaned up {len(deleted)} old checkpoints",
                level="INFO",
                context={'deleted_checkpoints': deleted}
            )
    
    def find_best_checkpoint(
        self,
        checkpoint_dir: str,
        metric: str = "loss",
        mode: str = "min",
    ) -> Optional[str]:
        """Find the best checkpoint based on metric.
        
        Args:
            checkpoint_dir: Directory to search
            metric: Metric to use
            mode: 'min' or 'max'
            
        Returns:
            Path to best checkpoint or None
        """
        best_checkpoint = self.checkpoint_port.get_best_checkpoint(
            directory=checkpoint_dir,
            metric=metric,
            mode=mode,
        )
        
        if best_checkpoint:
            self.monitoring_port.log_message(
                f"Found best checkpoint: {best_checkpoint}",
                level="INFO",
                context={'metric': metric, 'mode': mode}
            )
        
        return best_checkpoint
    
    def list_checkpoints(
        self,
        checkpoint_dir: str,
    ) -> List[Dict[str, Any]]:
        """List all available checkpoints.
        
        Args:
            checkpoint_dir: Directory to search
            
        Returns:
            List of checkpoint information
        """
        return self.checkpoint_port.list_checkpoints(checkpoint_dir)
    
    def compare_checkpoints(
        self,
        checkpoint_paths: List[str],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Compare multiple checkpoints.
        
        Args:
            checkpoint_paths: Paths to checkpoints
            metrics: Metrics to compare
            
        Returns:
            Comparison results
        """
        comparison = {}
        
        for path in checkpoint_paths:
            checkpoint = self.load_checkpoint(path)
            metadata = checkpoint.get('metadata', {})
            state = checkpoint.get('training_state', {})
            
            comparison[path] = {
                'epoch': state.epoch if hasattr(state, 'epoch') else metadata.get('epoch'),
                'step': state.global_step if hasattr(state, 'global_step') else metadata.get('global_step'),
                'best_metric': state.best_metric if hasattr(state, 'best_metric') else metadata.get('best_metric'),
                'is_best': metadata.get('is_best', False),
            }
            
            # Add specific metrics if requested
            if metrics and 'metadata' in checkpoint:
                for metric in metrics:
                    if metric in metadata:
                        comparison[path][metric] = metadata[metric]
        
        return comparison