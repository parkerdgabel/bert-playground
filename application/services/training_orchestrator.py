"""Training orchestrator service.

This application service orchestrates the training workflow by coordinating
between domain services and infrastructure adapters.
"""

from typing import Optional, Dict, Any, AsyncIterator
from dataclasses import dataclass
import asyncio
from datetime import datetime

from infrastructure.di import application_service
from domain.services.training_service import (
    TrainingService,
    TrainingDecision,
    TrainingProgress
)
from domain.entities.model import BertModel
from domain.entities.dataset import Dataset
from domain.entities.training import TrainingSession, CheckpointMetadata
from domain.value_objects.hyperparameters import Hyperparameters
from application.ports.secondary.training import (
    TrainingExecutor,
    TrainingMonitor,
    TrainingBatch,
    TrainingStepResult,
    EvaluationResult
)
from application.ports.secondary.storage import StorageService
from application.ports.secondary.monitoring import MonitoringService


@dataclass
class TrainingResult:
    """Result of a training run."""
    session: TrainingSession
    best_model_path: str
    final_model_path: str
    training_time_seconds: float
    final_metrics: Dict[str, float]


@application_service
class TrainingOrchestrator:
    """Orchestrates the training workflow.
    
    This service coordinates between:
    - Domain training service for business decisions
    - Training executor for actual training execution
    - Storage service for model persistence
    - Monitoring service for tracking progress
    """
    
    def __init__(
        self,
        training_service: TrainingService,
        training_executor: TrainingExecutor,
        storage_service: StorageService,
        monitoring_service: MonitoringService
    ):
        self.training_service = training_service
        self.executor = training_executor
        self.storage = storage_service
        self.monitoring = monitoring_service
        self.training_monitor: Optional[TrainingMonitor] = None
    
    def set_monitor(self, monitor: TrainingMonitor) -> None:
        """Set optional training monitor."""
        self.training_monitor = monitor
    
    async def train_model(
        self,
        model: BertModel,
        dataset: Dataset,
        hyperparameters: Hyperparameters,
        output_dir: str,
        resume_from: Optional[str] = None
    ) -> TrainingResult:
        """Execute the complete training workflow.
        
        Args:
            model: Model to train
            dataset: Training dataset
            hyperparameters: Training hyperparameters
            output_dir: Directory for outputs
            resume_from: Optional checkpoint to resume from
            
        Returns:
            Training result with paths and metrics
        """
        start_time = datetime.now()
        
        # Create training plan
        session = self.training_service.create_training_plan(
            model, dataset, hyperparameters
        )
        
        # Initialize training
        compiled_model, optimizer, scheduler = self.executor.initialize_training(
            model, hyperparameters
        )
        
        # Resume if specified
        if resume_from:
            compiled_model, optimizer, scheduler, state = \
                self.executor.load_checkpoint(resume_from)
            session.state = state
        
        # Start training
        session.start()
        if self.training_monitor:
            self.training_monitor.on_training_start(session)
        
        # Create data iterator
        data_iterator = self.executor.create_data_iterator(
            dataset,
            hyperparameters.batch_size,
            shuffle=True
        )
        
        # Training loop
        best_model_path = None
        try:
            async for epoch in self._epoch_iterator(session, hyperparameters):
                if self.training_monitor:
                    self.training_monitor.on_epoch_start(
                        epoch, hyperparameters.num_epochs
                    )
                
                # Train one epoch
                epoch_metrics = await self._train_epoch(
                    compiled_model,
                    data_iterator,
                    optimizer,
                    scheduler,
                    session,
                    hyperparameters
                )
                
                # Make training decision
                decision = self.training_service.make_training_decision(
                    session, epoch_metrics
                )
                
                # Handle decision
                if decision == TrainingDecision.EVALUATE:
                    eval_result = await self._evaluate(
                        compiled_model, dataset, session
                    )
                    if self.training_monitor:
                        self.training_monitor.on_evaluation_end(epoch, eval_result)
                
                if decision == TrainingDecision.CHECKPOINT:
                    checkpoint_path = await self._save_checkpoint(
                        compiled_model, optimizer, scheduler,
                        session, output_dir
                    )
                    if session.state.is_best_model:
                        best_model_path = checkpoint_path
                
                if decision == TrainingDecision.STOP_EARLY:
                    await self.monitoring.log_info("Early stopping triggered")
                    session.stop()
                    break
                
                if decision == TrainingDecision.COMPLETE:
                    session.complete()
                    break
            
            # Save final model
            final_path = await self._save_final_model(
                compiled_model, session, output_dir
            )
            
            # Get final metrics
            final_metrics = self._get_final_metrics(session)
            
            # Clean up
            self.executor.cleanup()
            
            # Create result
            training_time = (datetime.now() - start_time).total_seconds()
            
            if self.training_monitor:
                self.training_monitor.on_training_end(final_metrics)
            
            return TrainingResult(
                session=session,
                best_model_path=best_model_path or final_path,
                final_model_path=final_path,
                training_time_seconds=training_time,
                final_metrics=final_metrics
            )
            
        except Exception as e:
            session.fail(str(e))
            raise
    
    async def _epoch_iterator(
        self,
        session: TrainingSession,
        hyperparameters: Hyperparameters
    ) -> AsyncIterator[int]:
        """Iterate through training epochs."""
        for epoch in range(hyperparameters.num_epochs):
            if session.is_finished:
                break
            yield epoch
            session.state.increment_epoch()
    
    async def _train_epoch(
        self,
        model: Any,
        data_iterator: Any,
        optimizer: Any,
        scheduler: Any,
        session: TrainingSession,
        hyperparameters: Hyperparameters
    ) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in data_iterator:
            # Execute training step
            result = self.executor.training_step(
                model, batch, optimizer,
                accumulation_steps=hyperparameters.gradient_accumulation_steps,
                clip_grad_norm=hyperparameters.gradient_clipping_max_norm
            )
            
            # Update learning rate
            current_lr = self.executor.update_learning_rate(
                scheduler, session.state.current_step
            )
            result.learning_rate = current_lr
            
            # Track metrics
            epoch_loss += result.loss
            num_batches += 1
            
            # Update state
            session.state.increment_step()
            
            # Log batch progress
            if self.training_monitor and num_batches % 10 == 0:
                self.training_monitor.on_batch_end(
                    batch.batch_idx,
                    result,
                    epoch_loss / num_batches
                )
            
            # Check if should break
            decision = self.training_service.make_training_decision(
                session, {"loss": result.loss}
            )
            if decision in [TrainingDecision.STOP_EARLY, TrainingDecision.COMPLETE]:
                break
        
        # Return epoch metrics
        return {
            "loss": epoch_loss / num_batches,
            "learning_rate": current_lr,
            "epoch": session.state.current_epoch
        }
    
    async def _evaluate(
        self,
        model: Any,
        dataset: Dataset,
        session: TrainingSession
    ) -> EvaluationResult:
        """Evaluate the model."""
        eval_loss = 0.0
        num_batches = 0
        all_metrics = {}
        
        # Create evaluation data iterator
        eval_iterator = self.executor.create_data_iterator(
            dataset,
            session.hyperparameters.batch_size,
            shuffle=False
        )
        
        # Evaluation loop
        for batch in eval_iterator:
            loss, metrics = self.executor.evaluation_step(model, batch)
            eval_loss += loss
            num_batches += 1
            
            # Accumulate metrics
            for name, value in metrics.items():
                if name not in all_metrics:
                    all_metrics[name] = 0.0
                all_metrics[name] += value
        
        # Average metrics
        avg_loss = eval_loss / num_batches
        avg_metrics = {
            name: value / num_batches
            for name, value in all_metrics.items()
        }
        avg_metrics["loss"] = avg_loss
        
        # Update session metrics
        session.state.update_metrics(avg_metrics)
        
        return EvaluationResult(
            loss=avg_loss,
            metrics=avg_metrics,
            num_samples=dataset.size
        )
    
    async def _save_checkpoint(
        self,
        model: Any,
        optimizer: Any,
        scheduler: Any,
        session: TrainingSession,
        output_dir: str
    ) -> str:
        """Save a training checkpoint."""
        # Generate checkpoint path
        checkpoint_id = f"checkpoint-{session.state.current_step}"
        checkpoint_path = f"{output_dir}/{checkpoint_id}"
        
        # Save checkpoint
        self.executor.save_checkpoint(
            model, optimizer, scheduler,
            session.state, checkpoint_path
        )
        
        # Get file size
        file_size = await self.storage.get_file_size(checkpoint_path)
        
        # Create metadata
        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            session_id=session.id,
            epoch=session.state.current_epoch,
            step=session.state.current_step,
            loss=session.state.current_loss,
            metrics=dict(session.state.metric_history),
            is_best=session.state.is_best_model,
            created_at=datetime.now(),
            file_path=checkpoint_path,
            file_size_mb=file_size / (1024 * 1024)
        )
        
        # Add to session
        session.add_checkpoint(metadata)
        
        # Notify monitor
        if self.training_monitor:
            self.training_monitor.on_checkpoint_saved(
                checkpoint_path,
                session.state.is_best_model
            )
        
        return checkpoint_path
    
    async def _save_final_model(
        self,
        model: Any,
        session: TrainingSession,
        output_dir: str
    ) -> str:
        """Save the final model."""
        final_path = f"{output_dir}/model_final"
        
        # Save model state only (not optimizer)
        await self.storage.save_model(model, final_path)
        
        # Save training metadata
        metadata_path = f"{final_path}_metadata.json"
        await self.storage.save_json({
            "session_id": session.id,
            "hyperparameters": session.hyperparameters.to_dict(),
            "final_metrics": self._get_final_metrics(session),
            "training_duration_seconds": session.duration,
            "total_steps": session.state.current_step,
            "completed_at": session.completed_at.isoformat() if session.completed_at else None
        }, metadata_path)
        
        return final_path
    
    def _get_final_metrics(self, session: TrainingSession) -> Dict[str, float]:
        """Extract final metrics from session."""
        metrics = {
            "final_loss": session.state.current_loss,
            "best_loss": session.state.best_loss,
            "total_epochs": session.state.current_epoch,
            "total_steps": session.state.current_step
        }
        
        # Add latest values from metric history
        for name, history in session.state.metric_history.items():
            if history:
                metrics[f"final_{name}"] = history[-1]
        
        return metrics
    
    async def analyze_progress(
        self,
        session: TrainingSession
    ) -> TrainingProgress:
        """Get current training progress analysis."""
        return self.training_service.analyze_training_progress(session)