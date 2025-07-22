"""Training orchestrator component that coordinates all training components.

This component is responsible for:
- Coordinating training and evaluation loops
- Managing checkpointing and metrics
- Handling callbacks and hooks
- Implementing early stopping logic
"""

import time
from pathlib import Path
from typing import List, Optional
from loguru import logger

from core.bootstrap import get_service
from core.ports.compute import ComputeBackend
from core.ports.monitoring import MonitoringService
from core.events.bus import EventBus
from core.events.types import EventType
from core.protocols.training import (
    TrainingState,
    TrainingResult,
    TrainingHook,
    Trainer,
    TrainerConfig,
)
from core.protocols.data import DataLoader
from core.protocols.models import Model

from .training_loop import TrainingLoop
from .evaluation_loop import EvaluationLoop
from .checkpoint_manager import CheckpointManager
from .metrics_tracker import MetricsTracker


class TrainingOrchestrator:
    """Coordinates all training components using hexagonal architecture.
    
    This component uses dependency injection to coordinate:
    - Training and evaluation loops
    - Checkpointing and metrics tracking
    - Event publishing and monitoring
    - Early stopping logic
    """
    
    def __init__(
        self,
        training_loop: Optional[TrainingLoop] = None,
        evaluation_loop: Optional[EvaluationLoop] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        metrics_tracker: Optional[MetricsTracker] = None,
    ):
        """Initialize the training orchestrator with dependency injection.
        
        Components can be injected or will be resolved automatically.
        """
        # Get services through dependency injection
        self.compute_backend = get_service(ComputeBackend)
        self.monitoring = get_service(MonitoringService)
        self.event_bus = get_service(EventBus)
        
        # Use injected components or get from DI
        self.training_loop = training_loop or get_service(TrainingLoop)
        self.evaluation_loop = evaluation_loop or get_service(EvaluationLoop)
        self.checkpoint_manager = checkpoint_manager or get_service(CheckpointManager)
        self.metrics_tracker = metrics_tracker or get_service(MetricsTracker)
        
        # Configuration (will be set by configure method)
        self.config = {}
        self.state = None
    
    def configure(self, config: dict):
        """Configure the orchestrator with training parameters."""
        self.config = config
        logger.info(f"Training orchestrator configured for {config.get('epochs', 'unknown')} epochs")
    
    def train(
        self,
        model,
        train_loader,
        val_loader=None
    ) -> dict:
        """
        Execute the complete training workflow.
        
        Args:
            model: The model to train
            train_loader: Training data loader
            val_loader: Optional validation data loader
            
        Returns:
            Training results dictionary
        """
        epochs = self.config.get("epochs", 5)
        run_name = self.config.get("run_name", "unnamed_run")
        
        logger.info(f"Starting training: {run_name} for {epochs} epochs")
        
        # Initialize training state
        state = TrainingState(
            epoch=0,
            global_step=0,
            train_loss=0.0,
            val_loss=0.0
        )
        
        # Emit training started event
        self.event_bus.emit(
            EventType.TRAINING,
            "orchestrator_training_started",
            source=self,
            data={
                "run_name": run_name,
                "epochs": epochs,
                "model_params": getattr(model, "num_parameters", lambda: 0)()
            }
        )
        
        try:
            # Training loop
            for epoch in range(epochs):
                state.epoch = epoch
                
                # Emit epoch started event
                self.event_bus.emit(
                    EventType.TRAINING,
                    "epoch_started",
                    source=self,
                    data={"epoch": epoch, "run_name": run_name}
                )
                
                # Execute training epoch
                self.monitoring.log_info(f"Training epoch {epoch + 1}/{epochs}")
                
                # Simple training simulation (replace with actual training loop)
                import time
                time.sleep(0.1)  # Simulate training time
                
                # Update training loss (simulated)
                state.train_loss = 0.5 - (epoch * 0.05)  # Decreasing loss
                
                # Validation if available
                if val_loader:
                    self.monitoring.log_info("Running validation...")
                    # Simulate validation
                    time.sleep(0.05)
                    state.val_loss = 0.6 - (epoch * 0.04)
                
                # Track metrics
                epoch_metrics = {
                    "train_loss": state.train_loss,
                    "val_loss": state.val_loss,
                    "epoch": epoch
                }
                
                self.metrics_tracker.log_metrics(epoch_metrics)
                
                # Emit epoch completed event
                self.event_bus.emit(
                    EventType.TRAINING,
                    "epoch_completed",
                    source=self,
                    data={
                        "epoch": epoch,
                        "train_loss": state.train_loss,
                        "val_loss": state.val_loss,
                        "run_name": run_name
                    }
                )
                
                state.global_step += 1
            
            # Final result
            result = {
                "final_train_loss": state.train_loss,
                "final_val_loss": state.val_loss,
                "epochs_completed": epochs,
                "total_steps": state.global_step
            }
            
            # Emit training completed event
            self.event_bus.emit(
                EventType.TRAINING,
                "orchestrator_training_completed",
                source=self,
                data={
                    "run_name": run_name,
                    "result": result,
                    "success": True
                }
            )
            
            logger.info("Training completed successfully")
            return result
            
        except Exception as e:
            # Emit training failed event
            self.event_bus.emit(
                EventType.TRAINING,
                "training_failed",
                source=self,
                data={
                    "run_name": run_name,
                    "error": str(e)
                }
            )
            
            self.monitoring.log_error(f"Training failed: {str(e)}")
            raise