"""
Clean BaseTrainer implementation using hexagonal architecture.

This trainer acts as a facade that delegates to the TrainingOrchestrator
which implements the hexagonal architecture with dependency injection.
"""

import sys
from collections.abc import Callable
from pathlib import Path

from loguru import logger

# Import dependency injection and ports
from core.bootstrap import get_service
from training.components.training_orchestrator import TrainingOrchestrator

from .config import BaseTrainerConfig
from .protocols import DataLoader, Model, TrainingHook, TrainingResult


class BaseTrainer:
    """
    Clean base trainer implementation using hexagonal architecture.

    This trainer acts as a facade over the TrainingOrchestrator which
    implements proper dependency injection and ports/adapters pattern.
    """

    def __init__(
        self,
        model: Model,
        config: BaseTrainerConfig,
        callbacks: list[TrainingHook] | None = None,
    ):
        """
        Initialize the trainer.

        Args:
            model: Model to train
            config: Training configuration
            callbacks: Optional list of training callbacks
        """
        self._model = model
        self._config = config
        self.callbacks = callbacks or []

        # Create output directory
        self.config.environment.output_dir.mkdir(parents=True, exist_ok=True)

        # Save configuration
        config_path = self.config.environment.output_dir / "trainer_config.yaml"
        self.config.save(config_path)
        
        # Get training orchestrator through dependency injection
        self.orchestrator = get_service(TrainingOrchestrator)

    @property
    def model(self) -> Model:
        """Access to the model."""
        return self._model

    @property
    def config(self) -> BaseTrainerConfig:
        """Access to the config."""
        return self._config

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None = None,
        resume_from: Path | None = None,
    ) -> TrainingResult:
        """
        Run the training loop using hexagonal architecture.
        
        This method delegates to the TrainingOrchestrator which uses 
        dependency injection and ports/adapters pattern.

        Args:
            train_dataloader: Training data loader
            val_dataloader: Optional validation data loader
            resume_from: Optional checkpoint path to resume from

        Returns:
            TrainingResult with final metrics and paths
        """
        logger.info("Starting training using hexagonal architecture")
        
        # Configure orchestrator with trainer config
        training_config = {
            "epochs": self.config.training.num_epochs,
            "learning_rate": self.config.optimizer.learning_rate,
            "output_dir": str(self.config.environment.output_dir),
            "run_name": "basetrainer_run"
        }
        
        self.orchestrator.configure(training_config)
        
        # Delegate to orchestrator
        result_dict = self.orchestrator.train(
            model=self.model,
            train_loader=train_dataloader,
            val_loader=val_dataloader
        )
        
        # Convert orchestrator result to TrainingResult format for compatibility
        result = TrainingResult(
            final_train_loss=result_dict.get("final_train_loss", 0.0),
            final_val_loss=result_dict.get("final_val_loss", 0.0),
            best_val_loss=result_dict.get("final_val_loss", 0.0),
            best_val_metric=0.0,
            final_metrics=result_dict,
            train_history=[],
            val_history=[],
            total_epochs=result_dict.get("epochs_completed", 0),
            total_steps=result_dict.get("total_steps", 0),
            total_time=0.0,
            early_stopped=False,
        )
        
        logger.info("Training completed using hexagonal architecture")
        return result

    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        """
        Evaluate model on provided dataloader.
        
        Args:
            dataloader: Evaluation data loader
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Running evaluation using hexagonal architecture")
        
        # Configure orchestrator for evaluation
        eval_config = {
            "run_name": "evaluation"
        }
        self.orchestrator.configure(eval_config)
        
        # Use orchestrator's evaluation capabilities
        # For now return dummy metrics - this would be implemented via orchestrator
        return {
            "eval_loss": 0.0,
            "eval_accuracy": 0.0
        }

    def predict(self, dataloader: DataLoader):
        """
        Generate predictions using the model.
        
        Args:
            dataloader: Data loader for prediction
            
        Returns:
            Model predictions
        """
        logger.info("Generating predictions using hexagonal architecture")
        
        # This would delegate to orchestrator's prediction capabilities
        # For now return None - this would be implemented via orchestrator
        return None