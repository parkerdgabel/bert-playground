"""
MLflow integration callback for automatic experiment tracking.
"""

from pathlib import Path
from typing import Any

from loguru import logger

from ..core.protocols import Trainer, TrainingResult, TrainingState
from ..integrations.mlflow import (
    MLflowConfig,
    MLflowIntegration,
    create_mlflow_integration,
)
from .base import Callback


class MLflowCallback(Callback):
    """
    Callback for MLflow experiment tracking.

    This callback automatically:
    - Starts MLflow runs
    - Logs metrics during training
    - Logs model checkpoints
    - Logs training artifacts
    - Handles run lifecycle

    Args:
        mlflow_config: MLflow configuration
        log_every_n_steps: Log metrics every N steps
        log_model_checkpoints: Whether to log model checkpoints
        log_best_model: Whether to log the best model
        log_artifacts: Whether to log additional artifacts
    """

    def __init__(
        self,
        mlflow_config: MLflowConfig | None = None,
        log_every_n_steps: int = 1,
        log_model_checkpoints: bool = False,
        log_best_model: bool = True,
        log_artifacts: bool = True,
    ):
        super().__init__()
        self.mlflow_config = mlflow_config
        self.log_every_n_steps = log_every_n_steps
        self.log_model_checkpoints = log_model_checkpoints
        self.log_best_model = log_best_model
        self.log_artifacts = log_artifacts

        self.mlflow: MLflowIntegration | None = None
        self.run_context = None
        self.step_count = 0

    @property
    def priority(self) -> int:
        """MLflow logging should happen after metrics are computed."""
        return 70

    def on_train_begin(self, trainer: Trainer, state: TrainingState) -> None:
        """Start MLflow run."""
        # Create MLflow integration
        if self.mlflow_config is None:
            self.mlflow = create_mlflow_integration(trainer.config)
        else:
            self.mlflow = MLflowIntegration(self.mlflow_config)

        if self.mlflow is None:
            logger.warning("MLflow integration not enabled")
            return

        # Start run
        self.run_context = self.mlflow.start_run(trainer.config)
        self.run_context.__enter__()

        # Log initial state
        self.mlflow.log_training_state(state, step=0)

    def on_batch_end(self, trainer: Trainer, state: TrainingState, loss: float) -> None:
        """Log batch metrics."""
        if self.mlflow is None:
            return

        self.step_count += 1

        # Log metrics every N steps
        if self.step_count % self.log_every_n_steps == 0:
            metrics = {
                "train_loss": loss,
                "learning_rate": trainer.optimizer.learning_rate,
            }

            self.mlflow.log_metrics(metrics, step=state.global_step)

    def on_evaluate_end(
        self, trainer: Trainer, state: TrainingState, metrics: dict[str, float]
    ) -> None:
        """Log evaluation metrics."""
        if self.mlflow is None:
            return

        # Log eval metrics
        self.mlflow.log_metrics(metrics, step=state.global_step)

        # Update training state with metrics
        state.metrics.update(metrics)
        self.mlflow.log_training_state(state, step=state.global_step)

    def on_checkpoint_save(
        self, trainer: Trainer, state: TrainingState, checkpoint_path: str
    ) -> None:
        """Log checkpoint to MLflow."""
        if self.mlflow is None or not self.log_model_checkpoints:
            return

        # Determine if this is the best checkpoint
        is_best = "best" in checkpoint_path.lower()

        if is_best or not self.log_best_model:
            checkpoint_name = Path(checkpoint_path).name
            self.mlflow.log_checkpoint(
                checkpoint_path=Path(checkpoint_path),
                checkpoint_name=checkpoint_name,
                metrics=state.metrics,
            )

    def on_train_end(
        self, trainer: Trainer, state: TrainingState, result: TrainingResult
    ) -> None:
        """End MLflow run and log final artifacts."""
        if self.mlflow is None:
            return

        try:
            # Log training result
            self.mlflow.log_training_result(result)

            # Log best model if requested
            if self.log_best_model and result.best_model_path:
                logger.info(f"Logging best model from {result.best_model_path}")

                # Load best model
                best_model = trainer.model.__class__.load_pretrained(
                    result.best_model_path
                )

                # Log to MLflow
                self.mlflow.log_model(
                    model=best_model,
                    artifact_path="best_model",
                    registered_model_name=self.mlflow_config.model_name
                    if self.mlflow_config
                    else None,
                    metadata={
                        "best_metric": result.best_val_metric,
                        "best_loss": result.best_val_loss,
                        "total_epochs": result.total_epochs,
                        "total_steps": result.total_steps,
                    },
                )

            # Log additional artifacts
            if self.log_artifacts and trainer.config.environment.output_dir.exists():
                # Log config files
                config_files = list(
                    trainer.config.environment.output_dir.glob("*.yaml")
                )
                config_files.extend(
                    trainer.config.environment.output_dir.glob("*.json")
                )

                for config_file in config_files:
                    self.mlflow.log_artifact(str(config_file), "configs")

                # Log any plots or visualizations
                plot_files = list(trainer.config.environment.output_dir.glob("*.png"))
                plot_files.extend(trainer.config.environment.output_dir.glob("*.pdf"))

                for plot_file in plot_files:
                    self.mlflow.log_artifact(str(plot_file), "plots")

            # Update run with final info
            result.mlflow_run_id = self.mlflow.get_run_id()
            result.mlflow_experiment_id = self.mlflow.get_experiment_id()

        finally:
            # End run
            if self.run_context is not None:
                self.run_context.__exit__(None, None, None)
                self.run_context = None

    def on_log(
        self, trainer: Trainer, state: TrainingState, logs: dict[str, Any]
    ) -> None:
        """Log additional metrics."""
        if self.mlflow is None:
            return

        # Filter numeric values
        numeric_logs = {k: v for k, v in logs.items() if isinstance(v, (int, float))}

        if numeric_logs:
            self.mlflow.log_metrics(numeric_logs, step=state.global_step)
