"""
MLflow integration for experiment tracking and model registry.
"""

import json
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pytorch  # For compatibility, though we're using MLX
from loguru import logger

from ..core.config import BaseTrainerConfig
from ..core.protocols import Model, TrainingResult, TrainingState


@dataclass
class MLflowConfig:
    """Configuration for MLflow integration."""

    # MLflow settings
    tracking_uri: str | None = None
    experiment_name: str | None = None
    experiment_id: str | None = None
    run_name: str | None = None
    run_id: str | None = None

    # Logging settings
    log_models: bool = True
    log_checkpoints: bool = False
    log_metrics_every_n_steps: int = 1
    log_system_metrics: bool = True

    # Model registry
    register_model: bool = False
    model_name: str | None = None
    model_stage: str = "None"  # None, Staging, Production, Archived

    # Tags and params
    tags: dict[str, str] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)

    # Artifact settings
    log_artifacts: bool = True
    artifact_path: str = "artifacts"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tracking_uri": self.tracking_uri,
            "experiment_name": self.experiment_name,
            "experiment_id": self.experiment_id,
            "run_name": self.run_name,
            "run_id": self.run_id,
            "log_models": self.log_models,
            "log_checkpoints": self.log_checkpoints,
            "log_metrics_every_n_steps": self.log_metrics_every_n_steps,
            "log_system_metrics": self.log_system_metrics,
            "register_model": self.register_model,
            "model_name": self.model_name,
            "model_stage": self.model_stage,
            "tags": self.tags,
            "params": self.params,
            "log_artifacts": self.log_artifacts,
            "artifact_path": self.artifact_path,
        }


class MLflowIntegration:
    """
    MLflow integration for experiment tracking.

    This class provides:
    - Automatic experiment and run management
    - Metric logging
    - Model logging and registration
    - Artifact logging
    - System metrics tracking
    """

    def __init__(self, config: MLflowConfig):
        """
        Initialize MLflow integration.

        Args:
            config: MLflow configuration
        """
        self.config = config
        self.run = None
        self.step_count = 0

        # Set tracking URI if provided
        if config.tracking_uri:
            mlflow.set_tracking_uri(config.tracking_uri)

        # Set experiment
        if config.experiment_id:
            mlflow.set_experiment(experiment_id=config.experiment_id)
        elif config.experiment_name:
            mlflow.set_experiment(config.experiment_name)

        logger.info(
            f"Initialized MLflow integration with tracking URI: {mlflow.get_tracking_uri()}"
        )

    @contextmanager
    def start_run(self, trainer_config: BaseTrainerConfig | None = None):
        """
        Context manager for MLflow run.

        Args:
            trainer_config: Optional trainer configuration to log
        """
        try:
            # Start or resume run
            if self.config.run_id:
                self.run = mlflow.start_run(run_id=self.config.run_id)
            else:
                self.run = mlflow.start_run(run_name=self.config.run_name)

            # Log tags
            if self.config.tags:
                mlflow.set_tags(self.config.tags)

            # Log parameters
            if self.config.params:
                self._log_params(self.config.params)

            # Log trainer config if provided
            if trainer_config:
                self._log_trainer_config(trainer_config)

            # Log system info
            if self.config.log_system_metrics:
                self._log_system_info()

            logger.info(f"Started MLflow run: {self.run.info.run_id}")

            yield self.run

        finally:
            # End run
            if self.run:
                mlflow.end_run()
                logger.info(f"Ended MLflow run: {self.run.info.run_id}")

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
        prefix: str = "",
    ) -> None:
        """
        Log metrics to MLflow.

        Args:
            metrics: Dictionary of metrics
            step: Optional step number
            prefix: Optional prefix for metric names
        """
        if not self.run:
            logger.warning("No active MLflow run, skipping metric logging")
            return

        # Apply prefix if provided
        if prefix:
            metrics = {f"{prefix}{k}": v for k, v in metrics.items()}

        # Log each metric
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(name, value, step=step)

        self.step_count += 1

    def log_training_state(self, state: TrainingState, step: int | None = None) -> None:
        """
        Log training state to MLflow.

        Args:
            state: Current training state
            step: Optional step number
        """
        if not self.run:
            return

        # Log key metrics
        metrics = {
            "epoch": state.epoch,
            "global_step": state.global_step,
            "train_loss": state.train_loss,
            "val_loss": state.val_loss,
            "best_val_loss": state.best_val_loss,
            "best_val_metric": state.best_val_metric,
            "no_improvement_count": state.no_improvement_count,
        }

        # Add custom metrics
        metrics.update(state.metrics)

        # Filter out None values
        metrics = {k: v for k, v in metrics.items() if v is not None}

        self.log_metrics(metrics, step=step or state.global_step)

    def log_model(
        self,
        model: Model,
        artifact_path: str = "model",
        registered_model_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Log model to MLflow.

        Args:
            model: Model to log
            artifact_path: Path within artifacts
            registered_model_name: Optional name for model registry
            metadata: Optional metadata to log with model
        """
        if not self.run or not self.config.log_models:
            return

        # Save model to temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model"
            model_path.mkdir()

            # Save model weights
            model.save_pretrained(model_path)

            # Save metadata
            if metadata:
                metadata_path = model_path / "metadata.json"
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)

            # Log to MLflow
            mlflow.log_artifacts(str(model_path), artifact_path)

            # Register model if requested
            if registered_model_name or self.config.register_model:
                model_name = registered_model_name or self.config.model_name
                if model_name:
                    model_uri = f"runs:/{self.run.info.run_id}/{artifact_path}"
                    mlflow.register_model(model_uri, model_name)
                    logger.info(f"Registered model: {model_name}")

    def log_checkpoint(
        self,
        checkpoint_path: Path,
        checkpoint_name: str,
        metrics: dict[str, float] | None = None,
    ) -> None:
        """
        Log checkpoint to MLflow.

        Args:
            checkpoint_path: Path to checkpoint directory
            checkpoint_name: Name of checkpoint
            metrics: Optional metrics to log with checkpoint
        """
        if not self.run or not self.config.log_checkpoints:
            return

        # Log checkpoint artifacts
        artifact_path = f"checkpoints/{checkpoint_name}"
        mlflow.log_artifacts(str(checkpoint_path), artifact_path)

        # Log checkpoint metrics
        if metrics:
            checkpoint_metrics = {f"checkpoint.{k}": v for k, v in metrics.items()}
            self.log_metrics(checkpoint_metrics)

        logger.debug(f"Logged checkpoint to MLflow: {checkpoint_name}")

    def log_training_result(self, result: TrainingResult) -> None:
        """
        Log final training result to MLflow.

        Args:
            result: Training result
        """
        if not self.run:
            return

        # Log final metrics
        final_metrics = {
            "final_train_loss": result.final_train_loss,
            "final_val_loss": result.final_val_loss,
            "best_val_loss": result.best_val_loss,
            "best_val_metric": result.best_val_metric,
            "total_epochs": result.total_epochs,
            "total_steps": result.total_steps,
            "total_time_seconds": result.total_time,
            "early_stopped": result.early_stopped,
        }

        # Add custom final metrics
        for k, v in result.final_metrics.items():
            if not k.startswith("final_"):
                final_metrics[f"final_{k}"] = v

        self.log_metrics(final_metrics)

        # Log training summary
        summary = {
            "stop_reason": result.stop_reason,
            "final_model_path": str(result.final_model_path)
            if result.final_model_path
            else None,
            "best_model_path": str(result.best_model_path)
            if result.best_model_path
            else None,
        }

        # Log as artifacts
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(summary, f, indent=2)
            mlflow.log_artifact(f.name, "training_summary.json")

        # Log history as artifacts
        if self.config.log_artifacts:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Save train history
                train_history_path = Path(tmpdir) / "train_history.json"
                with open(train_history_path, "w") as f:
                    json.dump(result.train_history, f, indent=2)

                # Save val history
                val_history_path = Path(tmpdir) / "val_history.json"
                with open(val_history_path, "w") as f:
                    json.dump(result.val_history, f, indent=2)

                mlflow.log_artifacts(tmpdir, "history")

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """
        Log an artifact to MLflow.

        Args:
            local_path: Local path to artifact
            artifact_path: Optional path within artifacts
        """
        if not self.run or not self.config.log_artifacts:
            return

        mlflow.log_artifact(local_path, artifact_path)

    def log_artifacts(self, local_dir: str, artifact_path: str | None = None) -> None:
        """
        Log a directory of artifacts to MLflow.

        Args:
            local_dir: Local directory path
            artifact_path: Optional path within artifacts
        """
        if not self.run or not self.config.log_artifacts:
            return

        mlflow.log_artifacts(local_dir, artifact_path)

    def get_run_id(self) -> str | None:
        """Get current run ID."""
        return self.run.info.run_id if self.run else None

    def get_experiment_id(self) -> str | None:
        """Get current experiment ID."""
        return self.run.info.experiment_id if self.run else None

    def _log_trainer_config(self, config: BaseTrainerConfig) -> None:
        """Log trainer configuration as parameters."""
        # Log optimizer config
        optimizer_params = {
            f"optimizer.{k}": v for k, v in config.optimizer.to_dict().items()
        }
        self._log_params(optimizer_params)

        # Log scheduler config
        scheduler_params = {
            f"scheduler.{k}": v for k, v in config.scheduler.to_dict().items()
        }
        self._log_params(scheduler_params)

        # Log data config
        data_params = {f"data.{k}": v for k, v in config.data.to_dict().items()}
        self._log_params(data_params)

        # Log training config
        training_params = {
            f"training.{k}": v for k, v in config.training.to_dict().items()
        }
        self._log_params(training_params)

        # Log environment config (selective)
        env_params = {
            "environment.seed": config.environment.seed,
            "environment.output_dir": str(config.environment.output_dir),
        }
        self._log_params(env_params)

        # Log full config as artifact
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config.save(Path(f.name))
            mlflow.log_artifact(f.name, "config/trainer_config.yaml")

    def _log_params(self, params: dict[str, Any]) -> None:
        """Log parameters, handling type conversions."""
        for key, value in params.items():
            if value is None:
                continue

            # Convert to string for MLflow
            if isinstance(value, (list, dict)):
                value = json.dumps(value)
            elif isinstance(value, Path) or not isinstance(
                value, (str, int, float, bool)
            ):
                value = str(value)

            try:
                mlflow.log_param(key, value)
            except Exception as e:
                logger.warning(f"Failed to log parameter {key}: {e}")

    def _log_system_info(self) -> None:
        """Log system information."""
        import platform

        system_info = {
            "system.platform": platform.platform(),
            "system.python_version": platform.python_version(),
            "system.processor": platform.processor(),
            "system.node": platform.node(),
        }

        # Add MLX info if available
        try:
            import mlx

            system_info["system.mlx_version"] = mlx.__version__
        except:
            pass

        self._log_params(system_info)


def create_mlflow_integration(
    trainer_config: BaseTrainerConfig,
    mlflow_config: MLflowConfig | None = None,
) -> MLflowIntegration | None:
    """
    Create MLflow integration from trainer config.

    Args:
        trainer_config: Trainer configuration
        mlflow_config: Optional MLflow configuration

    Returns:
        MLflow integration or None if disabled
    """
    # Check if MLflow is enabled
    if "mlflow" not in trainer_config.training.report_to:
        return None

    # Create MLflow config if not provided
    if mlflow_config is None:
        mlflow_config = MLflowConfig(
            tracking_uri=trainer_config.environment.mlflow_tracking_uri,
            experiment_name=trainer_config.environment.experiment_name,
            run_name=trainer_config.environment.run_name,
            tags=trainer_config.environment.mlflow_tags,
        )

    return MLflowIntegration(mlflow_config)
