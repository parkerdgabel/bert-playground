"""Unified MLflow helper module for comprehensive experiment tracking."""

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from pathlib import Path
import os
from typing import Dict, Any, Optional, List, Union
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from loguru import logger
import subprocess
from enum import Enum


class TrackingMode(Enum):
    """MLflow tracking modes."""

    LOCAL = "local"
    REMOTE = "remote"
    FILE = "file"


class UnifiedMLflowTracker:
    """
    Unified MLflow tracker combining features from both implementations.
    Provides comprehensive experiment tracking with MLX model support.
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_mode: Union[str, TrackingMode] = TrackingMode.FILE,
        tracking_uri: Optional[str] = None,
        base_dir: str = "./mlflow",
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        artifact_location: Optional[str] = None,
    ):
        self.experiment_name = experiment_name
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.tags = tags or {}

        # Parse tracking mode
        if isinstance(tracking_mode, str):
            tracking_mode = TrackingMode(tracking_mode)
        self.tracking_mode = tracking_mode

        # Set up tracking URI based on mode
        if tracking_uri:
            self.tracking_uri = tracking_uri
        elif tracking_mode == TrackingMode.FILE:
            self.base_dir = Path(base_dir)
            self.base_dir.mkdir(exist_ok=True)
            self.tracking_dir = self.base_dir / "mlruns"
            self.tracking_dir.mkdir(exist_ok=True)
            self.tracking_uri = f"file://{self.tracking_dir.resolve()}"
        else:
            self.tracking_uri = tracking_uri or "http://localhost:5000"

        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        os.environ["MLFLOW_TRACKING_URI"] = self.tracking_uri
        self.client = MlflowClient(self.tracking_uri)

        # Set up artifact location
        if artifact_location is None and tracking_mode == TrackingMode.FILE:
            self.artifacts_dir = self.base_dir / "artifacts" / experiment_name
            self.artifacts_dir.mkdir(parents=True, exist_ok=True)
            artifact_location = str(self.artifacts_dir)
        self.artifact_location = artifact_location

        # Create or get experiment
        self.experiment_id = self._get_or_create_experiment()

        # Set experiment
        mlflow.set_experiment(experiment_name)

        # Track run state
        self.run = None
        self.is_active = False

        logger.info(f"MLflow tracking URI: {self.tracking_uri}")
        logger.info(f"Experiment: {experiment_name} (ID: {self.experiment_id})")

    def _get_or_create_experiment(self) -> str:
        """Get or create an MLflow experiment."""
        try:
            experiment = self.client.get_experiment_by_name(self.experiment_name)
            if experiment:
                return experiment.experiment_id
        except Exception:
            pass

        # Create new experiment
        experiment_id = self.client.create_experiment(
            self.experiment_name,
            artifact_location=self.artifact_location,
            tags=self.tags,
        )
        logger.info(f"Created new experiment: {self.experiment_name}")
        return experiment_id

    def start_run(
        self,
        run_name: Optional[str] = None,
        nested: bool = False,
        tags: Optional[Dict[str, str]] = None,
    ):
        """Start a new MLflow run."""
        if self.is_active and not nested:
            logger.warning("Run already active, ending previous run")
            self.end_run()

        run_tags = self.tags.copy()
        if tags:
            run_tags.update(tags)

        # Add metadata tags
        run_tags.update(
            {
                "framework": "MLX",
                "model_type": "ModernBERT",
                "task": "classification",
                "timestamp": datetime.now().isoformat(),
            }
        )

        self.run = mlflow.start_run(
            run_name=run_name or self.run_name,
            experiment_id=self.experiment_id,
            nested=nested,
            tags=run_tags,
        )
        self.is_active = True

        logger.info(f"Started MLflow run: {self.run.info.run_id}")
        return self.run

    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow."""
        if not self.is_active:
            logger.warning("No active run for logging parameters")
            return

        # Flatten nested parameters
        flat_params = self._flatten_dict(params)

        # Log parameters (MLflow has limits on param value length)
        for key, value in flat_params.items():
            if isinstance(value, (list, dict)):
                value = json.dumps(value)[:250]
            try:
                mlflow.log_param(key, value)
            except Exception as e:
                logger.warning(f"Failed to log param {key}: {e}")

        logger.debug(f"Logged {len(flat_params)} parameters to MLflow")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to MLflow."""
        if not self.is_active:
            logger.warning("No active run for logging metrics")
            return

        valid_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                valid_metrics[key] = float(value)
            elif isinstance(value, np.ndarray) and value.size == 1:
                valid_metrics[key] = float(value)

        if valid_metrics:
            mlflow.log_metrics(valid_metrics, step=step)
            logger.debug(f"Logged {len(valid_metrics)} metrics at step {step}")

    def log_artifacts(
        self, artifact_paths: Union[str, List[str]], artifact_path: Optional[str] = None
    ):
        """Log artifacts to MLflow."""
        if not self.is_active:
            logger.warning("No active run for logging artifacts")
            return

        if isinstance(artifact_paths, str):
            artifact_paths = [artifact_paths]

        for path in artifact_paths:
            if os.path.exists(path):
                if os.path.isfile(path):
                    mlflow.log_artifact(path, artifact_path)
                else:
                    mlflow.log_artifacts(path, artifact_path)
                logger.debug(f"Logged artifact: {path}")
            else:
                logger.warning(f"Artifact not found: {path}")

    def log_model(
        self,
        model_path: str,
        model_name: str,
        framework: str = "mlx",
        metadata: Optional[Dict[str, Any]] = None,
        model_version: Optional[str] = None,
    ):
        """Log MLX model to MLflow."""
        if not self.is_active:
            logger.warning("No active run for logging model")
            return

        artifact_path = f"models/{model_name}"

        # Log model files
        mlflow.log_artifacts(model_path, artifact_path)

        # Create and log model metadata
        model_info = {
            "model_type": "MLX ModernBERT",
            "model_name": model_name,
            "framework": framework,
            "version": model_version or "1.0",
            "saved_at": datetime.now().isoformat(),
        }

        if metadata:
            model_info.update(metadata)

        # Save model info
        mlflow.log_dict(model_info, f"{artifact_path}/model_info.json")

        logger.info(f"Logged model '{model_name}' to MLflow")

    def log_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None,
        step: Optional[int] = None,
    ):
        """Log confusion matrix visualization."""
        if not self.is_active:
            return

        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels or ["0", "1"],
            yticklabels=labels or ["0", "1"],
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")

        # Save and log
        filename = (
            f"confusion_matrix_step_{step}.png" if step else "confusion_matrix.png"
        )
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        mlflow.log_artifact(filename, "plots")
        plt.close()

        # Clean up
        os.remove(filename)

        logger.debug(f"Logged confusion matrix at step {step}")

    def log_roc_curve(
        self, y_true: np.ndarray, y_scores: np.ndarray, step: Optional[int] = None
    ):
        """Log ROC curve visualization."""
        if not self.is_active:
            return

        from sklearn.metrics import roc_curve, auc

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")

        # Save and log
        filename = f"roc_curve_step_{step}.png" if step else "roc_curve.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        mlflow.log_artifact(filename, "plots")
        plt.close()

        # Clean up
        os.remove(filename)

        logger.debug(f"Logged ROC curve at step {step}")

    def log_training_curves(
        self, history: Dict[str, List[float]], save_json: bool = True
    ):
        """Log comprehensive training history curves."""
        if not self.is_active:
            return

        # Save training history as JSON
        if save_json:
            history_data = {"history": history, "timestamp": datetime.now().isoformat()}
            mlflow.log_dict(history_data, "training_history.json")

        # Create comprehensive plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Training History", fontsize=16)

        # Loss curve
        if "train_loss" in history:
            axes[0, 0].plot(history["train_loss"], label="Train Loss")
            if "val_loss" in history:
                axes[0, 0].plot(history["val_loss"], label="Val Loss")
            axes[0, 0].set_xlabel("Epoch")
            axes[0, 0].set_ylabel("Loss")
            axes[0, 0].set_title("Loss Curve")
            axes[0, 0].legend()
            axes[0, 0].grid(True)

        # Accuracy curve
        if "train_accuracy" in history:
            axes[0, 1].plot(history["train_accuracy"], label="Train Accuracy")
            if "val_accuracy" in history:
                axes[0, 1].plot(history["val_accuracy"], label="Val Accuracy")
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylabel("Accuracy")
            axes[0, 1].set_title("Accuracy Curve")
            axes[0, 1].legend()
            axes[0, 1].grid(True)

        # Learning rate curve
        if "learning_rate" in history:
            axes[1, 0].plot(history["learning_rate"])
            axes[1, 0].set_xlabel("Step")
            axes[1, 0].set_ylabel("Learning Rate")
            axes[1, 0].set_title("Learning Rate Schedule")
            axes[1, 0].grid(True)
            axes[1, 0].set_yscale("log")

        # Additional metrics
        if any(key in history for key in ["val_f1", "val_precision", "val_recall"]):
            if "val_f1" in history:
                axes[1, 1].plot(history["val_f1"], label="F1 Score")
            if "val_precision" in history:
                axes[1, 1].plot(history["val_precision"], label="Precision")
            if "val_recall" in history:
                axes[1, 1].plot(history["val_recall"], label="Recall")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("Score")
            axes[1, 1].set_title("Validation Metrics")
            axes[1, 1].legend()
            axes[1, 1].grid(True)

        plt.tight_layout()

        # Save and log
        filename = "training_curves.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        mlflow.log_artifact(filename, "plots")
        plt.close()

        # Clean up
        os.remove(filename)

        logger.info("Logged training curves to MLflow")

    def log_gradient_statistics(
        self, gradient_stats: Dict[str, Dict[str, float]], step: int
    ):
        """Log gradient statistics for monitoring training stability."""
        if not self.is_active:
            return

        metrics = {}
        for component, stats in gradient_stats.items():
            for stat_name, value in stats.items():
                metrics[f"gradients/{component}_{stat_name}"] = value

        self.log_metrics(metrics, step=step)

    def log_dataset_info(self, dataset_info: Dict[str, Any]):
        """Log dataset information."""
        if not self.is_active:
            return

        mlflow.log_dict(dataset_info, "dataset_info.json")

        # Log numeric values as metrics
        for key, value in dataset_info.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"dataset_{key}", value)

    def end_run(self, status: str = "FINISHED"):
        """End the current MLflow run."""
        if self.is_active:
            mlflow.end_run(status=status)
            self.is_active = False
            logger.info(f"Ended MLflow run with status: {status}")

    def get_best_run(
        self, metric: str, mode: str = "min"
    ) -> Optional[mlflow.entities.Run]:
        """Get the best run from the experiment based on a metric."""
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"metrics.{metric} {'ASC' if mode == 'min' else 'DESC'}"],
            max_results=1,
        )

        if not runs.empty:
            run_id = runs.iloc[0]["run_id"]
            return self.client.get_run(run_id)
        return None

    def launch_ui(self, host: str = "127.0.0.1", port: int = 5000):
        """Launch MLflow UI server."""
        cmd = [
            "mlflow",
            "ui",
            "--backend-store-uri",
            self.tracking_uri,
            "--host",
            host,
            "--port",
            str(port),
        ]
        logger.info(f"Launching MLflow UI at http://{host}:{port}")
        subprocess.run(cmd)

    def _flatten_dict(
        self, d: Dict[str, Any], parent_key: str = "", sep: str = "."
    ) -> Dict[str, str]:
        """Flatten nested dictionary for parameter logging."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, str(v)))
        return dict(items)

    def __enter__(self):
        """Context manager entry."""
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is not None:
            self.end_run(status="FAILED")
        else:
            self.end_run(status="FINISHED")


# Factory functions for convenience
def create_experiment_tracker(
    experiment_name: str,
    tracking_uri: Optional[str] = None,
    base_dir: str = "./mlflow",
    **kwargs,
) -> UnifiedMLflowTracker:
    """Create a unified MLflow experiment tracker."""
    return UnifiedMLflowTracker(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        base_dir=base_dir,
        **kwargs,
    )


def create_experiment_tags(config: Dict[str, Any]) -> Dict[str, str]:
    """Create comprehensive experiment tags from configuration."""
    return {
        "dataset": config.get("dataset", "unknown"),
        "model_type": config.get("model_type", "ModernBERT"),
        "framework": "mlx",
        "task": config.get("task", "classification"),
        "optimizer": config.get("optimizer", "adamw"),
        "learning_rate": str(config.get("learning_rate", "unknown")),
        "batch_size": str(config.get("batch_size", "unknown")),
        "loss_function": config.get("loss_function", "cross_entropy"),
        "augmentation": str(config.get("augment", False)),
        "gradient_clipping": str(config.get("gradient_clipping", False)),
        "environment": config.get("environment", "development"),
        "version": config.get("version", "1.0.0"),
        "timestamp": datetime.now().isoformat(),
    }


# Decorators
def track_experiment(experiment_name: str, **kwargs):
    """Decorator to track a function as an MLflow experiment."""

    def decorator(func):
        def wrapper(*args, **inner_kwargs):
            with UnifiedMLflowTracker(experiment_name, **kwargs) as tracker:
                # Log function parameters
                params = {f"arg_{i}": str(arg)[:250] for i, arg in enumerate(args)}
                params.update(
                    {f"kwarg_{k}": str(v)[:250] for k, v in inner_kwargs.items()}
                )
                tracker.log_params(params)

                # Run function
                result = func(*args, **inner_kwargs)

                # Log any returned metrics
                if isinstance(result, dict) and "metrics" in result:
                    tracker.log_metrics(result["metrics"])

                return result

        return wrapper

    return decorator


# Backward compatibility
MLflowExperimentTracker = UnifiedMLflowTracker
EnhancedMLflowTracker = UnifiedMLflowTracker
setup_mlflow_tracking = create_experiment_tracker


def launch_mlflow_ui(base_dir="./mlflow", port=5000):
    """Launch MLflow UI interface."""
    return UnifiedMLflowTracker("temp", base_dir=base_dir).launch_ui(port=port)


# MLflowModelRegistry - stub for backward compatibility
MLflowModelRegistry = UnifiedMLflowTracker
