import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from pathlib import Path
import os
from typing import Dict, Any, Optional, List, Tuple
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd
from loguru import logger

# MLX doesn't have direct MLflow integration, so we'll create custom logging


class MLflowExperimentTracker:
    """MLflow experiment tracker for MLX models."""
    
    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str = "file:./mlruns",
        artifact_location: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.artifact_location = artifact_location
        self.tags = tags or {}
        
        # Set tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient(tracking_uri)
        
        # Create or get experiment
        self.experiment_id = self._get_or_create_experiment()
        
        # Set experiment
        mlflow.set_experiment(experiment_name)
        
        logger.info(f"MLflow tracking URI: {tracking_uri}")
        logger.info(f"Experiment: {experiment_name} (ID: {self.experiment_id})")
    
    def _get_or_create_experiment(self) -> str:
        """Get or create an MLflow experiment."""
        try:
            experiment = self.client.get_experiment_by_name(self.experiment_name)
            if experiment:
                return experiment.experiment_id
        except:
            pass
        
        # Create new experiment
        experiment_id = self.client.create_experiment(
            self.experiment_name,
            artifact_location=self.artifact_location,
            tags=self.tags
        )
        logger.info(f"Created new experiment: {self.experiment_name}")
        return experiment_id
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        nested: bool = False,
        tags: Optional[Dict[str, str]] = None
    ):
        """Start a new MLflow run."""
        run_tags = self.tags.copy()
        if tags:
            run_tags.update(tags)
        
        # Add metadata tags
        run_tags.update({
            "framework": "MLX",
            "model_type": "ModernBERT",
            "task": "classification",
            "timestamp": datetime.now().isoformat()
        })
        
        mlflow.start_run(
            run_name=run_name,
            nested=nested,
            tags=run_tags
        )
        
        logger.info(f"Started MLflow run: {mlflow.active_run().info.run_id}")
        return mlflow.active_run()
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow."""
        for key, value in params.items():
            # MLflow has a limit on param value length
            if isinstance(value, (list, dict)):
                value = json.dumps(value)[:250]
            mlflow.log_param(key, value)
        
        logger.debug(f"Logged {len(params)} parameters to MLflow")
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ):
        """Log metrics to MLflow."""
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value, step=step)
        
        logger.debug(f"Logged {len(metrics)} metrics to MLflow at step {step}")
    
    def log_artifacts(self, artifact_paths: List[str]):
        """Log artifacts to MLflow."""
        for path in artifact_paths:
            if os.path.exists(path):
                if os.path.isfile(path):
                    mlflow.log_artifact(path)
                else:
                    mlflow.log_artifacts(path)
                logger.debug(f"Logged artifact: {path}")
            else:
                logger.warning(f"Artifact not found: {path}")
    
    def log_model(
        self,
        model_path: str,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log MLX model to MLflow."""
        # Since MLX doesn't have direct MLflow integration,
        # we'll log the model files as artifacts
        artifact_path = f"models/{model_name}"
        
        # Log model files
        mlflow.log_artifacts(model_path, artifact_path)
        
        # Log model metadata
        if metadata:
            metadata_path = Path(model_path) / "model_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            mlflow.log_artifact(str(metadata_path), artifact_path)
        
        # Log model info
        model_info = {
            "model_type": "MLX ModernBERT",
            "model_name": model_name,
            "framework": "MLX",
            "saved_at": datetime.now().isoformat()
        }
        mlflow.log_dict(model_info, f"{artifact_path}/model_info.json")
        
        logger.info(f"Logged model '{model_name}' to MLflow")
    
    def log_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None,
        step: Optional[int] = None
    ):
        """Log confusion matrix visualization."""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels or ['0', '1'],
            yticklabels=labels or ['0', '1']
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save and log
        filename = f"confusion_matrix_step_{step}.png" if step else "confusion_matrix.png"
        plt.savefig(filename)
        mlflow.log_artifact(filename)
        plt.close()
        
        # Clean up
        os.remove(filename)
        
        logger.debug(f"Logged confusion matrix at step {step}")
    
    def log_roc_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        step: Optional[int] = None
    ):
        """Log ROC curve visualization."""
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        # Save and log
        filename = f"roc_curve_step_{step}.png" if step else "roc_curve.png"
        plt.savefig(filename)
        mlflow.log_artifact(filename)
        plt.close()
        
        # Clean up
        os.remove(filename)
        
        logger.debug(f"Logged ROC curve at step {step}")
    
    def log_training_curves(self, history: Dict[str, List[float]]):
        """Log training history curves."""
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16)
        
        # Loss curve
        if 'train_loss' in history:
            axes[0, 0].plot(history['train_loss'], label='Train Loss')
            if 'val_loss' in history:
                axes[0, 0].plot(history['val_loss'], label='Val Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Loss Curve')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # Accuracy curve
        if 'train_accuracy' in history:
            axes[0, 1].plot(history['train_accuracy'], label='Train Accuracy')
            if 'val_accuracy' in history:
                axes[0, 1].plot(history['val_accuracy'], label='Val Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].set_title('Accuracy Curve')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Learning rate curve
        if 'learning_rate' in history:
            axes[1, 0].plot(history['learning_rate'])
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].grid(True)
            axes[1, 0].set_yscale('log')
        
        # Additional metrics
        if 'val_f1' in history:
            axes[1, 1].plot(history['val_f1'], label='F1 Score')
            if 'val_precision' in history:
                axes[1, 1].plot(history['val_precision'], label='Precision')
            if 'val_recall' in history:
                axes[1, 1].plot(history['val_recall'], label='Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].set_title('Validation Metrics')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save and log
        filename = "training_curves.png"
        plt.savefig(filename)
        mlflow.log_artifact(filename)
        plt.close()
        
        # Clean up
        os.remove(filename)
        
        logger.info("Logged training curves to MLflow")
    
    def log_dataset_info(self, dataset_info: Dict[str, Any]):
        """Log dataset information."""
        mlflow.log_dict(dataset_info, "dataset_info.json")
        
        # Log as metrics for easy comparison
        for key, value in dataset_info.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"dataset_{key}", value)
    
    def end_run(self, status: str = "FINISHED"):
        """End the current MLflow run."""
        if mlflow.active_run():
            mlflow.end_run(status=status)
            logger.info(f"Ended MLflow run with status: {status}")
    
    def get_best_run(self, metric: str, mode: str = "min") -> Optional[mlflow.entities.Run]:
        """Get the best run from the experiment based on a metric."""
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"metrics.{metric} {'ASC' if mode == 'min' else 'DESC'}"],
            max_results=1
        )
        
        if not runs.empty:
            run_id = runs.iloc[0]['run_id']
            return self.client.get_run(run_id)
        return None
    
    def compare_runs(
        self,
        run_ids: List[str],
        metrics: List[str]
    ) -> pd.DataFrame:
        """Compare multiple runs by their metrics."""
        data = []
        
        for run_id in run_ids:
            run = self.client.get_run(run_id)
            row = {"run_id": run_id, "name": run.data.tags.get("mlflow.runName", "")}
            
            for metric in metrics:
                if metric in run.data.metrics:
                    row[metric] = run.data.metrics[metric]
            
            data.append(row)
        
        return pd.DataFrame(data)


class MLflowModelRegistry:
    """Model registry utilities for MLX models."""
    
    def __init__(self, tracking_uri: str = "file:./mlruns"):
        self.client = MlflowClient(tracking_uri)
        mlflow.set_tracking_uri(tracking_uri)
    
    def register_model(
        self,
        run_id: str,
        model_name: str,
        model_path: str,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """Register a model in the MLflow model registry."""
        # Create model version
        source = f"runs:/{run_id}/{model_path}"
        
        try:
            # Create registered model if it doesn't exist
            self.client.create_registered_model(
                model_name,
                tags=tags,
                description=f"MLX ModernBERT model for {model_name}"
            )
        except:
            # Model already exists
            pass
        
        # Create model version
        model_version = self.client.create_model_version(
            model_name,
            source,
            run_id,
            tags=tags
        )
        
        logger.info(f"Registered model '{model_name}' version {model_version.version}")
        return model_version.version
    
    def transition_model_stage(
        self,
        model_name: str,
        version: int,
        stage: str
    ):
        """Transition a model version to a new stage."""
        self.client.transition_model_version_stage(
            model_name,
            version,
            stage
        )
        logger.info(f"Transitioned {model_name} v{version} to {stage}")
    
    def get_latest_model_version(
        self,
        model_name: str,
        stage: Optional[str] = None
    ) -> Optional[mlflow.entities.model_registry.ModelVersion]:
        """Get the latest version of a model."""
        if stage:
            versions = self.client.get_latest_versions(model_name, stages=[stage])
        else:
            versions = self.client.get_latest_versions(model_name)
        
        return versions[0] if versions else None


# Convenience decorators
def track_experiment(experiment_name: str, **kwargs):
    """Decorator to track a function as an MLflow experiment."""
    def decorator(func):
        def wrapper(*args, **inner_kwargs):
            tracker = MLflowExperimentTracker(experiment_name, **kwargs)
            
            with mlflow.start_run():
                # Log function parameters
                params = {f"arg_{i}": str(arg) for i, arg in enumerate(args)}
                params.update({f"kwarg_{k}": str(v) for k, v in inner_kwargs.items()})
                tracker.log_params(params)
                
                # Run function
                result = func(*args, **inner_kwargs)
                
                # Log any returned metrics
                if isinstance(result, dict) and 'metrics' in result:
                    tracker.log_metrics(result['metrics'])
                
                return result
        
        return wrapper
    return decorator