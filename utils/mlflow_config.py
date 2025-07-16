"""
Centralized MLflow configuration for comprehensive experiment tracking.
"""

import os
import mlflow
import subprocess
from pathlib import Path
from typing import Dict, Optional, Any, List
from loguru import logger
import json
from datetime import datetime


class CentralizedMLflowConfig:
    """
    Centralized MLflow configuration for the project.
    Manages tracking URI, experiment organization, and model registry.
    """
    
    def __init__(self, base_dir: str = "./mlflow"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Set up directories
        self.tracking_dir = self.base_dir / "mlruns"
        self.artifacts_dir = self.base_dir / "artifacts"
        self.models_dir = self.base_dir / "models"
        
        for directory in [self.tracking_dir, self.artifacts_dir, self.models_dir]:
            directory.mkdir(exist_ok=True)
        
        # Configure MLflow
        self.tracking_uri = f"file://{self.tracking_dir.resolve()}"
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Set environment variables
        os.environ["MLFLOW_TRACKING_URI"] = self.tracking_uri
        
        logger.info(f"MLflow tracking URI set to: {self.tracking_uri}")
    
    def create_experiment_config(self, experiment_name: str) -> Dict[str, str]:
        """Create experiment configuration."""
        return {
            "experiment_name": experiment_name,
            "tracking_uri": self.tracking_uri,
            "artifact_location": str(self.artifacts_dir / experiment_name)
        }
    
    def launch_ui(self, host: str = "127.0.0.1", port: int = 5000) -> None:
        """Launch MLflow UI server."""
        cmd = [
            "mlflow", "ui",
            "--backend-store-uri", self.tracking_uri,
            "--host", host,
            "--port", str(port)
        ]
        logger.info(f"Launching MLflow UI at http://{host}:{port}")
        subprocess.run(cmd)
    
    def get_experiment_id(self, experiment_name: str) -> str:
        """Get or create experiment ID."""
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment:
                return experiment.experiment_id
        except Exception:
            pass
        
        # Create new experiment
        experiment_id = mlflow.create_experiment(
            name=experiment_name,
            artifact_location=str(self.artifacts_dir / experiment_name)
        )
        logger.info(f"Created experiment '{experiment_name}' with ID: {experiment_id}")
        return experiment_id
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments."""
        experiments = mlflow.search_experiments()
        return [
            {
                "name": exp.name,
                "experiment_id": exp.experiment_id,
                "lifecycle_stage": exp.lifecycle_stage,
                "artifact_location": exp.artifact_location,
                "creation_time": exp.creation_time,
                "last_update_time": exp.last_update_time
            }
            for exp in experiments
        ]
    
    def cleanup_old_experiments(self, days_old: int = 30) -> None:
        """Clean up old experiments."""
        # This would implement cleanup logic
        logger.info(f"Cleanup of experiments older than {days_old} days not implemented yet")


class EnhancedMLflowTracker:
    """
    Enhanced MLflow tracker with comprehensive logging capabilities.
    """
    
    def __init__(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        config: Optional[CentralizedMLflowConfig] = None
    ):
        self.config = config or CentralizedMLflowConfig()
        self.experiment_name = experiment_name
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Get or create experiment
        self.experiment_id = self.config.get_experiment_id(experiment_name)
        
        # Initialize run
        self.run = None
        self.is_active = False
    
    def start_run(self, tags: Optional[Dict[str, str]] = None) -> None:
        """Start MLflow run."""
        if self.is_active:
            logger.warning("Run already active, ending previous run")
            self.end_run()
        
        self.run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=self.run_name,
            tags=tags or {}
        )
        self.is_active = True
        logger.info(f"Started MLflow run: {self.run.info.run_id}")
    
    def end_run(self) -> None:
        """End MLflow run."""
        if self.is_active:
            mlflow.end_run()
            self.is_active = False
            logger.info("Ended MLflow run")
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters."""
        if not self.is_active:
            logger.warning("No active run for logging parameters")
            return
        
        # Convert all values to strings and handle nested dicts
        flat_params = self._flatten_dict(params)
        mlflow.log_params(flat_params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics."""
        if not self.is_active:
            logger.warning("No active run for logging metrics")
            return
        
        mlflow.log_metrics(metrics, step=step)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log artifact."""
        if not self.is_active:
            logger.warning("No active run for logging artifacts")
            return
        
        mlflow.log_artifact(local_path, artifact_path)
    
    def log_model(
        self,
        model_path: str,
        model_name: str,
        framework: str = "mlx",
        model_version: Optional[str] = None
    ) -> None:
        """Log model with metadata."""
        if not self.is_active:
            logger.warning("No active run for logging model")
            return
        
        # Log model directory as artifact
        mlflow.log_artifacts(model_path, f"models/{model_name}")
        
        # Log model metadata
        model_info = {
            "model_name": model_name,
            "framework": framework,
            "model_path": model_path,
            "version": model_version or "1.0",
            "timestamp": datetime.now().isoformat()
        }
        
        # Save model info as JSON
        model_info_path = Path(model_path) / "model_info.json"
        with open(model_info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        mlflow.log_artifact(str(model_info_path), "model_metadata")
    
    def log_training_curves(
        self,
        train_metrics: Dict[str, List[float]],
        val_metrics: Dict[str, List[float]],
        save_plots: bool = True
    ) -> None:
        """Log training curves and optionally save plots."""
        if not self.is_active:
            logger.warning("No active run for logging training curves")
            return
        
        # Log final metrics
        for metric_name, values in train_metrics.items():
            if values:
                mlflow.log_metric(f"final_train_{metric_name}", values[-1])
        
        for metric_name, values in val_metrics.items():
            if values:
                mlflow.log_metric(f"final_val_{metric_name}", values[-1])
        
        # Save training history as JSON
        training_history = {
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        history_path = Path.cwd() / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        mlflow.log_artifact(str(history_path), "training_data")
        
        # Optionally create and log plots
        if save_plots:
            try:
                self._create_training_plots(train_metrics, val_metrics)
            except Exception as e:
                logger.warning(f"Failed to create training plots: {e}")
    
    def log_gradient_statistics(
        self,
        gradient_stats: Dict[str, Dict[str, float]],
        step: int
    ) -> None:
        """Log gradient statistics."""
        if not self.is_active:
            return
        
        # Log component-wise gradient statistics
        for component, stats in gradient_stats.items():
            for stat_name, value in stats.items():
                mlflow.log_metric(f"gradients/{component}_{stat_name}", value, step=step)
    
    def log_system_metrics(
        self,
        memory_usage: float,
        training_speed: float,
        step: int
    ) -> None:
        """Log system performance metrics."""
        if not self.is_active:
            return
        
        system_metrics = {
            "system/memory_usage_mb": memory_usage,
            "system/training_speed_samples_per_sec": training_speed,
        }
        
        mlflow.log_metrics(system_metrics, step=step)
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, str]:
        """Flatten nested dictionary for parameter logging."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, str(v)))
        return dict(items)
    
    def _create_training_plots(
        self,
        train_metrics: Dict[str, List[float]],
        val_metrics: Dict[str, List[float]]
    ) -> None:
        """Create and log training plots."""
        try:
            import matplotlib.pyplot as plt
            
            # Create plots for each metric
            for metric_name in train_metrics.keys():
                if metric_name in val_metrics:
                    plt.figure(figsize=(10, 6))
                    
                    train_values = train_metrics[metric_name]
                    val_values = val_metrics[metric_name]
                    
                    plt.plot(train_values, label=f'Train {metric_name}', alpha=0.7)
                    plt.plot(val_values, label=f'Val {metric_name}', alpha=0.7)
                    
                    plt.xlabel('Epoch')
                    plt.ylabel(metric_name.title())
                    plt.title(f'Training Curve: {metric_name.title()}')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    # Save plot
                    plot_path = f"training_curve_{metric_name}.png"
                    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    # Log plot as artifact
                    mlflow.log_artifact(plot_path, "plots")
                    
                    # Clean up
                    Path(plot_path).unlink(missing_ok=True)
        
        except ImportError:
            logger.warning("matplotlib not available, skipping plot generation")
        except Exception as e:
            logger.error(f"Error creating training plots: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.end_run()


# Convenience functions
def setup_mlflow_tracking(
    experiment_name: str,
    run_name: Optional[str] = None,
    base_dir: str = "./mlflow"
) -> EnhancedMLflowTracker:
    """
    Setup centralized MLflow tracking with proper configuration.
    
    Args:
        experiment_name: Name of the experiment
        run_name: Optional run name
        base_dir: Base directory for MLflow data
    
    Returns:
        Configured EnhancedMLflowTracker instance
    """
    config = CentralizedMLflowConfig(base_dir)
    tracker = EnhancedMLflowTracker(experiment_name, run_name, config)
    return tracker


def create_experiment_tags(config: Dict[str, Any]) -> Dict[str, str]:
    """Create comprehensive experiment tags."""
    return {
        "dataset": config.get("dataset", "unknown"),
        "model_type": config.get("model_type", "unknown"),
        "framework": "mlx",
        "task": config.get("task", "classification"),
        "optimizer": config.get("optimizer", "adamw"),
        "learning_rate": str(config.get("learning_rate", "unknown")),
        "batch_size": str(config.get("batch_size", "unknown")),
        "loss_function": config.get("loss_function", "cross_entropy"),
        "augmentation": str(config.get("augment", False)),
        "gradient_clipping": str(config.get("gradient_clipping", False)),
        "environment": "development",
        "version": config.get("version", "1.0.0"),
        "timestamp": datetime.now().isoformat()
    }


def launch_mlflow_ui(base_dir: str = "./mlflow", port: int = 5000) -> None:
    """Launch MLflow UI with centralized configuration."""
    config = CentralizedMLflowConfig(base_dir)
    config.launch_ui(port=port)


def migrate_existing_experiments(old_paths: List[str], new_base_dir: str = "./mlflow") -> None:
    """
    Migrate existing MLflow experiments to centralized location.
    
    Args:
        old_paths: List of paths containing old mlruns directories
        new_base_dir: New centralized base directory
    """
    config = CentralizedMLflowConfig(new_base_dir)
    
    logger.info(f"Migrating experiments to {config.tracking_dir}")
    
    for old_path in old_paths:
        old_mlruns = Path(old_path) / "mlruns"
        if old_mlruns.exists():
            logger.info(f"Found mlruns at {old_mlruns}")
            # This would implement the actual migration logic
            # For now, just log the discovery
            logger.info("Migration logic not implemented yet - manual migration required")