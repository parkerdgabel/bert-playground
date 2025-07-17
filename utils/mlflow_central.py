"""Central MLflow configuration for unified experiment tracking.

This module provides a single source of truth for MLflow configuration,
ensuring all training runs use the same database and tracking location.
"""

import os
from pathlib import Path
from typing import Optional

import mlflow
from loguru import logger


class MLflowCentral:
    """Central MLflow configuration manager (Singleton)."""
    
    # Central tracking configuration
    TRACKING_URI = "sqlite:///mlruns/mlflow.db"
    ARTIFACT_ROOT = "./mlruns/artifacts"
    DEFAULT_EXPERIMENT = "mlx_training"
    
    _instance = None
    
    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize central MLflow configuration."""
        # Only initialize once
        if not hasattr(self, '_initialized'):
            self._initialized = False
            self._tracking_uri = None
            self._artifact_root = None
    
    def initialize(
        self,
        tracking_uri: Optional[str] = None,
        artifact_root: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ) -> None:
        """Initialize MLflow with central configuration.
        
        Args:
            tracking_uri: Override default tracking URI
            artifact_root: Override default artifact location
            experiment_name: Experiment name to use
        """
        if self._initialized:
            logger.debug("MLflow already initialized")
            return
        
        # Use environment variables if set, otherwise use defaults
        self._tracking_uri = (
            tracking_uri
            or os.getenv("MLFLOW_TRACKING_URI")
            or self.TRACKING_URI
        )
        self._artifact_root = (
            artifact_root
            or os.getenv("MLFLOW_ARTIFACT_ROOT")
            or self.ARTIFACT_ROOT
        )
        
        # Ensure directories exist
        Path(self._tracking_uri.replace("sqlite:///", "")).parent.mkdir(
            parents=True, exist_ok=True
        )
        Path(self._artifact_root).mkdir(parents=True, exist_ok=True)
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(self._tracking_uri)
        
        # Set experiment
        experiment = experiment_name or self.DEFAULT_EXPERIMENT
        mlflow.set_experiment(experiment)
        
        self._initialized = True
        
        logger.info(
            f"MLflow initialized with central configuration:\n"
            f"  Tracking URI: {self._tracking_uri}\n"
            f"  Artifact Root: {self._artifact_root}\n"
            f"  Experiment: {experiment}"
        )
    
    @property
    def tracking_uri(self) -> str:
        """Get current tracking URI."""
        return self._tracking_uri or self.TRACKING_URI
    
    @property
    def artifact_root(self) -> str:
        """Get current artifact root."""
        return self._artifact_root or self.ARTIFACT_ROOT
    
    def get_experiment_id(self, experiment_name: str) -> str:
        """Get or create experiment ID.
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            Experiment ID
        """
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=self.artifact_root,
            )
        else:
            experiment_id = experiment.experiment_id
        
        return experiment_id
    
    def list_experiments(self) -> list:
        """List all experiments in the central database."""
        return mlflow.search_experiments()
    
    def migrate_experiment(
        self,
        source_uri: str,
        experiment_name: str,
        target_experiment_name: Optional[str] = None,
    ) -> None:
        """Migrate an experiment from another tracking URI.
        
        Args:
            source_uri: Source MLflow tracking URI
            experiment_name: Name of experiment to migrate
            target_experiment_name: New name for experiment (optional)
        """
        # Save current tracking URI
        original_uri = mlflow.get_tracking_uri()
        
        try:
            # Connect to source
            mlflow.set_tracking_uri(source_uri)
            source_experiment = mlflow.get_experiment_by_name(experiment_name)
            
            if source_experiment is None:
                logger.warning(f"Experiment '{experiment_name}' not found in {source_uri}")
                return
            
            # Get all runs from source experiment
            runs = mlflow.search_runs(
                experiment_ids=[source_experiment.experiment_id],
                output_format="list",
            )
            
            # Switch to central tracking
            mlflow.set_tracking_uri(self.tracking_uri)
            
            # Create target experiment
            target_name = target_experiment_name or experiment_name
            target_exp_id = self.get_experiment_id(target_name)
            
            logger.info(
                f"Migrating {len(runs)} runs from '{experiment_name}' to '{target_name}'"
            )
            
            # Copy runs
            for run in runs:
                with mlflow.start_run(experiment_id=target_exp_id):
                    # Log parameters
                    for key, value in run.data.params.items():
                        mlflow.log_param(key, value)
                    
                    # Log metrics
                    for key, value in run.data.metrics.items():
                        mlflow.log_metric(key, value)
                    
                    # Log tags
                    for key, value in run.data.tags.items():
                        if not key.startswith("mlflow."):
                            mlflow.set_tag(key, value)
            
            logger.info(f"Successfully migrated experiment '{experiment_name}'")
            
        finally:
            # Restore original tracking URI
            mlflow.set_tracking_uri(original_uri)
    
    def cleanup_artifacts(self, experiment_name: str, keep_best_n: int = 5) -> None:
        """Clean up old artifacts, keeping only the best N runs.
        
        Args:
            experiment_name: Name of experiment to clean
            keep_best_n: Number of best runs to keep
        """
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.warning(f"Experiment '{experiment_name}' not found")
            return
        
        # Get all runs sorted by primary metric
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.val_accuracy DESC"],
            output_format="list",
        )
        
        if len(runs) <= keep_best_n:
            logger.info(f"Only {len(runs)} runs found, no cleanup needed")
            return
        
        # Delete artifacts for runs beyond keep_best_n
        runs_to_clean = runs[keep_best_n:]
        logger.info(f"Cleaning up artifacts for {len(runs_to_clean)} runs")
        
        for run in runs_to_clean:
            # Note: MLflow doesn't provide direct artifact deletion
            # This would need to be implemented based on storage backend
            logger.debug(f"Would clean artifacts for run {run.info.run_id}")


# Global instance
mlflow_central = MLflowCentral()


def setup_central_mlflow(
    experiment_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
) -> MLflowCentral:
    """Convenience function to setup central MLflow configuration.
    
    Args:
        experiment_name: Experiment name to use
        tracking_uri: Override tracking URI
        
    Returns:
        Configured MLflowCentral instance
    """
    mlflow_central.initialize(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
    )
    return mlflow_central