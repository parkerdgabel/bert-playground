"""MLflow implementation of MonitoringService."""

import os
import tempfile
from typing import Dict, Any, Optional, List
from datetime import datetime
from contextlib import contextmanager

from ..base import BaseMonitoringAdapter, BaseProgressBar
from .config import MLflowConfig
# object removed - not defined in ports
from domain.entities.training import TrainingSession


class MLflowMonitoringAdapter(BaseMonitoringAdapter):
    """MLflow implementation of the MonitoringService."""
    
    def __init__(self, config: Optional[MLflowConfig] = None):
        """Initialize MLflow monitoring adapter.
        
        Args:
            config: MLflow configuration
        """
        super().__init__()
        self.config = config or MLflowConfig()
        self._mlflow = None
        self._initialized = False
        self._active_run = None
        self._parent_run_id = None
        self._run_stack = []
        
    @property
    def mlflow(self):
        """Lazy import and initialization of MLflow."""
        if self._mlflow is None:
            try:
                import mlflow
                self._mlflow = mlflow
                
                # Configure MLflow
                if self.config.tracking_uri:
                    mlflow.set_tracking_uri(self.config.tracking_uri)
                
                if self.config.registry_uri:
                    mlflow.set_registry_uri(self.config.registry_uri)
                
                # Set or create experiment
                mlflow.set_experiment(self.config.experiment_name)
                
                self._initialized = True
                
            except ImportError:
                raise ImportError(
                    "MLflow is required for MLflow monitoring. "
                    "Install with: pip install mlflow"
                )
        
        return self._mlflow
    
    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
    ) -> None:
        """Log metrics.
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Optional global step
            epoch: Optional epoch number
        """
        if not self._active_run:
            return
        
        # Filter out non-numeric values
        numeric_metrics = {
            k: float(v) for k, v in metrics.items() 
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        
        # Add epoch to metric names if provided
        if epoch is not None:
            numeric_metrics["epoch"] = epoch
        
        # Log to MLflow
        self.mlflow.log_metrics(numeric_metrics, step=step)
        
        # Store in internal tracking
        for name, value in numeric_metrics.items():
            if name not in self._run_metrics:
                self._run_metrics[name] = []
            self._run_metrics[name].append({
                "value": value,
                "step": step,
                "epoch": epoch,
                "timestamp": datetime.now().isoformat()
            })
    
    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters.
        
        Args:
            params: Dictionary of hyperparameters
        """
        if not self._active_run:
            return
        
        # MLflow requires string values for params
        str_params = {}
        for k, v in params.items():
            if v is None:
                str_params[k] = "None"
            elif isinstance(v, (list, dict)):
                import json
                str_params[k] = json.dumps(v)
            else:
                str_params[k] = str(v)
        
        self.mlflow.log_params(str_params)
        self._run_params.update(params)
    
    def log_artifact(
        self,
        path: str,
        artifact_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an artifact (file, model, etc.).
        
        Args:
            path: Path to artifact
            artifact_type: Type of artifact
            metadata: Optional metadata
        """
        if not self._active_run:
            return
        
        # Log the artifact
        if os.path.isfile(path):
            self.mlflow.log_artifact(path)
        elif os.path.isdir(path):
            self.mlflow.log_artifacts(path)
        
        # Log metadata as tags if provided
        if metadata:
            tags = {f"artifact.{k}": str(v) for k, v in metadata.items()}
            if artifact_type:
                tags["artifact.type"] = artifact_type
            self.mlflow.set_tags(tags)
        
        self._run_artifacts.append(path)
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """Start a new monitoring run.
        
        Args:
            run_name: Optional run name
            tags: Optional tags for the run
            
        Returns:
            Run ID
        """
        # Handle nested runs
        nested = bool(self._active_run) and self.config.nested_runs
        
        # Start MLflow run
        mlflow_run = self.mlflow.start_run(
            run_name=run_name,
            nested=nested,
            tags=tags
        )
        
        self._active_run = mlflow_run
        self._active_run_id = mlflow_run.info.run_id
        
        # Track run stack for nested runs
        if nested:
            self._run_stack.append(self._active_run_id)
        else:
            self._run_stack = [self._active_run_id]
        
        # Set default tags
        default_tags = {
            "framework": "k-bert",
            "monitoring": "mlflow",
            "start_time": datetime.now().isoformat()
        }
        if tags:
            default_tags.update(tags)
        self.mlflow.set_tags(default_tags)
        
        return self._active_run_id
    
    def end_run(self, status: Optional[str] = None) -> None:
        """End current monitoring run.
        
        Args:
            status: Optional run status
        """
        if not self._active_run:
            return
        
        # Set final status if provided
        if status:
            self.mlflow.set_tag("run_status", status)
            self.mlflow.set_tag("end_time", datetime.now().isoformat())
        
        # End the MLflow run
        self.mlflow.end_run(status=status)
        
        # Update run stack
        if self._run_stack:
            self._run_stack.pop()
        
        # Reset active run
        if self._run_stack:
            self._active_run_id = self._run_stack[-1]
            # Note: We can't easily restore the run object, so we set it to None
            self._active_run = None
        else:
            self._active_run = None
            self._active_run_id = None
    
    def log_message(
        self,
        message: str,
        level: str = "INFO",
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a message.
        
        Args:
            message: Message to log
            level: Log level (DEBUG, INFO, WARNING, ERROR)
            context: Optional context information
        """
        if not self._active_run:
            return
        
        # Create a temporary file for the log message
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] [{level}] {message}\n"
        
        if context:
            log_entry += "Context:\n"
            for k, v in context.items():
                log_entry += f"  {k}: {v}\n"
        
        # Write to a temporary file and log as artifact
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.log',
            delete=False
        ) as f:
            f.write(log_entry)
            temp_path = f.name
        
        try:
            # Log as an artifact in a logs directory
            self.mlflow.log_artifact(temp_path, artifact_path="logs")
        finally:
            # Clean up temporary file
            os.unlink(temp_path)
    
    def create_progress_bar(
        self,
        total: int,
        description: str,
        unit: str = "it",
    ) -> object:
        """Create a progress bar.
        
        Args:
            total: Total number of items
            description: Progress bar description
            unit: Unit name
            
        Returns:
            Progress bar instance
        """
        # MLflow doesn't have native progress bars, return base implementation
        return MLflowProgressBar(
            total=total,
            description=description,
            unit=unit,
            adapter=self
        )
    
    def get_run_metrics(
        self,
        run_id: Optional[str] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get metrics for a run.
        
        Args:
            run_id: Optional run ID (current run if None)
            
        Returns:
            Dictionary of metric histories
        """
        if run_id is None:
            # Return cached metrics for current run
            return self._run_metrics
        
        # Get metrics from MLflow
        client = self.mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        
        metrics = {}
        for metric_key in run.data.metrics:
            history = client.get_metric_history(run_id, metric_key)
            metrics[metric_key] = [
                {
                    "value": m.value,
                    "step": m.step,
                    "timestamp": datetime.fromtimestamp(m.timestamp / 1000).isoformat()
                }
                for m in history
            ]
        
        return metrics
    
    def compare_runs(
        self,
        run_ids: List[str],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Compare multiple runs.
        
        Args:
            run_ids: List of run IDs to compare
            metrics: Optional list of metrics to compare
            
        Returns:
            Comparison results
        """
        client = self.mlflow.tracking.MlflowClient()
        comparison = {}
        
        for run_id in run_ids:
            run = client.get_run(run_id)
            run_data = {
                "params": run.data.params,
                "metrics": {},
                "tags": run.data.tags,
                "status": run.info.status,
                "start_time": datetime.fromtimestamp(run.info.start_time / 1000).isoformat(),
            }
            
            # Get final metric values
            if metrics:
                for metric in metrics:
                    if metric in run.data.metrics:
                        run_data["metrics"][metric] = run.data.metrics[metric]
            else:
                run_data["metrics"] = run.data.metrics
            
            comparison[run_id] = run_data
        
        return comparison
    
    def log_training_session(self, session: TrainingSession) -> None:
        """Log complete training session information.
        
        Args:
            session: Training session object
        """
        # First call parent implementation
        super().log_training_session(session)
        
        # Log model if configured
        if self.config.log_models and session.checkpoint_paths:
            # Log the final checkpoint as a model
            final_checkpoint = session.checkpoint_paths[-1]
            if os.path.exists(final_checkpoint):
                self.mlflow.log_artifact(
                    final_checkpoint,
                    artifact_path="model"
                )
                
                # Register model if configured
                if self.config.register_models and self.config.model_name:
                    model_uri = f"runs:/{self._active_run_id}/model"
                    self.mlflow.register_model(
                        model_uri,
                        self.config.model_name
                    )
    
    @contextmanager
    def nested_run(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for nested runs.
        
        Args:
            name: Run name
            tags: Optional tags
        """
        run_id = self.start_run(run_name=name, tags=tags)
        try:
            yield run_id
        finally:
            self.end_run()


class MLflowProgressBar(BaseProgressBar):
    """Progress bar that logs to MLflow."""
    
    def __init__(self, total: int, description: str, unit: str, adapter: MLflowMonitoringAdapter):
        """Initialize MLflow progress bar.
        
        Args:
            total: Total number of items
            description: Progress bar description
            unit: Unit name
            adapter: MLflow adapter instance
        """
        super().__init__(total, description)
        self.unit = unit
        self.adapter = adapter
        self.last_log_time = datetime.now()
        self.log_interval = 5.0  # Log every 5 seconds
    
    def update(self, n: int = 1) -> None:
        """Update progress and optionally log to MLflow.
        
        Args:
            n: Number of items completed
        """
        super().update(n)
        
        # Check if we should log
        now = datetime.now()
        time_since_last_log = (now - self.last_log_time).total_seconds()
        
        if time_since_last_log >= self.log_interval or self.current >= self.total:
            # Log progress as a metric
            progress_pct = (self.current / self.total) * 100
            self.adapter.log_metrics({
                f"progress/{self.description}": progress_pct
            })
            self.last_log_time = now
    
    def set_postfix(self, **kwargs: Any) -> None:
        """Set postfix values and log them as metrics.
        
        Args:
            **kwargs: Key-value pairs to display
        """
        super().set_postfix(**kwargs)
        
        # Log postfix values as metrics
        metrics = {}
        for k, v in kwargs.items():
            if isinstance(v, (int, float)):
                metrics[f"{self.description}/{k}"] = v
        
        if metrics:
            self.adapter.log_metrics(metrics)