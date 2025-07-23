"""Loguru monitoring adapter implementation.

This module provides loguru and MLflow implementations of the monitoring ports,
enabling structured logging, metrics tracking, and experiment management.
"""

import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from loguru import logger

from ports.secondary.monitoring import (
    ExperimentInfo,
    LogLevel,
    Metric,
    MetricType,
    MonitoringPort,
    RunInfo,
    RunStatus,
)


class LoguruMonitoringAdapter(MonitoringPort):
    """Loguru implementation of the MonitoringPort."""

    def __init__(self):
        """Initialize the monitoring adapter."""
        self._context: dict[str, Any] = {}
        self._metrics: dict[str, list[tuple[float, datetime]]] = {}
        self._active_timers: dict[str, float] = {}

    def log(
        self,
        level: LogLevel,
        message: str,
        extra: dict[str, Any] | None = None,
        error: Exception | None = None
    ) -> None:
        """Log a message using Loguru."""
        # Merge global and local context
        full_context = {**self._context, **(extra or {})}
        
        # Add error to context if provided
        if error:
            full_context["error"] = str(error)
            full_context["error_type"] = type(error).__name__
        
        # Log with appropriate level
        logger.opt(depth=1).log(level.value, message, **full_context)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self.log(LogLevel.DEBUG, message, kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self.log(LogLevel.INFO, message, kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self.log(LogLevel.WARNING, message, kwargs)

    def error(
        self,
        message: str,
        error: Exception | None = None,
        **kwargs: Any
    ) -> None:
        """Log error message."""
        self.log(LogLevel.ERROR, message, kwargs, error)

    def log_metric(
        self,
        name: str,
        value: float,
        step: int | None = None,
        timestamp: datetime | None = None,
        tags: dict[str, str] | None = None
    ) -> None:
        """Log a metric value."""
        timestamp = timestamp or datetime.now()
        
        # Store metric internally
        if name not in self._metrics:
            self._metrics[name] = []
        self._metrics[name].append((value, timestamp))
        
        # Log metric
        logger.info(
            f"Metric: {name}={value}",
            metric_name=name,
            metric_value=value,
            metric_step=step,
            metric_tags=tags or {},
            metric_timestamp=timestamp.isoformat()
        )

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
        timestamp: datetime | None = None,
        tags: dict[str, str] | None = None
    ) -> None:
        """Log multiple metrics at once."""
        for name, value in metrics.items():
            self.log_metric(name, value, step, timestamp, tags)

    def start_timer(self, name: str) -> None:
        """Start a timer."""
        self._active_timers[name] = time.time()

    def stop_timer(self, name: str) -> float:
        """Stop a timer and return elapsed time."""
        if name not in self._active_timers:
            raise ValueError(f"Timer '{name}' not started")
        
        elapsed = time.time() - self._active_timers[name]
        del self._active_timers[name]
        
        # Log the timing as a metric
        self.log_metric(f"timer.{name}", elapsed * 1000)  # Convert to ms
        
        return elapsed

    @contextmanager
    def timer(self, name: str):
        """Context manager for timing operations."""
        self.start_timer(name)
        try:
            yield
        finally:
            self.stop_timer(name)

    def set_context(self, **kwargs: Any) -> None:
        """Set global context values."""
        self._context.update(kwargs)

    def clear_context(self) -> None:
        """Clear global context."""
        self._context.clear()

    def flush(self) -> None:
        """Flush any buffered logs."""
        # Loguru handles flushing automatically
        pass


class MLflowExperimentTracker(MonitoringPort):
    """MLflow implementation for experiment tracking functionality."""

    def __init__(self):
        """Initialize MLflow tracker."""
        try:
            import mlflow
            self.mlflow = mlflow
            self._active_run_id: str | None = None
            self._experiment_id: str | None = None
        except ImportError:
            raise ImportError(
                "MLflow is required for experiment tracking. "
                "Install with: pip install mlflow"
            )

    def create_experiment(
        self,
        name: str,
        description: str | None = None,
        tags: dict[str, str] | None = None
    ) -> str:
        """Create a new experiment."""
        experiment_id = self.mlflow.create_experiment(
            name=name,
            artifact_location=None,
            tags=tags
        )
        
        if description:
            self.mlflow.set_experiment_tag("description", description)
        
        self._experiment_id = experiment_id
        return experiment_id

    def set_experiment(self, name: str) -> str:
        """Set the active experiment."""
        experiment = self.mlflow.set_experiment(name)
        self._experiment_id = experiment.experiment_id
        return experiment.experiment_id

    def get_experiment_info(self, experiment_id: str) -> ExperimentInfo:
        """Get experiment information."""
        exp = self.mlflow.get_experiment(experiment_id)
        
        return ExperimentInfo(
            id=exp.experiment_id,
            name=exp.name,
            artifact_location=exp.artifact_location,
            lifecycle_stage=exp.lifecycle_stage,
            tags=exp.tags,
            creation_time=datetime.fromtimestamp(exp.creation_time / 1000),
            last_update_time=datetime.fromtimestamp(exp.last_update_time / 1000) if exp.last_update_time else None,
        )

    def start_run(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
        description: str | None = None
    ) -> str:
        """Start a new run."""
        run = self.mlflow.start_run(
            run_name=run_name,
            tags=tags,
            description=description,
            experiment_id=self._experiment_id
        )
        self._active_run_id = run.info.run_id
        return run.info.run_id

    def end_run(self, status: RunStatus = RunStatus.FINISHED) -> None:
        """End the current run."""
        self.mlflow.end_run(status=status.value)
        self._active_run_id = None

    def get_run_info(self, run_id: str) -> RunInfo:
        """Get run information."""
        run = self.mlflow.get_run(run_id)
        
        return RunInfo(
            id=run.info.run_id,
            experiment_id=run.info.experiment_id,
            status=RunStatus(run.info.status),
            start_time=datetime.fromtimestamp(run.info.start_time / 1000),
            end_time=datetime.fromtimestamp(run.info.end_time / 1000) if run.info.end_time else None,
            artifact_uri=run.info.artifact_uri,
            tags=run.data.tags,
        )

    def log_params(self, params: dict[str, Any]) -> None:
        """Log run parameters."""
        # MLflow requires string values for params
        str_params = {k: str(v) for k, v in params.items()}
        self.mlflow.log_params(str_params)

    def log_metric(
        self,
        name: str,
        value: float,
        step: int | None = None,
        timestamp: datetime | None = None,
        tags: dict[str, str] | None = None
    ) -> None:
        """Log a metric value."""
        timestamp_ms = int(timestamp.timestamp() * 1000) if timestamp else None
        self.mlflow.log_metric(name, value, step=step, timestamp=timestamp_ms)
        
        if tags:
            for tag_key, tag_value in tags.items():
                self.mlflow.set_tag(f"metric.{name}.{tag_key}", tag_value)

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
        timestamp: datetime | None = None,
        tags: dict[str, str] | None = None
    ) -> None:
        """Log multiple metrics at once."""
        timestamp_ms = int(timestamp.timestamp() * 1000) if timestamp else None
        self.mlflow.log_metrics(metrics, step=step, timestamp=timestamp_ms)
        
        if tags:
            self.mlflow.set_tags(tags)

    def log_artifact(
        self,
        local_path: str,
        artifact_path: str | None = None
    ) -> None:
        """Log an artifact file or directory."""
        self.mlflow.log_artifact(local_path, artifact_path)

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        **kwargs: Any
    ) -> None:
        """Log a model."""
        # This is a simplified version - actual implementation
        # would use MLflow's model logging based on framework
        import pickle
        import tempfile
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
            pickle.dump(model, f)
            temp_path = f.name
        
        self.mlflow.log_artifact(temp_path, artifact_path)

    def set_tags(self, tags: dict[str, str]) -> None:
        """Set run tags."""
        self.mlflow.set_tags(tags)

    def get_metric_history(
        self,
        run_id: str,
        metric_name: str
    ) -> list[Metric]:
        """Get metric history for a run."""
        client = self.mlflow.tracking.MlflowClient()
        history = client.get_metric_history(run_id, metric_name)
        
        return [
            Metric(
                name=metric_name,
                value=m.value,
                timestamp=datetime.fromtimestamp(m.timestamp / 1000),
                step=m.step,
                type=MetricType.GAUGE,
            )
            for m in history
        ]

    # Implement remaining MonitoringPort methods with pass or basic implementation
    def log(self, level: LogLevel, message: str, extra: dict[str, Any] | None = None, error: Exception | None = None) -> None:
        """Log through MLflow (basic implementation)."""
        pass

    def debug(self, message: str, **kwargs: Any) -> None:
        """Debug logging."""
        pass

    def info(self, message: str, **kwargs: Any) -> None:
        """Info logging."""
        pass

    def warning(self, message: str, **kwargs: Any) -> None:
        """Warning logging."""
        pass

    def error(self, message: str, error: Exception | None = None, **kwargs: Any) -> None:
        """Error logging."""
        pass

    def start_timer(self, name: str) -> None:
        """Timer tracking not implemented in MLflow adapter."""
        pass

    def stop_timer(self, name: str) -> float:
        """Timer tracking not implemented in MLflow adapter."""
        return 0.0

    @contextmanager
    def timer(self, name: str):
        """Timer context manager."""
        yield

    def set_context(self, **kwargs: Any) -> None:
        """Context setting."""
        pass

    def clear_context(self) -> None:
        """Context clearing."""
        pass

    def flush(self) -> None:
        """Flush any buffered data."""
        pass