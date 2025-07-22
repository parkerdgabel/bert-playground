"""Loguru monitoring adapter implementation."""

import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from loguru import logger

from core.ports.monitoring import (
    Context,
    ExperimentTracker,
    LogLevel,
    LogSeverity,
    MetricValue,
    MonitoringService,
    Span,
    Tags,
    Timer,
)


class LoguruMonitoringAdapter:
    """Loguru implementation of the MonitoringService port."""

    def __init__(self):
        """Initialize the monitoring adapter."""
        self._context: Context = {}
        self._metrics: dict[str, list[tuple[MetricValue, datetime]]] = {}

    def log(
        self,
        level: LogSeverity | LogLevel,
        message: str,
        context: Context | None = None,
        error: Exception | None = None
    ) -> None:
        """Log a message using Loguru."""
        # Convert LogSeverity enum to string if needed
        if isinstance(level, LogSeverity):
            level = level.value
        
        # Merge global and local context
        full_context = {**self._context, **(context or {})}
        
        # Add error to context if provided
        if error:
            full_context["error"] = str(error)
            full_context["error_type"] = type(error).__name__
        
        # Log with appropriate level
        logger.opt(depth=1).log(level, message, **full_context)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self.log(LogSeverity.DEBUG, message, kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self.log(LogSeverity.INFO, message, kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self.log(LogSeverity.WARNING, message, kwargs)

    def error(
        self,
        message: str,
        error: Exception | None = None,
        **kwargs: Any
    ) -> None:
        """Log error message."""
        self.log(LogSeverity.ERROR, message, kwargs, error)

    def metric(
        self,
        name: str,
        value: MetricValue,
        tags: Tags | None = None,
        timestamp: datetime | None = None
    ) -> None:
        """Record a metric value."""
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
            metric_tags=tags or {},
            metric_timestamp=timestamp.isoformat()
        )

    def gauge(
        self,
        name: str,
        value: MetricValue,
        tags: Tags | None = None
    ) -> None:
        """Record a gauge metric."""
        self.metric(f"gauge.{name}", value, tags)

    def counter(
        self,
        name: str,
        value: MetricValue = 1,
        tags: Tags | None = None
    ) -> None:
        """Increment a counter metric."""
        self.metric(f"counter.{name}", value, tags)

    def histogram(
        self,
        name: str,
        value: MetricValue,
        tags: Tags | None = None
    ) -> None:
        """Record a histogram metric."""
        self.metric(f"histogram.{name}", value, tags)

    def timer(
        self,
        name: str,
        tags: Tags | None = None
    ) -> "LoguruTimer":
        """Create a timer context manager."""
        return LoguruTimer(self, name, tags)

    def span(
        self,
        name: str,
        context: Context | None = None
    ) -> "LoguruSpan":
        """Create a tracing span."""
        return LoguruSpan(self, name, context)

    def set_context(self, **kwargs: Any) -> None:
        """Set global context values."""
        self._context.update(kwargs)

    def clear_context(self) -> None:
        """Clear global context."""
        self._context.clear()


class LoguruTimer:
    """Timer implementation for Loguru adapter."""

    def __init__(
        self,
        monitor: LoguruMonitoringAdapter,
        name: str,
        tags: Tags | None = None
    ):
        """Initialize timer."""
        self.monitor = monitor
        self.name = name
        self.tags = tags
        self.start_time: float | None = None
        self._elapsed: float = 0.0

    def __enter__(self) -> "LoguruTimer":
        """Start the timer."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Stop the timer and record the metric."""
        if self.start_time is not None:
            self._elapsed = time.time() - self.start_time
            self.monitor.metric(
                f"timer.{self.name}",
                self._elapsed * 1000,  # Convert to milliseconds
                self.tags
            )

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is not None and self._elapsed == 0.0:
            return time.time() - self.start_time
        return self._elapsed


class LoguruSpan:
    """Span implementation for Loguru adapter."""

    def __init__(
        self,
        monitor: LoguruMonitoringAdapter,
        name: str,
        context: Context | None = None
    ):
        """Initialize span."""
        self.monitor = monitor
        self.name = name
        self.context = context or {}
        self.tags: Tags = {}
        self.start_time: datetime | None = None

    def __enter__(self) -> "LoguruSpan":
        """Enter the span."""
        self.start_time = datetime.now()
        self.monitor.info(
            f"Span started: {self.name}",
            span_name=self.name,
            span_context=self.context
        )
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the span."""
        duration = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        status = "error" if exc_type else "success"
        
        self.monitor.info(
            f"Span ended: {self.name}",
            span_name=self.name,
            span_duration=duration,
            span_status=status,
            span_tags=self.tags
        )

    def set_tag(self, key: str, value: str) -> None:
        """Set a span tag."""
        self.tags[key] = value

    def log(self, message: str, **kwargs: Any) -> None:
        """Log within the span context."""
        self.monitor.info(
            message,
            span_name=self.name,
            **kwargs
        )

    def set_status(self, status: str) -> None:
        """Set span status."""
        self.tags["status"] = status


class MLflowExperimentTracker:
    """MLflow implementation of the ExperimentTracker port."""

    def __init__(self):
        """Initialize MLflow tracker."""
        try:
            import mlflow
            self.mlflow = mlflow
            self._active_run_id: str | None = None
        except ImportError:
            raise ImportError(
                "MLflow is required for experiment tracking. "
                "Install with: pip install mlflow"
            )

    def start_run(
        self,
        run_name: str | None = None,
        tags: Tags | None = None,
        nested: bool = False
    ) -> str:
        """Start a new experiment run."""
        run = self.mlflow.start_run(
            run_name=run_name,
            tags=tags,
            nested=nested
        )
        self._active_run_id = run.info.run_id
        return run.info.run_id

    def end_run(self, status: str = "FINISHED") -> None:
        """End the current run."""
        self.mlflow.end_run(status=status)
        self._active_run_id = None

    def log_params(self, params: dict[str, Any]) -> None:
        """Log run parameters."""
        # MLflow requires string values for params
        str_params = {k: str(v) for k, v in params.items()}
        self.mlflow.log_params(str_params)

    def log_metrics(
        self,
        metrics: dict[str, MetricValue],
        step: int | None = None
    ) -> None:
        """Log run metrics."""
        for name, value in metrics.items():
            self.mlflow.log_metric(name, value, step=step)

    def log_artifact(
        self,
        path: str,
        artifact_type: str | None = None
    ) -> None:
        """Log an artifact file."""
        self.mlflow.log_artifact(path)
        if artifact_type:
            self.mlflow.set_tag("artifact_type", artifact_type)

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        metadata: dict[str, Any] | None = None
    ) -> None:
        """Log a model artifact."""
        # This is a simplified version - actual implementation
        # would depend on the model framework
        import pickle
        import tempfile
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
            pickle.dump(model, f)
            temp_path = f.name
        
        self.mlflow.log_artifact(temp_path, artifact_path)
        
        if metadata:
            for key, value in metadata.items():
                self.mlflow.set_tag(f"model_{key}", str(value))

    def get_run_id(self) -> str | None:
        """Get current run ID."""
        return self._active_run_id

    def set_tags(self, tags: Tags) -> None:
        """Set run tags."""
        self.mlflow.set_tags(tags)