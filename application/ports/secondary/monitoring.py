"""Secondary monitoring port - Monitoring services that the application depends on.

This port defines the monitoring interface that the application core uses
for logging, metrics, and experiment tracking. It's a driven port implemented
by adapters for different monitoring backends (loguru, MLflow, wandb, etc.).
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Protocol, runtime_checkable, Optional, Dict

from typing_extensions import TypeAlias
from infrastructure.di import port

# Type aliases
MetricValue: TypeAlias = float | int
Tags: TypeAlias = dict[str, str]
Context: TypeAlias = dict[str, Any]
LogLevel: TypeAlias = str


class LogSeverity(Enum):
    """Log severity levels."""
    
    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class RunStatus(Enum):
    """Status of an experiment run."""
    
    RUNNING = "RUNNING"
    SCHEDULED = "SCHEDULED"
    FINISHED = "FINISHED"
    FAILED = "FAILED"
    KILLED = "KILLED"


@dataclass
class ExperimentInfo:
    """Information about an experiment."""
    
    id: str
    name: str
    artifact_location: Optional[str] = None
    lifecycle_stage: Optional[str] = None
    tags: Optional[Dict[str, str]] = None
    creation_time: Optional[datetime] = None
    last_update_time: Optional[datetime] = None


@dataclass 
class Metric:
    """Metric data point."""
    
    name: str
    value: MetricValue
    step: Optional[int] = None
    timestamp: Optional[datetime] = None
    tags: Optional[Tags] = None


@dataclass
class RunInfo:
    """Information about an experiment run."""
    
    id: str
    experiment_id: str
    status: RunStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    artifact_uri: Optional[str] = None
    tags: Optional[Dict[str, str]] = None


@port()
@runtime_checkable
class MonitoringService(Protocol):
    """Secondary port for monitoring and logging operations.
    
    This interface is implemented by adapters for specific monitoring backends.
    The application core depends on this for all monitoring needs.
    """

    def log(
        self,
        level: LogSeverity | LogLevel,
        message: str,
        context: Optional[Context] = None,
        error: Optional[Exception] = None
    ) -> None:
        """Log a message.
        
        Args:
            level: Log severity level
            message: Log message
            context: Optional context data
            error: Optional exception
        """
        ...

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        ...

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        ...

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        ...

    def error(
        self,
        message: str,
        error: Optional[Exception] = None,
        **kwargs: Any
    ) -> None:
        """Log error message."""
        ...

    def metric(
        self,
        name: str,
        value: MetricValue,
        tags: Optional[Tags] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Record a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional metric tags
            timestamp: Optional timestamp
        """
        ...

    def gauge(
        self,
        name: str,
        value: MetricValue,
        tags: Optional[Tags] = None
    ) -> None:
        """Record a gauge metric (point-in-time value)."""
        ...

    def counter(
        self,
        name: str,
        value: MetricValue = 1,
        tags: Optional[Tags] = None
    ) -> None:
        """Increment a counter metric."""
        ...

    def histogram(
        self,
        name: str,
        value: MetricValue,
        tags: Optional[Tags] = None
    ) -> None:
        """Record a histogram metric."""
        ...

    def timer(
        self,
        name: str,
        tags: Optional[Tags] = None
    ) -> "Timer":
        """Create a timer context manager.
        
        Args:
            name: Timer metric name
            tags: Optional tags
            
        Returns:
            Timer context manager
        """
        ...

    def span(
        self,
        name: str,
        context: Optional[Context] = None
    ) -> "Span":
        """Create a tracing span.
        
        Args:
            name: Span name
            context: Optional span context
            
        Returns:
            Span context manager
        """
        ...

    def set_context(self, **kwargs: Any) -> None:
        """Set global context values.
        
        Args:
            **kwargs: Context key-value pairs
        """
        ...

    def clear_context(self) -> None:
        """Clear global context."""
        ...


@port()
@runtime_checkable
class Timer(Protocol):
    """Timer context manager for measuring durations."""

    def __enter__(self) -> "Timer":
        """Start the timer."""
        ...

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Stop the timer and record the metric."""
        ...

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        ...


@port()
@runtime_checkable
class Span(Protocol):
    """Tracing span for distributed tracing."""

    def __enter__(self) -> "Span":
        """Enter the span."""
        ...

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the span."""
        ...

    def set_tag(self, key: str, value: str) -> None:
        """Set a span tag."""
        ...

    def log(self, message: str, **kwargs: Any) -> None:
        """Log within the span context."""
        ...

    def set_status(self, status: str) -> None:
        """Set span status."""
        ...


@port()
@runtime_checkable
class ExperimentTracker(Protocol):
    """Experiment tracking port for ML experiments.
    
    This specialized monitoring port handles ML-specific tracking needs
    like hyperparameters, metrics over time, and model artifacts.
    """

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Tags] = None,
        nested: bool = False
    ) -> str:
        """Start a new experiment run.
        
        Args:
            run_name: Optional run name
            tags: Optional run tags
            nested: Whether this is a nested run
            
        Returns:
            Run ID
        """
        ...

    def end_run(self, status: str = "FINISHED") -> None:
        """End the current run.
        
        Args:
            status: Run status
        """
        ...

    def log_params(self, params: dict[str, Any]) -> None:
        """Log run parameters.
        
        Args:
            params: Parameters to log
        """
        ...

    def log_metrics(
        self,
        metrics: dict[str, MetricValue],
        step: Optional[int] = None
    ) -> None:
        """Log run metrics.
        
        Args:
            metrics: Metrics to log
            step: Optional step number
        """
        ...

    def log_artifact(
        self,
        path: str,
        artifact_type: Optional[str] = None
    ) -> None:
        """Log an artifact file.
        
        Args:
            path: Path to artifact
            artifact_type: Optional artifact type
        """
        ...

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        metadata: dict[str, Any] | None = None
    ) -> None:
        """Log a model artifact.
        
        Args:
            model: Model to log
            artifact_path: Artifact path in run
            metadata: Optional model metadata
        """
        ...

    def get_run_id(self) -> Optional[str]:
        """Get current run ID."""
        ...

    def set_tags(self, tags: Tags) -> None:
        """Set run tags."""
        ...

    def log_figure(
        self,
        figure: Any,
        name: str,
        step: Optional[int] = None
    ) -> None:
        """Log a figure/plot.
        
        Args:
            figure: Figure to log (matplotlib, plotly, etc.)
            name: Figure name
            step: Optional step number
        """
        ...

    def log_text(
        self,
        text: str,
        name: str,
        step: Optional[int] = None
    ) -> None:
        """Log text data.
        
        Args:
            text: Text to log
            name: Text name
            step: Optional step number
        """
        ...

    def log_table(
        self,
        data: dict[str, list[Any]] | Any,
        name: str,
        step: Optional[int] = None
    ) -> None:
        """Log tabular data.
        
        Args:
            data: Table data (dict or pandas DataFrame)
            name: Table name
            step: Optional step number
        """
        ...

    def get_experiment_id(self) -> Optional[str]:
        """Get current experiment ID."""
        ...

    def set_experiment(self, name: str) -> str:
        """Set or create an experiment.
        
        Args:
            name: Experiment name
            
        Returns:
            Experiment ID
        """
        ...