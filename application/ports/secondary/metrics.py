"""Secondary metrics port - Metrics computation services that the application depends on.

This port defines the metrics interface that the application core uses
for computing evaluation metrics. It's a driven port implemented by
adapters for different metric libraries.
"""

from enum import Enum
from typing import Any, Protocol, runtime_checkable, Optional

from infrastructure.di import port
from .compute import Array


class MetricType(Enum):
    """Types of metrics."""
    
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    ROC_AUC = "roc_auc"
    PR_AUC = "pr_auc"
    MSE = "mse"
    MAE = "mae"
    R2 = "r2"
    BLEU = "bleu"
    ROUGE = "rouge"
    PERPLEXITY = "perplexity"
    CUSTOM = "custom"


@port()
@runtime_checkable
class Metric(Protocol):
    """Secondary port for individual metric computation.
    
    This interface is implemented by adapters for specific metrics.
    The application core depends on this for evaluation.
    """

    @property
    def name(self) -> str:
        """Name of the metric."""
        ...

    @property
    def metric_type(self) -> MetricType:
        """Type of the metric."""
        ...

    @property
    def requires_probabilities(self) -> bool:
        """Whether metric requires probability outputs."""
        ...

    def update(self, predictions: Array, targets: Array, **kwargs: Any) -> None:
        """Update metric with batch results.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            **kwargs: Additional metric-specific arguments
        """
        ...

    def compute(self) -> float | dict[str, float]:
        """Compute final metric value.
        
        Returns:
            Metric value or dict of values for multi-value metrics
        """
        ...

    def reset(self) -> None:
        """Reset metric state."""
        ...

    def merge(self, other: "Metric") -> None:
        """Merge state from another metric instance.
        
        Args:
            other: Other metric instance to merge from
        """
        ...

    def state_dict(self) -> dict[str, Any]:
        """Get metric state for serialization.
        
        Returns:
            State dictionary
        """
        ...

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load metric state.
        
        Args:
            state: State dictionary to load
        """
        ...


@port()
@runtime_checkable
class MetricsCollector(Protocol):
    """Secondary port for collecting and managing multiple metrics.
    
    This interface is implemented by adapters that manage collections
    of metrics. The application core depends on this for tracking metrics.
    """

    def add_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Add a single metric value.
        
        Args:
            name: Metric name
            value: Metric value
            step: Optional step/epoch number
        """
        ...

    def add_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        """Add multiple metric values.
        
        Args:
            metrics: Dictionary of metric name to value
            step: Optional step/epoch number
        """
        ...

    def get_metric(self, name: str) -> list[tuple[int, float]]:
        """Get metric history.
        
        Args:
            name: Metric name
            
        Returns:
            List of (step, value) tuples
        """
        ...

    def get_latest_metrics(self) -> dict[str, float]:
        """Get latest value for all metrics.
        
        Returns:
            Dictionary of metric name to latest value
        """
        ...

    def get_metrics_at_step(self, step: int) -> dict[str, float]:
        """Get all metrics at a specific step.
        
        Args:
            step: Step number
            
        Returns:
            Dictionary of metric values at that step
        """
        ...

    def get_best_metric(
        self,
        name: str,
        mode: str = "max"
    ) -> tuple[int, float] | None:
        """Get best metric value and its step.
        
        Args:
            name: Metric name
            mode: 'max' or 'min'
            
        Returns:
            Tuple of (step, value) or None if metric not found
        """
        ...

    def clear(self) -> None:
        """Clear all metrics."""
        ...

    def clear_metric(self, name: str) -> None:
        """Clear a specific metric.
        
        Args:
            name: Metric name to clear
        """
        ...

    def save(self, path: str) -> None:
        """Save metrics to file.
        
        Args:
            path: Save path
        """
        ...

    def load(self, path: str) -> None:
        """Load metrics from file.
        
        Args:
            path: Load path
        """
        ...

    def to_dataframe(self) -> Any:
        """Convert metrics to pandas DataFrame.
        
        Returns:
            DataFrame with metrics (if pandas available)
        """
        ...

    def plot_metric(
        self,
        name: str,
        show: bool = True,
        save_path: Optional[str] = None
    ) -> None:
        """Plot a metric over time.
        
        Args:
            name: Metric name
            show: Whether to display plot
            save_path: Optional path to save plot
        """
        ...

    def summary(self, markdown: bool = False) -> str:
        """Get summary of all metrics.
        
        Args:
            markdown: Whether to format as markdown
            
        Returns:
            Summary string
        """
        ...