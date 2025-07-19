"""
Base classes for metrics computation.
"""

from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod
import mlx.core as mx
import numpy as np
from collections import defaultdict
from pathlib import Path
import json


class Metric(ABC):
    """
    Base class for metrics.
    
    Metrics accumulate values over batches and compute final results.
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize metric.
        
        Args:
            name: Optional metric name
        """
        self.name = name or self.__class__.__name__.lower()
        self.reset()
    
    @abstractmethod
    def update(self, predictions: mx.array, targets: mx.array) -> None:
        """
        Update metric with batch results.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
        """
        pass
    
    @abstractmethod
    def compute(self) -> float:
        """
        Compute final metric value.
        
        Returns:
            Metric value
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset metric state."""
        pass
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.name}: {self.compute():.4f}"


class MetricsCollector:
    """
    Collects and manages multiple metrics.
    """
    
    def __init__(self, metrics: Optional[List[Metric]] = None):
        """
        Initialize metrics collector.
        
        Args:
            metrics: List of metrics to track
        """
        self.metrics = {}
        if metrics:
            for metric in metrics:
                self.add_metric(metric)
        
        # History tracking
        self.history: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        self.step_count = 0
    
    def add_metric(self, metric: Metric) -> None:
        """Add a metric to track."""
        self.metrics[metric.name] = metric
    
    def update(self, predictions: mx.array, targets: mx.array) -> None:
        """Update all metrics with batch results."""
        for metric in self.metrics.values():
            metric.update(predictions, targets)
    
    def compute(self) -> Dict[str, float]:
        """Compute all metric values."""
        return {name: metric.compute() for name, metric in self.metrics.items()}
    
    def reset(self) -> None:
        """Reset all metrics."""
        for metric in self.metrics.values():
            metric.reset()
    
    def log_step(self, step: Optional[int] = None) -> Dict[str, float]:
        """Log current metric values and add to history."""
        if step is None:
            step = self.step_count
            self.step_count += 1
        
        values = self.compute()
        for name, value in values.items():
            self.history[name].append((step, value))
        
        return values
    
    def get_history(self, metric_name: str) -> List[Tuple[int, float]]:
        """Get history for a specific metric."""
        return self.history.get(metric_name, [])
    
    def save(self, path: Path) -> None:
        """Save metrics history to file."""
        data = {
            "metrics": list(self.metrics.keys()),
            "history": {k: list(v) for k, v in self.history.items()},
            "step_count": self.step_count,
        }
        
        path = Path(path)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: Path) -> None:
        """Load metrics history from file."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        
        self.history = defaultdict(list, {k: v for k, v in data["history"].items()})
        self.step_count = data["step_count"]


class AveragedMetric(Metric):
    """Base class for metrics that average over batches."""
    
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.total = 0.0
        self.count = 0
    
    def reset(self) -> None:
        """Reset metric state."""
        self.total = 0.0
        self.count = 0
    
    def compute(self) -> float:
        """Compute average value."""
        if self.count == 0:
            return 0.0
        return self.total / self.count