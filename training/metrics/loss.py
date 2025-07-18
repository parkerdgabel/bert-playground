"""
Loss metrics for training evaluation.
"""

import mlx.core as mx
from typing import Optional, List

from .base import AveragedMetric


class Loss(AveragedMetric):
    """Generic loss metric."""
    
    def __init__(self, name: str = "loss"):
        super().__init__(name)
    
    def update(self, loss_value: float, batch_size: Optional[int] = None) -> None:
        """
        Update metric with loss value.
        
        Note: This metric is different as it takes the loss value directly
        rather than predictions and targets.
        
        Args:
            loss_value: Loss value for the batch
            batch_size: Batch size (default 1)
        """
        if batch_size is None:
            batch_size = 1
        
        self.total += loss_value * batch_size
        self.count += batch_size


class MeanLoss(Loss):
    """Mean loss metric - alias for Loss."""
    pass


class SmoothLoss(Loss):
    """
    Exponentially smoothed loss metric.
    
    Useful for getting a smoother view of the loss trajectory.
    """
    
    def __init__(self, name: str = "smooth_loss", smoothing: float = 0.98):
        """
        Initialize smoothed loss metric.
        
        Args:
            name: Metric name
            smoothing: Smoothing factor (0-1, higher = more smoothing)
        """
        super().__init__(name)
        self.smoothing = smoothing
        self.smoothed_value = None
    
    def update(self, loss_value: float, batch_size: Optional[int] = None) -> None:
        """Update metric with loss value."""
        super().update(loss_value, batch_size)
        
        # Update smoothed value
        if self.smoothed_value is None:
            self.smoothed_value = loss_value
        else:
            self.smoothed_value = (
                self.smoothing * self.smoothed_value +
                (1 - self.smoothing) * loss_value
            )
    
    def compute(self) -> float:
        """Compute smoothed loss value."""
        if self.smoothed_value is None:
            return super().compute()
        return self.smoothed_value
    
    def reset(self) -> None:
        """Reset metric state."""
        super().reset()
        self.smoothed_value = None


class LossCollection:
    """Collection of loss metrics."""
    
    def __init__(
        self,
        metrics: Optional[List] = None,
        prefix: str = "",
    ):
        self.prefix = prefix
        
        # Default metrics
        if metrics is None:
            self.metrics = {
                "loss": Loss(),
                "smooth_loss": SmoothLoss(),
            }
        else:
            self.metrics = {m.name: m for m in metrics}
    
    def update(self, loss_value: float, batch_size: Optional[int] = None) -> None:
        """Update all loss metrics."""
        for metric in self.metrics.values():
            metric.update(loss_value, batch_size)
    
    def compute(self) -> dict:
        """Compute all metrics."""
        results = {}
        for name, metric in self.metrics.items():
            results[f"{self.prefix}{name}"] = metric.compute()
        return results
    
    def reset(self) -> None:
        """Reset all metrics."""
        for metric in self.metrics.values():
            metric.reset()