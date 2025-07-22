"""Metrics tracker component for collecting and reporting training metrics.

This component is responsible for:
- Collecting training and validation metrics
- Computing moving averages
- Tracking best metrics
- Reporting metrics to various sinks
"""

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from loguru import logger

from core.protocols.training import MetricsCollector


class MetricsTracker(MetricsCollector):
    """Handles metrics collection and reporting.
    
    This component manages:
    - Metric accumulation over time
    - Moving average computation
    - Best metric tracking
    - Metric persistence
    """
    
    def __init__(
        self,
        window_size: int = 100,
        output_dir: Path | None = None,
    ):
        """Initialize the metrics tracker.
        
        Args:
            window_size: Size of moving average window
            output_dir: Optional directory to save metrics
        """
        self.window_size = window_size
        self.output_dir = output_dir
        
        # Metric storage
        self._metrics: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        self._moving_averages: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self._best_metrics: Dict[str, Tuple[float, int]] = {}  # metric -> (value, step)
        
        # Metric configuration
        self._metric_modes: Dict[str, str] = {}  # metric -> "min" or "max"
        
        logger.debug("Initialized MetricsTracker")
        
    def add_metric(self, name: str, value: float, step: int | None = None) -> None:
        """Add a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            step: Optional step number
        """
        if step is None:
            step = len(self._metrics[name])
            
        self._metrics[name].append((step, value))
        self._moving_averages[name].append(value)
        
        # Update best metric if configured
        if name in self._metric_modes:
            self._update_best_metric(name, value, step)
            
    def add_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Add multiple metrics.
        
        Args:
            metrics: Dictionary of metrics
            step: Optional step number
        """
        for name, value in metrics.items():
            self.add_metric(name, value, step)
            
    def get_metric(self, name: str) -> list[tuple[int, float]]:
        """Get metric history.
        
        Args:
            name: Metric name
            
        Returns:
            List of (step, value) tuples
        """
        return self._metrics.get(name, [])
        
    def get_latest_metrics(self) -> dict[str, float]:
        """Get latest metric values.
        
        Returns:
            Dictionary of latest metrics
        """
        latest = {}
        for name, history in self._metrics.items():
            if history:
                latest[name] = history[-1][1]
        return latest
        
    def get_moving_average(self, name: str) -> float | None:
        """Get moving average for a metric.
        
        Args:
            name: Metric name
            
        Returns:
            Moving average or None
        """
        if name in self._moving_averages and self._moving_averages[name]:
            return sum(self._moving_averages[name]) / len(self._moving_averages[name])
        return None
        
    def get_best_metric(self, name: str) -> tuple[float, int] | None:
        """Get best value for a metric.
        
        Args:
            name: Metric name
            
        Returns:
            Tuple of (best_value, step) or None
        """
        return self._best_metrics.get(name)
        
    def configure_metric(self, name: str, mode: str = "min") -> None:
        """Configure how to track best values for a metric.
        
        Args:
            name: Metric name
            mode: "min" or "max" for best value tracking
        """
        if mode not in ["min", "max"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'min' or 'max'")
        self._metric_modes[name] = mode
        
    def is_best_metric(self, name: str, value: float) -> bool:
        """Check if a value is the best for a metric.
        
        Args:
            name: Metric name
            value: Value to check
            
        Returns:
            True if this is the best value
        """
        if name not in self._metric_modes:
            return False
            
        if name not in self._best_metrics:
            return True
            
        best_value, _ = self._best_metrics[name]
        mode = self._metric_modes[name]
        
        if mode == "min":
            return value < best_value
        else:
            return value > best_value
            
    def get_summary(self) -> dict[str, Any]:
        """Get a summary of all metrics.
        
        Returns:
            Dictionary with metric summaries
        """
        summary = {
            "latest": self.get_latest_metrics(),
            "moving_averages": {},
            "best": {},
        }
        
        # Add moving averages
        for name in self._metrics:
            avg = self.get_moving_average(name)
            if avg is not None:
                summary["moving_averages"][name] = avg
                
        # Add best metrics
        for name, (value, step) in self._best_metrics.items():
            summary["best"][name] = {"value": value, "step": step}
            
        return summary
        
    def clear(self) -> None:
        """Clear all metrics."""
        self._metrics.clear()
        self._moving_averages.clear()
        self._best_metrics.clear()
        logger.debug("Cleared all metrics")
        
    def save(self, path: Path) -> None:
        """Save metrics to file.
        
        Args:
            path: Path to save metrics
        """
        data = {
            "metrics": dict(self._metrics),
            "best_metrics": dict(self._best_metrics),
            "metric_modes": dict(self._metric_modes),
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
            
        logger.debug(f"Saved metrics to {path}")
        
    def load(self, path: Path) -> None:
        """Load metrics from file.
        
        Args:
            path: Path to load metrics from
        """
        with open(path, "r") as f:
            data = json.load(f)
            
        self._metrics = defaultdict(list, data.get("metrics", {}))
        self._best_metrics = data.get("best_metrics", {})
        self._metric_modes = data.get("metric_modes", {})
        
        # Rebuild moving averages
        self._moving_averages.clear()
        for name, history in self._metrics.items():
            for step, value in history[-self.window_size:]:
                self._moving_averages[name].append(value)
                
        logger.debug(f"Loaded metrics from {path}")
        
    def log_to_console(self, metrics: dict[str, float], prefix: str = "") -> None:
        """Log metrics to console.
        
        Args:
            metrics: Metrics to log
            prefix: Optional prefix for log message
        """
        # Format metrics for logging
        metric_strs = []
        for name, value in sorted(metrics.items()):
            if isinstance(value, float):
                metric_strs.append(f"{name}: {value:.4f}")
            else:
                metric_strs.append(f"{name}: {value}")
                
        message = " | ".join(metric_strs)
        if prefix:
            message = f"{prefix} - {message}"
            
        logger.info(message)
        
    def _update_best_metric(self, name: str, value: float, step: int) -> None:
        """Update best metric tracking.
        
        Args:
            name: Metric name
            value: Metric value
            step: Step number
        """
        if name not in self._best_metrics:
            self._best_metrics[name] = (value, step)
            return
            
        best_value, _ = self._best_metrics[name]
        mode = self._metric_modes[name]
        
        should_update = (mode == "min" and value < best_value) or (mode == "max" and value > best_value)
        
        if should_update:
            self._best_metrics[name] = (value, step)
            logger.debug(f"New best {name}: {value:.4f} at step {step}")
            
    def save_epoch_metrics(self, epoch: int, train_metrics: dict[str, float], val_metrics: dict[str, float] | None = None) -> None:
        """Save metrics for an epoch.
        
        Args:
            epoch: Epoch number
            train_metrics: Training metrics
            val_metrics: Optional validation metrics
        """
        if self.output_dir:
            # Save to JSONL file for easy streaming
            metrics_file = self.output_dir / "metrics.jsonl"
            
            record = {
                "epoch": epoch,
                "train": train_metrics,
            }
            
            if val_metrics:
                record["validation"] = val_metrics
                
            with open(metrics_file, "a") as f:
                f.write(json.dumps(record) + "\n")