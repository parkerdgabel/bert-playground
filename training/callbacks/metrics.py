"""
Metrics logging callback for tracking and aggregating training metrics.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from datetime import datetime

from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd

from .base import Callback
from ..core.protocols import Trainer, TrainingState, TrainingResult


class MetricsLogger(Callback):
    """
    Callback for comprehensive metrics logging and visualization.
    
    This callback:
    - Collects all metrics during training
    - Saves metrics to CSV/JSON files
    - Creates visualizations
    - Computes aggregate statistics
    
    Args:
        log_dir: Directory to save metrics
        save_format: Format to save metrics ('csv', 'json', or 'both')
        plot_metrics: Whether to create metric plots
        plot_freq: How often to update plots ('epoch', 'end', or int for steps)
        smooth_factor: Smoothing factor for plots (0-1)
        aggregate_metrics: Whether to compute aggregate statistics
    """
    
    def __init__(
        self,
        log_dir: Optional[Path] = None,
        save_format: str = "both",
        plot_metrics: bool = True,
        plot_freq: str | int = "epoch",
        smooth_factor: float = 0.0,
        aggregate_metrics: bool = True,
    ):
        super().__init__()
        self.log_dir = log_dir
        self.save_format = save_format
        self.plot_metrics = plot_metrics
        self.plot_freq = plot_freq
        self.smooth_factor = smooth_factor
        self.aggregate_metrics = aggregate_metrics
        
        # Metrics storage
        self.metrics_history: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        self.epoch_metrics: Dict[str, List[float]] = defaultdict(list)
        self.batch_metrics: List[Dict[str, Any]] = []
        
        # Timing
        self.epoch_times: List[float] = []
        self.batch_times: List[float] = []
        self.last_batch_time = None
    
    @property
    def priority(self) -> int:
        """Metrics logging should happen after all metrics are computed."""
        return 90
    
    def on_train_begin(self, trainer: Trainer, state: TrainingState) -> None:
        """Initialize metrics logging."""
        if self.log_dir is None:
            self.log_dir = trainer.config.environment.output_dir / "metrics"
        
        self.log_dir = Path(self.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Log training start
        self.start_time = datetime.now()
        logger.info(f"MetricsLogger: saving metrics to {self.log_dir}")
    
    def on_epoch_begin(self, trainer: Trainer, state: TrainingState) -> None:
        """Record epoch start time."""
        self.epoch_start_time = datetime.now()
    
    def on_batch_begin(self, trainer: Trainer, state: TrainingState, batch: Dict[str, Any]) -> None:
        """Record batch start time."""
        self.last_batch_time = datetime.now()
    
    def on_batch_end(self, trainer: Trainer, state: TrainingState, loss: float) -> None:
        """Log batch metrics."""
        # Calculate batch time
        if self.last_batch_time:
            batch_time = (datetime.now() - self.last_batch_time).total_seconds()
            self.batch_times.append(batch_time)
        
        # Record batch metrics
        batch_metrics = {
            "step": state.global_step,
            "epoch": state.epoch,
            "loss": loss,
            "learning_rate": trainer.optimizer.learning_rate,
            "batch_time": batch_time if self.last_batch_time else 0,
        }
        
        # Add any additional metrics from state
        if hasattr(state, 'metrics'):
            for k, v in state.metrics.items():
                if k.startswith('train_') and isinstance(v, (int, float)):
                    batch_metrics[k] = v
        
        self.batch_metrics.append(batch_metrics)
        
        # Update history
        for key, value in batch_metrics.items():
            if isinstance(value, (int, float)):
                self.metrics_history[f"batch_{key}"].append((state.global_step, value))
        
        # Check if should update plots
        if self.plot_metrics and isinstance(self.plot_freq, int):
            if state.global_step % self.plot_freq == 0:
                self._update_plots()
    
    def on_epoch_end(self, trainer: Trainer, state: TrainingState) -> None:
        """Log epoch metrics."""
        # Calculate epoch time
        epoch_time = (datetime.now() - self.epoch_start_time).total_seconds()
        self.epoch_times.append(epoch_time)
        
        # Collect epoch metrics
        epoch_metrics = {
            "epoch": state.epoch,
            "train_loss": state.train_loss,
            "epoch_time": epoch_time,
        }
        
        # Add validation metrics
        if state.val_loss > 0:
            epoch_metrics["val_loss"] = state.val_loss
        
        # Add all metrics from state
        epoch_metrics.update(state.metrics)
        
        # Store metrics
        for key, value in epoch_metrics.items():
            if isinstance(value, (int, float)):
                self.epoch_metrics[key].append(value)
                self.metrics_history[f"epoch_{key}"].append((state.epoch, value))
        
        # Save metrics
        self._save_metrics()
        
        # Update plots if needed
        if self.plot_metrics and self.plot_freq == "epoch":
            self._update_plots()
    
    def on_evaluate_end(self, trainer: Trainer, state: TrainingState, metrics: Dict[str, float]) -> None:
        """Log evaluation metrics."""
        # Record eval metrics with step
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.metrics_history[key].append((state.global_step, value))
    
    def on_train_end(self, trainer: Trainer, state: TrainingState, result: TrainingResult) -> None:
        """Save final metrics and create summary."""
        # Calculate total training time
        total_time = (datetime.now() - self.start_time).total_seconds()
        
        # Create summary
        summary = {
            "total_time_seconds": total_time,
            "total_epochs": result.total_epochs,
            "total_steps": result.total_steps,
            "final_train_loss": result.final_train_loss,
            "final_val_loss": result.final_val_loss,
            "best_val_loss": result.best_val_loss,
            "best_val_metric": result.best_val_metric,
            "early_stopped": result.early_stopped,
            "average_epoch_time": sum(self.epoch_times) / len(self.epoch_times) if self.epoch_times else 0,
            "average_batch_time": sum(self.batch_times) / len(self.batch_times) if self.batch_times else 0,
        }
        
        # Add aggregate metrics
        if self.aggregate_metrics:
            aggregates = self._compute_aggregates()
            summary["aggregates"] = aggregates
        
        # Save summary
        summary_path = self.log_dir / "training_summary.json"
        # Convert to JSON-serializable format
        serializable_summary = self._make_json_serializable(summary)
        with open(summary_path, "w") as f:
            json.dump(serializable_summary, f, indent=2)
        
        # Final save
        self._save_metrics()
        
        # Final plots
        if self.plot_metrics:
            self._update_plots(final=True)
        
        logger.info(f"MetricsLogger: saved all metrics to {self.log_dir}")
    
    def _make_json_serializable(self, obj):
        """Convert MLX arrays and other non-serializable objects to JSON-serializable format."""
        import mlx.core as mx
        import numpy as np
        
        # Check if it's an MLX array directly
        if isinstance(obj, mx.array):
            try:
                return obj.item() if obj.size == 1 else obj.tolist()
            except:
                return float(obj)
        # Check if it's a numpy array
        elif isinstance(obj, np.ndarray):
            try:
                return obj.item() if obj.size == 1 else obj.tolist()
            except:
                return float(obj)
        # Check if it's an MLX array by module
        elif hasattr(obj, '__module__') and obj.__module__ and 'mlx' in obj.__module__:
            if hasattr(obj, 'item') and hasattr(obj, 'size'):
                try:
                    return obj.item() if obj.size == 1 else obj.tolist()
                except:
                    return float(obj)
            else:
                # For other MLX objects, convert to string
                return str(obj)
        elif hasattr(obj, 'item') and hasattr(obj, 'size'):
            # Handle numpy arrays or similar
            try:
                return obj.item() if obj.size == 1 else obj.tolist()
            except:
                return float(obj)
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, tuple):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            # Handle numpy scalar types
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            # For any other type, try to convert to string if not already serializable
            try:
                json.dumps(obj)
                return obj
            except:
                return str(obj)

    def _save_metrics(self) -> None:
        """Save metrics to files."""
        # Save as CSV
        if self.save_format in ["csv", "both"]:
            # Epoch metrics
            if self.epoch_metrics:
                df_epochs = pd.DataFrame(self.epoch_metrics)
                df_epochs.to_csv(self.log_dir / "epoch_metrics.csv", index=False)
            
            # Batch metrics
            if self.batch_metrics:
                df_batches = pd.DataFrame(self.batch_metrics)
                df_batches.to_csv(self.log_dir / "batch_metrics.csv", index=False)
        
        # Save as JSON
        if self.save_format in ["json", "both"]:
            # All metrics history
            metrics_data = {
                "epoch_metrics": self.epoch_metrics,
                "batch_metrics": self.batch_metrics,
                "metrics_history": {k: list(v) for k, v in self.metrics_history.items()},
            }
            
            # Convert to JSON-serializable format
            serializable_data = self._make_json_serializable(metrics_data)
            
            with open(self.log_dir / "metrics.json", "w") as f:
                json.dump(serializable_data, f, indent=2)
    
    def _update_plots(self, final: bool = False) -> None:
        """Create or update metric plots."""
        if not self.metrics_history:
            return
        
        # Group metrics by type
        loss_metrics = [k for k in self.metrics_history if 'loss' in k]
        accuracy_metrics = [k for k in self.metrics_history if any(x in k for x in ['accuracy', 'acc', 'auc', 'f1'])]
        other_metrics = [k for k in self.metrics_history if k not in loss_metrics + accuracy_metrics]
        
        # Create figure with subplots
        num_plots = sum([len(loss_metrics) > 0, len(accuracy_metrics) > 0, len(other_metrics) > 0])
        if num_plots == 0:
            return
        
        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots))
        if num_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Plot losses
        if loss_metrics:
            ax = axes[plot_idx]
            for metric_name in loss_metrics:
                steps, values = zip(*self.metrics_history[metric_name])
                
                # Apply smoothing if needed
                if self.smooth_factor > 0:
                    values = self._smooth_values(values, self.smooth_factor)
                
                ax.plot(steps, values, label=metric_name)
            
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.set_title("Loss Metrics")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        
        # Plot accuracy metrics
        if accuracy_metrics:
            ax = axes[plot_idx]
            for metric_name in accuracy_metrics:
                steps, values = zip(*self.metrics_history[metric_name])
                
                # Apply smoothing if needed
                if self.smooth_factor > 0:
                    values = self._smooth_values(values, self.smooth_factor)
                
                ax.plot(steps, values, label=metric_name)
            
            ax.set_xlabel("Step")
            ax.set_ylabel("Score")
            ax.set_title("Performance Metrics")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        
        # Plot other metrics
        if other_metrics and plot_idx < len(axes):
            ax = axes[plot_idx]
            for metric_name in other_metrics[:5]:  # Limit to 5 metrics
                steps, values = zip(*self.metrics_history[metric_name])
                
                # Apply smoothing if needed
                if self.smooth_factor > 0:
                    values = self._smooth_values(values, self.smooth_factor)
                
                ax.plot(steps, values, label=metric_name)
            
            ax.set_xlabel("Step")
            ax.set_ylabel("Value")
            ax.set_title("Other Metrics")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_name = "metrics_final.png" if final else "metrics_latest.png"
        plt.savefig(self.log_dir / plot_name, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _smooth_values(self, values: List[float], factor: float) -> List[float]:
        """Apply exponential moving average smoothing."""
        smoothed = []
        last = values[0]
        
        for value in values:
            smoothed_val = last * factor + (1 - factor) * value
            smoothed.append(smoothed_val)
            last = smoothed_val
        
        return smoothed
    
    def _compute_aggregates(self) -> Dict[str, Any]:
        """Compute aggregate statistics for metrics."""
        aggregates = {}
        
        for metric_name, history in self.metrics_history.items():
            if not history:
                continue
            
            _, values = zip(*history)
            
            aggregates[metric_name] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "final": values[-1],
                "count": len(values),
            }
        
        return aggregates