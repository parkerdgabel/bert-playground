"""
Regression metrics for training evaluation.
"""

import mlx.core as mx
import numpy as np

from .base import AveragedMetric, Metric


class MSE(AveragedMetric):
    """Mean Squared Error metric."""

    def __init__(self, name: str = "mse"):
        super().__init__(name)

    def update(self, predictions: mx.array, targets: mx.array) -> None:
        """Update metric with batch results."""
        # Ensure same shape
        predictions = predictions.squeeze()
        targets = targets.squeeze()

        # Calculate MSE
        squared_errors = (predictions - targets) ** 2
        batch_mse = mx.mean(squared_errors).item()
        batch_size = targets.shape[0]

        self.total += batch_mse * batch_size
        self.count += batch_size


class RMSE(MSE):
    """Root Mean Squared Error metric."""

    def __init__(self, name: str = "rmse"):
        super().__init__(name)

    def compute(self) -> float:
        """Compute RMSE."""
        mse = super().compute()
        return np.sqrt(mse)


class MAE(AveragedMetric):
    """Mean Absolute Error metric."""

    def __init__(self, name: str = "mae"):
        super().__init__(name)

    def update(self, predictions: mx.array, targets: mx.array) -> None:
        """Update metric with batch results."""
        # Ensure same shape
        predictions = predictions.squeeze()
        targets = targets.squeeze()

        # Calculate MAE
        absolute_errors = mx.abs(predictions - targets)
        batch_mae = mx.mean(absolute_errors).item()
        batch_size = targets.shape[0]

        self.total += batch_mae * batch_size
        self.count += batch_size


class R2Score(Metric):
    """R-squared (coefficient of determination) metric."""

    def __init__(self, name: str = "r2"):
        super().__init__(name)
        self.sum_squared_residuals = 0.0
        self.sum_squared_total = 0.0
        self.target_sum = 0.0
        self.target_count = 0
        self.predictions = []
        self.targets = []

    def reset(self) -> None:
        """Reset metric state."""
        self.sum_squared_residuals = 0.0
        self.sum_squared_total = 0.0
        self.target_sum = 0.0
        self.target_count = 0
        self.predictions = []
        self.targets = []

    def update(self, predictions: mx.array, targets: mx.array) -> None:
        """Update metric with batch results."""
        # Ensure same shape
        predictions = predictions.squeeze()
        targets = targets.squeeze()

        # Store for final computation
        self.predictions.extend(predictions.tolist())
        self.targets.extend(targets.tolist())

        # Update running statistics
        self.target_sum += mx.sum(targets).item()
        self.target_count += targets.shape[0]

    def compute(self) -> float:
        """Compute R² score."""
        if self.target_count == 0:
            return 0.0

        # Convert to numpy for easier computation
        y_true = np.array(self.targets)
        y_pred = np.array(self.predictions)

        # Calculate mean
        y_mean = self.target_sum / self.target_count

        # Calculate sum of squared residuals
        ss_res = np.sum((y_true - y_pred) ** 2)

        # Calculate total sum of squares
        ss_tot = np.sum((y_true - y_mean) ** 2)

        # Calculate R²
        if ss_tot == 0:
            return 0.0

        r2 = 1 - (ss_res / ss_tot)
        return r2
