"""Regression heads for BERT models in Kaggle competitions.

This module implements regression heads for various types of continuous
target prediction tasks with competition-specific optimizations.
"""

from typing import Dict, Optional, List, Any
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .base_head import BaseKaggleHead, HeadType, HeadConfig, ActivationType
from .head_registry import register_head_class, CompetitionType
from loguru import logger


@register_head_class(
    name="regression",
    head_type=HeadType.REGRESSION,
    competition_types=[CompetitionType.REGRESSION],
    priority=10,
    description="Standard regression head with MSE loss and optional uncertainty estimation"
)
class RegressionHead(BaseKaggleHead):
    """Standard regression head for continuous target prediction.
    
    Features:
    - Linear output layer without activation
    - MSE, MAE, or Huber loss options
    - Optional uncertainty estimation
    - Competition-specific optimizations (RMSE, MAE, etc.)
    """
    
    def __init__(self, config: HeadConfig):
        """Initialize regression head.
        
        Args:
            config: Head configuration
        """
        # Ensure output size is 1 for standard regression
        if config.output_size != 1:
            config.output_size = 1
        
        # Competition-specific parameters (must be set before super().__init__())
        self.loss_type = "mse"  # Options: "mse", "mae", "huber"
        self.huber_delta = 1.0
        self.use_uncertainty = config.use_uncertainty
        
        # Target normalization parameters (learned during training)
        self.target_mean = 0.0
        self.target_std = 1.0
            
        super().__init__(config)
        
    def _build_output_layer(self):
        """Build the output layer for regression."""
        self.regressor = nn.Linear(self.projection_output_size, self.config.output_size)
        
        # Optional uncertainty head
        if self.use_uncertainty:
            self.uncertainty_head = nn.Linear(self.projection_output_size, 1)
        
        # Optional output scaling/normalization
        if self.config.use_competition_tricks:
            # Learnable output scaling
            self.output_scale = mx.ones(1)  # Will be learnable parameter
            self.output_bias = mx.zeros(1)  # Will be learnable parameter
        else:
            self.output_scale = None
            self.output_bias = None
    
    def _build_loss_function(self):
        """Build loss function for regression."""
        if self.loss_type == "mse":
            self.loss_fn = self._mse_loss
        elif self.loss_type == "mae":
            self.loss_fn = self._mae_loss
        elif self.loss_type == "huber":
            self.loss_fn = HuberLoss(delta=self.huber_delta)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
    
    def forward(self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None, **kwargs) -> Dict[str, mx.array]:
        """Forward pass through regression head.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask [batch_size, seq_len]
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with predictions and optional uncertainty
        """
        # Apply pooling
        pooled_output = self._apply_pooling(hidden_states, attention_mask)
        
        # Apply projection layers
        projected = self.projection(pooled_output)
        
        # Get predictions
        predictions = self.regressor(projected)  # [batch_size, 1]
        
        # Apply output scaling if enabled
        if self.output_scale is not None:
            predictions = predictions * self.output_scale + self.output_bias
        
        result = {
            "predictions": predictions,
        }
        
        # Add uncertainty if enabled
        if self.use_uncertainty:
            log_variance = self.uncertainty_head(projected)
            variance = mx.exp(log_variance)
            uncertainty = mx.sqrt(variance)
            
            result.update({
                "uncertainty": uncertainty,
                "log_variance": log_variance,
                "variance": variance,
            })
        
        return result
    
    def compute_loss(self, predictions: Dict[str, mx.array], targets: mx.array, **kwargs) -> mx.array:
        """Compute loss for regression.
        
        Args:
            predictions: Predictions from forward pass
            targets: Ground truth targets [batch_size, 1]
            **kwargs: Additional arguments
            
        Returns:
            Loss value
        """
        pred_values = predictions["predictions"]
        targets = targets.astype(mx.float32)
        
        # Standard regression loss
        regression_loss = self.loss_fn(pred_values, targets)
        
        # Add uncertainty loss if enabled
        if self.use_uncertainty and "log_variance" in predictions:
            log_variance = predictions["log_variance"]
            variance = mx.exp(log_variance)
            
            # Heteroscedastic uncertainty loss
            uncertainty_loss = 0.5 * (mx.square(pred_values - targets) / variance + log_variance)
            uncertainty_loss = uncertainty_loss.mean()
            
            # Combine losses
            total_loss = regression_loss + uncertainty_loss
            
            return total_loss
        
        return regression_loss
    
    def compute_metrics(self, predictions: Dict[str, mx.array], targets: mx.array, **kwargs) -> Dict[str, float]:
        """Compute evaluation metrics for regression.
        
        Args:
            predictions: Predictions from forward pass
            targets: Ground truth targets
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of metric values
        """
        pred_values = predictions["predictions"]
        
        # Convert to numpy for metric computation
        pred_np = np.array(pred_values).flatten()
        targets_np = np.array(targets).flatten()
        
        # Basic metrics
        mse = np.mean((pred_np - targets_np) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(pred_np - targets_np))
        
        # R-squared
        ss_res = np.sum((targets_np - pred_np) ** 2)
        ss_tot = np.sum((targets_np - np.mean(targets_np)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        metrics = {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
        }
        
        # Add uncertainty metrics if available
        if self.use_uncertainty and "uncertainty" in predictions:
            uncertainty_np = np.array(predictions["uncertainty"]).flatten()
            avg_uncertainty = np.mean(uncertainty_np)
            
            # Calibration metric (simplified)
            # Ideally, we'd use proper calibration plots
            calibration_score = self._compute_calibration_score(pred_np, targets_np, uncertainty_np)
            
            metrics.update({
                "avg_uncertainty": float(avg_uncertainty),
                "calibration_score": float(calibration_score),
            })
        
        return metrics
    
    def _compute_calibration_score(self, predictions: np.ndarray, targets: np.ndarray, uncertainties: np.ndarray) -> float:
        """Compute calibration score for uncertainty estimates."""
        # Simple calibration check: does uncertainty correlate with error?
        errors = np.abs(predictions - targets)
        
        # Spearman correlation between uncertainty and error
        from scipy.stats import spearmanr
        try:
            correlation, _ = spearmanr(uncertainties, errors)
            return correlation if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def set_target_normalization(self, mean: float, std: float):
        """Set target normalization parameters.
        
        Args:
            mean: Target mean for normalization
            std: Target standard deviation for normalization
        """
        self.target_mean = mean
        self.target_std = std
        logger.info(f"Set target normalization: mean={mean:.4f}, std={std:.4f}")
    
    def _mse_loss(self, predictions: mx.array, targets: mx.array) -> mx.array:
        """Custom MSE loss implementation."""
        diff = predictions - targets
        return mx.mean(mx.square(diff))
    
    def _mae_loss(self, predictions: mx.array, targets: mx.array) -> mx.array:
        """Custom MAE loss implementation."""
        diff = mx.abs(predictions - targets)
        return mx.mean(diff)


@register_head_class(
    name="ordinal_regression",
    head_type=HeadType.ORDINAL_REGRESSION,
    competition_types=[CompetitionType.ORDINAL_REGRESSION],
    priority=10,
    description="Ordinal regression head for ordered categorical targets"
)
class OrdinalRegressionHead(BaseKaggleHead):
    """Ordinal regression head for ordered categorical targets.
    
    Features:
    - Ordinal loss function respecting order
    - Cumulative logits approach
    - Support for various ordinal loss functions
    - Competition-specific optimizations
    """
    
    def __init__(self, config: HeadConfig):
        """Initialize ordinal regression head.
        
        Args:
            config: Head configuration
        """
        super().__init__(config)
        
        # Number of ordinal classes
        self.num_classes = config.output_size
        
        # Competition-specific parameters
        self.ordinal_loss_type = "cumulative_logits"  # Options: "cumulative_logits", "threshold"
        
    def _build_output_layer(self):
        """Build the output layer for ordinal regression."""
        if self.ordinal_loss_type == "cumulative_logits":
            # Cumulative logits approach: predict K-1 thresholds
            self.classifier = nn.Linear(self.projection_output_size, self.num_classes - 1)
        else:
            # Threshold approach: predict thresholds directly
            self.classifier = nn.Linear(self.projection_output_size, self.num_classes - 1)
        
        # Bias terms for thresholds (learnable)
        self.threshold_bias = mx.arange(self.num_classes - 1, dtype=mx.float32)  # Will be learnable parameter
    
    def _build_loss_function(self):
        """Build loss function for ordinal regression."""
        self.ordinal_loss = OrdinalLoss(num_classes=self.num_classes)
    
    def forward(self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None, **kwargs) -> Dict[str, mx.array]:
        """Forward pass through ordinal regression head.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask [batch_size, seq_len]
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with predictions and probabilities
        """
        # Apply pooling
        pooled_output = self._apply_pooling(hidden_states, attention_mask)
        
        # Apply projection layers
        projected = self.projection(pooled_output)
        
        # Get threshold logits
        threshold_logits = self.classifier(projected)  # [batch_size, num_classes-1]
        
        # Add bias terms
        threshold_logits = threshold_logits + self.threshold_bias
        
        # Convert to cumulative probabilities
        cumulative_probs = mx.sigmoid(threshold_logits)  # [batch_size, num_classes-1]
        
        # Convert to class probabilities
        class_probs = self._cumulative_to_class_probs(cumulative_probs)
        
        # Get predictions
        predictions = mx.argmax(class_probs, axis=-1)
        
        return {
            "threshold_logits": threshold_logits,
            "cumulative_probabilities": cumulative_probs,
            "class_probabilities": class_probs,
            "predictions": predictions,
        }
    
    def _cumulative_to_class_probs(self, cumulative_probs: mx.array) -> mx.array:
        """Convert cumulative probabilities to class probabilities.
        
        Args:
            cumulative_probs: Cumulative probabilities [batch_size, num_classes-1]
            
        Returns:
            Class probabilities [batch_size, num_classes]
        """
        batch_size = cumulative_probs.shape[0]
        
        # Add boundaries (P(Y <= 0) = 0, P(Y <= K) = 1)
        zeros = mx.zeros((batch_size, 1))
        ones = mx.ones((batch_size, 1))
        
        # Padded cumulative probabilities
        padded_cumulative = mx.concatenate([zeros, cumulative_probs, ones], axis=1)
        
        # Class probabilities: P(Y = k) = P(Y <= k) - P(Y <= k-1)
        class_probs = padded_cumulative[:, 1:] - padded_cumulative[:, :-1]
        
        return class_probs
    
    def compute_loss(self, predictions: Dict[str, mx.array], targets: mx.array, **kwargs) -> mx.array:
        """Compute loss for ordinal regression.
        
        Args:
            predictions: Predictions from forward pass
            targets: Ground truth targets [batch_size] (ordinal class indices)
            **kwargs: Additional arguments
            
        Returns:
            Loss value
        """
        threshold_logits = predictions["threshold_logits"]
        targets = targets.astype(mx.int32)
        
        return self.ordinal_loss(threshold_logits, targets)
    
    def compute_metrics(self, predictions: Dict[str, mx.array], targets: mx.array, **kwargs) -> Dict[str, float]:
        """Compute evaluation metrics for ordinal regression.
        
        Args:
            predictions: Predictions from forward pass
            targets: Ground truth targets
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of metric values
        """
        class_probs = predictions["class_probabilities"]
        preds = predictions["predictions"]
        
        # Convert to numpy for metric computation
        probs_np = np.array(class_probs)
        preds_np = np.array(preds)
        targets_np = np.array(targets)
        
        # Exact accuracy
        accuracy = (preds_np == targets_np).mean()
        
        # Mean absolute error (treating as ordinal)
        mae = np.mean(np.abs(preds_np - targets_np))
        
        # Ordinal accuracy (within 1 class)
        ordinal_accuracy = np.mean(np.abs(preds_np - targets_np) <= 1)
        
        # Kendall's tau (rank correlation)
        from scipy.stats import kendalltau
        try:
            tau, _ = kendalltau(preds_np, targets_np)
            tau = tau if not np.isnan(tau) else 0.0
        except:
            tau = 0.0
        
        return {
            "accuracy": float(accuracy),
            "mae": float(mae),
            "ordinal_accuracy": float(ordinal_accuracy),
            "kendall_tau": float(tau),
        }


@register_head_class(
    name="time_series_regression",
    head_type=HeadType.TIME_SERIES,
    competition_types=[CompetitionType.TIME_SERIES],
    priority=10,
    description="Time series regression head with temporal modeling"
)
class TimeSeriesRegressionHead(BaseKaggleHead):
    """Time series regression head for temporal prediction tasks.
    
    Features:
    - Temporal modeling with LSTM/GRU layers
    - Multi-step ahead prediction
    - Seasonal decomposition
    - Competition-specific optimizations
    """
    
    def __init__(self, config: HeadConfig):
        """Initialize time series regression head.
        
        Args:
            config: Head configuration
        """
        super().__init__(config)
        
        # Time series specific parameters
        self.sequence_length = getattr(config, 'sequence_length', 1)
        self.forecast_horizon = getattr(config, 'forecast_horizon', 1)
        self.use_temporal_features = getattr(config, 'use_temporal_features', True)
        
        # Temporal modeling architecture
        self.temporal_model_type = "lstm"  # Options: "lstm", "gru", "transformer"
        self.temporal_hidden_size = config.input_size // 2
        
    def _build_output_layer(self):
        """Build the output layer for time series regression."""
        # Simplified temporal modeling using linear layers
        # (MLX doesn't have LSTM/GRU, so we use feedforward approximation)
        self.temporal_layer = nn.Sequential(
            nn.Linear(self.projection_output_size, self.temporal_hidden_size),
            nn.ReLU(),
            nn.Linear(self.temporal_hidden_size, self.temporal_hidden_size),
            nn.ReLU(),
        )
        
        # Final regression layer
        self.regressor = nn.Linear(self.temporal_hidden_size, self.forecast_horizon)
        
        # Optional seasonal decomposition
        if self.use_temporal_features:
            self.seasonal_layer = nn.Linear(self.temporal_hidden_size, self.forecast_horizon)
            self.trend_layer = nn.Linear(self.temporal_hidden_size, self.forecast_horizon)
    
    def _build_loss_function(self):
        """Build loss function for time series regression."""
        # MSE loss with optional temporal regularization
        self.mse_loss = self._mse_loss
        
        # Temporal smoothness regularization
        self.temporal_regularization = 0.1
    
    def forward(self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None, **kwargs) -> Dict[str, mx.array]:
        """Forward pass through time series regression head.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask [batch_size, seq_len]
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with predictions and temporal components
        """
        # Apply pooling to get sequence representation
        pooled_output = self._apply_pooling(hidden_states, attention_mask)
        
        # Apply projection layers
        projected = self.projection(pooled_output)
        
        # Reshape for temporal modeling if needed
        if len(projected.shape) == 2:
            # Add sequence dimension
            projected = projected.unsqueeze(1)  # [batch_size, 1, hidden_size]
        
        # Apply temporal modeling
        if hasattr(self, 'temporal_layer'):
            temporal_features = self.temporal_layer(projected.squeeze(1))
        else:
            temporal_features = projected.squeeze(1)
        
        # Get predictions
        predictions = self.regressor(temporal_features)
        
        result = {
            "predictions": predictions,
        }
        
        # Add seasonal decomposition if enabled
        if self.use_temporal_features:
            seasonal = self.seasonal_layer(temporal_features)
            trend = self.trend_layer(temporal_features)
            
            result.update({
                "seasonal": seasonal,
                "trend": trend,
                "residual": predictions - seasonal - trend,
            })
        
        return result
    
    def compute_loss(self, predictions: Dict[str, mx.array], targets: mx.array, **kwargs) -> mx.array:
        """Compute loss for time series regression.
        
        Args:
            predictions: Predictions from forward pass
            targets: Ground truth targets [batch_size, forecast_horizon]
            **kwargs: Additional arguments
            
        Returns:
            Loss value
        """
        pred_values = predictions["predictions"]
        targets = targets.astype(mx.float32)
        
        # Main regression loss
        regression_loss = self.mse_loss(pred_values, targets)
        
        # Add temporal regularization
        if self.forecast_horizon > 1:
            # Encourage smooth predictions
            pred_diff = pred_values[:, 1:] - pred_values[:, :-1]
            smoothness_loss = mx.mean(mx.square(pred_diff))
            
            total_loss = regression_loss + self.temporal_regularization * smoothness_loss
            return total_loss
        
        return regression_loss
    
    def compute_metrics(self, predictions: Dict[str, mx.array], targets: mx.array, **kwargs) -> Dict[str, float]:
        """Compute evaluation metrics for time series regression.
        
        Args:
            predictions: Predictions from forward pass
            targets: Ground truth targets
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of metric values
        """
        pred_values = predictions["predictions"]
        
        # Convert to numpy for metric computation
        pred_np = np.array(pred_values)
        targets_np = np.array(targets)
        
        # Flatten for multi-step predictions
        pred_flat = pred_np.flatten()
        targets_flat = targets_np.flatten()
        
        # Basic metrics
        mse = np.mean((pred_flat - targets_flat) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(pred_flat - targets_flat))
        
        # Mean absolute percentage error (MAPE)
        mape = np.mean(np.abs((targets_flat - pred_flat) / (targets_flat + 1e-8))) * 100
        
        # Directional accuracy (for multi-step)
        if self.forecast_horizon > 1:
            pred_direction = np.sign(pred_np[:, 1:] - pred_np[:, :-1])
            target_direction = np.sign(targets_np[:, 1:] - targets_np[:, :-1])
            directional_accuracy = np.mean(pred_direction == target_direction)
        else:
            directional_accuracy = 0.0
        
        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "mape": float(mape),
            "directional_accuracy": float(directional_accuracy),
        }


# Loss function implementations
class HuberLoss(nn.Module):
    """Huber loss for robust regression."""
    
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta
    
    def __call__(self, predictions: mx.array, targets: mx.array) -> mx.array:
        residual = mx.abs(predictions - targets)
        condition = residual <= self.delta
        
        squared_loss = 0.5 * mx.square(residual)
        linear_loss = self.delta * residual - 0.5 * self.delta ** 2
        
        return mx.where(condition, squared_loss, linear_loss).mean()


class OrdinalLoss(nn.Module):
    """Ordinal loss function for ordinal regression."""
    
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
    
    def __call__(self, threshold_logits: mx.array, targets: mx.array) -> mx.array:
        """Compute ordinal loss using cumulative logits approach."""
        batch_size = threshold_logits.shape[0]
        
        # Create target matrix for cumulative probabilities
        # target_matrix[i, j] = 1 if target[i] > j, else 0
        target_matrix = mx.zeros((batch_size, self.num_classes - 1))
        
        for i in range(batch_size):
            target_class = targets[i]
            # Set all thresholds up to target_class-1 to 1
            if target_class > 0:
                target_matrix = target_matrix.at[i, :target_class].set(1.0)
        
        # Binary cross-entropy loss for each threshold
        loss = mx.sigmoid_cross_entropy_with_logits(threshold_logits, target_matrix)
        
        return loss.mean()