"""Regression heads for BERT models.

This module provides various regression heads following the clean
architecture patterns from the BERT module.
"""

from typing import Dict, Optional, Union
import mlx.core as mx
import mlx.nn as nn

from .base import BaseHead
from .config import RegressionConfig, HeadConfig


class RegressionHead(BaseHead):
    """Standard regression head for continuous target prediction.
    
    This head performs standard regression with optional uncertainty estimation.
    """
    
    def __init__(self, config: Union[RegressionConfig, HeadConfig]):
        """Initialize regression head.
        
        Args:
            config: Head configuration
        """
        # Ensure regression settings
        config.output_size = 1
        config.head_type = "regression"
        
        # Set loss type
        self.loss_type = getattr(config, "loss_type", "mse")
        self.huber_delta = getattr(config, "huber_delta", 1.0)
        
        super().__init__(config)
        
        # Initialize uncertainty if enabled
        self.use_uncertainty = getattr(config, "use_uncertainty", False)
        if self.use_uncertainty:
            self._build_uncertainty_head()
    
    def _build_output_layer(self):
        """Build the output layer for regression."""
        self.regressor = nn.Linear(
            self.projection_output_size,
            self.config.output_size,
            bias=self.config.use_bias
        )
    
    def _build_uncertainty_head(self):
        """Build uncertainty estimation head."""
        self.uncertainty_head = nn.Linear(
            self.projection_output_size,
            self.config.output_size,
            bias=self.config.use_bias
        )
    
    def _forward_output(self, features: mx.array) -> Dict[str, mx.array]:
        """Forward pass through the output layer.
        
        Args:
            features: Features after projection [batch_size, projection_output_size]
            
        Returns:
            Dictionary containing predictions and optional uncertainty
        """
        # Get predictions
        predictions = self.regressor(features)  # [batch_size, 1]
        
        result = {
            "predictions": predictions.squeeze(-1),  # [batch_size]
            "logits": predictions,  # Keep original shape for compatibility
        }
        
        # Add uncertainty if enabled
        if self.use_uncertainty:
            log_variance = self.uncertainty_head(features)
            variance = mx.exp(log_variance)
            uncertainty = mx.sqrt(variance)
            
            result.update({
                "uncertainty": uncertainty.squeeze(-1),
                "log_variance": log_variance.squeeze(-1),
                "variance": variance.squeeze(-1),
            })
        
        return result
    
    def compute_loss(
        self,
        predictions: Dict[str, mx.array],
        targets: mx.array,
        **kwargs
    ) -> mx.array:
        """Compute loss for regression.
        
        Args:
            predictions: Output from forward pass
            targets: Ground truth values [batch_size]
            **kwargs: Additional arguments
            
        Returns:
            Loss value
        """
        pred_values = predictions["predictions"]
        
        # Compute base regression loss
        if self.loss_type == "mse":
            base_loss = mx.mean((pred_values - targets) ** 2)
        elif self.loss_type == "mae":
            base_loss = mx.mean(mx.abs(pred_values - targets))
        elif self.loss_type == "huber":
            base_loss = self._huber_loss(pred_values, targets, self.huber_delta)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Add uncertainty loss if enabled
        if self.use_uncertainty and "log_variance" in predictions:
            log_variance = predictions["log_variance"]
            variance = mx.exp(log_variance)
            
            # Heteroscedastic uncertainty loss
            uncertainty_loss = 0.5 * mx.mean(
                (pred_values - targets) ** 2 / variance + log_variance
            )
            
            return base_loss + uncertainty_loss
        
        return base_loss
    
    def _huber_loss(self, predictions: mx.array, targets: mx.array, delta: float) -> mx.array:
        """Compute Huber loss.
        
        Args:
            predictions: Predicted values
            targets: True values
            delta: Huber delta parameter
            
        Returns:
            Huber loss value
        """
        error = predictions - targets
        abs_error = mx.abs(error)
        
        # Huber loss computation
        quadratic = 0.5 * error ** 2
        linear = delta * abs_error - 0.5 * delta ** 2
        
        loss = mx.where(abs_error <= delta, quadratic, linear)
        return mx.mean(loss)


class OrdinalRegressionHead(BaseHead):
    """Ordinal regression head for ordered categorical targets.
    
    This head uses the cumulative link model approach for ordinal regression.
    """
    
    def __init__(self, config: Union[HeadConfig, RegressionConfig], num_classes: int):
        """Initialize ordinal regression head.
        
        Args:
            config: Head configuration
            num_classes: Number of ordinal classes
        """
        self.num_classes = num_classes
        config.output_size = num_classes - 1  # K-1 thresholds
        config.head_type = "ordinal_regression"
        
        super().__init__(config)
        
        # Initialize thresholds (will be sorted during forward pass)
        self.threshold_bias = mx.zeros(num_classes - 1)
    
    def _build_output_layer(self):
        """Build the output layer for ordinal regression."""
        # Single linear projection (shared across all thresholds)
        self.projection_layer = nn.Linear(
            self.projection_output_size,
            1,
            bias=False  # Bias is handled separately as thresholds
        )
    
    def _forward_output(self, features: mx.array) -> Dict[str, mx.array]:
        """Forward pass through the output layer.
        
        Args:
            features: Features after projection [batch_size, projection_output_size]
            
        Returns:
            Dictionary containing cumulative probabilities and predictions
        """
        # Get base projection
        base_logits = self.projection_layer(features)  # [batch_size, 1]
        
        # Sort thresholds to ensure ordering
        sorted_thresholds = mx.sort(self.threshold_bias)
        
        # Compute cumulative logits
        cumulative_logits = base_logits - sorted_thresholds  # [batch_size, K-1]
        
        # Get cumulative probabilities P(Y <= k)
        cumulative_probs = mx.sigmoid(cumulative_logits)
        
        # Convert to class probabilities
        # P(Y = 0) = P(Y <= 0)
        # P(Y = k) = P(Y <= k) - P(Y <= k-1) for k = 1, ..., K-2
        # P(Y = K-1) = 1 - P(Y <= K-2)
        
        # Pad with 0 and 1 for boundary conditions
        padded_probs = mx.concatenate([
            mx.zeros((cumulative_probs.shape[0], 1)),
            cumulative_probs,
            mx.ones((cumulative_probs.shape[0], 1))
        ], axis=1)
        
        # Compute differences to get class probabilities
        class_probs = padded_probs[:, 1:] - padded_probs[:, :-1]
        
        # Get predictions
        predictions = mx.argmax(class_probs, axis=-1)
        
        return {
            "cumulative_logits": cumulative_logits,
            "cumulative_probabilities": cumulative_probs,
            "probabilities": class_probs,
            "predictions": predictions,
            "thresholds": sorted_thresholds,
        }
    
    def compute_loss(
        self,
        predictions: Dict[str, mx.array],
        targets: mx.array,
        **kwargs
    ) -> mx.array:
        """Compute loss for ordinal regression.
        
        Args:
            predictions: Output from forward pass
            targets: Ground truth ordinal labels [batch_size]
            **kwargs: Additional arguments
            
        Returns:
            Loss value
        """
        cumulative_logits = predictions["cumulative_logits"]
        
        # Create cumulative targets
        # For target class k: y_cumulative[j] = 1 if j < k, else 0
        cumulative_targets = mx.zeros_like(cumulative_logits)
        for i in range(cumulative_logits.shape[0]):
            target_class = int(targets[i])
            if target_class > 0:
                cumulative_targets[i, :target_class] = 1
        
        # Binary cross-entropy for each threshold
        probs = mx.sigmoid(cumulative_logits)
        eps = 1e-7
        probs = mx.clip(probs, eps, 1 - eps)
        
        loss = -(cumulative_targets * mx.log(probs) + 
                 (1 - cumulative_targets) * mx.log(1 - probs))
        
        return mx.mean(loss)


class QuantileRegressionHead(BaseHead):
    """Quantile regression head for prediction intervals.
    
    This head predicts multiple quantiles for uncertainty quantification.
    """
    
    def __init__(
        self,
        config: Union[HeadConfig, RegressionConfig],
        quantiles: Optional[list] = None
    ):
        """Initialize quantile regression head.
        
        Args:
            config: Head configuration
            quantiles: List of quantiles to predict (default: [0.1, 0.5, 0.9])
        """
        self.quantiles = quantiles or [0.1, 0.5, 0.9]
        config.output_size = len(self.quantiles)
        config.head_type = "quantile_regression"
        
        super().__init__(config)
    
    def _build_output_layer(self):
        """Build the output layer for quantile regression."""
        self.quantile_heads = nn.Linear(
            self.projection_output_size,
            len(self.quantiles),
            bias=self.config.use_bias
        )
    
    def _forward_output(self, features: mx.array) -> Dict[str, mx.array]:
        """Forward pass through the output layer.
        
        Args:
            features: Features after projection [batch_size, projection_output_size]
            
        Returns:
            Dictionary containing quantile predictions
        """
        # Get quantile predictions
        quantile_preds = self.quantile_heads(features)  # [batch_size, num_quantiles]
        
        # Ensure quantiles are ordered (optional, can be enforced during training)
        # This is a simple approach; more sophisticated methods exist
        sorted_preds = mx.sort(quantile_preds, axis=-1)
        
        # Get median prediction (or closest quantile to 0.5)
        median_idx = len(self.quantiles) // 2
        median_pred = sorted_preds[:, median_idx]
        
        return {
            "quantile_predictions": sorted_preds,
            "predictions": median_pred,  # Use median as point estimate
            "quantiles": mx.array(self.quantiles),
        }
    
    def compute_loss(
        self,
        predictions: Dict[str, mx.array],
        targets: mx.array,
        **kwargs
    ) -> mx.array:
        """Compute pinball loss for quantile regression.
        
        Args:
            predictions: Output from forward pass
            targets: Ground truth values [batch_size]
            **kwargs: Additional arguments
            
        Returns:
            Loss value
        """
        quantile_preds = predictions["quantile_predictions"]
        
        # Expand targets to match quantile predictions
        targets_expanded = mx.expand_dims(targets, axis=-1)
        
        # Compute pinball loss for each quantile
        errors = targets_expanded - quantile_preds
        
        # Create quantile array matching predictions shape
        quantiles_array = mx.array(self.quantiles)
        quantiles_expanded = mx.expand_dims(quantiles_array, axis=0)
        
        # Pinball loss
        loss = mx.maximum(quantiles_expanded * errors, (quantiles_expanded - 1) * errors)
        
        return mx.mean(loss)


# Factory functions

def create_regression_head(
    config: Union[RegressionConfig, HeadConfig],
    head_type: Optional[str] = None
) -> BaseHead:
    """Create a regression head based on configuration.
    
    Args:
        config: Head configuration
        head_type: Optional override for head type
        
    Returns:
        Regression head instance
        
    Raises:
        ValueError: If head type is unknown
    """
    # Determine head type
    if head_type is None:
        head_type = getattr(config, "head_type", "regression")
    
    # Create appropriate head
    if head_type == "regression":
        return RegressionHead(config)
    elif head_type == "ordinal_regression":
        if not hasattr(config, "num_classes"):
            raise ValueError("num_classes must be specified for ordinal regression")
        return OrdinalRegressionHead(config, config.num_classes)
    elif head_type == "quantile_regression":
        quantiles = getattr(config, "quantiles", None)
        return QuantileRegressionHead(config, quantiles)
    else:
        raise ValueError(f"Unknown regression head type: {head_type}")


def create_standard_regression_head(
    input_size: int,
    **kwargs
) -> RegressionHead:
    """Create a standard regression head with sensible defaults.
    
    Args:
        input_size: Size of input features
        **kwargs: Additional configuration parameters
        
    Returns:
        Standard regression head
    """
    from .config import get_regression_preset_config
    
    config = get_regression_preset_config(input_size)
    
    # Update with any provided kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return RegressionHead(config)


def create_ordinal_regression_head(
    input_size: int,
    num_classes: int,
    **kwargs
) -> OrdinalRegressionHead:
    """Create an ordinal regression head.
    
    Args:
        input_size: Size of input features
        num_classes: Number of ordinal classes
        **kwargs: Additional configuration parameters
        
    Returns:
        Ordinal regression head
    """
    from .config import get_base_config
    
    config = get_base_config(input_size, num_classes - 1)
    config.pooling_type = "mean"  # Mean pooling often works well for ordinal
    
    # Update with any provided kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Add num_classes to config for OrdinalRegressionHead
    config.num_classes = num_classes
    
    return OrdinalRegressionHead(config, num_classes)


def create_quantile_regression_head(
    input_size: int,
    quantiles: Optional[list] = None,
    **kwargs
) -> QuantileRegressionHead:
    """Create a quantile regression head.
    
    Args:
        input_size: Size of input features
        quantiles: List of quantiles to predict
        **kwargs: Additional configuration parameters
        
    Returns:
        Quantile regression head
    """
    from .config import get_base_config
    
    quantiles = quantiles or [0.1, 0.5, 0.9]
    config = get_base_config(input_size, len(quantiles))
    config.pooling_type = "mean"
    
    # Update with any provided kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return QuantileRegressionHead(config, quantiles)


__all__ = [
    # Head classes
    "RegressionHead",
    "OrdinalRegressionHead",
    "QuantileRegressionHead",
    # Factory functions
    "create_regression_head",
    "create_standard_regression_head",
    "create_ordinal_regression_head",
    "create_quantile_regression_head",
]