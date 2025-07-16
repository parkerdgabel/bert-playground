"""
Gradient clipping and monitoring utilities for MLX models.
Optimized for CNN-BERT hybrid architectures.
"""

import mlx.core as mx
from typing import Dict, List, Tuple, Optional, Any
from loguru import logger
import numpy as np


def clip_gradients(
    gradients: Dict[str, mx.array], max_norm: float = 1.0, norm_type: str = "l2"
) -> Tuple[Dict[str, mx.array], float]:
    """
    Clip gradients by global norm to prevent gradient explosion.

    Args:
        gradients: Dictionary mapping parameter names to gradients (possibly nested)
        max_norm: Maximum allowed gradient norm
        norm_type: Type of norm to use ('l2' or 'inf')

    Returns:
        Tuple of (clipped_gradients, total_norm)
    """
    from mlx.utils import tree_flatten

    # Flatten gradients to handle nested structures
    flat_grads = tree_flatten(gradients)

    if norm_type == "l2":
        # Calculate L2 norm across all gradients
        total_norm = 0.0
        for key, grad in flat_grads:
            if isinstance(grad, mx.array):
                param_norm = mx.sqrt(mx.sum(grad * grad))
                total_norm += param_norm * param_norm
        total_norm = mx.sqrt(total_norm)
    elif norm_type == "inf":
        # Calculate infinity norm (max absolute value)
        total_norm = 0.0
        for key, grad in flat_grads:
            if isinstance(grad, mx.array):
                param_norm = mx.max(mx.abs(grad))
                total_norm = mx.maximum(total_norm, param_norm)
    else:
        raise ValueError(f"Unsupported norm type: {norm_type}")

    # Clip gradients if norm exceeds threshold
    clip_coef = max_norm / (total_norm + 1e-8)

    if float(total_norm) > max_norm:
        # Create a function to apply clipping recursively
        def apply_clipping(grad_dict):
            clipped = {}
            for key, value in grad_dict.items():
                if isinstance(value, mx.array):
                    clipped[key] = value * clip_coef
                elif isinstance(value, dict):
                    clipped[key] = apply_clipping(value)
                else:
                    clipped[key] = value
            return clipped

        clipped_gradients = apply_clipping(gradients)
        return clipped_gradients, float(total_norm)
    else:
        return gradients, float(total_norm)


def monitor_gradient_components(
    gradients: Dict[str, mx.array],
    model_components: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Monitor gradient statistics by model component.

    Args:
        gradients: Dictionary mapping parameter names to gradients (possibly nested)
        model_components: Dictionary mapping component names to parameter patterns

    Returns:
        Dictionary with gradient statistics for each component
    """
    from mlx.utils import tree_flatten

    if model_components is None:
        # Default component mapping for CNN-BERT hybrid
        model_components = {
            "embeddings": ["embeddings", "word_embeddings", "position_embeddings"],
            "bert": ["bert", "transformer_layers", "attention", "feed_forward"],
            "cnn": ["conv", "cnn", "dilated_conv", "kernel"],
            "fusion": ["fusion", "attention_fusion", "feature_projection"],
            "classifier": ["classifier", "dense", "output"],
        }

    # Flatten gradients to handle nested structures
    flat_grads = tree_flatten(gradients)

    component_stats = {}

    for component_name, patterns in model_components.items():
        component_grads = []

        # Find gradients matching component patterns
        for param_name, grad in flat_grads:
            if any(pattern in param_name.lower() for pattern in patterns):
                component_grads.append(grad)

        if component_grads:
            # Calculate statistics for this component
            all_norms = []
            all_values = []

            for grad in component_grads:
                if isinstance(grad, mx.array):
                    all_norms.append(mx.sqrt(mx.sum(grad * grad)))
                    all_values.append(mx.flatten(grad))

            if all_norms and all_values:
                all_values_concat = mx.concatenate(all_values)

                component_stats[component_name] = {
                    "num_params": len(all_norms),
                    "mean_norm": float(mx.mean(mx.stack(all_norms))),
                    "max_norm": float(mx.max(mx.stack(all_norms))),
                    "min_norm": float(mx.min(mx.stack(all_norms))),
                    "std_norm": float(mx.std(mx.stack(all_norms))),
                    "mean_value": float(mx.mean(all_values_concat)),
                    "std_value": float(mx.std(all_values_concat)),
                    "max_abs_value": float(mx.max(mx.abs(all_values_concat))),
                    "total_norm": float(
                        mx.sqrt(mx.sum(all_values_concat * all_values_concat))
                    ),
                }

    return component_stats


def log_gradient_statistics(
    gradient_stats: Dict[str, Dict[str, float]],
    step: int,
    total_norm: float,
    clipped: bool = False,
) -> None:
    """
    Log gradient statistics in a structured format.

    Args:
        gradient_stats: Component-wise gradient statistics
        step: Current training step
        total_norm: Total gradient norm
        clipped: Whether gradients were clipped
    """
    logger.info(f"Step {step} - Gradient Statistics:")
    logger.info(
        f"  Total gradient norm: {total_norm:.4f} {'(CLIPPED)' if clipped else ''}"
    )

    for component, stats in gradient_stats.items():
        logger.info(
            f"  {component:12s}: "
            f"norm={stats['total_norm']:.4f}, "
            f"mean_norm={stats['mean_norm']:.4f}, "
            f"max_norm={stats['max_norm']:.4f}, "
            f"params={stats['num_params']}"
        )


def detect_gradient_anomalies(
    gradient_stats: Dict[str, Dict[str, float]],
    total_norm: float,
    step: int,
    anomaly_threshold: float = 10.0,
) -> List[str]:
    """
    Detect potential gradient anomalies that might indicate training problems.

    Args:
        gradient_stats: Component-wise gradient statistics
        total_norm: Total gradient norm
        step: Current training step
        anomaly_threshold: Threshold for detecting anomalies

    Returns:
        List of anomaly descriptions
    """
    anomalies = []

    # Check for exploding gradients
    if total_norm > anomaly_threshold:
        anomalies.append(f"Exploding gradients detected: total_norm={total_norm:.4f}")

    # Check for vanishing gradients
    if total_norm < 1e-6:
        anomalies.append(f"Vanishing gradients detected: total_norm={total_norm:.4f}")

    # Check component-specific anomalies
    for component, stats in gradient_stats.items():
        # Check for component-specific exploding gradients
        if stats["max_norm"] > anomaly_threshold:
            anomalies.append(
                f"{component} component has exploding gradients: "
                f"max_norm={stats['max_norm']:.4f}"
            )

        # Check for component-specific vanishing gradients
        if stats["mean_norm"] < 1e-7:
            anomalies.append(
                f"{component} component has vanishing gradients: "
                f"mean_norm={stats['mean_norm']:.4f}"
            )

        # Check for abnormal gradient distribution
        if stats["std_norm"] > stats["mean_norm"] * 5:
            anomalies.append(
                f"{component} component has highly variable gradients: "
                f"std/mean={stats['std_norm'] / stats['mean_norm']:.2f}"
            )

    return anomalies


class GradientMonitor:
    """
    Comprehensive gradient monitoring class for training stability.
    """

    def __init__(
        self,
        max_norm: float = 1.0,
        log_interval: int = 50,
        anomaly_threshold: float = 10.0,
        component_mapping: Optional[Dict[str, List[str]]] = None,
    ):
        self.max_norm = max_norm
        self.log_interval = log_interval
        self.anomaly_threshold = anomaly_threshold
        self.component_mapping = component_mapping

        # History tracking
        self.gradient_history = []
        self.clip_history = []
        self.anomaly_history = []

        # Statistics
        self.total_clips = 0
        self.total_steps = 0

    def process_gradients(
        self, gradients: Dict[str, mx.array], step: int, force_log: bool = False
    ) -> Tuple[Dict[str, mx.array], Dict[str, Any]]:
        """
        Process gradients with clipping and monitoring.

        Args:
            gradients: Raw gradients from backward pass
            step: Current training step
            force_log: Force logging even if not at log interval

        Returns:
            Tuple of (processed_gradients, monitoring_info)
        """
        self.total_steps += 1

        # Clip gradients
        clipped_gradients, total_norm = clip_gradients(
            gradients, max_norm=self.max_norm
        )

        # Track clipping statistics
        was_clipped = total_norm > self.max_norm
        if was_clipped:
            self.total_clips += 1

        # Monitor component statistics
        component_stats = monitor_gradient_components(gradients, self.component_mapping)

        # Detect anomalies
        anomalies = detect_gradient_anomalies(
            component_stats, total_norm, step, self.anomaly_threshold
        )

        # Store history
        self.gradient_history.append(total_norm)
        self.clip_history.append(was_clipped)
        self.anomaly_history.extend(anomalies)

        # Log if needed
        should_log = (step % self.log_interval == 0) or force_log or anomalies
        if should_log:
            log_gradient_statistics(component_stats, step, total_norm, was_clipped)

            # Log anomalies
            if anomalies:
                logger.warning(f"Gradient anomalies detected at step {step}:")
                for anomaly in anomalies:
                    logger.warning(f"  - {anomaly}")

        # Create monitoring info
        monitoring_info = {
            "total_norm": total_norm,
            "was_clipped": was_clipped,
            "component_stats": component_stats,
            "anomalies": anomalies,
            "clip_percentage": self.total_clips / self.total_steps * 100,
        }

        return clipped_gradients, monitoring_info

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the entire training run."""
        if not self.gradient_history:
            return {}

        gradient_norms = np.array(self.gradient_history)

        return {
            "total_steps": self.total_steps,
            "total_clips": self.total_clips,
            "clip_percentage": self.total_clips / self.total_steps * 100,
            "mean_gradient_norm": np.mean(gradient_norms),
            "std_gradient_norm": np.std(gradient_norms),
            "max_gradient_norm": np.max(gradient_norms),
            "min_gradient_norm": np.min(gradient_norms),
            "total_anomalies": len(self.anomaly_history),
            "recent_gradient_trend": self._calculate_trend(),
        }

    def _calculate_trend(self, window_size: int = 100) -> str:
        """Calculate recent gradient norm trend."""
        if len(self.gradient_history) < window_size:
            return "insufficient_data"

        recent_norms = self.gradient_history[-window_size:]
        first_half = np.mean(recent_norms[: window_size // 2])
        second_half = np.mean(recent_norms[window_size // 2 :])

        if second_half > first_half * 1.1:
            return "increasing"
        elif second_half < first_half * 0.9:
            return "decreasing"
        else:
            return "stable"

    def reset(self):
        """Reset monitoring history."""
        self.gradient_history = []
        self.clip_history = []
        self.anomaly_history = []
        self.total_clips = 0
        self.total_steps = 0


# Convenience function for creating gradient monitor with sensible defaults
def create_gradient_monitor(
    max_norm: float = 1.0, log_interval: int = 50, model_type: str = "cnn_hybrid"
) -> GradientMonitor:
    """
    Create gradient monitor with appropriate settings for different model types.

    Args:
        max_norm: Maximum gradient norm for clipping
        log_interval: Steps between logging
        model_type: Type of model for component mapping

    Returns:
        Configured GradientMonitor instance
    """
    component_mappings = {
        "cnn_hybrid": {
            "embeddings": ["embeddings", "word_embeddings", "position_embeddings"],
            "bert": [
                "bert",
                "transformer_layers",
                "attention",
                "feed_forward",
                "self_attn",
            ],
            "cnn": ["conv", "cnn", "dilated_conv", "kernel"],
            "fusion": ["fusion", "attention_fusion", "feature_projection"],
            "classifier": ["classifier", "dense", "output"],
        },
        "bert": {
            "embeddings": ["embeddings", "word_embeddings", "position_embeddings"],
            "transformer": [
                "transformer_layers",
                "attention",
                "feed_forward",
                "self_attn",
            ],
            "classifier": ["classifier", "dense", "output"],
        },
    }

    component_mapping = component_mappings.get(model_type, None)

    return GradientMonitor(
        max_norm=max_norm,
        log_interval=log_interval,
        component_mapping=component_mapping,
    )
