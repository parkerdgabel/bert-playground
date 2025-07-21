"""
Custom loss functions for handling class imbalance in binary classification.
Optimized for MLX framework.
"""

import mlx.core as mx
import mlx.nn as nn
from loguru import logger


def focal_loss(
    logits: mx.array,
    labels: mx.array,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> mx.array:
    """
    Focal loss for binary classification.

    FL(pt) = -α(1-pt)^γ * log(pt)

    Args:
        logits: Model output logits of shape [batch_size, 2]
        labels: Ground truth labels of shape [batch_size]
        alpha: Weighting factor for positive class (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Loss value (scalar if reduction is 'mean' or 'sum')
    """
    # Ensure labels are integers
    labels = labels.astype(mx.int32)

    # Convert logits to probabilities
    probs = mx.softmax(logits, axis=-1)

    # Get probability of the true class for each sample
    batch_size = labels.shape[0]
    batch_indices = mx.arange(batch_size)
    pt = probs[batch_indices, labels]

    # Calculate focal term: (1 - pt)^gamma
    focal_term = mx.power(1 - pt, gamma)

    # Calculate alpha term for class weighting
    # alpha for positive class, (1-alpha) for negative class
    alpha_t = mx.where(labels == 1, alpha, 1 - alpha)

    # Cross entropy loss: -log(pt)
    ce_loss = -mx.log(pt + 1e-8)  # Add epsilon for numerical stability

    # Combine all terms
    focal_loss = alpha_t * focal_term * ce_loss

    # Apply reduction
    if reduction == "mean":
        return mx.mean(focal_loss)
    elif reduction == "sum":
        return mx.sum(focal_loss)
    else:
        return focal_loss


def weighted_cross_entropy(
    logits: mx.array,
    labels: mx.array,
    class_weights: mx.array | None = None,
    reduction: str = "mean",
) -> mx.array:
    """
    Weighted cross-entropy loss for handling class imbalance.

    Args:
        logits: Model output logits of shape [batch_size, 2]
        labels: Ground truth labels of shape [batch_size]
        class_weights: Weights for each class [weight_class_0, weight_class_1]
                      Default: [1.56, 1.0] for Titanic dataset
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Loss value
    """
    if class_weights is None:
        # Default weights for Titanic dataset (39% class 0, 61% class 1)
        class_weights = mx.array([1.56, 1.0])

    # Ensure labels are integers
    labels = labels.astype(mx.int32)

    # Standard cross-entropy loss
    ce_loss = nn.losses.cross_entropy(logits, labels, reduction="none")

    # Apply class weights
    weights = class_weights[labels]
    weighted_loss = ce_loss * weights

    # Apply reduction
    if reduction == "mean":
        return mx.mean(weighted_loss)
    elif reduction == "sum":
        return mx.sum(weighted_loss)
    else:
        return weighted_loss


def label_smoothing_cross_entropy(
    logits: mx.array, labels: mx.array, smoothing: float = 0.1, reduction: str = "mean"
) -> mx.array:
    """
    Cross-entropy loss with label smoothing.

    Args:
        logits: Model output logits of shape [batch_size, 2]
        labels: Ground truth labels of shape [batch_size]
        smoothing: Label smoothing factor (default: 0.1)
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Loss value
    """
    num_classes = logits.shape[-1]
    batch_size = labels.shape[0]

    # Ensure labels are integers
    labels = labels.astype(mx.int32)

    # Create smoothed label distribution
    confidence = 1.0 - smoothing
    smooth_labels = mx.ones((batch_size, num_classes)) * (smoothing / num_classes)

    # Set confidence for true labels
    batch_indices = mx.arange(batch_size)
    smooth_labels[batch_indices, labels] = confidence

    # Calculate log probabilities
    log_probs = mx.log(mx.softmax(logits, axis=-1) + 1e-8)

    # Calculate loss: -sum(smooth_labels * log_probs)
    loss = -mx.sum(smooth_labels * log_probs, axis=-1)

    # Apply reduction
    if reduction == "mean":
        return mx.mean(loss)
    elif reduction == "sum":
        return mx.sum(loss)
    else:
        return loss


def focal_loss_with_label_smoothing(
    logits: mx.array,
    labels: mx.array,
    alpha: float = 0.25,
    gamma: float = 2.0,
    smoothing: float = 0.1,
    reduction: str = "mean",
) -> mx.array:
    """
    Combines focal loss with label smoothing for optimal performance.

    Args:
        logits: Model output logits of shape [batch_size, 2]
        labels: Ground truth labels of shape [batch_size]
        alpha: Weighting factor for positive class (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
        smoothing: Label smoothing factor (default: 0.1)
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Loss value
    """
    num_classes = logits.shape[-1]
    batch_size = labels.shape[0]

    # Ensure labels are integers
    labels = labels.astype(mx.int32)

    # Create smoothed label distribution
    confidence = 1.0 - smoothing
    smooth_labels = mx.ones((batch_size, num_classes)) * (smoothing / num_classes)
    batch_indices = mx.arange(batch_size)
    smooth_labels[batch_indices, labels] = confidence

    # Calculate probabilities
    probs = mx.softmax(logits, axis=-1)

    # Get probability for true class (with smoothing)
    pt = mx.sum(probs * smooth_labels, axis=-1)

    # Calculate focal term
    focal_term = mx.power(1 - pt, gamma)

    # Calculate alpha term
    alpha_t = mx.where(labels == 1, alpha, 1 - alpha)

    # Cross-entropy with smoothed labels
    ce_loss = -mx.sum(smooth_labels * mx.log(probs + 1e-8), axis=-1)

    # Combine all terms
    loss = alpha_t * focal_term * ce_loss

    # Apply reduction
    if reduction == "mean":
        return mx.mean(loss)
    elif reduction == "sum":
        return mx.sum(loss)
    else:
        return loss


def dice_loss(
    logits: mx.array, labels: mx.array, smooth: float = 1.0, reduction: str = "mean"
) -> mx.array:
    """
    Dice loss (F1 loss) for binary classification.
    Particularly effective for imbalanced datasets.

    Args:
        logits: Model output logits of shape [batch_size, 2]
        labels: Ground truth labels of shape [batch_size]
        smooth: Smoothing factor to avoid division by zero
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Loss value
    """
    # Get probabilities for positive class
    probs = mx.softmax(logits, axis=-1)[:, 1]

    # Ensure labels are floats for multiplication
    labels = labels.astype(mx.float32)

    # Calculate Dice coefficient
    intersection = mx.sum(probs * labels, axis=0)
    union = mx.sum(probs, axis=0) + mx.sum(labels, axis=0)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    loss = 1.0 - dice

    if reduction == "mean":
        return loss
    elif reduction == "sum":
        return loss * labels.shape[0]
    else:
        return mx.broadcast_to(loss, labels.shape)


def tversky_loss(
    logits: mx.array,
    labels: mx.array,
    alpha: float = 0.7,
    beta: float = 0.3,
    smooth: float = 1.0,
    reduction: str = "mean",
) -> mx.array:
    """
    Tversky loss - a generalization of Dice loss with adjustable
    false positive and false negative penalties.

    Args:
        logits: Model output logits of shape [batch_size, 2]
        labels: Ground truth labels of shape [batch_size]
        alpha: Weight for false positives (default: 0.7)
        beta: Weight for false negatives (default: 0.3)
        smooth: Smoothing factor
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Loss value
    """
    # Get probabilities for positive class
    probs = mx.softmax(logits, axis=-1)[:, 1]

    # Ensure labels are floats
    labels = labels.astype(mx.float32)

    # Calculate Tversky index components
    true_pos = mx.sum(probs * labels, axis=0)
    false_pos = mx.sum(probs * (1 - labels), axis=0)
    false_neg = mx.sum((1 - probs) * labels, axis=0)

    tversky = (true_pos + smooth) / (
        true_pos + alpha * false_pos + beta * false_neg + smooth
    )
    loss = 1.0 - tversky

    if reduction == "mean":
        return loss
    elif reduction == "sum":
        return loss * labels.shape[0]
    else:
        return mx.broadcast_to(loss, labels.shape)


class AdaptiveLoss:
    """
    Adaptive loss that automatically adjusts based on training progress.
    Starts with weighted cross-entropy and gradually transitions to focal loss.
    """

    def __init__(
        self,
        initial_alpha: float = 0.5,
        final_alpha: float = 0.25,
        initial_gamma: float = 0.0,
        final_gamma: float = 2.0,
        warmup_steps: int = 1000,
        class_weights: mx.array | None = None,
    ):
        self.initial_alpha = initial_alpha
        self.final_alpha = final_alpha
        self.initial_gamma = initial_gamma
        self.final_gamma = final_gamma
        self.warmup_steps = warmup_steps
        self.class_weights = class_weights
        self.current_step = 0

    def __call__(
        self, logits: mx.array, labels: mx.array, reduction: str = "mean"
    ) -> mx.array:
        """
        Calculate adaptive loss based on current training step.
        """
        # Calculate interpolation factor
        progress = min(self.current_step / self.warmup_steps, 1.0)

        # Interpolate alpha and gamma
        alpha = self.initial_alpha + (self.final_alpha - self.initial_alpha) * progress
        gamma = self.initial_gamma + (self.final_gamma - self.initial_gamma) * progress

        # Use focal loss with current parameters
        loss = focal_loss(logits, labels, alpha=alpha, gamma=gamma, reduction=reduction)

        # Increment step counter
        self.current_step += 1

        # Log progress occasionally
        if self.current_step % 100 == 0:
            logger.debug(
                f"Adaptive loss at step {self.current_step}: alpha={alpha:.3f}, gamma={gamma:.3f}"
            )

        return loss

    def reset(self):
        """Reset the step counter."""
        self.current_step = 0


def get_loss_function(loss_type: str = "focal", **kwargs) -> callable | AdaptiveLoss:
    """
    Factory function to get the appropriate loss function.

    Args:
        loss_type: One of 'focal', 'weighted_ce', 'label_smooth',
                  'focal_smooth', 'dice', 'tversky', 'adaptive'
        **kwargs: Additional arguments for the specific loss function

    Returns:
        Loss function callable
    """
    loss_functions = {
        "focal": focal_loss,
        "weighted_ce": weighted_cross_entropy,
        "label_smooth": label_smoothing_cross_entropy,
        "focal_smooth": focal_loss_with_label_smoothing,
        "dice": dice_loss,
        "tversky": tversky_loss,
        "standard": lambda logits, labels, reduction="mean": mx.mean(
            nn.losses.cross_entropy(logits, labels, reduction="none")
        ),
    }

    if loss_type == "adaptive":
        return AdaptiveLoss(**kwargs)
    elif loss_type in loss_functions:
        # Return a partial function with the kwargs
        loss_fn = loss_functions[loss_type]
        return lambda logits, labels, reduction="mean": loss_fn(
            logits, labels, reduction=reduction, **kwargs
        )
    else:
        raise ValueError(
            f"Unknown loss type: {loss_type}. "
            f"Choose from: {list(loss_functions.keys()) + ['adaptive']}"
        )


# Convenience function for Titanic dataset with optimal settings
def get_titanic_loss(loss_type: str = "focal") -> callable:
    """
    Get loss function with optimal settings for Titanic dataset.

    Args:
        loss_type: Type of loss function to use

    Returns:
        Configured loss function
    """
    titanic_configs = {
        "focal": {"alpha": 0.39, "gamma": 2.0},  # 39% minority class
        "weighted_ce": {"class_weights": mx.array([1.56, 1.0])},
        "focal_smooth": {"alpha": 0.39, "gamma": 2.0, "smoothing": 0.1},
        "label_smooth": {"smoothing": 0.1},
        "dice": {"smooth": 1.0},
        "tversky": {"alpha": 0.7, "beta": 0.3},
        "adaptive": {
            "initial_alpha": 0.5,
            "final_alpha": 0.39,
            "initial_gamma": 0.0,
            "final_gamma": 2.0,
            "warmup_steps": 500,
        },
    }

    config = titanic_configs.get(loss_type, {})
    return get_loss_function(loss_type, **config)
