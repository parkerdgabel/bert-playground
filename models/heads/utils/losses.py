"""Loss functions for BERT heads.

This module provides various loss functions commonly used in Kaggle competitions
and machine learning tasks.
"""

from collections.abc import Callable

import mlx.core as mx
import mlx.nn as nn

# Base loss functions


def binary_cross_entropy_loss(
    predictions: mx.array, targets: mx.array, reduction: str = "mean"
) -> mx.array:
    """Binary cross-entropy loss.

    Args:
        predictions: Model predictions (logits or probabilities)
        targets: Binary targets (0 or 1)
        reduction: Reduction method ('mean', 'sum', or 'none')

    Returns:
        Loss value
    """
    # Ensure predictions are in probability space
    if len(predictions.shape) > 1 and predictions.shape[-1] == 2:
        # Two-class output, use softmax
        probs = mx.softmax(predictions, axis=-1)
        probs = probs[:, 1]  # Probability of positive class
    else:
        # Single output, use sigmoid (already 1D, no need to squeeze)
        probs = mx.sigmoid(predictions)

    # Clip for numerical stability
    probs = mx.clip(probs, 1e-7, 1 - 1e-7)

    # Compute loss
    loss = -targets * mx.log(probs) - (1 - targets) * mx.log(1 - probs)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


def cross_entropy_loss(
    predictions: mx.array,
    targets: mx.array,
    num_classes: int | None = None,
    reduction: str = "mean",
) -> mx.array:
    """Cross-entropy loss for multiclass classification.

    Args:
        predictions: Model predictions (logits) of shape [batch_size, num_classes]
        targets: Target class indices of shape [batch_size]
        num_classes: Number of classes (inferred if not provided)
        reduction: Reduction method ('mean', 'sum', or 'none')

    Returns:
        Loss value
    """
    if num_classes is None:
        num_classes = predictions.shape[-1]

    # Apply softmax to get probabilities
    log_probs = mx.log_softmax(predictions, axis=-1)

    # Convert targets to one-hot if needed
    if targets.ndim == 1:
        targets_one_hot = mx.zeros((targets.shape[0], num_classes))
        targets_one_hot[mx.arange(targets.shape[0]), targets] = 1
    else:
        targets_one_hot = targets

    # Compute loss
    loss = -(targets_one_hot * log_probs).sum(axis=-1)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


def multilabel_bce_loss(
    predictions: mx.array, targets: mx.array, reduction: str = "mean"
) -> mx.array:
    """Binary cross-entropy loss for multilabel classification.

    Args:
        predictions: Model predictions (logits) of shape [batch_size, num_labels]
        targets: Binary targets of shape [batch_size, num_labels]
        reduction: Reduction method ('mean', 'sum', or 'none')

    Returns:
        Loss value
    """
    probs = mx.sigmoid(predictions)
    probs = mx.clip(probs, 1e-7, 1 - 1e-7)

    # Compute BCE for each label
    loss = -targets * mx.log(probs) - (1 - targets) * mx.log(1 - probs)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


# Specialized loss functions


class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance.

    Focal loss applies a modulating term to the cross-entropy loss to focus
    learning on hard misclassified examples.
    """

    def __init__(
        self,
        alpha: float | mx.array | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        num_classes: int | None = None,
    ):
        """Initialize focal loss.

        Args:
            alpha: Weighting factor in [0, 1] to balance classes
            gamma: Focusing parameter (γ ≥ 0)
            reduction: Reduction method ('mean', 'sum', or 'none')
            num_classes: Number of classes for multiclass problems
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.num_classes = num_classes

    def __call__(self, predictions: mx.array, targets: mx.array) -> mx.array:
        """Apply focal loss.

        Args:
            predictions: Model predictions (logits)
            targets: Ground truth targets

        Returns:
            Loss value
        """
        if self.num_classes is None and predictions.shape[-1] > 1:
            self.num_classes = predictions.shape[-1]

        if self.num_classes and self.num_classes > 2:
            # Multiclass focal loss
            return self._multiclass_focal_loss(predictions, targets)
        else:
            # Binary focal loss
            return self._binary_focal_loss(predictions, targets)

    def _binary_focal_loss(self, predictions: mx.array, targets: mx.array) -> mx.array:
        """Compute binary focal loss."""
        # Get probabilities
        if predictions.shape[-1] == 2:
            probs = mx.softmax(predictions, axis=-1)[:, 1]
        else:
            probs = mx.sigmoid(predictions.squeeze(-1))

        probs = mx.clip(probs, 1e-7, 1 - 1e-7)

        # Compute focal weights
        pt = mx.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma

        # Compute loss
        ce_loss = -targets * mx.log(probs) - (1 - targets) * mx.log(1 - probs)
        loss = focal_weight * ce_loss

        # Apply alpha weighting if specified
        if self.alpha is not None:
            alpha_t = mx.where(targets == 1, self.alpha, 1 - self.alpha)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

    def _multiclass_focal_loss(
        self, predictions: mx.array, targets: mx.array
    ) -> mx.array:
        """Compute multiclass focal loss."""
        # Get probabilities
        probs = mx.softmax(predictions, axis=-1)

        # Convert targets to one-hot if needed
        if targets.ndim == 1:
            targets_one_hot = mx.zeros_like(probs)
            targets_one_hot[mx.arange(targets.shape[0]), targets] = 1
        else:
            targets_one_hot = targets

        # Get probabilities for true class
        pt = (probs * targets_one_hot).sum(axis=-1)
        pt = mx.clip(pt, 1e-7, 1 - 1e-7)

        # Compute focal weight
        focal_weight = (1 - pt) ** self.gamma

        # Compute cross-entropy
        log_probs = mx.log(probs)
        ce_loss = -(targets_one_hot * log_probs).sum(axis=-1)

        # Apply focal weight
        loss = focal_weight * ce_loss

        # Apply alpha weighting if specified
        if self.alpha is not None:
            if (
                isinstance(self.alpha, mx.array)
                and self.alpha.shape[0] == self.num_classes
            ):
                # Per-class alpha
                alpha_t = (self.alpha * targets_one_hot).sum(axis=-1)
            else:
                # Single alpha value
                alpha_t = self.alpha
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class LabelSmoothingLoss(nn.Module):
    """Cross-entropy loss with label smoothing.

    Label smoothing prevents the model from becoming too confident by
    distributing some probability mass to other classes.
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        num_classes: int | None = None,
        reduction: str = "mean",
    ):
        """Initialize label smoothing loss.

        Args:
            smoothing: Smoothing parameter in [0, 1)
            num_classes: Number of classes
            reduction: Reduction method
        """
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.reduction = reduction

    def __call__(self, predictions: mx.array, targets: mx.array) -> mx.array:
        """Apply label smoothing loss.

        Args:
            predictions: Model predictions (logits)
            targets: Ground truth targets

        Returns:
            Loss value
        """
        if self.num_classes is None:
            self.num_classes = predictions.shape[-1]

        # Apply log softmax
        log_probs = mx.log_softmax(predictions, axis=-1)

        # Convert targets to one-hot if needed
        if targets.ndim == 1:
            targets_one_hot = mx.zeros((targets.shape[0], self.num_classes))
            targets_one_hot[mx.arange(targets.shape[0]), targets] = 1
        else:
            targets_one_hot = targets

        # Apply label smoothing
        smoothed_targets = (
            1.0 - self.smoothing
        ) * targets_one_hot + self.smoothing / self.num_classes

        # Compute loss
        loss = -(smoothed_targets * log_probs).sum(axis=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class ContrastiveLoss(nn.Module):
    """Contrastive loss for similarity learning.

    This loss encourages similar pairs to have small distances and
    dissimilar pairs to have large distances.
    """

    def __init__(self, margin: float = 1.0, reduction: str = "mean"):
        """Initialize contrastive loss.

        Args:
            margin: Margin for dissimilar pairs
            reduction: Reduction method
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def __call__(
        self, embeddings1: mx.array, embeddings2: mx.array, labels: mx.array
    ) -> mx.array:
        """Apply contrastive loss.

        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            labels: Binary labels (1 for similar, 0 for dissimilar)

        Returns:
            Loss value
        """
        # Compute euclidean distance
        distances = mx.sqrt(((embeddings1 - embeddings2) ** 2).sum(axis=-1) + 1e-8)

        # Compute loss
        positive_loss = labels * distances**2
        negative_loss = (1 - labels) * mx.maximum(0, self.margin - distances) ** 2

        loss = 0.5 * (positive_loss + negative_loss)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class TripletLoss(nn.Module):
    """Triplet loss for ranking and similarity learning.

    This loss operates on triplets of (anchor, positive, negative) examples.
    """

    def __init__(self, margin: float = 1.0, reduction: str = "mean"):
        """Initialize triplet loss.

        Args:
            margin: Margin between positive and negative distances
            reduction: Reduction method
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def __call__(
        self, anchor: mx.array, positive: mx.array, negative: mx.array
    ) -> mx.array:
        """Apply triplet loss.

        Args:
            anchor: Anchor embeddings
            positive: Positive embeddings (similar to anchor)
            negative: Negative embeddings (dissimilar to anchor)

        Returns:
            Loss value
        """
        # Compute distances
        pos_dist = mx.sqrt(((anchor - positive) ** 2).sum(axis=-1) + 1e-8)
        neg_dist = mx.sqrt(((anchor - negative) ** 2).sum(axis=-1) + 1e-8)

        # Compute loss
        loss = mx.maximum(0, pos_dist - neg_dist + self.margin)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


# Loss factory and utilities


def compute_class_weights(targets: mx.array, num_classes: int) -> mx.array:
    """Compute class weights for imbalanced datasets.

    Args:
        targets: Target labels
        num_classes: Number of classes

    Returns:
        Class weights array
    """
    class_counts = mx.zeros(num_classes)

    # Count occurrences of each class
    for i in range(num_classes):
        class_counts[i] = (targets == i).sum()

    # Compute weights (inverse frequency)
    total_samples = targets.shape[0]
    class_weights = total_samples / (num_classes * class_counts + 1e-8)

    # Normalize
    class_weights = class_weights / class_weights.mean()

    return class_weights


def create_loss_function(loss_type: str, **kwargs) -> nn.Module | Callable:
    """Create a loss function by name.

    Args:
        loss_type: Type of loss function
        **kwargs: Additional arguments for the loss function

    Returns:
        Loss function

    Raises:
        ValueError: If loss type is unknown
    """
    loss_type = loss_type.lower()

    if loss_type == "bce" or loss_type == "binary_cross_entropy":
        return lambda p, t: binary_cross_entropy_loss(p, t, **kwargs)
    elif loss_type == "ce" or loss_type == "cross_entropy":
        return lambda p, t: cross_entropy_loss(p, t, **kwargs)
    elif loss_type == "multilabel_bce":
        return lambda p, t: multilabel_bce_loss(p, t, **kwargs)
    elif loss_type == "focal":
        return FocalLoss(**kwargs)
    elif loss_type == "label_smoothing":
        return LabelSmoothingLoss(**kwargs)
    elif loss_type == "contrastive":
        return ContrastiveLoss(**kwargs)
    elif loss_type == "triplet":
        return TripletLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


__all__ = [
    # Base loss functions
    "binary_cross_entropy_loss",
    "cross_entropy_loss",
    "multilabel_bce_loss",
    # Loss classes
    "FocalLoss",
    "LabelSmoothingLoss",
    "ContrastiveLoss",
    "TripletLoss",
    # Utilities
    "compute_class_weights",
    "create_loss_function",
]
