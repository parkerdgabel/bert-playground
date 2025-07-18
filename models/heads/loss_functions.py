"""Competition-optimized loss functions for BERT heads.

This module provides loss functions specifically designed for Kaggle competitions,
including focal loss, label smoothing, and other competition-specific optimizations.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Dict, Any, Callable
import numpy as np


class CompetitionLoss(nn.Module):
    """Base class for competition-optimized loss functions."""
    
    def __init__(self, name: str = "competition_loss"):
        super().__init__()
        self.name = name
    
    def __call__(self, predictions: mx.array, targets: mx.array, **kwargs) -> mx.array:
        """Compute loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            **kwargs: Additional arguments
            
        Returns:
            Loss value
        """
        raise NotImplementedError


class FocalLoss(CompetitionLoss):
    """Focal loss for addressing class imbalance.
    
    Paper: "Focal Loss for Dense Object Detection"
    Particularly effective for binary classification with class imbalance.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__("focal_loss")
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def __call__(self, logits: mx.array, targets: mx.array, **kwargs) -> mx.array:
        """Compute focal loss.
        
        Args:
            logits: Raw logits [batch_size, num_classes] or [batch_size] for binary
            targets: Target labels [batch_size]
            
        Returns:
            Focal loss value
        """
        if len(logits.shape) == 1 or logits.shape[-1] == 1:
            # Binary classification
            return self._binary_focal_loss(logits, targets)
        else:
            # Multiclass classification
            return self._multiclass_focal_loss(logits, targets)
    
    def _binary_focal_loss(self, logits: mx.array, targets: mx.array) -> mx.array:
        """Compute binary focal loss."""
        # Ensure logits are 1D
        if len(logits.shape) > 1:
            logits = logits.squeeze(-1)
        
        # Convert to probabilities
        probs = mx.sigmoid(logits)
        targets = targets.astype(mx.float32)
        
        # Compute focal loss components
        pt = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_weight = alpha_t * mx.power(1 - pt, self.gamma)
        
        # Binary cross-entropy
        bce = -(targets * mx.log(probs + 1e-8) + (1 - targets) * mx.log(1 - probs + 1e-8))
        
        # Apply focal weighting
        focal_loss = focal_weight * bce
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
    
    def _multiclass_focal_loss(self, logits: mx.array, targets: mx.array) -> mx.array:
        """Compute multiclass focal loss."""
        # Convert to probabilities
        probs = mx.softmax(logits, axis=-1)
        targets = targets.astype(mx.int32)
        
        # Get probabilities for true classes
        batch_size = logits.shape[0]
        batch_indices = mx.arange(batch_size)
        pt = probs[batch_indices, targets]
        
        # Compute focal loss
        focal_weight = self.alpha * mx.power(1 - pt, self.gamma)
        
        # Cross-entropy
        ce = -mx.log(pt + 1e-8)
        
        # Apply focal weighting
        focal_loss = focal_weight * ce
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(CompetitionLoss):
    """Label smoothing cross-entropy loss.
    
    Helps prevent overconfidence and improves generalization.
    """
    
    def __init__(self, smoothing: float = 0.1, reduction: str = "mean"):
        super().__init__("label_smoothing_loss")
        self.smoothing = smoothing
        self.reduction = reduction
    
    def __call__(self, logits: mx.array, targets: mx.array, **kwargs) -> mx.array:
        """Compute label smoothing loss.
        
        Args:
            logits: Raw logits [batch_size, num_classes]
            targets: Target labels [batch_size]
            
        Returns:
            Label smoothing loss value
        """
        num_classes = logits.shape[-1]
        targets = targets.astype(mx.int32)
        
        # Create smoothed target distribution
        confidence = 1.0 - self.smoothing
        smoothed_targets = mx.ones_like(logits) * (self.smoothing / num_classes)
        
        # Set confidence for true classes
        batch_size = logits.shape[0]
        batch_indices = mx.arange(batch_size)
        smoothed_targets = smoothed_targets.at[batch_indices, targets].set(confidence)
        
        # Compute cross-entropy with smoothed targets
        log_probs = mx.log_softmax(logits, axis=-1)
        loss = -mx.sum(smoothed_targets * log_probs, axis=-1)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class WeightedLoss(CompetitionLoss):
    """Weighted loss for class imbalance.
    
    Applies different weights to different classes based on their frequency.
    """
    
    def __init__(self, weights: Optional[mx.array] = None, base_loss: str = "cross_entropy"):
        super().__init__("weighted_loss")
        self.weights = weights
        self.base_loss = base_loss
        
        # Initialize base loss function
        if base_loss == "cross_entropy":
            self.loss_fn = nn.CrossEntropyLoss(reduction="none")
        elif base_loss == "binary_cross_entropy":
            self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        else:
            raise ValueError(f"Unknown base loss: {base_loss}")
    
    def __call__(self, logits: mx.array, targets: mx.array, **kwargs) -> mx.array:
        """Compute weighted loss.
        
        Args:
            logits: Raw logits
            targets: Target labels
            
        Returns:
            Weighted loss value
        """
        # Compute base loss
        if self.base_loss == "cross_entropy":
            losses = self.loss_fn(logits, targets.astype(mx.int32))
        else:
            losses = self.loss_fn(logits, targets.astype(mx.float32))
        
        # Apply weights if provided
        if self.weights is not None:
            if self.base_loss == "cross_entropy":
                # Use target indices to get weights
                sample_weights = self.weights[targets.astype(mx.int32)]
            else:
                # Binary case: use weights based on target values
                sample_weights = targets * self.weights[1] + (1 - targets) * self.weights[0]
            
            losses = losses * sample_weights
        
        return losses.mean()


class ContrastiveLoss(CompetitionLoss):
    """Contrastive loss for similarity learning.
    
    Useful for ranking and recommendation tasks.
    """
    
    def __init__(self, margin: float = 1.0, reduction: str = "mean"):
        super().__init__("contrastive_loss")
        self.margin = margin
        self.reduction = reduction
    
    def __call__(self, embeddings1: mx.array, embeddings2: mx.array, targets: mx.array, **kwargs) -> mx.array:
        """Compute contrastive loss.
        
        Args:
            embeddings1: First set of embeddings [batch_size, embedding_dim]
            embeddings2: Second set of embeddings [batch_size, embedding_dim]
            targets: Binary targets (1 for similar, 0 for dissimilar) [batch_size]
            
        Returns:
            Contrastive loss value
        """
        # Compute euclidean distance
        diff = embeddings1 - embeddings2
        distance = mx.sqrt(mx.sum(mx.square(diff), axis=-1) + 1e-8)
        
        # Compute contrastive loss
        targets = targets.astype(mx.float32)
        
        # Loss for similar pairs (minimize distance)
        positive_loss = targets * mx.square(distance)
        
        # Loss for dissimilar pairs (maximize distance up to margin)
        negative_loss = (1 - targets) * mx.square(mx.maximum(0, self.margin - distance))
        
        loss = positive_loss + negative_loss
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class TripletLoss(CompetitionLoss):
    """Triplet loss for ranking and similarity learning.
    
    Ensures that positive pairs are closer than negative pairs by a margin.
    """
    
    def __init__(self, margin: float = 1.0, reduction: str = "mean"):
        super().__init__("triplet_loss")
        self.margin = margin
        self.reduction = reduction
    
    def __call__(self, anchor: mx.array, positive: mx.array, negative: mx.array, **kwargs) -> mx.array:
        """Compute triplet loss.
        
        Args:
            anchor: Anchor embeddings [batch_size, embedding_dim]
            positive: Positive embeddings [batch_size, embedding_dim]
            negative: Negative embeddings [batch_size, embedding_dim]
            
        Returns:
            Triplet loss value
        """
        # Compute distances
        pos_distance = mx.sum(mx.square(anchor - positive), axis=-1)
        neg_distance = mx.sum(mx.square(anchor - negative), axis=-1)
        
        # Compute triplet loss
        loss = mx.maximum(0, pos_distance - neg_distance + self.margin)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class RankingLoss(CompetitionLoss):
    """Ranking loss for learning to rank problems.
    
    Optimizes for ranking metrics like NDCG and MAP.
    """
    
    def __init__(self, margin: float = 1.0, reduction: str = "mean"):
        super().__init__("ranking_loss")
        self.margin = margin
        self.reduction = reduction
    
    def __call__(self, scores: mx.array, targets: mx.array, **kwargs) -> mx.array:
        """Compute ranking loss.
        
        Args:
            scores: Predicted scores [batch_size, num_items]
            targets: Relevance targets [batch_size, num_items]
            
        Returns:
            Ranking loss value
        """
        # Pairwise ranking loss
        batch_size, num_items = scores.shape
        
        # Create all pairs
        scores_i = scores.reshape(batch_size, num_items, 1)
        scores_j = scores.reshape(batch_size, 1, num_items)
        
        targets_i = targets.reshape(batch_size, num_items, 1)
        targets_j = targets.reshape(batch_size, 1, num_items)
        
        # Compute pairwise differences
        score_diff = scores_i - scores_j
        target_diff = targets_i - targets_j
        
        # Only consider pairs where targets differ
        valid_pairs = mx.abs(target_diff) > 0
        
        # Compute hinge loss for valid pairs
        # If target_i > target_j, then score_i should be > score_j
        should_rank_higher = target_diff > 0
        
        # Hinge loss: max(0, margin - score_diff) when should_rank_higher
        hinge_loss = mx.maximum(0, self.margin - score_diff)
        ranking_loss = mx.where(should_rank_higher & valid_pairs, hinge_loss, 0)
        
        # Average over valid pairs
        num_valid_pairs = valid_pairs.sum() + 1e-8
        total_loss = ranking_loss.sum() / num_valid_pairs
        
        if self.reduction == "mean":
            return total_loss.mean()
        elif self.reduction == "sum":
            return total_loss.sum()
        else:
            return total_loss


class UncertaintyLoss(CompetitionLoss):
    """Uncertainty-aware loss for regression with heteroscedastic uncertainty.
    
    Learns both the prediction and its uncertainty.
    """
    
    def __init__(self, reduction: str = "mean"):
        super().__init__("uncertainty_loss")
        self.reduction = reduction
    
    def __call__(self, predictions: mx.array, log_variance: mx.array, targets: mx.array, **kwargs) -> mx.array:
        """Compute uncertainty loss.
        
        Args:
            predictions: Predicted values [batch_size, output_dim]
            log_variance: Log variance predictions [batch_size, output_dim]
            targets: Ground truth targets [batch_size, output_dim]
            
        Returns:
            Uncertainty loss value
        """
        # Compute variance
        variance = mx.exp(log_variance)
        
        # Uncertainty loss: combines prediction error with uncertainty
        squared_error = mx.square(predictions - targets)
        uncertainty_loss = 0.5 * (squared_error / variance + log_variance)
        
        if self.reduction == "mean":
            return uncertainty_loss.mean()
        elif self.reduction == "sum":
            return uncertainty_loss.sum()
        else:
            return uncertainty_loss


class DistillationLoss(CompetitionLoss):
    """Knowledge distillation loss for model compression.
    
    Combines student loss with teacher knowledge transfer.
    """
    
    def __init__(self, alpha: float = 0.7, temperature: float = 3.0, reduction: str = "mean"):
        super().__init__("distillation_loss")
        self.alpha = alpha
        self.temperature = temperature
        self.reduction = reduction
        
        # Base loss for student
        self.student_loss = nn.CrossEntropyLoss(reduction=reduction)
    
    def __call__(self, student_logits: mx.array, teacher_logits: mx.array, targets: mx.array, **kwargs) -> mx.array:
        """Compute distillation loss.
        
        Args:
            student_logits: Student model logits [batch_size, num_classes]
            teacher_logits: Teacher model logits [batch_size, num_classes]
            targets: Ground truth targets [batch_size]
            
        Returns:
            Distillation loss value
        """
        # Student loss (standard cross-entropy)
        student_loss = self.student_loss(student_logits, targets.astype(mx.int32))
        
        # Distillation loss (KL divergence between student and teacher)
        # Apply temperature scaling
        student_soft = mx.softmax(student_logits / self.temperature, axis=-1)
        teacher_soft = mx.softmax(teacher_logits / self.temperature, axis=-1)
        
        # KL divergence
        kl_loss = mx.sum(teacher_soft * mx.log(teacher_soft / (student_soft + 1e-8) + 1e-8), axis=-1)
        
        if self.reduction == "mean":
            kl_loss = kl_loss.mean()
        elif self.reduction == "sum":
            kl_loss = kl_loss.sum()
        
        # Combine losses
        total_loss = (1 - self.alpha) * student_loss + self.alpha * kl_loss * (self.temperature ** 2)
        
        return total_loss


# Loss factory for easy creation
class LossFactory:
    """Factory for creating competition-optimized loss functions."""
    
    _LOSS_REGISTRY = {
        "focal": FocalLoss,
        "label_smoothing": LabelSmoothingLoss,
        "weighted": WeightedLoss,
        "contrastive": ContrastiveLoss,
        "triplet": TripletLoss,
        "ranking": RankingLoss,
        "uncertainty": UncertaintyLoss,
        "distillation": DistillationLoss,
    }
    
    @classmethod
    def create_loss(cls, loss_type: str, **kwargs) -> CompetitionLoss:
        """Create a loss function by name.
        
        Args:
            loss_type: Name of the loss function
            **kwargs: Arguments for the loss function
            
        Returns:
            Loss function instance
        """
        if loss_type not in cls._LOSS_REGISTRY:
            raise ValueError(f"Unknown loss type: {loss_type}. Available: {list(cls._LOSS_REGISTRY.keys())}")
        
        return cls._LOSS_REGISTRY[loss_type](**kwargs)
    
    @classmethod
    def register_loss(cls, name: str, loss_class: type):
        """Register a new loss function.
        
        Args:
            name: Name of the loss function
            loss_class: Loss function class
        """
        cls._LOSS_REGISTRY[name] = loss_class
    
    @classmethod
    def list_losses(cls) -> list:
        """List all available loss functions."""
        return list(cls._LOSS_REGISTRY.keys())


# Utility functions for loss computation
def compute_class_weights(targets: mx.array, method: str = "inverse_freq") -> mx.array:
    """Compute class weights for imbalanced datasets.
    
    Args:
        targets: Target labels [batch_size]
        method: Method for computing weights ("inverse_freq", "effective_num")
        
    Returns:
        Class weights [num_classes]
    """
    targets_np = np.array(targets)
    unique_classes, counts = np.unique(targets_np, return_counts=True)
    
    if method == "inverse_freq":
        # Inverse frequency weighting
        weights = 1.0 / counts
        weights = weights / weights.sum() * len(unique_classes)  # Normalize
    elif method == "effective_num":
        # Effective number of samples
        beta = 0.999
        effective_num = (1 - beta ** counts) / (1 - beta)
        weights = 1.0 / effective_num
        weights = weights / weights.sum() * len(unique_classes)  # Normalize
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    
    return mx.array(weights)


def get_competition_loss(
    competition_type: str,
    num_classes: int,
    class_weights: Optional[mx.array] = None,
    **kwargs
) -> CompetitionLoss:
    """Get optimized loss function for a competition type.
    
    Args:
        competition_type: Type of competition
        num_classes: Number of classes
        class_weights: Optional class weights
        **kwargs: Additional arguments
        
    Returns:
        Optimized loss function
    """
    if competition_type == "binary_classification":
        if class_weights is not None:
            return LossFactory.create_loss("weighted", weights=class_weights, base_loss="binary_cross_entropy")
        else:
            return LossFactory.create_loss("focal", alpha=0.25, gamma=2.0)
    
    elif competition_type == "multiclass_classification":
        if class_weights is not None:
            return LossFactory.create_loss("weighted", weights=class_weights, base_loss="cross_entropy")
        else:
            return LossFactory.create_loss("label_smoothing", smoothing=0.1)
    
    elif competition_type == "multilabel_classification":
        return LossFactory.create_loss("focal", alpha=0.25, gamma=2.0)
    
    elif competition_type == "regression":
        return UncertaintyLoss()
    
    elif competition_type == "ranking":
        return LossFactory.create_loss("ranking", margin=1.0)
    
    else:
        # Default to cross-entropy with label smoothing
        return LossFactory.create_loss("label_smoothing", smoothing=0.1)