"""Utilities module for BERT heads.

This module provides loss functions, metrics, and other utilities.
"""

from .losses import (
    ContrastiveLoss,
    # Loss classes
    FocalLoss,
    LabelSmoothingLoss,
    TripletLoss,
    # Base loss functions
    binary_cross_entropy_loss,
    # Utilities
    compute_class_weights,
    create_loss_function,
    cross_entropy_loss,
    multilabel_bce_loss,
)
from .metrics import (
    # Classification metrics
    accuracy,
    auc,
    # Utility functions
    compute_metrics,
    f1_score,
    get_metrics_for_task,
    # Multilabel metrics
    hamming_loss,
    # Ordinal regression metrics
    kendall_tau,
    # Regression metrics
    mae,
    mape,
    # Ranking metrics
    mean_average_precision,
    mse,
    ndcg,
    precision,
    r2_score,
    recall,
    rmse,
    subset_accuracy,
)

__all__ = [
    # Loss functions
    "binary_cross_entropy_loss",
    "cross_entropy_loss",
    "multilabel_bce_loss",
    "FocalLoss",
    "LabelSmoothingLoss",
    "ContrastiveLoss",
    "TripletLoss",
    "compute_class_weights",
    "create_loss_function",
    # Metrics
    "accuracy",
    "auc",
    "f1_score",
    "precision",
    "recall",
    "mae",
    "mse",
    "rmse",
    "r2_score",
    "mape",
    "mean_average_precision",
    "ndcg",
    "hamming_loss",
    "subset_accuracy",
    "kendall_tau",
    "compute_metrics",
    "get_metrics_for_task",
]
