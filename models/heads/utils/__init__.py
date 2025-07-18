"""Utilities module for BERT heads.

This module provides loss functions, metrics, and other utilities.
"""

from .losses import (
    # Base loss functions
    binary_cross_entropy_loss,
    cross_entropy_loss,
    multilabel_bce_loss,
    # Loss classes
    FocalLoss,
    LabelSmoothingLoss,
    ContrastiveLoss,
    TripletLoss,
    # Utilities
    compute_class_weights,
    create_loss_function,
)

from .metrics import (
    # Classification metrics
    accuracy,
    auc,
    f1_score,
    precision,
    recall,
    # Regression metrics
    mae,
    mse,
    rmse,
    r2_score,
    mape,
    # Ranking metrics
    mean_average_precision,
    ndcg,
    # Multilabel metrics
    hamming_loss,
    subset_accuracy,
    # Ordinal regression metrics
    kendall_tau,
    # Utility functions
    compute_metrics,
    get_metrics_for_task,
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