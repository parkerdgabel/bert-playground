"""Evaluation metrics for BERT heads.

This module provides various evaluation metrics commonly used in
machine learning competitions.
"""

import mlx.core as mx
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any


def _to_numpy(array: Union[mx.array, np.ndarray]) -> np.ndarray:
    """Convert MLX array to numpy if needed."""
    if isinstance(array, mx.array):
        return np.array(array)
    return array


# Classification metrics

def accuracy(predictions: Union[mx.array, np.ndarray], targets: Union[mx.array, np.ndarray]) -> float:
    """Compute accuracy.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        
    Returns:
        Accuracy score
    """
    pred_np = _to_numpy(predictions)
    targets_np = _to_numpy(targets)
    
    return float(np.mean(pred_np == targets_np))


def auc(probabilities: Union[mx.array, np.ndarray], targets: Union[mx.array, np.ndarray]) -> float:
    """Compute Area Under the ROC Curve.
    
    Args:
        probabilities: Model predicted probabilities
        targets: Binary ground truth targets
        
    Returns:
        AUC score
    """
    probs_np = _to_numpy(probabilities)
    targets_np = _to_numpy(targets)
    
    if len(np.unique(targets_np)) < 2:
        return 0.5  # No positive or negative examples
    
    # Sort by probability
    sorted_indices = np.argsort(probs_np)
    sorted_probs = probs_np[sorted_indices]
    sorted_targets = targets_np[sorted_indices]
    
    # Compute TPR and FPR
    n_pos = (targets_np == 1).sum()
    n_neg = (targets_np == 0).sum()
    
    if n_pos == 0 or n_neg == 0:
        return 0.5
    
    # Compute TPR and FPR at each threshold
    tpr = []
    fpr = []
    
    for i in range(len(sorted_probs)):
        threshold = sorted_probs[i]
        pred_pos = (sorted_probs >= threshold)
        
        tp = (pred_pos & (sorted_targets == 1)).sum()
        fp = (pred_pos & (sorted_targets == 0)).sum()
        
        tpr.append(tp / n_pos)
        fpr.append(fp / n_neg)
    
    # Add boundary points
    tpr = [0] + tpr + [1]
    fpr = [0] + fpr + [1]
    
    # Compute AUC using trapezoidal rule
    auc_score = 0.0
    for i in range(len(tpr) - 1):
        auc_score += (fpr[i + 1] - fpr[i]) * (tpr[i + 1] + tpr[i]) / 2
    
    return max(0.0, min(1.0, auc_score))


def f1_score(
    predictions: Union[mx.array, np.ndarray],
    targets: Union[mx.array, np.ndarray],
    average: str = "binary"
) -> float:
    """Compute F1 score.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        average: Averaging strategy ('binary', 'macro', 'micro')
        
    Returns:
        F1 score
    """
    pred_np = _to_numpy(predictions)
    targets_np = _to_numpy(targets)
    
    if average == "binary":
        tp = ((pred_np == 1) & (targets_np == 1)).sum()
        fp = ((pred_np == 1) & (targets_np == 0)).sum()
        fn = ((pred_np == 0) & (targets_np == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    elif average == "macro":
        unique_classes = np.unique(np.concatenate([pred_np, targets_np]))
        f1_scores = []
        
        for cls in unique_classes:
            tp = ((pred_np == cls) & (targets_np == cls)).sum()
            fp = ((pred_np == cls) & (targets_np != cls)).sum()
            fn = ((pred_np != cls) & (targets_np == cls)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            f1_scores.append(f1)
        
        return np.mean(f1_scores)
    
    elif average == "micro":
        # For multiclass, micro F1 = accuracy
        if len(np.unique(targets_np)) > 2:
            return float((pred_np == targets_np).mean())
        else:
            # Binary case
            tp = ((pred_np == 1) & (targets_np == 1)).sum()
            fp = ((pred_np == 1) & (targets_np == 0)).sum()
            fn = ((pred_np == 0) & (targets_np == 1)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    else:
        raise ValueError(f"Unknown average type: {average}")


def precision(
    predictions: Union[mx.array, np.ndarray],
    targets: Union[mx.array, np.ndarray],
    average: str = "binary"
) -> float:
    """Compute precision.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        average: Averaging strategy ('binary', 'macro')
        
    Returns:
        Precision score
    """
    pred_np = _to_numpy(predictions)
    targets_np = _to_numpy(targets)
    
    if average == "binary":
        tp = ((pred_np == 1) & (targets_np == 1)).sum()
        fp = ((pred_np == 1) & (targets_np == 0)).sum()
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    elif average == "macro":
        unique_classes = np.unique(np.concatenate([pred_np, targets_np]))
        precisions = []
        
        for cls in unique_classes:
            tp = ((pred_np == cls) & (targets_np == cls)).sum()
            fp = ((pred_np == cls) & (targets_np != cls)).sum()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            precisions.append(prec)
        
        return np.mean(precisions)
    else:
        raise ValueError(f"Unknown average type: {average}")


def recall(
    predictions: Union[mx.array, np.ndarray],
    targets: Union[mx.array, np.ndarray],
    average: str = "binary"
) -> float:
    """Compute recall.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        average: Averaging strategy ('binary', 'macro')
        
    Returns:
        Recall score
    """
    pred_np = _to_numpy(predictions)
    targets_np = _to_numpy(targets)
    
    if average == "binary":
        tp = ((pred_np == 1) & (targets_np == 1)).sum()
        fn = ((pred_np == 0) & (targets_np == 1)).sum()
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    elif average == "macro":
        unique_classes = np.unique(np.concatenate([pred_np, targets_np]))
        recalls = []
        
        for cls in unique_classes:
            tp = ((pred_np == cls) & (targets_np == cls)).sum()
            fn = ((pred_np != cls) & (targets_np == cls)).sum()
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            recalls.append(rec)
        
        return np.mean(recalls)
    else:
        raise ValueError(f"Unknown average type: {average}")


# Regression metrics

def mae(predictions: Union[mx.array, np.ndarray], targets: Union[mx.array, np.ndarray]) -> float:
    """Compute Mean Absolute Error.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        
    Returns:
        MAE score
    """
    pred_np = _to_numpy(predictions)
    targets_np = _to_numpy(targets)
    
    return float(np.mean(np.abs(pred_np - targets_np)))


def mse(predictions: Union[mx.array, np.ndarray], targets: Union[mx.array, np.ndarray]) -> float:
    """Compute Mean Squared Error.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        
    Returns:
        MSE score
    """
    pred_np = _to_numpy(predictions)
    targets_np = _to_numpy(targets)
    
    return float(np.mean((pred_np - targets_np) ** 2))


def rmse(predictions: Union[mx.array, np.ndarray], targets: Union[mx.array, np.ndarray]) -> float:
    """Compute Root Mean Squared Error.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        
    Returns:
        RMSE score
    """
    pred_np = _to_numpy(predictions)
    targets_np = _to_numpy(targets)
    
    return float(np.sqrt(np.mean((pred_np - targets_np) ** 2)))


def r2_score(predictions: Union[mx.array, np.ndarray], targets: Union[mx.array, np.ndarray]) -> float:
    """Compute R-squared score.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        
    Returns:
        R2 score
    """
    pred_np = _to_numpy(predictions)
    targets_np = _to_numpy(targets)
    
    ss_res = np.sum((targets_np - pred_np) ** 2)
    ss_tot = np.sum((targets_np - np.mean(targets_np)) ** 2)
    
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


def mape(predictions: Union[mx.array, np.ndarray], targets: Union[mx.array, np.ndarray]) -> float:
    """Compute Mean Absolute Percentage Error.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        
    Returns:
        MAPE score (as percentage)
    """
    pred_np = _to_numpy(predictions)
    targets_np = _to_numpy(targets)
    
    # Avoid division by zero
    mask = targets_np != 0
    if not np.any(mask):
        return 0.0
    
    return float(np.mean(np.abs((targets_np[mask] - pred_np[mask]) / targets_np[mask])) * 100)


# Ranking metrics

def mean_average_precision(scores: Union[mx.array, np.ndarray], targets: Union[mx.array, np.ndarray]) -> float:
    """Compute Mean Average Precision.
    
    Args:
        scores: Ranking scores
        targets: Binary relevance labels
        
    Returns:
        MAP score
    """
    scores_np = _to_numpy(scores)
    targets_np = _to_numpy(targets)
    
    # Sort by scores in descending order
    sorted_indices = np.argsort(scores_np)[::-1]
    sorted_targets = targets_np[sorted_indices]
    
    # Compute average precision
    relevant_count = 0
    precision_sum = 0.0
    
    for i, is_relevant in enumerate(sorted_targets):
        if is_relevant:
            relevant_count += 1
            precision_at_i = relevant_count / (i + 1)
            precision_sum += precision_at_i
    
    total_relevant = np.sum(targets_np)
    return precision_sum / total_relevant if total_relevant > 0 else 0.0


def ndcg(scores: Union[mx.array, np.ndarray], targets: Union[mx.array, np.ndarray], k: int = 10) -> float:
    """Compute Normalized Discounted Cumulative Gain.
    
    Args:
        scores: Ranking scores
        targets: Relevance labels
        k: Cutoff position
        
    Returns:
        NDCG@k score
    """
    scores_np = _to_numpy(scores)
    targets_np = _to_numpy(targets)
    
    # Sort by scores in descending order
    sorted_indices = np.argsort(scores_np)[::-1]
    sorted_targets = targets_np[sorted_indices]
    
    # Compute DCG@k
    dcg = 0.0
    for i in range(min(k, len(sorted_targets))):
        relevance = sorted_targets[i]
        dcg += (2 ** relevance - 1) / np.log2(i + 2)
    
    # Compute IDCG@k (ideal DCG)
    ideal_targets = np.sort(targets_np)[::-1]
    idcg = 0.0
    for i in range(min(k, len(ideal_targets))):
        relevance = ideal_targets[i]
        idcg += (2 ** relevance - 1) / np.log2(i + 2)
    
    return dcg / idcg if idcg > 0 else 0.0


# Multilabel metrics

def hamming_loss(predictions: Union[mx.array, np.ndarray], targets: Union[mx.array, np.ndarray]) -> float:
    """Compute Hamming loss for multilabel classification.
    
    Args:
        predictions: Binary predictions
        targets: Binary targets
        
    Returns:
        Hamming loss
    """
    pred_np = _to_numpy(predictions)
    targets_np = _to_numpy(targets)
    
    return float(np.mean(pred_np != targets_np))


def subset_accuracy(predictions: Union[mx.array, np.ndarray], targets: Union[mx.array, np.ndarray]) -> float:
    """Compute subset accuracy for multilabel classification.
    
    Args:
        predictions: Binary predictions [batch_size, num_labels]
        targets: Binary targets [batch_size, num_labels]
        
    Returns:
        Subset accuracy
    """
    pred_np = _to_numpy(predictions)
    targets_np = _to_numpy(targets)
    
    return float(np.mean(np.all(pred_np == targets_np, axis=1)))


# Ordinal regression metrics

def kendall_tau(predictions: Union[mx.array, np.ndarray], targets: Union[mx.array, np.ndarray]) -> float:
    """Compute Kendall's tau for ordinal regression.
    
    Args:
        predictions: Predicted ordinal values
        targets: True ordinal values
        
    Returns:
        Kendall's tau coefficient
    """
    pred_np = _to_numpy(predictions)
    targets_np = _to_numpy(targets)
    
    try:
        from scipy.stats import kendalltau
        tau, _ = kendalltau(pred_np, targets_np)
        return tau if not np.isnan(tau) else 0.0
    except ImportError:
        # Fallback to simple concordance calculation
        n = len(pred_np)
        concordant = 0
        discordant = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                pred_diff = pred_np[i] - pred_np[j]
                target_diff = targets_np[i] - targets_np[j]
                
                if pred_diff * target_diff > 0:
                    concordant += 1
                elif pred_diff * target_diff < 0:
                    discordant += 1
        
        total_pairs = n * (n - 1) / 2
        return (concordant - discordant) / total_pairs if total_pairs > 0 else 0.0


# Utility functions

def compute_metrics(
    predictions: Dict[str, Union[mx.array, np.ndarray]],
    targets: Union[mx.array, np.ndarray],
    metrics: List[str],
    **kwargs
) -> Dict[str, float]:
    """Compute multiple metrics.
    
    Args:
        predictions: Dictionary with prediction arrays
        targets: Ground truth targets
        metrics: List of metric names to compute
        **kwargs: Additional arguments for specific metrics
        
    Returns:
        Dictionary of metric values
    """
    results = {}
    
    for metric_name in metrics:
        try:
            if metric_name == "accuracy":
                pred = predictions.get("predictions", predictions.get("logits"))
                if pred is not None:
                    results[metric_name] = accuracy(pred, targets)
                    
            elif metric_name == "auc":
                prob = predictions.get("probabilities", predictions.get("logits"))
                if prob is not None:
                    if len(prob.shape) > 1 and prob.shape[-1] == 2:
                        prob = prob[:, 1]  # Use positive class probability
                    results[metric_name] = auc(prob, targets)
                    
            elif metric_name.startswith("f1"):
                pred = predictions.get("predictions", predictions.get("logits"))
                if pred is not None:
                    average = metric_name.split("_")[1] if "_" in metric_name else "binary"
                    results[metric_name] = f1_score(pred, targets, average)
                    
            elif metric_name.startswith("precision"):
                pred = predictions.get("predictions", predictions.get("logits"))
                if pred is not None:
                    average = metric_name.split("_")[1] if "_" in metric_name else "binary"
                    results[metric_name] = precision(pred, targets, average)
                    
            elif metric_name.startswith("recall"):
                pred = predictions.get("predictions", predictions.get("logits"))
                if pred is not None:
                    average = metric_name.split("_")[1] if "_" in metric_name else "binary"
                    results[metric_name] = recall(pred, targets, average)
                    
            elif metric_name == "mae":
                pred = predictions.get("predictions", predictions.get("logits"))
                if pred is not None:
                    results[metric_name] = mae(pred, targets)
                    
            elif metric_name == "mse":
                pred = predictions.get("predictions", predictions.get("logits"))
                if pred is not None:
                    results[metric_name] = mse(pred, targets)
                    
            elif metric_name == "rmse":
                pred = predictions.get("predictions", predictions.get("logits"))
                if pred is not None:
                    results[metric_name] = rmse(pred, targets)
                    
            elif metric_name == "r2":
                pred = predictions.get("predictions", predictions.get("logits"))
                if pred is not None:
                    results[metric_name] = r2_score(pred, targets)
                    
            elif metric_name == "mape":
                pred = predictions.get("predictions", predictions.get("logits"))
                if pred is not None:
                    results[metric_name] = mape(pred, targets)
                    
            elif metric_name == "map":
                scores = predictions.get("scores", predictions.get("logits"))
                if scores is not None:
                    results[metric_name] = mean_average_precision(scores, targets)
                    
            elif metric_name.startswith("ndcg"):
                scores = predictions.get("scores", predictions.get("logits"))
                if scores is not None:
                    k = kwargs.get("k", 10)
                    if "@" in metric_name:
                        k = int(metric_name.split("@")[1])
                    results[metric_name] = ndcg(scores, targets, k)
                    
            elif metric_name == "hamming_loss":
                pred = predictions.get("predictions", predictions.get("logits"))
                if pred is not None:
                    results[metric_name] = hamming_loss(pred, targets)
                    
            elif metric_name == "subset_accuracy":
                pred = predictions.get("predictions", predictions.get("logits"))
                if pred is not None:
                    results[metric_name] = subset_accuracy(pred, targets)
                    
            elif metric_name == "kendall_tau":
                pred = predictions.get("predictions", predictions.get("logits"))
                if pred is not None:
                    results[metric_name] = kendall_tau(pred, targets)
                    
        except Exception as e:
            # Skip metrics that fail
            results[metric_name] = 0.0
    
    return results


def get_metrics_for_task(task_type: str) -> List[str]:
    """Get appropriate metrics for a task type.
    
    Args:
        task_type: Type of task
        
    Returns:
        List of metric names
    """
    metrics_map = {
        "binary_classification": ["accuracy", "auc", "f1_binary", "precision_binary", "recall_binary"],
        "multiclass_classification": ["accuracy", "f1_macro", "f1_micro", "precision_macro", "recall_macro"],
        "multilabel_classification": ["subset_accuracy", "hamming_loss", "f1_macro", "f1_micro"],
        "regression": ["mae", "mse", "rmse", "r2", "mape"],
        "ordinal_regression": ["accuracy", "mae", "kendall_tau"],
        "ranking": ["map", "ndcg@10", "ndcg@5"],
        "time_series": ["mae", "mse", "rmse", "mape"],
    }
    
    return metrics_map.get(task_type, ["accuracy", "f1_macro"])


__all__ = [
    # Classification metrics
    "accuracy",
    "auc",
    "f1_score",
    "precision",
    "recall",
    # Regression metrics
    "mae",
    "mse",
    "rmse",
    "r2_score",
    "mape",
    # Ranking metrics
    "mean_average_precision",
    "ndcg",
    # Multilabel metrics
    "hamming_loss",
    "subset_accuracy",
    # Ordinal regression metrics
    "kendall_tau",
    # Utility functions
    "compute_metrics",
    "get_metrics_for_task",
]