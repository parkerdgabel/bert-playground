"""Kaggle-specific evaluation metrics for BERT heads.

This module provides comprehensive evaluation metrics optimized for
Kaggle competitions, including AUC, F1, NDCG, and other competition metrics.
"""

import mlx.core as mx
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from abc import ABC, abstractmethod


class CompetitionMetric(ABC):
    """Base class for competition metrics."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def compute(self, predictions: Union[mx.array, np.ndarray], targets: Union[mx.array, np.ndarray], **kwargs) -> float:
        """Compute the metric.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            **kwargs: Additional arguments
            
        Returns:
            Metric value
        """
        pass
    
    def _to_numpy(self, array: Union[mx.array, np.ndarray]) -> np.ndarray:
        """Convert MLX array to numpy if needed."""
        if isinstance(array, mx.array):
            return np.array(array)
        return array


class AccuracyMetric(CompetitionMetric):
    """Standard accuracy metric."""
    
    def __init__(self):
        super().__init__("accuracy")
    
    def compute(self, predictions: Union[mx.array, np.ndarray], targets: Union[mx.array, np.ndarray], **kwargs) -> float:
        """Compute accuracy."""
        pred_np = self._to_numpy(predictions)
        targets_np = self._to_numpy(targets)
        
        return float(np.mean(pred_np == targets_np))


class AUCMetric(CompetitionMetric):
    """Area Under the ROC Curve metric."""
    
    def __init__(self):
        super().__init__("auc")
    
    def compute(self, probabilities: Union[mx.array, np.ndarray], targets: Union[mx.array, np.ndarray], **kwargs) -> float:
        """Compute AUC score."""
        probs_np = self._to_numpy(probabilities)
        targets_np = self._to_numpy(targets)
        
        return self._compute_auc(probs_np, targets_np)
    
    def _compute_auc(self, probabilities: np.ndarray, targets: np.ndarray) -> float:
        """Compute AUC using trapezoidal rule."""
        if len(np.unique(targets)) < 2:
            return 0.5  # No positive or negative examples
        
        # Sort by probability
        sorted_indices = np.argsort(probabilities)
        sorted_probs = probabilities[sorted_indices]
        sorted_targets = targets[sorted_indices]
        
        # Compute TPR and FPR
        n_pos = (targets == 1).sum()
        n_neg = (targets == 0).sum()
        
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
        auc = 0.0
        for i in range(len(tpr) - 1):
            auc += (fpr[i + 1] - fpr[i]) * (tpr[i + 1] + tpr[i]) / 2
        
        return max(0.0, min(1.0, auc))


class F1Metric(CompetitionMetric):
    """F1 score metric."""
    
    def __init__(self, average: str = "binary"):
        super().__init__(f"f1_{average}")
        self.average = average
    
    def compute(self, predictions: Union[mx.array, np.ndarray], targets: Union[mx.array, np.ndarray], **kwargs) -> float:
        """Compute F1 score."""
        pred_np = self._to_numpy(predictions)
        targets_np = self._to_numpy(targets)
        
        if self.average == "binary":
            return self._compute_binary_f1(pred_np, targets_np)
        elif self.average == "macro":
            return self._compute_macro_f1(pred_np, targets_np)
        elif self.average == "micro":
            return self._compute_micro_f1(pred_np, targets_np)
        else:
            raise ValueError(f"Unknown average type: {self.average}")
    
    def _compute_binary_f1(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute binary F1 score."""
        tp = ((predictions == 1) & (targets == 1)).sum()
        fp = ((predictions == 1) & (targets == 0)).sum()
        fn = ((predictions == 0) & (targets == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    def _compute_macro_f1(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute macro F1 score."""
        unique_classes = np.unique(np.concatenate([predictions, targets]))
        f1_scores = []
        
        for cls in unique_classes:
            tp = ((predictions == cls) & (targets == cls)).sum()
            fp = ((predictions == cls) & (targets != cls)).sum()
            fn = ((predictions != cls) & (targets == cls)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            f1_scores.append(f1)
        
        return np.mean(f1_scores)
    
    def _compute_micro_f1(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute micro F1 score."""
        # For multiclass, micro F1 = accuracy
        if len(np.unique(targets)) > 2:
            return (predictions == targets).mean()
        else:
            return self._compute_binary_f1(predictions, targets)


class PrecisionMetric(CompetitionMetric):
    """Precision metric."""
    
    def __init__(self, average: str = "binary"):
        super().__init__(f"precision_{average}")
        self.average = average
    
    def compute(self, predictions: Union[mx.array, np.ndarray], targets: Union[mx.array, np.ndarray], **kwargs) -> float:
        """Compute precision."""
        pred_np = self._to_numpy(predictions)
        targets_np = self._to_numpy(targets)
        
        if self.average == "binary":
            tp = ((pred_np == 1) & (targets_np == 1)).sum()
            fp = ((pred_np == 1) & (targets_np == 0)).sum()
            return tp / (tp + fp) if (tp + fp) > 0 else 0.0
        elif self.average == "macro":
            unique_classes = np.unique(np.concatenate([pred_np, targets_np]))
            precisions = []
            
            for cls in unique_classes:
                tp = ((pred_np == cls) & (targets_np == cls)).sum()
                fp = ((pred_np == cls) & (targets_np != cls)).sum()
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                precisions.append(precision)
            
            return np.mean(precisions)
        else:
            raise ValueError(f"Unknown average type: {self.average}")


class RecallMetric(CompetitionMetric):
    """Recall metric."""
    
    def __init__(self, average: str = "binary"):
        super().__init__(f"recall_{average}")
        self.average = average
    
    def compute(self, predictions: Union[mx.array, np.ndarray], targets: Union[mx.array, np.ndarray], **kwargs) -> float:
        """Compute recall."""
        pred_np = self._to_numpy(predictions)
        targets_np = self._to_numpy(targets)
        
        if self.average == "binary":
            tp = ((pred_np == 1) & (targets_np == 1)).sum()
            fn = ((pred_np == 0) & (targets_np == 1)).sum()
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0
        elif self.average == "macro":
            unique_classes = np.unique(np.concatenate([pred_np, targets_np]))
            recalls = []
            
            for cls in unique_classes:
                tp = ((pred_np == cls) & (targets_np == cls)).sum()
                fn = ((pred_np != cls) & (targets_np == cls)).sum()
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                recalls.append(recall)
            
            return np.mean(recalls)
        else:
            raise ValueError(f"Unknown average type: {self.average}")


class MAEMetric(CompetitionMetric):
    """Mean Absolute Error metric."""
    
    def __init__(self):
        super().__init__("mae")
    
    def compute(self, predictions: Union[mx.array, np.ndarray], targets: Union[mx.array, np.ndarray], **kwargs) -> float:
        """Compute MAE."""
        pred_np = self._to_numpy(predictions)
        targets_np = self._to_numpy(targets)
        
        return float(np.mean(np.abs(pred_np - targets_np)))


class MSEMetric(CompetitionMetric):
    """Mean Squared Error metric."""
    
    def __init__(self):
        super().__init__("mse")
    
    def compute(self, predictions: Union[mx.array, np.ndarray], targets: Union[mx.array, np.ndarray], **kwargs) -> float:
        """Compute MSE."""
        pred_np = self._to_numpy(predictions)
        targets_np = self._to_numpy(targets)
        
        return float(np.mean((pred_np - targets_np) ** 2))


class RMSEMetric(CompetitionMetric):
    """Root Mean Squared Error metric."""
    
    def __init__(self):
        super().__init__("rmse")
    
    def compute(self, predictions: Union[mx.array, np.ndarray], targets: Union[mx.array, np.ndarray], **kwargs) -> float:
        """Compute RMSE."""
        pred_np = self._to_numpy(predictions)
        targets_np = self._to_numpy(targets)
        
        return float(np.sqrt(np.mean((pred_np - targets_np) ** 2)))


class R2Metric(CompetitionMetric):
    """R-squared metric."""
    
    def __init__(self):
        super().__init__("r2")
    
    def compute(self, predictions: Union[mx.array, np.ndarray], targets: Union[mx.array, np.ndarray], **kwargs) -> float:
        """Compute R-squared."""
        pred_np = self._to_numpy(predictions)
        targets_np = self._to_numpy(targets)
        
        ss_res = np.sum((targets_np - pred_np) ** 2)
        ss_tot = np.sum((targets_np - np.mean(targets_np)) ** 2)
        
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


class MAPMetric(CompetitionMetric):
    """Mean Average Precision metric for ranking."""
    
    def __init__(self):
        super().__init__("map")
    
    def compute(self, scores: Union[mx.array, np.ndarray], targets: Union[mx.array, np.ndarray], **kwargs) -> float:
        """Compute MAP."""
        scores_np = self._to_numpy(scores)
        targets_np = self._to_numpy(targets)
        
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


class NDCGMetric(CompetitionMetric):
    """Normalized Discounted Cumulative Gain metric."""
    
    def __init__(self, k: int = 10):
        super().__init__(f"ndcg@{k}")
        self.k = k
    
    def compute(self, scores: Union[mx.array, np.ndarray], targets: Union[mx.array, np.ndarray], **kwargs) -> float:
        """Compute NDCG@k."""
        scores_np = self._to_numpy(scores)
        targets_np = self._to_numpy(targets)
        
        # Sort by scores in descending order
        sorted_indices = np.argsort(scores_np)[::-1]
        sorted_targets = targets_np[sorted_indices]
        
        # Compute DCG@k
        dcg = 0.0
        for i in range(min(self.k, len(sorted_targets))):
            relevance = sorted_targets[i]
            dcg += (2 ** relevance - 1) / np.log2(i + 2)
        
        # Compute IDCG@k (ideal DCG)
        ideal_targets = np.sort(targets_np)[::-1]
        idcg = 0.0
        for i in range(min(self.k, len(ideal_targets))):
            relevance = ideal_targets[i]
            idcg += (2 ** relevance - 1) / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0


class HammingLossMetric(CompetitionMetric):
    """Hamming loss for multilabel classification."""
    
    def __init__(self):
        super().__init__("hamming_loss")
    
    def compute(self, predictions: Union[mx.array, np.ndarray], targets: Union[mx.array, np.ndarray], **kwargs) -> float:
        """Compute Hamming loss."""
        pred_np = self._to_numpy(predictions)
        targets_np = self._to_numpy(targets)
        
        return float(np.mean(pred_np != targets_np))


class SubsetAccuracyMetric(CompetitionMetric):
    """Subset accuracy for multilabel classification."""
    
    def __init__(self):
        super().__init__("subset_accuracy")
    
    def compute(self, predictions: Union[mx.array, np.ndarray], targets: Union[mx.array, np.ndarray], **kwargs) -> float:
        """Compute subset accuracy."""
        pred_np = self._to_numpy(predictions)
        targets_np = self._to_numpy(targets)
        
        return float(np.mean(np.all(pred_np == targets_np, axis=1)))


class KendallTauMetric(CompetitionMetric):
    """Kendall's tau for ordinal regression."""
    
    def __init__(self):
        super().__init__("kendall_tau")
    
    def compute(self, predictions: Union[mx.array, np.ndarray], targets: Union[mx.array, np.ndarray], **kwargs) -> float:
        """Compute Kendall's tau."""
        pred_np = self._to_numpy(predictions)
        targets_np = self._to_numpy(targets)
        
        from scipy.stats import kendalltau
        try:
            tau, _ = kendalltau(pred_np, targets_np)
            return tau if not np.isnan(tau) else 0.0
        except:
            return 0.0


class MAPEMetric(CompetitionMetric):
    """Mean Absolute Percentage Error metric."""
    
    def __init__(self):
        super().__init__("mape")
    
    def compute(self, predictions: Union[mx.array, np.ndarray], targets: Union[mx.array, np.ndarray], **kwargs) -> float:
        """Compute MAPE."""
        pred_np = self._to_numpy(predictions)
        targets_np = self._to_numpy(targets)
        
        # Avoid division by zero
        mask = targets_np != 0
        if not np.any(mask):
            return 0.0
        
        return float(np.mean(np.abs((targets_np[mask] - pred_np[mask]) / targets_np[mask])) * 100)


class MetricComputer:
    """Utility class for computing multiple metrics."""
    
    def __init__(self):
        self.metrics = {}
    
    def add_metric(self, metric: CompetitionMetric):
        """Add a metric to the computer."""
        self.metrics[metric.name] = metric
    
    def compute_all(self, predictions: Dict[str, Union[mx.array, np.ndarray]], targets: Union[mx.array, np.ndarray], **kwargs) -> Dict[str, float]:
        """Compute all metrics.
        
        Args:
            predictions: Dictionary of predictions with different keys
            targets: Ground truth targets
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of metric values
        """
        results = {}
        
        for metric_name, metric in self.metrics.items():
            try:
                # Determine which prediction to use
                if metric_name in ["auc"]:
                    # Use probabilities for AUC
                    if "probabilities" in predictions:
                        pred = predictions["probabilities"]
                        if len(pred.shape) > 1 and pred.shape[-1] == 2:
                            pred = pred[:, 1]  # Use positive class probability
                    else:
                        continue
                elif metric_name in ["map", "ndcg"]:
                    # Use scores for ranking metrics
                    if "scores" in predictions:
                        pred = predictions["scores"]
                    else:
                        continue
                else:
                    # Use predictions for other metrics
                    pred = predictions.get("predictions", None)
                    if pred is None:
                        continue
                
                # Compute metric
                value = metric.compute(pred, targets, **kwargs)
                results[metric_name] = value
                
            except Exception as e:
                # Skip metrics that fail
                results[metric_name] = 0.0
        
        return results


def get_competition_metrics(competition_type: str) -> MetricComputer:
    """Get appropriate metrics for a competition type.
    
    Args:
        competition_type: Type of competition
        
    Returns:
        MetricComputer with appropriate metrics
    """
    computer = MetricComputer()
    
    if competition_type == "binary_classification":
        computer.add_metric(AccuracyMetric())
        computer.add_metric(AUCMetric())
        computer.add_metric(F1Metric("binary"))
        computer.add_metric(PrecisionMetric("binary"))
        computer.add_metric(RecallMetric("binary"))
        
    elif competition_type == "multiclass_classification":
        computer.add_metric(AccuracyMetric())
        computer.add_metric(F1Metric("macro"))
        computer.add_metric(F1Metric("micro"))
        computer.add_metric(PrecisionMetric("macro"))
        computer.add_metric(RecallMetric("macro"))
        
    elif competition_type == "multilabel_classification":
        computer.add_metric(SubsetAccuracyMetric())
        computer.add_metric(HammingLossMetric())
        computer.add_metric(F1Metric("macro"))
        computer.add_metric(F1Metric("micro"))
        
    elif competition_type == "regression":
        computer.add_metric(MAEMetric())
        computer.add_metric(MSEMetric())
        computer.add_metric(RMSEMetric())
        computer.add_metric(R2Metric())
        computer.add_metric(MAPEMetric())
        
    elif competition_type == "ordinal_regression":
        computer.add_metric(AccuracyMetric())
        computer.add_metric(MAEMetric())
        computer.add_metric(KendallTauMetric())
        
    elif competition_type == "ranking":
        computer.add_metric(MAPMetric())
        computer.add_metric(NDCGMetric(k=10))
        computer.add_metric(NDCGMetric(k=5))
        
    elif competition_type == "time_series":
        computer.add_metric(MAEMetric())
        computer.add_metric(MSEMetric())
        computer.add_metric(RMSEMetric())
        computer.add_metric(MAPEMetric())
        
    else:
        # Default metrics
        computer.add_metric(AccuracyMetric())
        computer.add_metric(F1Metric("macro"))
    
    return computer


def compute_competition_metrics(
    predictions: Dict[str, Union[mx.array, np.ndarray]],
    targets: Union[mx.array, np.ndarray],
    competition_type: str,
    **kwargs
) -> Dict[str, float]:
    """Compute metrics for a competition type.
    
    Args:
        predictions: Dictionary of predictions
        targets: Ground truth targets
        competition_type: Type of competition
        **kwargs: Additional arguments
        
    Returns:
        Dictionary of metric values
    """
    computer = get_competition_metrics(competition_type)
    return computer.compute_all(predictions, targets, **kwargs)