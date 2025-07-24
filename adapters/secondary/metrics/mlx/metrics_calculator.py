"""MLX implementation of MetricsCalculatorPort."""

from typing import Dict, Any, List, Optional, Union
import mlx.core as mx
import numpy as np

from infrastructure.di import adapter, Scope
from ports.secondary.metrics import MetricsCalculatorPort
from adapters.secondary.compute.mlx.utils import convert_from_mlx_array


@adapter(MetricsCalculatorPort, scope=Scope.SINGLETON)
class MLXMetricsCalculator(MetricsCalculatorPort):
    """MLX implementation for metrics calculation."""
    
    def calculate_accuracy(
        self,
        predictions: Union[List[int], Any],
        labels: Union[List[int], Any],
    ) -> float:
        """Calculate accuracy.
        
        Args:
            predictions: Predicted labels
            labels: True labels
            
        Returns:
            Accuracy score
        """
        # Convert to MLX arrays if needed
        preds = self._to_mlx_array(predictions)
        targets = self._to_mlx_array(labels)
        
        # Calculate accuracy
        correct = mx.sum(preds == targets)
        total = targets.size
        
        return (correct / total).item()
    
    def calculate_precision_recall_f1(
        self,
        predictions: Union[List[int], Any],
        labels: Union[List[int], Any],
        average: str = "macro",
    ) -> Dict[str, float]:
        """Calculate precision, recall, and F1 scores.
        
        Args:
            predictions: Predicted labels
            labels: True labels
            average: Averaging method ('macro', 'micro', 'weighted')
            
        Returns:
            Dictionary with precision, recall, and f1 scores
        """
        # Convert to numpy for sklearn compatibility
        preds_np = self._to_numpy(predictions)
        labels_np = self._to_numpy(labels)
        
        # Get unique classes
        classes = np.unique(np.concatenate([preds_np, labels_np]))
        
        if average == "micro":
            # Micro-averaging
            tp = np.sum(preds_np == labels_np)
            total = len(labels_np)
            precision = recall = f1 = tp / total if total > 0 else 0.0
        
        elif average == "macro":
            # Macro-averaging
            precisions = []
            recalls = []
            f1s = []
            
            for cls in classes:
                tp = np.sum((preds_np == cls) & (labels_np == cls))
                fp = np.sum((preds_np == cls) & (labels_np != cls))
                fn = np.sum((preds_np != cls) & (labels_np == cls))
                
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                
                precisions.append(prec)
                recalls.append(rec)
                f1s.append(f1)
            
            precision = np.mean(precisions)
            recall = np.mean(recalls)
            f1 = np.mean(f1s)
        
        else:  # weighted
            # Weighted averaging
            precisions = []
            recalls = []
            f1s = []
            weights = []
            
            for cls in classes:
                tp = np.sum((preds_np == cls) & (labels_np == cls))
                fp = np.sum((preds_np == cls) & (labels_np != cls))
                fn = np.sum((preds_np != cls) & (labels_np == cls))
                support = np.sum(labels_np == cls)
                
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                
                precisions.append(prec)
                recalls.append(rec)
                f1s.append(f1)
                weights.append(support)
            
            total_weight = sum(weights)
            precision = sum(p * w for p, w in zip(precisions, weights)) / total_weight if total_weight > 0 else 0.0
            recall = sum(r * w for r, w in zip(recalls, weights)) / total_weight if total_weight > 0 else 0.0
            f1 = sum(f * w for f, w in zip(f1s, weights)) / total_weight if total_weight > 0 else 0.0
        
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    
    def calculate_confusion_matrix(
        self,
        predictions: Union[List[int], Any],
        labels: Union[List[int], Any],
    ) -> List[List[int]]:
        """Calculate confusion matrix.
        
        Args:
            predictions: Predicted labels
            labels: True labels
            
        Returns:
            Confusion matrix
        """
        # Convert to numpy
        preds_np = self._to_numpy(predictions)
        labels_np = self._to_numpy(labels)
        
        # Get unique classes
        classes = np.unique(np.concatenate([preds_np, labels_np]))
        n_classes = len(classes)
        
        # Create confusion matrix
        cm = np.zeros((n_classes, n_classes), dtype=int)
        
        for i, true_class in enumerate(classes):
            for j, pred_class in enumerate(classes):
                cm[i, j] = np.sum((labels_np == true_class) & (preds_np == pred_class))
        
        return cm.tolist()
    
    def calculate_auc_roc(
        self,
        probabilities: Union[List[float], Any],
        labels: Union[List[int], Any],
        multi_class: str = "ovr",
    ) -> float:
        """Calculate AUC-ROC score.
        
        Args:
            probabilities: Predicted probabilities
            labels: True labels
            multi_class: Multi-class strategy ('ovr', 'ovo')
            
        Returns:
            AUC-ROC score
        """
        # Convert to numpy
        probs_np = self._to_numpy(probabilities)
        labels_np = self._to_numpy(labels)
        
        # Binary classification
        if probs_np.ndim == 1 or (probs_np.ndim == 2 and probs_np.shape[1] == 1):
            return self._binary_auc_roc(probs_np.flatten(), labels_np)
        
        # Multi-class classification
        n_classes = probs_np.shape[1]
        
        if multi_class == "ovr":
            # One-vs-rest
            auc_scores = []
            for i in range(n_classes):
                binary_labels = (labels_np == i).astype(int)
                class_probs = probs_np[:, i]
                auc = self._binary_auc_roc(class_probs, binary_labels)
                auc_scores.append(auc)
            
            return np.mean(auc_scores)
        
        else:  # ovo
            # One-vs-one
            auc_scores = []
            for i in range(n_classes):
                for j in range(i + 1, n_classes):
                    # Filter samples for classes i and j
                    mask = (labels_np == i) | (labels_np == j)
                    if not np.any(mask):
                        continue
                    
                    binary_labels = (labels_np[mask] == j).astype(int)
                    binary_probs = probs_np[mask, j] / (probs_np[mask, i] + probs_np[mask, j])
                    
                    auc = self._binary_auc_roc(binary_probs, binary_labels)
                    auc_scores.append(auc)
            
            return np.mean(auc_scores) if auc_scores else 0.5
    
    def calculate_auc_pr(
        self,
        probabilities: Union[List[float], Any],
        labels: Union[List[int], Any],
    ) -> float:
        """Calculate AUC-PR (Area Under Precision-Recall curve).
        
        Args:
            probabilities: Predicted probabilities
            labels: True labels
            
        Returns:
            AUC-PR score
        """
        # Convert to numpy
        probs_np = self._to_numpy(probabilities)
        labels_np = self._to_numpy(labels)
        
        # Handle multi-class by averaging
        if probs_np.ndim > 1 and probs_np.shape[1] > 1:
            auc_scores = []
            for i in range(probs_np.shape[1]):
                binary_labels = (labels_np == i).astype(int)
                class_probs = probs_np[:, i]
                auc = self._binary_auc_pr(class_probs, binary_labels)
                auc_scores.append(auc)
            return np.mean(auc_scores)
        
        # Binary case
        return self._binary_auc_pr(probs_np.flatten(), labels_np)
    
    def calculate_mse(
        self,
        predictions: Union[List[float], Any],
        targets: Union[List[float], Any],
    ) -> float:
        """Calculate mean squared error.
        
        Args:
            predictions: Predicted values
            targets: True values
            
        Returns:
            MSE value
        """
        # Convert to MLX arrays
        preds = self._to_mlx_array(predictions)
        targs = self._to_mlx_array(targets)
        
        # Calculate MSE
        mse = mx.mean((preds - targs) ** 2)
        
        return mse.item()
    
    def calculate_mae(
        self,
        predictions: Union[List[float], Any],
        targets: Union[List[float], Any],
    ) -> float:
        """Calculate mean absolute error.
        
        Args:
            predictions: Predicted values
            targets: True values
            
        Returns:
            MAE value
        """
        # Convert to MLX arrays
        preds = self._to_mlx_array(predictions)
        targs = self._to_mlx_array(targets)
        
        # Calculate MAE
        mae = mx.mean(mx.abs(preds - targs))
        
        return mae.item()
    
    def calculate_r2_score(
        self,
        predictions: Union[List[float], Any],
        targets: Union[List[float], Any],
    ) -> float:
        """Calculate R-squared score.
        
        Args:
            predictions: Predicted values
            targets: True values
            
        Returns:
            R-squared score
        """
        # Convert to MLX arrays
        preds = self._to_mlx_array(predictions)
        targs = self._to_mlx_array(targets)
        
        # Calculate R-squared
        ss_res = mx.sum((targs - preds) ** 2)
        ss_tot = mx.sum((targs - mx.mean(targs)) ** 2)
        
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return r2.item()
    
    def calculate_per_class_metrics(
        self,
        predictions: Union[List[int], Any],
        labels: Union[List[int], Any],
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Calculate per-class metrics.
        
        Args:
            predictions: Predicted labels
            labels: True labels
            class_names: Optional class names
            
        Returns:
            Dictionary mapping class to metrics
        """
        # Convert to numpy
        preds_np = self._to_numpy(predictions)
        labels_np = self._to_numpy(labels)
        
        # Get unique classes
        classes = np.unique(np.concatenate([preds_np, labels_np]))
        
        # Use class names if provided
        if class_names is None:
            class_names = [str(cls) for cls in classes]
        
        per_class_metrics = {}
        
        for cls, name in zip(classes, class_names):
            tp = np.sum((preds_np == cls) & (labels_np == cls))
            fp = np.sum((preds_np == cls) & (labels_np != cls))
            fn = np.sum((preds_np != cls) & (labels_np == cls))
            tn = np.sum((preds_np != cls) & (labels_np != cls))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            support = np.sum(labels_np == cls)
            
            per_class_metrics[name] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": int(support),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_negatives": int(tn),
            }
        
        return per_class_metrics
    
    def calculate_loss(
        self,
        logits: Any,
        labels: Any,
        loss_type: str = "cross_entropy",
        **kwargs: Any,
    ) -> float:
        """Calculate loss value.
        
        Args:
            logits: Model predictions
            labels: True labels
            loss_type: Type of loss function
            **kwargs: Additional loss parameters
            
        Returns:
            Loss value
        """
        # Convert to MLX arrays
        logits_mx = self._to_mlx_array(logits)
        labels_mx = self._to_mlx_array(labels)
        
        if loss_type == "cross_entropy":
            # Cross-entropy loss
            import mlx.nn as nn
            loss = nn.losses.cross_entropy(logits_mx, labels_mx, **kwargs)
            
        elif loss_type == "binary_cross_entropy":
            # Binary cross-entropy
            import mlx.nn as nn
            loss = nn.losses.binary_cross_entropy(logits_mx, labels_mx, **kwargs)
            
        elif loss_type == "mse":
            # Mean squared error
            loss = mx.mean((logits_mx - labels_mx) ** 2)
            
        elif loss_type == "mae":
            # Mean absolute error
            loss = mx.mean(mx.abs(logits_mx - labels_mx))
            
        elif loss_type == "huber":
            # Huber loss
            delta = kwargs.get("delta", 1.0)
            diff = mx.abs(logits_mx - labels_mx)
            loss = mx.where(
                diff <= delta,
                0.5 * diff ** 2,
                delta * diff - 0.5 * delta ** 2
            )
            loss = mx.mean(loss)
            
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        return loss.item()
    
    def aggregate_metrics(
        self,
        metrics_list: List[Dict[str, float]],
    ) -> Dict[str, float]:
        """Aggregate metrics from multiple batches.
        
        Args:
            metrics_list: List of metric dictionaries
            
        Returns:
            Aggregated metrics
        """
        if not metrics_list:
            return {}
        
        # Get all unique metric names
        all_metrics = set()
        for metrics in metrics_list:
            all_metrics.update(metrics.keys())
        
        # Aggregate each metric
        aggregated = {}
        for metric_name in all_metrics:
            values = [m.get(metric_name, 0.0) for m in metrics_list]
            
            # Use mean aggregation
            aggregated[metric_name] = float(np.mean(values))
        
        return aggregated
    
    # Private helper methods
    
    def _to_mlx_array(self, data: Union[List, mx.array, np.ndarray]) -> mx.array:
        """Convert data to MLX array."""
        if isinstance(data, mx.array):
            return data
        elif isinstance(data, np.ndarray):
            return mx.array(data)
        else:
            return mx.array(data)
    
    def _to_numpy(self, data: Union[List, mx.array, np.ndarray]) -> np.ndarray:
        """Convert data to numpy array."""
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, mx.array):
            return convert_from_mlx_array(data, to_numpy=True)
        else:
            return np.array(data)
    
    def _binary_auc_roc(self, scores: np.ndarray, labels: np.ndarray) -> float:
        """Calculate binary AUC-ROC."""
        # Sort by scores
        sorted_indices = np.argsort(-scores)
        sorted_labels = labels[sorted_indices]
        
        # Calculate TPR and FPR
        n_pos = np.sum(labels)
        n_neg = len(labels) - n_pos
        
        if n_pos == 0 or n_neg == 0:
            return 0.5
        
        tpr = []
        fpr = []
        
        tp = 0
        fp = 0
        
        for label in sorted_labels:
            if label == 1:
                tp += 1
            else:
                fp += 1
            
            tpr.append(tp / n_pos)
            fpr.append(fp / n_neg)
        
        # Add endpoints
        tpr = [0] + tpr + [1]
        fpr = [0] + fpr + [1]
        
        # Calculate AUC using trapezoidal rule
        auc = 0
        for i in range(1, len(fpr)):
            auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2
        
        return auc
    
    def _binary_auc_pr(self, scores: np.ndarray, labels: np.ndarray) -> float:
        """Calculate binary AUC-PR."""
        # Sort by scores
        sorted_indices = np.argsort(-scores)
        sorted_labels = labels[sorted_indices]
        
        # Calculate precision and recall
        n_pos = np.sum(labels)
        
        if n_pos == 0:
            return 0.0
        
        precision = []
        recall = []
        
        tp = 0
        fp = 0
        
        for i, label in enumerate(sorted_labels):
            if label == 1:
                tp += 1
            else:
                fp += 1
            
            prec = tp / (tp + fp)
            rec = tp / n_pos
            
            precision.append(prec)
            recall.append(rec)
        
        # Add endpoints
        precision = [1] + precision
        recall = [0] + recall
        
        # Calculate AUC using trapezoidal rule
        auc = 0
        for i in range(1, len(recall)):
            auc += (recall[i] - recall[i-1]) * (precision[i] + precision[i-1]) / 2
        
        return auc