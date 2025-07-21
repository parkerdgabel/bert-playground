"""
Classification metrics for training evaluation.
"""

import mlx.core as mx
import numpy as np

from .base import AveragedMetric, Metric


class Accuracy(AveragedMetric):
    """
    Accuracy metric for classification tasks.
    """

    def __init__(self, name: str = "accuracy", threshold: float = 0.5):
        """
        Initialize accuracy metric.

        Args:
            name: Metric name
            threshold: Threshold for binary classification
        """
        super().__init__(name)
        self.threshold = threshold

    def update(self, predictions: mx.array, targets: mx.array) -> None:
        """Update metric with batch results."""
        # Handle different prediction formats
        if predictions.ndim == 1:
            # Binary classification with single output
            preds = (predictions > self.threshold).astype(mx.int32)
        elif predictions.shape[-1] == 1:
            # Binary classification with single output dimension
            preds = (predictions.squeeze(-1) > self.threshold).astype(mx.int32)
        elif predictions.shape[-1] == 2:
            # Binary classification with two outputs
            preds = mx.argmax(predictions, axis=-1)
        else:
            # Multiclass classification
            preds = mx.argmax(predictions, axis=-1)

        # Ensure targets are integers
        if targets.dtype != mx.int32:
            targets = targets.astype(mx.int32)

        # Calculate correct predictions
        correct = mx.sum(preds == targets).item()
        batch_size = targets.shape[0]

        self.total += correct
        self.count += batch_size


class F1Score(Metric):
    """
    F1 score metric for classification tasks.
    """

    def __init__(
        self,
        name: str = "f1",
        num_classes: int | None = None,
        average: str = "binary",
        threshold: float = 0.5,
    ):
        """
        Initialize F1 score metric.

        Args:
            name: Metric name
            num_classes: Number of classes
            average: Averaging method ('binary', 'macro', 'micro', 'weighted')
            threshold: Threshold for binary classification
        """
        # Set attributes before calling super().__init__
        self.num_classes = num_classes
        self.average = average
        self.threshold = threshold

        # Initialize confusion matrix components
        self.true_positives = None
        self.false_positives = None
        self.false_negatives = None
        self.support = None

        # Now call super().__init__ which will call reset()
        super().__init__(name)

    def reset(self) -> None:
        """Reset metric state."""
        if self.num_classes:
            self.true_positives = mx.zeros(self.num_classes)
            self.false_positives = mx.zeros(self.num_classes)
            self.false_negatives = mx.zeros(self.num_classes)
            self.support = mx.zeros(self.num_classes)
        else:
            self.true_positives = 0
            self.false_positives = 0
            self.false_negatives = 0
            self.support = 0

    def update(self, predictions: mx.array, targets: mx.array) -> None:
        """Update metric with batch results."""
        # Get predicted classes
        if self.average == "binary":
            if predictions.ndim > 1 and predictions.shape[-1] == 2:
                # Binary classification with two outputs
                preds = mx.argmax(predictions, axis=-1)
            else:
                # Binary classification with single output
                preds = (predictions.squeeze() > self.threshold).astype(mx.int32)

            # Calculate for binary case
            tp = mx.sum((preds == 1) & (targets == 1)).item()
            fp = mx.sum((preds == 1) & (targets == 0)).item()
            fn = mx.sum((preds == 0) & (targets == 1)).item()

            self.true_positives += tp
            self.false_positives += fp
            self.false_negatives += fn
            self.support += targets.shape[0]
        else:
            # Multiclass case
            if predictions.shape[-1] > 1:
                preds = mx.argmax(predictions, axis=-1)
            else:
                preds = predictions.squeeze().astype(mx.int32)

            # Update per-class statistics
            for c in range(self.num_classes):
                tp = mx.sum((preds == c) & (targets == c)).item()
                fp = mx.sum((preds == c) & (targets != c)).item()
                fn = mx.sum((preds != c) & (targets == c)).item()

                self.true_positives[c] += tp
                self.false_positives[c] += fp
                self.false_negatives[c] += fn
                self.support[c] += mx.sum(targets == c).item()

    def compute(self) -> float:
        """Compute F1 score."""
        if self.average == "binary":
            # Binary F1
            precision = self.true_positives / max(
                self.true_positives + self.false_positives, 1e-10
            )
            recall = self.true_positives / max(
                self.true_positives + self.false_negatives, 1e-10
            )
            f1 = 2 * precision * recall / max(precision + recall, 1e-10)
            return f1

        elif self.average == "macro":
            # Macro-averaged F1
            f1_scores = []
            for c in range(self.num_classes):
                tp = self.true_positives[c].item()
                fp = self.false_positives[c].item()
                fn = self.false_negatives[c].item()

                precision = tp / max(tp + fp, 1e-10)
                recall = tp / max(tp + fn, 1e-10)
                f1 = 2 * precision * recall / max(precision + recall, 1e-10)
                f1_scores.append(f1)

            return np.mean(f1_scores)

        elif self.average == "micro":
            # Micro-averaged F1
            tp_total = mx.sum(self.true_positives).item()
            fp_total = mx.sum(self.false_positives).item()
            fn_total = mx.sum(self.false_negatives).item()

            precision = tp_total / max(tp_total + fp_total, 1e-10)
            recall = tp_total / max(tp_total + fn_total, 1e-10)
            f1 = 2 * precision * recall / max(precision + recall, 1e-10)
            return f1

        else:
            raise ValueError(f"Unknown average method: {self.average}")


class AUC(Metric):
    """
    Area Under the ROC Curve metric for binary classification.
    """

    def __init__(self, name: str = "auc"):
        super().__init__(name)
        self.predictions = []
        self.targets = []

    def reset(self) -> None:
        """Reset metric state."""
        self.predictions = []
        self.targets = []

    def update(self, predictions: mx.array, targets: mx.array) -> None:
        """Update metric with batch results."""
        # Get probabilities for positive class
        if predictions.ndim > 1 and predictions.shape[-1] == 2:
            # Binary classification with two outputs - use softmax
            probs = mx.softmax(predictions, axis=-1)[:, 1]
        else:
            # Binary classification with single output - use sigmoid
            probs = mx.sigmoid(predictions.squeeze())

        self.predictions.extend(probs.tolist())
        self.targets.extend(targets.tolist())

    def compute(self) -> float:
        """Compute AUC score."""
        if not self.predictions:
            return 0.0

        # Simple AUC calculation using trapezoidal rule
        # Sort by predictions
        sorted_indices = np.argsort(self.predictions)
        sorted_preds = np.array(self.predictions)[sorted_indices]
        sorted_targets = np.array(self.targets)[sorted_indices]

        # Calculate TPR and FPR
        n_pos = np.sum(sorted_targets)
        n_neg = len(sorted_targets) - n_pos

        if n_pos == 0 or n_neg == 0:
            return 0.5  # Random performance if only one class

        tpr = []
        fpr = []

        for threshold in sorted_preds:
            preds = (np.array(self.predictions) >= threshold).astype(int)
            tp = np.sum((preds == 1) & (np.array(self.targets) == 1))
            fp = np.sum((preds == 1) & (np.array(self.targets) == 0))

            tpr.append(tp / n_pos)
            fpr.append(fp / n_neg)

        # Add endpoints
        tpr = [0] + tpr + [1]
        fpr = [0] + fpr + [1]

        # Calculate AUC using trapezoidal rule
        auc = 0.0
        for i in range(1, len(fpr)):
            auc += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2

        return auc


class PrecisionRecall(Metric):
    """
    Precision and Recall metrics for classification.
    """

    def __init__(
        self,
        name: str = "precision_recall",
        num_classes: int | None = None,
        average: str = "binary",
        threshold: float = 0.5,
    ):
        super().__init__(name)
        self.num_classes = num_classes
        self.average = average
        self.threshold = threshold

        self.true_positives = None
        self.false_positives = None
        self.false_negatives = None

    def reset(self) -> None:
        """Reset metric state."""
        if self.num_classes:
            self.true_positives = mx.zeros(self.num_classes)
            self.false_positives = mx.zeros(self.num_classes)
            self.false_negatives = mx.zeros(self.num_classes)
        else:
            self.true_positives = 0
            self.false_positives = 0
            self.false_negatives = 0

    def update(self, predictions: mx.array, targets: mx.array) -> None:
        """Update metric with batch results."""
        # Similar to F1Score update
        if self.average == "binary":
            if predictions.ndim > 1 and predictions.shape[-1] == 2:
                preds = mx.argmax(predictions, axis=-1)
            else:
                preds = (predictions.squeeze() > self.threshold).astype(mx.int32)

            tp = mx.sum((preds == 1) & (targets == 1)).item()
            fp = mx.sum((preds == 1) & (targets == 0)).item()
            fn = mx.sum((preds == 0) & (targets == 1)).item()

            self.true_positives += tp
            self.false_positives += fp
            self.false_negatives += fn

    def compute(self) -> dict[str, float]:
        """Compute precision and recall."""
        precision = self.true_positives / max(
            self.true_positives + self.false_positives, 1e-10
        )
        recall = self.true_positives / max(
            self.true_positives + self.false_negatives, 1e-10
        )

        return {
            "precision": precision,
            "recall": recall,
        }


class Precision(AveragedMetric):
    """Precision metric for classification."""

    def __init__(self, name: str = "precision", threshold: float = 0.5):
        super().__init__(name)
        self.threshold = threshold
        self.true_positives = 0
        self.false_positives = 0

    def reset(self) -> None:
        """Reset metric state."""
        super().reset()
        self.true_positives = 0
        self.false_positives = 0

    def update(self, predictions: mx.array, targets: mx.array) -> None:
        """Update metric with batch results."""
        if predictions.ndim > 1 and predictions.shape[-1] == 2:
            preds = mx.argmax(predictions, axis=-1)
        else:
            preds = (predictions.squeeze() > self.threshold).astype(mx.int32)

        tp = mx.sum((preds == 1) & (targets == 1)).item()
        fp = mx.sum((preds == 1) & (targets == 0)).item()

        self.true_positives += tp
        self.false_positives += fp
        self.total = self.true_positives
        self.count = self.true_positives + self.false_positives

    def compute(self) -> float:
        """Compute precision."""
        if self.count == 0:
            return 0.0
        return self.total / self.count


class Recall(AveragedMetric):
    """Recall metric for classification."""

    def __init__(self, name: str = "recall", threshold: float = 0.5):
        super().__init__(name)
        self.threshold = threshold
        self.true_positives = 0
        self.false_negatives = 0

    def reset(self) -> None:
        """Reset metric state."""
        super().reset()
        self.true_positives = 0
        self.false_negatives = 0

    def update(self, predictions: mx.array, targets: mx.array) -> None:
        """Update metric with batch results."""
        if predictions.ndim > 1 and predictions.shape[-1] == 2:
            preds = mx.argmax(predictions, axis=-1)
        else:
            preds = (predictions.squeeze() > self.threshold).astype(mx.int32)

        tp = mx.sum((preds == 1) & (targets == 1)).item()
        fn = mx.sum((preds == 0) & (targets == 1)).item()

        self.true_positives += tp
        self.false_negatives += fn
        self.total = self.true_positives
        self.count = self.true_positives + self.false_negatives

    def compute(self) -> float:
        """Compute recall."""
        if self.count == 0:
            return 0.0
        return self.total / self.count


class MulticlassAccuracy(AveragedMetric):
    """Multiclass accuracy metric."""

    def __init__(self, name: str = "multiclass_accuracy", num_classes: int = None):
        super().__init__(name)
        self.num_classes = num_classes

    def update(self, predictions: mx.array, targets: mx.array) -> None:
        """Update metric with batch results."""
        # Get predicted classes
        if predictions.ndim == 1:
            preds = predictions.astype(mx.int32)
        else:
            preds = mx.argmax(predictions, axis=-1)

        # Ensure targets are integers
        if targets.dtype != mx.int32:
            targets = targets.astype(mx.int32)

        # Calculate correct predictions
        correct = mx.sum(preds == targets).item()
        batch_size = targets.shape[0]

        self.total += correct
        self.count += batch_size


class TopKAccuracy(Metric):
    """Top-K accuracy metric for multiclass classification."""

    def __init__(self, k: int = 5, name: str = None):
        name = name or f"top{k}_accuracy"
        super().__init__(name)
        self.k = k
        self.correct = 0
        self.total = 0

    def reset(self) -> None:
        """Reset metric state."""
        self.correct = 0
        self.total = 0

    def update(self, predictions: mx.array, targets: mx.array) -> None:
        """Update metric with batch results."""
        # Get top-k predictions
        top_k = mx.argpartition(-predictions, self.k, axis=-1)[:, : self.k]

        # Check if target is in top-k
        targets_expanded = targets.reshape(-1, 1)
        correct = mx.any(top_k == targets_expanded, axis=1)

        self.correct += mx.sum(correct).item()
        self.total += targets.shape[0]

    def compute(self) -> float:
        """Compute top-k accuracy."""
        if self.total == 0:
            return 0.0
        return self.correct / self.total


class ClassificationMetricsCollection(Metric):
    """Collection of classification metrics."""

    def __init__(
        self,
        num_classes: int | None = None,
        metrics: list | None = None,
        prefix: str = "",
    ):
        # Initialize metrics first before calling super().__init__
        self.num_classes = num_classes
        self.prefix = prefix

        # Default metrics
        if metrics is None:
            if num_classes == 2 or num_classes is None:
                # Binary classification
                self.metrics = {
                    "accuracy": Accuracy(),
                    "precision": Precision(),
                    "recall": Recall(),
                    "f1": F1Score(average="binary"),
                }
            else:
                # Multiclass
                self.metrics = {
                    "accuracy": MulticlassAccuracy(num_classes=num_classes),
                    "f1_macro": F1Score(num_classes=num_classes, average="macro"),
                    "f1_micro": F1Score(num_classes=num_classes, average="micro"),
                }

                # Add top-k accuracy for multiclass
                if num_classes > 5:
                    self.metrics["top5_accuracy"] = TopKAccuracy(k=5)
        else:
            self.metrics = {m.name: m for m in metrics}

        # Now call super().__init__
        super().__init__("classification_metrics")

    def reset(self) -> None:
        """Reset all metrics."""
        for metric in self.metrics.values():
            metric.reset()

    def update(self, predictions: mx.array, targets: mx.array) -> None:
        """Update all metrics."""
        for metric in self.metrics.values():
            metric.update(predictions, targets)

    def compute(self) -> dict[str, float]:
        """Compute all metrics."""
        results = {}
        for name, metric in self.metrics.items():
            value = metric.compute()
            if isinstance(value, dict):
                # Handle metrics that return multiple values
                for k, v in value.items():
                    results[f"{self.prefix}{name}_{k}"] = v
            else:
                results[f"{self.prefix}{name}"] = value

        return results
