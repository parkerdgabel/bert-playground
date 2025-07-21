"""Unit tests for classification metrics."""

import mlx.core as mx

from training.metrics.classification import (
    Accuracy,
    ClassificationMetricsCollection,
    F1Score,
    MulticlassAccuracy,
    Precision,
    Recall,
    TopKAccuracy,
)


class TestAccuracy:
    """Test accuracy metric."""

    def test_binary_accuracy(self):
        """Test binary classification accuracy."""
        metric = Accuracy()

        # Perfect predictions
        y_true = mx.array([0, 1, 0, 1])
        y_pred = mx.array([0, 1, 0, 1])

        metric.update(y_pred, y_true)
        assert metric.compute() == 1.0

    def test_binary_accuracy_imperfect(self):
        """Test binary accuracy with errors."""
        metric = Accuracy()

        # 3/4 correct
        y_true = mx.array([0, 1, 0, 1])
        y_pred = mx.array([0, 1, 1, 1])

        metric.update(y_pred, y_true)
        assert metric.compute() == 0.75

    def test_accuracy_accumulation(self):
        """Test accuracy accumulation over batches."""
        metric = Accuracy()

        # Batch 1: 2/2 correct
        metric.update(mx.array([0, 1]), mx.array([0, 1]))

        # Batch 2: 1/2 correct
        metric.update(mx.array([0, 0]), mx.array([0, 1]))

        # Total: 3/4 correct
        assert metric.compute() == 0.75

    def test_accuracy_reset(self):
        """Test resetting accuracy."""
        metric = Accuracy()

        metric.update(mx.array([0, 1]), mx.array([1, 0]))
        assert metric.compute() == 0.0

        metric.reset()
        assert metric.compute() == 0.0

        metric.update(mx.array([1, 1]), mx.array([1, 1]))
        assert metric.compute() == 1.0


class TestPrecisionRecall:
    """Test precision and recall metrics."""

    def test_binary_precision(self):
        """Test binary precision."""
        metric = Precision()

        # TP=2, FP=1, Precision = 2/3
        y_true = mx.array([0, 1, 0, 1])
        y_pred = mx.array([0, 1, 1, 1])

        metric.update(y_pred, y_true)
        assert abs(metric.compute() - 2 / 3) < 1e-6

    def test_binary_recall(self):
        """Test binary recall."""
        metric = Recall()

        # TP=1, FN=1, Recall = 1/2
        y_true = mx.array([0, 1, 0, 1])
        y_pred = mx.array([0, 1, 0, 0])

        metric.update(y_pred, y_true)
        assert metric.compute() == 0.5

    def test_perfect_precision_recall(self):
        """Test perfect precision and recall."""
        precision = Precision()
        recall = Recall()

        y_true = mx.array([0, 1, 0, 1])
        y_pred = mx.array([0, 1, 0, 1])

        precision.update(y_pred, y_true)
        recall.update(y_pred, y_true)

        assert precision.compute() == 1.0
        assert recall.compute() == 1.0


class TestF1Score:
    """Test F1 score metric."""

    def test_perfect_f1(self):
        """Test perfect F1 score."""
        metric = F1Score()

        y_true = mx.array([0, 1, 0, 1])
        y_pred = mx.array([0, 1, 0, 1])

        metric.update(y_pred, y_true)
        assert metric.compute() == 1.0

    def test_zero_f1(self):
        """Test zero F1 score."""
        metric = F1Score()

        # All wrong
        y_true = mx.array([0, 1, 0, 1])
        y_pred = mx.array([1, 0, 1, 0])

        metric.update(y_pred, y_true)
        assert metric.compute() == 0.0

    def test_f1_calculation(self):
        """Test F1 score calculation."""
        metric = F1Score()

        # TP=2, FP=1, FN=0
        # Precision = 2/3, Recall = 2/2 = 1
        # F1 = 2 * (2/3 * 1) / (2/3 + 1) = 0.8
        y_true = mx.array([0, 1, 0, 1])
        y_pred = mx.array([0, 1, 1, 1])

        metric.update(y_pred, y_true)
        assert abs(metric.compute() - 0.8) < 1e-6


class TestMulticlassAccuracy:
    """Test multiclass accuracy."""

    def test_multiclass_perfect(self):
        """Test perfect multiclass accuracy."""
        metric = MulticlassAccuracy(num_classes=3)

        y_true = mx.array([0, 1, 2, 0, 1, 2])
        y_pred = mx.array([0, 1, 2, 0, 1, 2])

        metric.update(y_pred, y_true)
        assert metric.compute() == 1.0

    def test_multiclass_partial(self):
        """Test partial multiclass accuracy."""
        metric = MulticlassAccuracy(num_classes=3)

        # 4/6 correct
        y_true = mx.array([0, 1, 2, 0, 1, 2])
        y_pred = mx.array([0, 1, 2, 1, 2, 2])

        metric.update(y_pred, y_true)
        assert abs(metric.compute() - 4 / 6) < 1e-6


class TestTopKAccuracy:
    """Test top-k accuracy."""

    def test_top1_accuracy(self):
        """Test top-1 accuracy."""
        metric = TopKAccuracy(k=1)

        # Logits for 3 classes
        logits = mx.array(
            [
                [2.0, 1.0, 0.0],  # Pred: 0
                [0.0, 2.0, 1.0],  # Pred: 1
                [1.0, 0.0, 2.0],  # Pred: 2
            ]
        )
        y_true = mx.array([0, 1, 2])

        metric.update(logits, y_true)
        assert metric.compute() == 1.0

    def test_top2_accuracy(self):
        """Test top-2 accuracy."""
        metric = TopKAccuracy(k=2)

        # Logits where true label is in top-2
        logits = mx.array(
            [
                [2.0, 1.0, 0.0],  # True: 0, in top-2
                [2.0, 1.0, 0.0],  # True: 1, in top-2
                [2.0, 1.0, 0.0],  # True: 2, not in top-2
            ]
        )
        y_true = mx.array([0, 1, 2])

        metric.update(logits, y_true)
        assert abs(metric.compute() - 2 / 3) < 1e-6


class TestClassificationMetricsCollection:
    """Test metrics collection."""

    def test_collection_update(self):
        """Test updating multiple metrics."""
        collection = ClassificationMetricsCollection()

        y_true = mx.array([0, 1, 0, 1])
        y_pred = mx.array([0, 1, 0, 1])

        collection.update(y_pred, y_true)

        results = collection.compute()
        assert results["accuracy"] == 1.0
        assert results["precision"] == 1.0
        assert results["recall"] == 1.0
        assert results["f1"] == 1.0

    def test_collection_reset(self):
        """Test resetting collection."""
        collection = ClassificationMetricsCollection()

        # Add some data
        collection.update(mx.array([0, 1]), mx.array([1, 0]))

        # Reset
        collection.reset()

        # Add perfect predictions
        collection.update(mx.array([1, 1]), mx.array([1, 1]))

        results = collection.compute()
        assert results["accuracy"] == 1.0
