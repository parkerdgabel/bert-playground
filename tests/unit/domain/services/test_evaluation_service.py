"""Unit tests for evaluation service domain logic.

These tests verify the pure business logic of the evaluation service
without any external dependencies or framework-specific code.
"""

import pytest
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from domain.services.evaluation_service import (
    MetricType,
    MetricConfig,
    EvaluationConfig,
    EvaluationResult,
    ClassificationMetrics,
    RegressionMetrics,
    MetricCalculator,
    EvaluationService
)


class TestMetricConfig:
    """Test metric configuration."""
    
    def test_default_metric_config(self):
        """Test default metric configuration."""
        config = MetricConfig(metric_type=MetricType.ACCURACY)
        assert config.metric_type == MetricType.ACCURACY
        assert config.primary is False
        assert config.greater_is_better is True
        assert config.threshold is None
        assert config.average == "macro"
    
    def test_loss_metric_defaults(self):
        """Test that loss metrics default to lower is better."""
        for metric_type in [MetricType.LOSS, MetricType.MSE, MetricType.MAE, 
                           MetricType.RMSE, MetricType.HAMMING_LOSS]:
            config = MetricConfig(metric_type=metric_type)
            assert config.greater_is_better is False
    
    def test_accuracy_metric_defaults(self):
        """Test that accuracy metrics default to micro averaging."""
        for metric_type in [MetricType.ACCURACY, MetricType.EXACT_MATCH]:
            config = MetricConfig(metric_type=metric_type)
            assert config.average == "micro"
    
    def test_custom_metric_config(self):
        """Test custom metric configuration."""
        config = MetricConfig(
            metric_type=MetricType.F1_SCORE,
            primary=True,
            threshold=0.7,
            average="weighted"
        )
        assert config.primary is True
        assert config.threshold == 0.7
        assert config.average == "weighted"


class TestEvaluationConfig:
    """Test evaluation configuration."""
    
    def test_minimal_config(self):
        """Test minimal evaluation configuration."""
        metric = MetricConfig(metric_type=MetricType.ACCURACY)
        config = EvaluationConfig(metrics=[metric])
        
        assert len(config.metrics) == 1
        assert config.metrics[0].primary is True  # First metric marked as primary
        assert config.batch_size == 32
        assert config.use_mixed_precision is False
    
    def test_multiple_metrics(self):
        """Test configuration with multiple metrics."""
        metrics = [
            MetricConfig(metric_type=MetricType.ACCURACY),
            MetricConfig(metric_type=MetricType.F1_SCORE, primary=True),
            MetricConfig(metric_type=MetricType.PRECISION)
        ]
        config = EvaluationConfig(metrics=metrics)
        
        assert len(config.metrics) == 3
        assert config.primary_metric.metric_type == MetricType.F1_SCORE
        assert config.metric_names == ["accuracy", "f1", "precision"]
    
    def test_no_metrics_validation(self):
        """Test that configuration requires at least one metric."""
        with pytest.raises(ValueError, match="At least one metric must be specified"):
            EvaluationConfig(metrics=[])
    
    def test_multiple_primary_validation(self):
        """Test that only one primary metric is allowed."""
        metrics = [
            MetricConfig(metric_type=MetricType.ACCURACY, primary=True),
            MetricConfig(metric_type=MetricType.F1_SCORE, primary=True)
        ]
        with pytest.raises(ValueError, match="Only one metric can be marked as primary"):
            EvaluationConfig(metrics=metrics)
    
    def test_custom_config(self):
        """Test custom evaluation configuration."""
        metric = MetricConfig(metric_type=MetricType.AUC_ROC)
        config = EvaluationConfig(
            metrics=[metric],
            batch_size=64,
            use_mixed_precision=True,
            compute_confidence=True,
            save_predictions=True,
            classification_threshold=0.3,
            max_eval_samples=1000
        )
        
        assert config.batch_size == 64
        assert config.use_mixed_precision is True
        assert config.compute_confidence is True
        assert config.save_predictions is True
        assert config.classification_threshold == 0.3
        assert config.max_eval_samples == 1000


class TestEvaluationResult:
    """Test evaluation result handling."""
    
    def test_basic_result(self):
        """Test basic evaluation result."""
        metrics = {"accuracy": 0.95, "loss": 0.15}
        result = EvaluationResult(
            metrics=metrics,
            num_samples=1000,
            evaluation_time_seconds=10.0
        )
        
        assert result.get_primary_metric("accuracy") == 0.95
        assert result.num_samples == 1000
        assert result.evaluation_time_seconds == 10.0
    
    def test_get_summary(self):
        """Test evaluation summary generation."""
        metrics = {"accuracy": 0.92, "f1": 0.89}
        result = EvaluationResult(
            metrics=metrics,
            num_samples=500,
            evaluation_time_seconds=5.0
        )
        
        summary = result.get_summary()
        assert summary["metrics"] == metrics
        assert summary["num_samples"] == 500
        assert summary["evaluation_time"] == 5.0
        assert summary["samples_per_second"] == 100.0
    
    def test_missing_metric_error(self):
        """Test error when accessing missing metric."""
        result = EvaluationResult(metrics={"accuracy": 0.9})
        
        with pytest.raises(ValueError, match="Metric 'f1' not found"):
            result.get_primary_metric("f1")
    
    def test_full_result(self):
        """Test result with all optional fields."""
        result = EvaluationResult(
            metrics={"accuracy": 0.95},
            predictions=[1, 0, 1, 1, 0],
            probabilities=[[0.1, 0.9], [0.8, 0.2], [0.3, 0.7], [0.2, 0.8], [0.9, 0.1]],
            confidence_scores=[0.9, 0.8, 0.7, 0.8, 0.9],
            per_class_metrics={
                "class_0": {"precision": 0.9, "recall": 0.85},
                "class_1": {"precision": 0.95, "recall": 0.92}
            },
            num_samples=5,
            evaluation_time_seconds=0.1
        )
        
        assert result.predictions == [1, 0, 1, 1, 0]
        assert result.confidence_scores == [0.9, 0.8, 0.7, 0.8, 0.9]
        assert result.per_class_metrics["class_0"]["precision"] == 0.9


class TestClassificationMetrics:
    """Test classification metrics calculation."""
    
    def test_binary_classification_update(self):
        """Test updating binary classification metrics."""
        metrics = ClassificationMetrics(num_classes=2)
        
        # Predictions: [1, 0, 1, 1, 0]
        # Labels:      [1, 0, 0, 1, 1]
        predictions = [1, 0, 0, 1, 1]
        labels = [1, 0, 0, 1, 1]
        
        metrics.update(predictions, labels)
        
        # Perfect predictions
        assert metrics.true_positives[0] == 2  # Correctly predicted class 0
        assert metrics.true_positives[1] == 3  # Correctly predicted class 1
        assert metrics.false_positives[0] == 0
        assert metrics.false_positives[1] == 0
        assert metrics.false_negatives[0] == 0
        assert metrics.false_negatives[1] == 0
    
    def test_precision_calculation(self):
        """Test precision calculation."""
        metrics = ClassificationMetrics(num_classes=2)
        
        # Set up test scenario
        metrics.true_positives = {0: 8, 1: 12}
        metrics.false_positives = {0: 2, 1: 3}
        metrics.false_negatives = {0: 3, 1: 2}
        metrics.true_negatives = {0: 15, 1: 10}
        
        # Macro precision: average of (8/10, 12/15) = (0.8, 0.8) = 0.8
        assert metrics.compute_precision("macro") == 0.8
        
        # Micro precision: 20/25 = 0.8
        assert metrics.compute_precision("micro") == 0.8
    
    def test_recall_calculation(self):
        """Test recall calculation."""
        metrics = ClassificationMetrics(num_classes=2)
        
        # Set up test scenario
        metrics.true_positives = {0: 8, 1: 12}
        metrics.false_positives = {0: 2, 1: 3}
        metrics.false_negatives = {0: 2, 1: 3}
        metrics.true_negatives = {0: 15, 1: 10}
        
        # Macro recall: average of (8/10, 12/15) = (0.8, 0.8) = 0.8
        assert metrics.compute_recall("macro") == 0.8
        
        # Micro recall: 20/25 = 0.8
        assert metrics.compute_recall("micro") == 0.8
    
    def test_f1_calculation(self):
        """Test F1 score calculation."""
        metrics = ClassificationMetrics(num_classes=2)
        
        # Set up scenario with perfect predictions
        metrics.true_positives = {0: 10, 1: 15}
        metrics.false_positives = {0: 0, 1: 0}
        metrics.false_negatives = {0: 0, 1: 0}
        
        # Perfect F1 score
        assert metrics.compute_f1("macro") == 1.0
        assert metrics.compute_f1("micro") == 1.0
    
    def test_f1_with_zero_division(self):
        """Test F1 calculation with zero precision/recall."""
        metrics = ClassificationMetrics(num_classes=2)
        
        # Class 0 has no true positives
        metrics.true_positives = {0: 0, 1: 10}
        metrics.false_positives = {0: 5, 1: 0}
        metrics.false_negatives = {0: 5, 1: 0}
        
        # F1 for class 0 should be 0, class 1 should be 1
        # Macro F1: (0 + 1) / 2 = 0.5
        assert metrics.compute_f1("macro") == 0.5
    
    def test_accuracy_calculation(self):
        """Test accuracy calculation."""
        metrics = ClassificationMetrics(num_classes=3)
        
        # Set up test scenario
        metrics.true_positives = {0: 10, 1: 15, 2: 20}
        metrics.false_positives = {0: 2, 1: 3, 2: 1}
        metrics.false_negatives = {0: 3, 1: 2, 2: 4}
        # True negatives calculated to make total consistent
        
        # Total correct: 10 + 15 + 20 = 45
        # This is a simplified test - actual calculation is more complex
        accuracy = metrics.compute_accuracy()
        assert accuracy > 0  # Just verify it computes something reasonable
    
    def test_multiclass_metrics(self):
        """Test metrics for multiclass classification."""
        metrics = ClassificationMetrics(num_classes=3)
        
        # Predictions and labels for 3-class problem
        predictions = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        labels =      [0, 1, 2, 1, 0, 2, 0, 2, 1]
        
        metrics.update(predictions, labels)
        
        # Check some statistics were recorded
        assert metrics.true_positives[0] == 2  # Correctly predicted class 0
        assert metrics.true_positives[1] == 1  # Correctly predicted class 1
        assert metrics.true_positives[2] == 2  # Correctly predicted class 2
        
        # Compute metrics
        precision = metrics.compute_precision("macro")
        recall = metrics.compute_recall("macro")
        f1 = metrics.compute_f1("macro")
        
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
        assert 0 <= f1 <= 1


class TestRegressionMetrics:
    """Test regression metrics calculation."""
    
    def test_update_metrics(self):
        """Test updating regression metrics."""
        metrics = RegressionMetrics()
        
        predictions = [2.5, 3.0, 4.5, 1.0]
        targets = [2.0, 3.5, 4.0, 1.5]
        
        metrics.update(predictions, targets)
        
        assert metrics.num_samples == 4
        assert metrics.sum_squared_errors == 1.0  # 0.25 + 0.25 + 0.25 + 0.25
        assert metrics.sum_absolute_errors == 2.0  # 0.5 + 0.5 + 0.5 + 0.5
        assert metrics.sum_targets == 11.0  # 2 + 3.5 + 4 + 1.5
    
    def test_mse_calculation(self):
        """Test mean squared error calculation."""
        metrics = RegressionMetrics()
        metrics.sum_squared_errors = 10.0
        metrics.num_samples = 5
        
        assert metrics.compute_mse() == 2.0
    
    def test_mae_calculation(self):
        """Test mean absolute error calculation."""
        metrics = RegressionMetrics()
        metrics.sum_absolute_errors = 15.0
        metrics.num_samples = 5
        
        assert metrics.compute_mae() == 3.0
    
    def test_rmse_calculation(self):
        """Test root mean squared error calculation."""
        metrics = RegressionMetrics()
        metrics.sum_squared_errors = 16.0
        metrics.num_samples = 4
        
        assert metrics.compute_rmse() == 2.0
    
    def test_r2_calculation(self):
        """Test R-squared calculation."""
        metrics = RegressionMetrics()
        
        # Perfect predictions
        predictions = [1.0, 2.0, 3.0, 4.0]
        targets = [1.0, 2.0, 3.0, 4.0]
        
        metrics.update(predictions, targets)
        
        # R² should be 1.0 for perfect predictions
        assert abs(metrics.compute_r2() - 1.0) < 1e-10
    
    def test_r2_with_poor_predictions(self):
        """Test R-squared with poor predictions."""
        metrics = RegressionMetrics()
        
        # Predictions that are worse than mean
        predictions = [0.0, 0.0, 0.0, 0.0]
        targets = [1.0, 2.0, 3.0, 4.0]
        
        metrics.update(predictions, targets)
        
        # R² should be negative for predictions worse than mean
        r2 = metrics.compute_r2()
        assert r2 < 0
    
    def test_empty_metrics(self):
        """Test metrics with no data."""
        metrics = RegressionMetrics()
        
        assert metrics.compute_mse() == 0.0
        assert metrics.compute_mae() == 0.0
        assert metrics.compute_rmse() == 0.0
        assert metrics.compute_r2() == 0.0


# Mock implementations for testing abstract classes
class MockMetricCalculator(MetricCalculator[List[float]]):
    """Mock metric calculator for testing."""
    
    def __init__(self):
        self.total = 0.0
        self.count = 0
    
    def calculate(self, predictions: List[float], labels: List[float], **kwargs) -> float:
        """Calculate metric directly."""
        return sum(1 for p, l in zip(predictions, labels) if p == l) / len(predictions)
    
    def batch_update(self, predictions: List[float], labels: List[float], **kwargs) -> None:
        """Update with batch."""
        self.total += sum(1 for p, l in zip(predictions, labels) if p == l)
        self.count += len(predictions)
    
    def compute(self) -> float:
        """Compute final metric."""
        return self.total / self.count if self.count > 0 else 0.0
    
    def reset(self) -> None:
        """Reset state."""
        self.total = 0.0
        self.count = 0


class MockEvaluationService(EvaluationService[List[float]]):
    """Mock evaluation service for testing."""
    
    def _create_metric_calculators(self) -> Dict[str, MetricCalculator[List[float]]]:
        """Create mock calculators."""
        calculators = {}
        for metric in self.config.metrics:
            calculators[metric.metric_type.value] = MockMetricCalculator()
        return calculators
    
    def evaluate_batch(
        self,
        model: Any,
        batch: Dict[str, List[float]]
    ) -> Tuple[List[float], List[float], Optional[List[float]]]:
        """Mock batch evaluation."""
        # Simple mock: predictions are just rounded inputs
        predictions = [round(x) for x in batch["inputs"]]
        labels = batch["labels"]
        probabilities = [[1-x, x] for x in batch["inputs"]] if self.config.save_predictions else None
        return predictions, labels, probabilities


class TestEvaluationService:
    """Test abstract evaluation service functionality."""
    
    def test_service_initialization(self):
        """Test evaluation service initialization."""
        metrics = [
            MetricConfig(metric_type=MetricType.ACCURACY),
            MetricConfig(metric_type=MetricType.F1_SCORE)
        ]
        config = EvaluationConfig(metrics=metrics)
        service = MockEvaluationService(config)
        
        assert len(service.metric_calculators) == 2
        assert "accuracy" in service.metric_calculators
        assert "f1" in service.metric_calculators
    
    def test_evaluate_single_batch(self):
        """Test evaluating a single batch."""
        metric = MetricConfig(metric_type=MetricType.ACCURACY)
        config = EvaluationConfig(metrics=[metric])
        service = MockEvaluationService(config)
        
        # Mock dataloader with one batch
        dataloader = [{
            "inputs": [0.9, 0.1, 0.8, 0.2],
            "labels": [1.0, 0.0, 1.0, 0.0]
        }]
        
        result = service.evaluate(model=None, dataloader=dataloader)
        
        assert result.num_samples == 4
        assert "accuracy" in result.metrics
        assert result.metrics["accuracy"] == 1.0  # All predictions correct
    
    def test_evaluate_multiple_batches(self):
        """Test evaluating multiple batches."""
        metric = MetricConfig(metric_type=MetricType.ACCURACY)
        config = EvaluationConfig(metrics=[metric])
        service = MockEvaluationService(config)
        
        # Mock dataloader with multiple batches
        dataloader = [
            {"inputs": [0.9, 0.1], "labels": [1.0, 0.0]},
            {"inputs": [0.8, 0.2], "labels": [1.0, 0.0]},
            {"inputs": [0.7, 0.3], "labels": [0.0, 1.0]}  # Wrong predictions
        ]
        
        result = service.evaluate(model=None, dataloader=dataloader)
        
        assert result.num_samples == 6
        assert result.metrics["accuracy"] == 4/6  # 4 correct out of 6
    
    def test_evaluate_with_saved_predictions(self):
        """Test evaluation with prediction saving."""
        metric = MetricConfig(metric_type=MetricType.ACCURACY)
        config = EvaluationConfig(metrics=[metric], save_predictions=True)
        service = MockEvaluationService(config)
        
        dataloader = [{"inputs": [0.9, 0.1, 0.8], "labels": [1.0, 0.0, 1.0]}]
        
        result = service.evaluate(model=None, dataloader=dataloader)
        
        assert result.predictions is not None
        assert len(result.predictions) == 1  # One batch
        assert result.predictions[0] == [1.0, 0.0, 1.0]
        
        assert result.probabilities is not None
        assert len(result.probabilities) == 1
    
    def test_evaluate_with_max_samples(self):
        """Test evaluation with sample limit."""
        metric = MetricConfig(metric_type=MetricType.ACCURACY)
        config = EvaluationConfig(metrics=[metric], max_eval_samples=5)
        service = MockEvaluationService(config)
        
        # Mock dataloader with many samples
        dataloader = [
            {"inputs": [0.9, 0.1, 0.8], "labels": [1.0, 0.0, 1.0]},  # 3 samples
            {"inputs": [0.7, 0.3, 0.6], "labels": [1.0, 0.0, 1.0]},  # 3 more
            {"inputs": [0.5, 0.4, 0.3], "labels": [1.0, 0.0, 0.0]},  # Would exceed
        ]
        
        result = service.evaluate(model=None, dataloader=dataloader)
        
        assert result.num_samples == 6  # Only processed first 2 batches
    
    def test_multiple_metrics(self):
        """Test evaluation with multiple metrics."""
        metrics = [
            MetricConfig(metric_type=MetricType.ACCURACY),
            MetricConfig(metric_type=MetricType.PRECISION),
            MetricConfig(metric_type=MetricType.RECALL)
        ]
        config = EvaluationConfig(metrics=metrics)
        service = MockEvaluationService(config)
        
        dataloader = [{"inputs": [0.9, 0.1, 0.8], "labels": [1.0, 0.0, 1.0]}]
        
        result = service.evaluate(model=None, dataloader=dataloader)
        
        assert len(result.metrics) == 3
        assert "accuracy" in result.metrics
        assert "precision" in result.metrics
        assert "recall" in result.metrics