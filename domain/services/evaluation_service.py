"""Domain service for model evaluation logic.

This module contains the pure business logic for evaluating BERT models,
free from any framework dependencies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, TypeVar, Generic, Tuple
from enum import Enum
import math
from collections import defaultdict


TArray = TypeVar('TArray')


class MetricType(Enum):
    """Types of evaluation metrics."""
    # Classification metrics
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1"
    AUC_ROC = "auc_roc"
    AUC_PR = "auc_pr"
    
    # Regression metrics
    MSE = "mse"
    MAE = "mae"
    RMSE = "rmse"
    R2 = "r2"
    MAPE = "mape"
    
    # Ranking metrics
    MRR = "mrr"
    NDCG = "ndcg"
    MAP = "map"
    
    # Multi-label metrics
    HAMMING_LOSS = "hamming_loss"
    EXACT_MATCH = "exact_match"
    
    # General metrics
    LOSS = "loss"
    PERPLEXITY = "perplexity"


@dataclass
class MetricConfig:
    """Configuration for a metric."""
    
    metric_type: MetricType
    primary: bool = False  # Is this the primary metric for model selection?
    greater_is_better: bool = True
    threshold: Optional[float] = None  # For binary classification
    average: str = "macro"  # For multi-class metrics: "micro", "macro", "weighted"
    top_k: Optional[int] = None  # For ranking metrics
    
    def __post_init__(self):
        """Set default values based on metric type."""
        if self.metric_type in [MetricType.LOSS, MetricType.MSE, MetricType.MAE, 
                                MetricType.RMSE, MetricType.HAMMING_LOSS]:
            self.greater_is_better = False
        
        if self.metric_type in [MetricType.ACCURACY, MetricType.EXACT_MATCH]:
            self.average = "micro"  # These are typically computed at instance level


@dataclass
class EvaluationConfig:
    """Configuration for evaluation process."""
    
    metrics: List[MetricConfig]
    batch_size: int = 32
    use_mixed_precision: bool = False
    compute_confidence: bool = False
    save_predictions: bool = False
    save_embeddings: bool = False
    
    # Thresholds and settings
    classification_threshold: float = 0.5
    confidence_threshold: Optional[float] = None
    
    # Sampling
    max_eval_samples: Optional[int] = None
    stratified_sampling: bool = False
    
    def __post_init__(self):
        """Validate configuration."""
        if not self.metrics:
            raise ValueError("At least one metric must be specified")
            
        primary_metrics = [m for m in self.metrics if m.primary]
        if len(primary_metrics) > 1:
            raise ValueError("Only one metric can be marked as primary")
        elif len(primary_metrics) == 0:
            # Mark first metric as primary
            self.metrics[0].primary = True
    
    @property
    def primary_metric(self) -> MetricConfig:
        """Get the primary metric for model selection."""
        return next(m for m in self.metrics if m.primary)
    
    @property
    def metric_names(self) -> List[str]:
        """Get list of metric names."""
        return [m.metric_type.value for m in self.metrics]


@dataclass
class EvaluationResult:
    """Results from model evaluation."""
    
    # Computed metrics
    metrics: Dict[str, float]
    
    # Detailed results
    predictions: Optional[TArray] = None
    probabilities: Optional[TArray] = None
    embeddings: Optional[TArray] = None
    
    # Confidence estimates
    confidence_scores: Optional[TArray] = None
    uncertainty_estimates: Optional[TArray] = None
    
    # Performance breakdown
    per_class_metrics: Optional[Dict[str, Dict[str, float]]] = None
    confusion_matrix: Optional[TArray] = None
    
    # Metadata
    num_samples: int = 0
    evaluation_time_seconds: float = 0.0
    
    def get_primary_metric(self, metric_name: str) -> float:
        """Get the primary metric value."""
        if metric_name not in self.metrics:
            raise ValueError(f"Metric '{metric_name}' not found in results")
        return self.metrics[metric_name]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get evaluation summary."""
        return {
            "metrics": self.metrics,
            "num_samples": self.num_samples,
            "evaluation_time": self.evaluation_time_seconds,
            "samples_per_second": (
                self.num_samples / self.evaluation_time_seconds 
                if self.evaluation_time_seconds > 0 else 0
            ),
        }


class MetricCalculator(ABC, Generic[TArray]):
    """Abstract base for metric calculation."""
    
    @abstractmethod
    def calculate(
        self,
        predictions: TArray,
        labels: TArray,
        **kwargs
    ) -> float:
        """Calculate metric value."""
        pass
    
    @abstractmethod
    def batch_update(
        self,
        predictions: TArray,
        labels: TArray,
        **kwargs
    ) -> None:
        """Update metric state with batch results."""
        pass
    
    @abstractmethod
    def compute(self) -> float:
        """Compute final metric from accumulated state."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset metric state."""
        pass


@dataclass
class ClassificationMetrics:
    """Metrics specific to classification tasks."""
    
    num_classes: int
    threshold: float = 0.5
    
    # Accumulated statistics
    true_positives: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    false_positives: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    false_negatives: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    true_negatives: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    
    def update(self, predictions: List[int], labels: List[int]) -> None:
        """Update metrics with batch results."""
        for pred, label in zip(predictions, labels):
            for class_idx in range(self.num_classes):
                if label == class_idx:
                    if pred == class_idx:
                        self.true_positives[class_idx] += 1
                    else:
                        self.false_negatives[class_idx] += 1
                else:
                    if pred == class_idx:
                        self.false_positives[class_idx] += 1
                    else:
                        self.true_negatives[class_idx] += 1
    
    def compute_precision(self, average: str = "macro") -> float:
        """Compute precision metric."""
        precisions = []
        for class_idx in range(self.num_classes):
            tp = self.true_positives[class_idx]
            fp = self.false_positives[class_idx]
            if tp + fp > 0:
                precisions.append(tp / (tp + fp))
            else:
                precisions.append(0.0)
        
        if average == "macro":
            return sum(precisions) / len(precisions)
        elif average == "micro":
            total_tp = sum(self.true_positives.values())
            total_fp = sum(self.false_positives.values())
            return total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
        else:
            return precisions
    
    def compute_recall(self, average: str = "macro") -> float:
        """Compute recall metric."""
        recalls = []
        for class_idx in range(self.num_classes):
            tp = self.true_positives[class_idx]
            fn = self.false_negatives[class_idx]
            if tp + fn > 0:
                recalls.append(tp / (tp + fn))
            else:
                recalls.append(0.0)
        
        if average == "macro":
            return sum(recalls) / len(recalls)
        elif average == "micro":
            total_tp = sum(self.true_positives.values())
            total_fn = sum(self.false_negatives.values())
            return total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
        else:
            return recalls
    
    def compute_f1(self, average: str = "macro") -> float:
        """Compute F1 score."""
        precision = self.compute_precision(average)
        recall = self.compute_recall(average)
        
        if isinstance(precision, list):
            f1_scores = []
            for p, r in zip(precision, recall):
                if p + r > 0:
                    f1_scores.append(2 * p * r / (p + r))
                else:
                    f1_scores.append(0.0)
            return f1_scores
        else:
            if precision + recall > 0:
                return 2 * precision * recall / (precision + recall)
            else:
                return 0.0
    
    def compute_accuracy(self) -> float:
        """Compute accuracy metric."""
        total_correct = sum(self.true_positives.values())
        total_samples = sum(
            self.true_positives.values() + 
            self.false_positives.values() + 
            self.false_negatives.values() + 
            self.true_negatives.values()
        ) / self.num_classes  # Adjust for multi-class counting
        
        return total_correct / total_samples if total_samples > 0 else 0.0


@dataclass
class RegressionMetrics:
    """Metrics specific to regression tasks."""
    
    # Accumulated statistics
    sum_squared_errors: float = 0.0
    sum_absolute_errors: float = 0.0
    sum_targets: float = 0.0
    sum_squared_targets: float = 0.0
    num_samples: int = 0
    
    def update(self, predictions: List[float], targets: List[float]) -> None:
        """Update metrics with batch results."""
        for pred, target in zip(predictions, targets):
            error = pred - target
            self.sum_squared_errors += error ** 2
            self.sum_absolute_errors += abs(error)
            self.sum_targets += target
            self.sum_squared_targets += target ** 2
            self.num_samples += 1
    
    def compute_mse(self) -> float:
        """Compute mean squared error."""
        return self.sum_squared_errors / self.num_samples if self.num_samples > 0 else 0.0
    
    def compute_mae(self) -> float:
        """Compute mean absolute error."""
        return self.sum_absolute_errors / self.num_samples if self.num_samples > 0 else 0.0
    
    def compute_rmse(self) -> float:
        """Compute root mean squared error."""
        return math.sqrt(self.compute_mse())
    
    def compute_r2(self) -> float:
        """Compute R-squared metric."""
        if self.num_samples == 0:
            return 0.0
            
        mean_target = self.sum_targets / self.num_samples
        total_variance = self.sum_squared_targets - self.num_samples * mean_target ** 2
        
        if total_variance == 0:
            return 0.0
            
        explained_variance = total_variance - self.sum_squared_errors
        return explained_variance / total_variance


class EvaluationService(ABC, Generic[TArray]):
    """Abstract evaluation service defining evaluation logic."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.metric_calculators = self._create_metric_calculators()
    
    @abstractmethod
    def _create_metric_calculators(self) -> Dict[str, MetricCalculator[TArray]]:
        """Create metric calculator instances."""
        pass
    
    @abstractmethod
    def evaluate_batch(
        self,
        model: Any,
        batch: Dict[str, TArray]
    ) -> Tuple[TArray, TArray, Optional[TArray]]:
        """Evaluate single batch, returning predictions, labels, and optional probs."""
        pass
    
    def evaluate(
        self,
        model: Any,
        dataloader: Any
    ) -> EvaluationResult:
        """Run complete evaluation."""
        # Reset all metric calculators
        for calculator in self.metric_calculators.values():
            calculator.reset()
        
        all_predictions = []
        all_probabilities = []
        num_samples = 0
        
        # Evaluate all batches
        for batch in dataloader:
            predictions, labels, probabilities = self.evaluate_batch(model, batch)
            
            # Update metrics
            for metric_name, calculator in self.metric_calculators.items():
                calculator.batch_update(predictions, labels)
            
            # Store results if needed
            if self.config.save_predictions:
                all_predictions.append(predictions)
            if self.config.save_predictions and probabilities is not None:
                all_probabilities.append(probabilities)
            
            num_samples += len(predictions)
            
            if (self.config.max_eval_samples is not None and 
                num_samples >= self.config.max_eval_samples):
                break
        
        # Compute final metrics
        metrics = {}
        for metric_name, calculator in self.metric_calculators.items():
            metrics[metric_name] = calculator.compute()
        
        return EvaluationResult(
            metrics=metrics,
            predictions=all_predictions if all_predictions else None,
            probabilities=all_probabilities if all_probabilities else None,
            num_samples=num_samples,
        )


class ConfidenceEstimator(ABC, Generic[TArray]):
    """Abstract base for confidence estimation."""
    
    @abstractmethod
    def estimate_confidence(
        self,
        probabilities: TArray,
        predictions: TArray
    ) -> TArray:
        """Estimate confidence scores for predictions."""
        pass


class UncertaintyEstimator(ABC, Generic[TArray]):
    """Abstract base for uncertainty estimation."""
    
    @abstractmethod
    def estimate_uncertainty(
        self,
        probabilities: TArray,
        predictions: TArray,
        embeddings: Optional[TArray] = None
    ) -> TArray:
        """Estimate uncertainty for predictions."""
        pass