"""Evaluation engine service for model evaluation logic.

This service contains the business logic for evaluating models,
independent of any specific ML framework.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from collections import defaultdict

from domain.entities.model import BertModel, TaskType
from domain.entities.dataset import Dataset, DataBatch
from domain.entities.metrics import EvaluationMetrics


class MetricType(Enum):
    """Types of evaluation metrics."""
    # Classification metrics
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    AUC_ROC = "auc_roc"
    AUC_PR = "auc_pr"
    
    # Regression metrics
    MSE = "mse"
    MAE = "mae"
    RMSE = "rmse"
    R2_SCORE = "r2_score"
    
    # General metrics
    LOSS = "loss"
    PERPLEXITY = "perplexity"


@dataclass
class EvaluationPlan:
    """Plan for model evaluation."""
    dataset_name: str
    dataset_split: str
    task_type: TaskType
    metrics_to_compute: List[MetricType]
    batch_size: int
    num_batches: int
    total_samples: int
    
    # Options
    save_predictions: bool = False
    compute_confidence: bool = False
    per_class_metrics: bool = True
    
    @property
    def is_classification(self) -> bool:
        """Check if this is a classification task."""
        return self.task_type in [
            TaskType.CLASSIFICATION,
            TaskType.TOKEN_CLASSIFICATION
        ]
    
    @property
    def is_regression(self) -> bool:
        """Check if this is a regression task."""
        return self.task_type == TaskType.REGRESSION


@dataclass
class PredictionResult:
    """Result of a single prediction."""
    sample_id: str
    prediction: Any  # Class index, probability distribution, or regression value
    ground_truth: Optional[Any] = None
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = None


class EvaluationEngine:
    """Engine for model evaluation logic.
    
    This service manages the evaluation process and metrics
    calculation, independent of ML framework specifics.
    """
    
    def create_evaluation_plan(
        self,
        model: BertModel,
        dataset: Dataset,
        batch_size: int = 32,
        metrics: Optional[List[str]] = None
    ) -> EvaluationPlan:
        """Create an evaluation plan.
        
        Args:
            model: Model to evaluate
            dataset: Dataset to evaluate on
            batch_size: Batch size for evaluation
            metrics: Optional list of specific metrics to compute
            
        Returns:
            EvaluationPlan with computed values
        """
        # Determine task type
        task_type = model.task_type or TaskType.CLASSIFICATION
        
        # Determine metrics to compute
        if metrics:
            metrics_to_compute = [MetricType(m) for m in metrics]
        else:
            metrics_to_compute = self._get_default_metrics(task_type)
        
        # Calculate batches
        num_batches = dataset.size // batch_size
        if dataset.size % batch_size != 0:
            num_batches += 1
        
        return EvaluationPlan(
            dataset_name=dataset.name,
            dataset_split=dataset.split.value,
            task_type=task_type,
            metrics_to_compute=metrics_to_compute,
            batch_size=batch_size,
            num_batches=num_batches,
            total_samples=dataset.size,
            per_class_metrics=task_type == TaskType.CLASSIFICATION
        )
    
    def aggregate_predictions(
        self,
        predictions: List[PredictionResult],
        plan: EvaluationPlan
    ) -> Dict[str, Any]:
        """Aggregate predictions into final metrics.
        
        Args:
            predictions: List of prediction results
            plan: Evaluation plan
            
        Returns:
            Dictionary of aggregated metrics
        """
        if not predictions:
            return {}
        
        # Separate predictions and ground truth
        y_pred = [p.prediction for p in predictions]
        y_true = [p.ground_truth for p in predictions if p.ground_truth is not None]
        
        if not y_true:
            # No ground truth, can only return predictions
            return {"predictions": y_pred}
        
        # Calculate metrics based on task type
        if plan.is_classification:
            return self._calculate_classification_metrics(
                y_true, y_pred, plan
            )
        elif plan.is_regression:
            return self._calculate_regression_metrics(
                y_true, y_pred, plan
            )
        else:
            return {"predictions": y_pred}
    
    def calculate_confusion_matrix(
        self,
        y_true: List[int],
        y_pred: List[int],
        num_classes: int
    ) -> List[List[int]]:
        """Calculate confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            num_classes: Number of classes
            
        Returns:
            Confusion matrix as 2D list
        """
        matrix = [[0] * num_classes for _ in range(num_classes)]
        
        for true, pred in zip(y_true, y_pred):
            if 0 <= true < num_classes and 0 <= pred < num_classes:
                matrix[true][pred] += 1
        
        return matrix
    
    def calculate_per_class_metrics(
        self,
        confusion_matrix: List[List[int]],
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """Calculate per-class metrics from confusion matrix.
        
        Args:
            confusion_matrix: Confusion matrix
            class_names: Optional class names
            
        Returns:
            Per-class precision, recall, and F1 scores
        """
        num_classes = len(confusion_matrix)
        if class_names is None:
            class_names = [f"class_{i}" for i in range(num_classes)]
        
        per_class = {}
        
        for i in range(num_classes):
            # True positives
            tp = confusion_matrix[i][i]
            
            # False positives (predicted as i but not actually i)
            fp = sum(confusion_matrix[j][i] for j in range(num_classes) if j != i)
            
            # False negatives (actually i but not predicted as i)
            fn = sum(confusion_matrix[i][j] for j in range(num_classes) if j != i)
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            per_class[class_names[i]] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "support": tp + fn  # Total actual instances of this class
            }
        
        return per_class
    
    def validate_evaluation_setup(
        self,
        model: BertModel,
        dataset: Dataset
    ) -> List[str]:
        """Validate evaluation setup.
        
        Args:
            model: Model to evaluate
            dataset: Dataset to evaluate on
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Check model has task head for non-MLM tasks
        if model.task_type != TaskType.MASKED_LM and not model.task_head:
            errors.append("Model must have task head for evaluation")
        
        # Check dataset compatibility
        if dataset.is_empty:
            errors.append("Dataset cannot be empty")
        
        if model.task_head and dataset.num_classes:
            if model.num_labels != dataset.num_classes:
                errors.append(
                    f"Model labels ({model.num_labels}) don't match "
                    f"dataset classes ({dataset.num_classes})"
                )
        
        return errors
    
    def _get_default_metrics(self, task_type: TaskType) -> List[MetricType]:
        """Get default metrics for task type."""
        if task_type == TaskType.CLASSIFICATION:
            return [
                MetricType.ACCURACY,
                MetricType.PRECISION,
                MetricType.RECALL,
                MetricType.F1_SCORE,
                MetricType.LOSS
            ]
        elif task_type == TaskType.REGRESSION:
            return [
                MetricType.MSE,
                MetricType.MAE,
                MetricType.R2_SCORE,
                MetricType.LOSS
            ]
        elif task_type == TaskType.TOKEN_CLASSIFICATION:
            return [
                MetricType.ACCURACY,
                MetricType.F1_SCORE,
                MetricType.LOSS
            ]
        else:
            return [MetricType.LOSS]
    
    def _calculate_classification_metrics(
        self,
        y_true: List[int],
        y_pred: List[int],
        plan: EvaluationPlan
    ) -> Dict[str, Any]:
        """Calculate classification metrics."""
        metrics = {}
        
        # Accuracy
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        metrics["accuracy"] = correct / len(y_true) if y_true else 0.0
        
        # Get unique classes
        classes = sorted(set(y_true + y_pred))
        num_classes = len(classes)
        
        # Confusion matrix
        if plan.per_class_metrics and num_classes > 1:
            conf_matrix = self.calculate_confusion_matrix(
                y_true, y_pred, num_classes
            )
            metrics["confusion_matrix"] = conf_matrix
            
            # Per-class metrics
            per_class = self.calculate_per_class_metrics(conf_matrix)
            metrics["per_class_metrics"] = per_class
            
            # Macro averages
            metrics["precision"] = sum(
                m["precision"] for m in per_class.values()
            ) / len(per_class)
            metrics["recall"] = sum(
                m["recall"] for m in per_class.values()
            ) / len(per_class)
            metrics["f1_score"] = sum(
                m["f1_score"] for m in per_class.values()
            ) / len(per_class)
        
        return metrics
    
    def _calculate_regression_metrics(
        self,
        y_true: List[float],
        y_pred: List[float],
        plan: EvaluationPlan
    ) -> Dict[str, float]:
        """Calculate regression metrics."""
        metrics = {}
        n = len(y_true)
        
        if n == 0:
            return metrics
        
        # MSE
        mse = sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / n
        metrics["mse"] = mse
        
        # RMSE
        metrics["rmse"] = mse ** 0.5
        
        # MAE
        mae = sum(abs(t - p) for t, p in zip(y_true, y_pred)) / n
        metrics["mae"] = mae
        
        # R² score
        y_mean = sum(y_true) / n
        ss_tot = sum((t - y_mean) ** 2 for t in y_true)
        ss_res = sum((t - p) ** 2 for t, p in zip(y_true, y_pred))
        
        if ss_tot > 0:
            metrics["r2_score"] = 1 - (ss_res / ss_tot)
        else:
            metrics["r2_score"] = 0.0
        
        return metrics