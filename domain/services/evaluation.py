"""Evaluation service for model assessment."""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from domain.entities.model import BertModel
from domain.entities.dataset import Dataset, DataBatch
from domain.entities.metrics import EvaluationMetrics
from domain.ports.compute import ComputePort
from domain.ports.data import DataLoaderPort
from domain.ports.monitoring import MonitoringPort
from domain.ports.metrics import MetricsCalculatorPort


@dataclass
class EvaluationService:
    """Service for evaluating BERT models."""
    compute_port: ComputePort
    data_loader_port: DataLoaderPort
    monitoring_port: MonitoringPort
    metrics_port: MetricsCalculatorPort
    
    def evaluate(
        self,
        model: BertModel,
        dataset: Dataset,
        batch_size: int = 32,
        num_workers: int = 0,
    ) -> EvaluationMetrics:
        """Evaluate model on dataset.
        
        Args:
            model: Model to evaluate
            dataset: Evaluation dataset
            batch_size: Batch size for evaluation
            num_workers: Number of data loading workers
            
        Returns:
            Evaluation metrics
        """
        # Create data loader
        data_loader = self.data_loader_port.create_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            prefetch_size=2,
        )
        
        # Initialize tracking
        all_predictions = []
        all_labels = []
        all_probabilities = []
        total_loss = 0.0
        num_samples = 0
        
        # Create progress bar
        progress = self.monitoring_port.create_progress_bar(
            total=len(data_loader),
            description=f"Evaluating {dataset.name}",
            unit="batch",
        )
        
        # Evaluation loop
        for batch in data_loader:
            # Forward pass (no gradients needed)
            outputs = self.compute_port.forward(
                model=model,
                batch=batch,
                training=False,
            )
            
            # Extract outputs
            logits = outputs['logits']
            loss = outputs.get('loss', 0.0)
            
            # Accumulate loss
            batch_size = batch.batch_size
            total_loss += loss * batch_size
            num_samples += batch_size
            
            # Process predictions based on task type
            if dataset.is_classification:
                # Convert logits to predictions
                predictions = self._logits_to_predictions(logits)
                probabilities = self._logits_to_probabilities(logits)
                
                all_predictions.extend(predictions)
                all_probabilities.extend(probabilities)
            else:
                # Regression - logits are predictions
                all_predictions.extend(logits)
            
            # Collect labels
            if batch.labels is not None:
                all_labels.extend(batch.labels)
            
            progress.update(1)
        
        progress.close()
        
        # Calculate metrics
        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        
        metrics = EvaluationMetrics(
            dataset_name=dataset.name,
            split=dataset.split.value,
            loss=avg_loss,
            total_samples=num_samples,
        )
        
        # Calculate task-specific metrics
        if all_labels:
            if dataset.is_classification:
                self._add_classification_metrics(
                    metrics,
                    all_predictions,
                    all_labels,
                    all_probabilities,
                    dataset.num_classes,
                )
            else:
                self._add_regression_metrics(
                    metrics,
                    all_predictions,
                    all_labels,
                )
        
        # Log metrics
        self.monitoring_port.log_evaluation_metrics(metrics)
        
        return metrics
    
    def _logits_to_predictions(self, logits: Any) -> List[int]:
        """Convert logits to class predictions."""
        # Implementation depends on framework
        # This is placeholder logic
        return [0] * len(logits)
    
    def _logits_to_probabilities(self, logits: Any) -> List[List[float]]:
        """Convert logits to probabilities."""
        # Implementation depends on framework
        # This is placeholder logic
        return [[0.0] * 2 for _ in range(len(logits))]
    
    def _add_classification_metrics(
        self,
        metrics: EvaluationMetrics,
        predictions: List[int],
        labels: List[int],
        probabilities: List[List[float]],
        num_classes: Optional[int],
    ) -> None:
        """Add classification metrics to evaluation results."""
        # Calculate accuracy
        metrics.accuracy = self.metrics_port.calculate_accuracy(predictions, labels)
        
        # Calculate precision, recall, F1
        prf_scores = self.metrics_port.calculate_precision_recall_f1(
            predictions, labels, average="macro"
        )
        metrics.precision = prf_scores['precision']
        metrics.recall = prf_scores['recall']
        metrics.f1_score = prf_scores['f1']
        
        # Calculate confusion matrix
        metrics.confusion_matrix = self.metrics_port.calculate_confusion_matrix(
            predictions, labels
        )
        
        # Calculate AUC scores if binary or probabilities available
        if num_classes == 2 or (num_classes and num_classes <= 10):
            try:
                # Flatten probabilities for binary, or use multi-class
                if num_classes == 2:
                    binary_probs = [p[1] for p in probabilities]
                    metrics.auc_roc = self.metrics_port.calculate_auc_roc(
                        binary_probs, labels
                    )
                    metrics.auc_pr = self.metrics_port.calculate_auc_pr(
                        binary_probs, labels
                    )
                else:
                    metrics.auc_roc = self.metrics_port.calculate_auc_roc(
                        probabilities, labels, multi_class="ovr"
                    )
            except Exception:
                # AUC calculation might fail for some cases
                pass
        
        # Calculate per-class metrics
        per_class = self.metrics_port.calculate_per_class_metrics(
            predictions, labels
        )
        metrics.per_class_precision = per_class.get('precision')
        metrics.per_class_recall = per_class.get('recall')
        metrics.per_class_f1 = per_class.get('f1')
    
    def _add_regression_metrics(
        self,
        metrics: EvaluationMetrics,
        predictions: List[float],
        targets: List[float],
    ) -> None:
        """Add regression metrics to evaluation results."""
        # Calculate MSE
        metrics.mse = self.metrics_port.calculate_mse(predictions, targets)
        
        # Calculate MAE
        metrics.mae = self.metrics_port.calculate_mae(predictions, targets)
        
        # Calculate R2 score
        metrics.r2_score = self.metrics_port.calculate_r2_score(predictions, targets)
    
    def evaluate_with_tta(
        self,
        model: BertModel,
        dataset: Dataset,
        num_augmentations: int = 5,
        batch_size: int = 32,
    ) -> EvaluationMetrics:
        """Evaluate with test-time augmentation.
        
        Args:
            model: Model to evaluate
            dataset: Evaluation dataset
            num_augmentations: Number of augmented versions
            batch_size: Batch size
            
        Returns:
            Aggregated evaluation metrics
        """
        all_metrics = []
        
        for i in range(num_augmentations):
            # Each iteration would use different augmentation
            # This is simplified - actual implementation would
            # apply different augmentations
            metrics = self.evaluate(
                model=model,
                dataset=dataset,
                batch_size=batch_size,
            )
            all_metrics.append(metrics)
        
        # Aggregate metrics
        # This is simplified - would need proper aggregation logic
        return all_metrics[0]
    
    def cross_validate(
        self,
        model: BertModel,
        dataset: Dataset,
        num_folds: int = 5,
        batch_size: int = 32,
    ) -> Dict[str, List[float]]:
        """Perform k-fold cross-validation.
        
        Args:
            model: Model to evaluate
            dataset: Full dataset
            num_folds: Number of folds
            batch_size: Batch size
            
        Returns:
            Dictionary of metric name to list of values per fold
        """
        # This would split dataset and evaluate on each fold
        # Simplified implementation
        metrics = self.evaluate(model, dataset, batch_size)
        
        return {
            "accuracy": [metrics.accuracy] * num_folds if metrics.accuracy else [],
            "f1_score": [metrics.f1_score] * num_folds if metrics.f1_score else [],
            "loss": [metrics.loss] * num_folds,
        }