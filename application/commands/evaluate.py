"""Evaluate Model Command - orchestrates model evaluation."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from application.dto.evaluation import EvaluationRequestDTO, EvaluationResponseDTO
from domain.entities.model import BertModel
from domain.entities.dataset import Dataset
from domain.entities.metrics import EvaluationMetrics
from ports.secondary.metrics import MetricType
from domain.services import EvaluationService
from ports.secondary.data import DataLoaderPort
from ports.secondary.compute import ComputeBackend as ComputePort
from ports.secondary.monitoring import MonitoringService as MonitoringPort
from ports.secondary.storage import StorageService as StoragePort
from ports.secondary.metrics import MetricsCollector as MetricsCalculatorPort


@dataclass
class EvaluateModelCommand:
    """Command to evaluate a trained model.
    
    This command orchestrates the evaluation workflow by:
    1. Loading the model from checkpoint
    2. Loading and preparing test data
    3. Running evaluation
    4. Computing metrics
    5. Generating evaluation report
    """
    
    # Domain services
    evaluation_service: EvaluationService
    
    # Ports
    data_loader_port: DataLoaderPort
    compute_port: ComputePort
    monitoring_port: MonitoringPort
    storage_port: StoragePort
    metrics_port: MetricsCalculatorPort
    
    async def execute(self, request: EvaluationRequestDTO) -> EvaluationResponseDTO:
        """Execute the evaluation command.
        
        Args:
            request: Evaluation request with model and data paths
            
        Returns:
            Response with evaluation metrics and results
        """
        start_time = datetime.now()
        
        try:
            # 1. Validate request
            errors = request.validate()
            if errors:
                return EvaluationResponseDTO(
                    success=False,
                    error_message=f"Validation errors: {', '.join(errors)}"
                )
            
            # 2. Initialize monitoring
            await self.monitoring_port.start_run(
                name=f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                experiment=request.experiment_name,
                tags={"type": "evaluation", "model": str(request.model_path)}
            )
            
            # 3. Load model
            model = await self._load_model(request.model_path)
            
            # 4. Initialize compute backend
            await self.compute_port.initialize(model)
            
            # 5. Load test dataset
            test_dataset = await self._load_dataset(request.test_data_path)
            
            # 6. Create data loader
            test_loader = await self.data_loader_port.create_loader(
                test_dataset,
                batch_size=request.batch_size,
                shuffle=False,
                num_workers=request.num_workers
            )
            
            # 7. Run evaluation
            eval_results = await self.evaluation_service.evaluate(
                model=model,
                data_loader=test_loader,
                compute_port=self.compute_port,
                metrics_to_compute=request.metrics or self._get_default_metrics(test_dataset)
            )
            
            # 8. Calculate additional metrics if requested
            if request.compute_confidence:
                confidence_metrics = await self._compute_confidence_metrics(
                    model, test_loader, eval_results
                )
                eval_results.confidence_metrics = confidence_metrics
            
            if request.compute_per_class_metrics:
                per_class_metrics = await self._compute_per_class_metrics(
                    eval_results, test_dataset
                )
                eval_results.per_class_metrics = per_class_metrics
            
            # 9. Generate confusion matrix if classification task
            confusion_matrix = None
            if test_dataset.task_type == "classification" and request.generate_confusion_matrix:
                confusion_matrix = await self._generate_confusion_matrix(
                    eval_results, test_dataset
                )
            
            # 10. Save results if requested
            if request.save_results:
                await self._save_results(
                    eval_results,
                    confusion_matrix,
                    request.output_dir or Path("evaluation_results")
                )
            
            # 11. Create response
            response = self._create_response(
                eval_results,
                confusion_matrix,
                start_time,
                datetime.now(),
                test_dataset
            )
            
            # 12. Log final metrics
            await self.monitoring_port.log_metrics(response.metrics)
            
            return response
            
        except Exception as e:
            await self.monitoring_port.log_error(f"Evaluation failed: {str(e)}")
            return EvaluationResponseDTO.from_error(e)
        finally:
            await self.monitoring_port.end_run()
    
    async def _load_model(self, model_path: Path) -> BertModel:
        """Load model from checkpoint or saved model directory."""
        # Check if it's a checkpoint or final model
        if (model_path / "checkpoint.json").exists():
            # Load from checkpoint
            checkpoint_data = await self.storage_port.load_json(model_path / "checkpoint.json")
            model_config = checkpoint_data["model_config"]
            model = BertModel.from_config(model_config)
            model.load_state(checkpoint_data["model_state"])
        else:
            # Load from saved model directory
            model = await self.storage_port.load_model(model_path)
        
        return model
    
    async def _load_dataset(self, data_path: Path) -> Dataset:
        """Load and prepare test dataset."""
        raw_data = await self.storage_port.load_data(data_path)
        
        # Create dataset (tokenization might be needed)
        # This would follow similar logic to training
        if self._is_text_data(raw_data):
            # Text data - needs tokenization
            return Dataset.from_text(raw_data)
        else:
            # Tabular data
            return Dataset.from_tabular(raw_data)
    
    def _is_text_data(self, data: Any) -> bool:
        """Check if data is text-based."""
        # Implementation would check data format
        return False  # Placeholder
    
    def _get_default_metrics(self, dataset: Dataset) -> List[MetricType]:
        """Get default metrics based on task type."""
        if dataset.task_type == "classification":
            if dataset.num_labels == 2:
                return [
                    MetricType.ACCURACY,
                    MetricType.PRECISION,
                    MetricType.RECALL,
                    MetricType.F1,
                    MetricType.AUC_ROC
                ]
            else:
                return [
                    MetricType.ACCURACY,
                    MetricType.MACRO_F1,
                    MetricType.WEIGHTED_F1
                ]
        elif dataset.task_type == "regression":
            return [
                MetricType.MSE,
                MetricType.MAE,
                MetricType.RMSE,
                MetricType.R2
            ]
        else:
            return [MetricType.LOSS]
    
    async def _compute_confidence_metrics(
        self,
        model: BertModel,
        data_loader: Any,
        eval_results: EvaluationMetrics
    ) -> Dict[str, float]:
        """Compute confidence and uncertainty metrics."""
        confidence_scores = []
        uncertainties = []
        
        # Run inference with uncertainty estimation
        for batch in data_loader:
            with self.compute_port.no_grad():
                outputs = await self.compute_port.forward(model, batch)
                
                # Calculate confidence (e.g., max probability for classification)
                if hasattr(outputs, "probabilities"):
                    confidence = outputs.probabilities.max(dim=-1).values
                    confidence_scores.extend(confidence.tolist())
                    
                    # Calculate entropy as uncertainty measure
                    entropy = -(outputs.probabilities * outputs.probabilities.log()).sum(dim=-1)
                    uncertainties.extend(entropy.tolist())
        
        return {
            "mean_confidence": sum(confidence_scores) / len(confidence_scores),
            "min_confidence": min(confidence_scores),
            "max_confidence": max(confidence_scores),
            "mean_uncertainty": sum(uncertainties) / len(uncertainties),
            "calibration_error": await self._compute_calibration_error(
                confidence_scores, eval_results
            )
        }
    
    async def _compute_per_class_metrics(
        self,
        eval_results: EvaluationMetrics,
        dataset: Dataset
    ) -> Dict[str, Dict[str, float]]:
        """Compute per-class metrics for classification tasks."""
        per_class = {}
        
        for class_idx, class_name in enumerate(dataset.label_names):
            # Filter predictions for this class
            class_mask = eval_results.true_labels == class_idx
            class_preds = eval_results.predictions[class_mask]
            class_labels = eval_results.true_labels[class_mask]
            
            # Calculate metrics for this class
            per_class[class_name] = await self.metrics_port.calculate(
                predictions=class_preds,
                labels=class_labels,
                metrics=[MetricType.PRECISION, MetricType.RECALL, MetricType.F1]
            )
        
        return per_class
    
    async def _generate_confusion_matrix(
        self,
        eval_results: EvaluationMetrics,
        dataset: Dataset
    ) -> Dict[str, Any]:
        """Generate confusion matrix for classification tasks."""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(
            eval_results.true_labels,
            eval_results.predictions
        )
        
        return {
            "matrix": cm.tolist(),
            "labels": dataset.label_names,
            "normalized": (cm / cm.sum(axis=1, keepdims=True)).tolist()
        }
    
    async def _compute_calibration_error(
        self,
        confidence_scores: List[float],
        eval_results: EvaluationMetrics
    ) -> float:
        """Compute expected calibration error."""
        # Simplified implementation
        # Real implementation would bin predictions by confidence
        # and compare accuracy in each bin
        return 0.0  # Placeholder
    
    async def _save_results(
        self,
        eval_results: EvaluationMetrics,
        confusion_matrix: Optional[Dict[str, Any]],
        output_dir: Path
    ) -> None:
        """Save evaluation results to disk."""
        await self.storage_port.create_directory(output_dir)
        
        # Save metrics
        metrics_path = output_dir / "metrics.json"
        await self.storage_port.save_json(eval_results.to_dict(), metrics_path)
        
        # Save confusion matrix if available
        if confusion_matrix:
            cm_path = output_dir / "confusion_matrix.json"
            await self.storage_port.save_json(confusion_matrix, cm_path)
        
        # Save predictions
        predictions_path = output_dir / "predictions.csv"
        await self.storage_port.save_csv({
            "prediction": eval_results.predictions.tolist(),
            "true_label": eval_results.true_labels.tolist(),
            "confidence": eval_results.confidence_scores.tolist() if hasattr(eval_results, "confidence_scores") else None
        }, predictions_path)
        
        await self.monitoring_port.log_info(f"Evaluation results saved to {output_dir}")
    
    def _create_response(
        self,
        eval_results: EvaluationMetrics,
        confusion_matrix: Optional[Dict[str, Any]],
        start_time: datetime,
        end_time: datetime,
        dataset: Dataset
    ) -> EvaluationResponseDTO:
        """Create response DTO from evaluation results."""
        return EvaluationResponseDTO(
            success=True,
            metrics=eval_results.metrics,
            loss=eval_results.loss,
            confusion_matrix=confusion_matrix,
            per_class_metrics=eval_results.per_class_metrics,
            confidence_metrics=eval_results.confidence_metrics,
            predictions_count=len(eval_results.predictions),
            task_type=dataset.task_type,
            num_classes=dataset.num_labels if dataset.task_type == "classification" else None,
            evaluation_time_seconds=(end_time - start_time).total_seconds(),
            dataset_info={
                "path": str(dataset.path) if hasattr(dataset, "path") else None,
                "size": len(dataset),
                "features": dataset.feature_names if hasattr(dataset, "feature_names") else None
            }
        )