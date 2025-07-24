"""Evaluate Model Use Case.

This use case orchestrates model evaluation by coordinating
between domain services, ports, and adapters.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from infrastructure.di import use_case
from application.dto.evaluation import EvaluationRequestDTO, EvaluationResponseDTO
from domain.services.evaluation_service import EvaluationService
from application.ports.secondary.storage import StorageService
from application.ports.secondary.monitoring import MonitoringService
from application.ports.secondary.metrics import MetricsCollector


@use_case
class EvaluateModelUseCase:
    """Use case for evaluating a trained model.
    
    This orchestrates the evaluation workflow by:
    1. Loading the trained model
    2. Preparing evaluation data
    3. Running evaluation
    4. Computing metrics
    5. Generating analysis and reports
    """
    
    def __init__(
        self,
        evaluation_service: EvaluationService,
        storage_port: StorageService,
        monitoring_port: MonitoringService,
        metrics_port: MetricsCollector,
    ):
        """Initialize the use case with required dependencies.
        
        Args:
            evaluation_service: Domain service for evaluation logic
            storage_port: Port for file storage operations
            monitoring_port: Port for logging and monitoring
            metrics_port: Port for metrics calculation
        """
        self.evaluation_service = evaluation_service
        self.storage = storage_port
        self.monitoring = monitoring_port
        self.metrics = metrics_port
    
    async def execute(self, request: EvaluationRequestDTO) -> EvaluationResponseDTO:
        """Execute the evaluation use case.
        
        Args:
            request: Evaluation request from external actor
            
        Returns:
            Evaluation response with results
        """
        start_time = datetime.now()
        
        try:
            # Validate request
            errors = request.validate()
            if errors:
                return EvaluationResponseDTO(
                    success=False,
                    error_message=f"Validation errors: {', '.join(errors)}"
                )
            
            # Set up monitoring
            run_id = await self._setup_monitoring(request)
            
            # Load model
            model = await self._load_model(request.model_path)
            model_config = await self._load_model_config(request.model_path)
            
            # Prepare data loader
            data_loader = await self._create_data_loader(
                request.data_path,
                request.batch_size,
                request.data_split
            )
            
            # Run evaluation
            evaluation_results = await self._run_evaluation(
                model,
                data_loader,
                request.metrics
            )
            
            # Perform additional analysis if requested
            analysis_results = {}
            if request.error_analysis:
                analysis_results['errors'] = await self._perform_error_analysis(
                    model, data_loader, evaluation_results
                )
            
            if request.confidence_analysis:
                analysis_results['confidence'] = await self._perform_confidence_analysis(
                    model, data_loader
                )
            
            if request.feature_importance:
                analysis_results['features'] = await self._compute_feature_importance(
                    model, data_loader
                )
            
            # Save outputs if requested
            output_paths = {}
            if request.output_dir:
                output_paths = await self._save_outputs(
                    evaluation_results,
                    analysis_results,
                    request
                )
            
            # Create response
            end_time = datetime.now()
            response = self._create_response(
                evaluation_results,
                analysis_results,
                output_paths,
                start_time,
                end_time,
                run_id,
                request,
                model_config
            )
            
            # Log completion
            await self.monitoring.log_info(
                f"Evaluation completed successfully: {run_id}"
            )
            
            return response
            
        except Exception as e:
            await self.monitoring.log_error(
                f"Evaluation failed: {str(e)}"
            )
            return EvaluationResponseDTO.from_error(e)
    
    async def _setup_monitoring(self, request: EvaluationRequestDTO) -> str:
        """Set up monitoring and tracking.
        
        Returns:
            Run ID for tracking
        """
        run_name = request.run_name or f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize monitoring
        await self.monitoring.start_run(
            name=run_name,
            experiment=request.experiment_name
        )
        
        # Log parameters
        await self.monitoring.log_params({
            "model_path": str(request.model_path),
            "data_path": str(request.data_path),
            "data_split": request.data_split,
            "batch_size": request.batch_size,
            "metrics": ",".join(request.metrics),
        })
        
        return run_name
    
    async def _load_model(self, model_path: Path) -> Any:
        """Load the trained model."""
        # This would use the model loader/factory
        # Placeholder for actual implementation
        raise NotImplementedError("Model loading to be implemented")
    
    async def _load_model_config(self, model_path: Path) -> Dict[str, Any]:
        """Load model configuration."""
        config_path = model_path / "config.json"
        if await self.storage.exists(config_path):
            return await self.storage.load_json(config_path)
        return {}
    
    async def _create_data_loader(
        self,
        data_path: Path,
        batch_size: int,
        data_split: str
    ) -> Any:
        """Create data loader for evaluation."""
        # This would use the data factory
        # Placeholder for actual implementation
        raise NotImplementedError("Data loader creation to be implemented")
    
    async def _run_evaluation(
        self,
        model: Any,
        data_loader: Any,
        metric_names: List[str]
    ) -> Dict[str, Any]:
        """Run the evaluation loop."""
        # Initialize metrics
        metric_calculators = {
            name: self.metrics.create_metric(name)
            for name in metric_names
        }
        
        # Evaluation loop
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0
        
        for batch in data_loader:
            # Get predictions
            outputs = self.evaluation_service.evaluate_batch(model, batch)
            
            # Accumulate for metrics
            all_predictions.extend(outputs['predictions'])
            all_labels.extend(outputs['labels'])
            total_loss += outputs.get('loss', 0.0)
            num_batches += 1
            
            # Log progress
            if num_batches % 100 == 0:
                await self.monitoring.log_info(
                    f"Evaluated {num_batches} batches"
                )
        
        # Calculate final metrics
        metrics = {
            'loss': total_loss / num_batches if num_batches > 0 else 0.0
        }
        
        for name, calculator in metric_calculators.items():
            metrics[name] = calculator.compute(all_predictions, all_labels)
        
        # Log metrics
        await self.monitoring.log_metrics(metrics)
        
        return {
            'metrics': metrics,
            'predictions': all_predictions,
            'labels': all_labels,
            'num_samples': len(all_predictions),
        }
    
    async def _perform_error_analysis(
        self,
        model: Any,
        data_loader: Any,
        evaluation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze prediction errors."""
        predictions = evaluation_results['predictions']
        labels = evaluation_results['labels']
        
        # Find misclassified samples
        errors = []
        for i, (pred, label) in enumerate(zip(predictions, labels)):
            if pred != label:
                errors.append({
                    'index': i,
                    'predicted': pred,
                    'actual': label,
                })
        
        # Compute error distribution
        error_distribution = {}
        for error in errors:
            key = f"{error['actual']}_to_{error['predicted']}"
            error_distribution[key] = error_distribution.get(key, 0) + 1
        
        return {
            'error_samples': errors[:100],  # Limit to first 100
            'error_distribution': error_distribution,
            'error_rate': len(errors) / len(predictions) if predictions else 0.0,
        }
    
    async def _perform_confidence_analysis(
        self,
        model: Any,
        data_loader: Any
    ) -> Dict[str, Any]:
        """Analyze prediction confidence."""
        # This would analyze confidence scores
        # Placeholder for actual implementation
        return {
            'confidence_distribution': {},
            'calibration_metrics': {},
        }
    
    async def _compute_feature_importance(
        self,
        model: Any,
        data_loader: Any
    ) -> Dict[str, Any]:
        """Compute feature importance scores."""
        # This would compute feature importance
        # Placeholder for actual implementation
        return {
            'feature_scores': {},
        }
    
    async def _save_outputs(
        self,
        evaluation_results: Dict[str, Any],
        analysis_results: Dict[str, Any],
        request: EvaluationRequestDTO
    ) -> Dict[str, Path]:
        """Save evaluation outputs."""
        output_dir = request.output_dir
        await self.storage.create_directory(output_dir)
        
        paths = {}
        
        # Save metrics
        metrics_path = output_dir / "metrics.json"
        await self.storage.save_json(evaluation_results['metrics'], metrics_path)
        paths['metrics'] = metrics_path
        
        # Save predictions if requested
        if request.save_predictions:
            predictions_path = output_dir / "predictions.json"
            await self.storage.save_json({
                'predictions': evaluation_results['predictions'],
                'labels': evaluation_results['labels'],
            }, predictions_path)
            paths['predictions'] = predictions_path
        
        # Save confusion matrix if requested
        if request.save_confusion_matrix and 'confusion_matrix' in evaluation_results:
            cm_path = output_dir / "confusion_matrix.json"
            await self.storage.save_json(
                evaluation_results['confusion_matrix'],
                cm_path
            )
            paths['confusion_matrix'] = cm_path
        
        # Save error analysis
        if 'errors' in analysis_results:
            errors_path = output_dir / "error_analysis.json"
            await self.storage.save_json(analysis_results['errors'], errors_path)
            paths['error_analysis'] = errors_path
        
        return paths
    
    def _create_response(
        self,
        evaluation_results: Dict[str, Any],
        analysis_results: Dict[str, Any],
        output_paths: Dict[str, Path],
        start_time: datetime,
        end_time: datetime,
        run_id: str,
        request: EvaluationRequestDTO,
        model_config: Dict[str, Any]
    ) -> EvaluationResponseDTO:
        """Create the response DTO."""
        total_time = (end_time - start_time).total_seconds()
        num_samples = evaluation_results['num_samples']
        
        response = EvaluationResponseDTO(
            success=True,
            metrics=evaluation_results['metrics'],
            num_samples_evaluated=num_samples,
            evaluation_time_seconds=total_time,
            samples_per_second=num_samples / total_time if total_time > 0 else 0.0,
            start_time=start_time,
            end_time=end_time,
            model_path=request.model_path,
            model_config=model_config,
            data_path=request.data_path,
            data_split=request.data_split,
            run_id=run_id,
        )
        
        # Add analysis results
        if 'errors' in analysis_results:
            response.error_samples = analysis_results['errors']['error_samples']
            response.error_distribution = analysis_results['errors']['error_distribution']
        
        if 'confidence' in analysis_results:
            response.confidence_distribution = analysis_results['confidence']['confidence_distribution']
            response.calibration_metrics = analysis_results['confidence']['calibration_metrics']
        
        # Add output paths
        if output_paths:
            response.predictions_path = output_paths.get('predictions')
        
        return response