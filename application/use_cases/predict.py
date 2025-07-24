"""Predict Use Case.

This use case orchestrates prediction generation by coordinating
between domain services, ports, and adapters.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import csv

from infrastructure.di import use_case
from application.dto.prediction import (
    PredictionRequestDTO, 
    PredictionResponseDTO,
    PredictionFormat
)
from ports.secondary.storage import StorageService
from ports.secondary.monitoring import MonitoringService


@use_case
class PredictUseCase:
    """Use case for generating predictions with a trained model.
    
    This orchestrates the prediction workflow by:
    1. Loading the trained model
    2. Preparing input data
    3. Generating predictions
    4. Post-processing results
    5. Saving outputs in requested format
    """
    
    def __init__(
        self,
        storage_port: StorageService,
        monitoring_port: MonitoringService,
    ):
        """Initialize the use case with required dependencies.
        
        Args:
            storage_port: Port for file storage operations
            monitoring_port: Port for logging and monitoring
        """
        self.storage = storage_port
        self.monitoring = monitoring_port
    
    async def execute(self, request: PredictionRequestDTO) -> PredictionResponseDTO:
        """Execute the prediction use case.
        
        Args:
            request: Prediction request from external actor
            
        Returns:
            Prediction response with results
        """
        start_time = datetime.now()
        
        try:
            # Validate request
            errors = request.validate()
            if errors:
                return PredictionResponseDTO(
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
                request.data_format
            )
            
            # Generate predictions
            prediction_results = await self._generate_predictions(
                model,
                data_loader,
                request
            )
            
            # Post-process predictions
            processed_results = await self._post_process_predictions(
                prediction_results,
                request
            )
            
            # Save predictions
            output_path = await self._save_predictions(
                processed_results,
                request
            )
            
            # Compute statistics
            statistics = self._compute_statistics(processed_results)
            
            # Create response
            end_time = datetime.now()
            response = self._create_response(
                processed_results,
                statistics,
                output_path,
                start_time,
                end_time,
                run_id,
                request,
                model_config
            )
            
            # Log completion
            await self.monitoring.log_info(
                f"Prediction completed successfully: {run_id}"
            )
            
            return response
            
        except Exception as e:
            await self.monitoring.log_error(
                f"Prediction failed: {str(e)}"
            )
            return PredictionResponseDTO.from_error(e)
    
    async def _setup_monitoring(self, request: PredictionRequestDTO) -> str:
        """Set up monitoring and tracking.
        
        Returns:
            Run ID for tracking
        """
        run_name = request.run_name or f"predict_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if request.track_predictions:
            # Initialize monitoring
            await self.monitoring.start_run(
                name=run_name,
                experiment=request.experiment_name
            )
            
            # Log parameters
            await self.monitoring.log_params({
                "model_path": str(request.model_path),
                "data_path": str(request.data_path),
                "batch_size": request.batch_size,
                "output_format": request.output_format.value,
                "include_probabilities": request.include_probabilities,
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
        data_format: Optional[str]
    ) -> Any:
        """Create data loader for prediction."""
        # This would use the data factory
        # Placeholder for actual implementation
        raise NotImplementedError("Data loader creation to be implemented")
    
    async def _generate_predictions(
        self,
        model: Any,
        data_loader: Any,
        request: PredictionRequestDTO
    ) -> Dict[str, Any]:
        """Generate predictions for all data."""
        all_predictions = []
        all_probabilities = []
        all_embeddings = []
        all_attention_weights = []
        all_input_ids = []
        
        num_batches = 0
        
        for batch in data_loader:
            # Get model outputs
            outputs = model.predict(batch)
            
            # Extract requested components
            predictions = outputs['predictions']
            all_predictions.extend(predictions)
            
            if request.include_probabilities:
                all_probabilities.extend(outputs.get('probabilities', []))
            
            if request.include_embeddings:
                all_embeddings.extend(outputs.get('embeddings', []))
            
            if request.include_attention_weights:
                all_attention_weights.extend(outputs.get('attention_weights', []))
            
            if request.include_input_ids:
                all_input_ids.extend(batch.get('input_ids', []))
            
            num_batches += 1
            
            # Log progress
            if num_batches % 100 == 0:
                await self.monitoring.log_info(
                    f"Processed {num_batches} batches"
                )
        
        return {
            'predictions': all_predictions,
            'probabilities': all_probabilities if request.include_probabilities else None,
            'embeddings': all_embeddings if request.include_embeddings else None,
            'attention_weights': all_attention_weights if request.include_attention_weights else None,
            'input_ids': all_input_ids if request.include_input_ids else None,
            'num_samples': len(all_predictions),
        }
    
    async def _post_process_predictions(
        self,
        prediction_results: Dict[str, Any],
        request: PredictionRequestDTO
    ) -> Dict[str, Any]:
        """Post-process predictions based on request settings."""
        predictions = prediction_results['predictions']
        probabilities = prediction_results.get('probabilities')
        
        # Apply probability threshold if specified
        if request.probability_threshold is not None and probabilities:
            filtered_predictions = []
            for pred, prob in zip(predictions, probabilities):
                if max(prob) >= request.probability_threshold:
                    filtered_predictions.append(pred)
                else:
                    filtered_predictions.append(None)  # Or a default value
            predictions = filtered_predictions
        
        # Get top-k predictions if specified
        if request.top_k_predictions is not None and probabilities:
            top_k_predictions = []
            for prob in probabilities:
                # Get indices of top-k probabilities
                top_k_indices = sorted(
                    range(len(prob)), 
                    key=lambda i: prob[i], 
                    reverse=True
                )[:request.top_k_predictions]
                top_k_predictions.append(top_k_indices)
            prediction_results['top_k_predictions'] = top_k_predictions
        
        prediction_results['predictions'] = predictions
        return prediction_results
    
    async def _save_predictions(
        self,
        processed_results: Dict[str, Any],
        request: PredictionRequestDTO
    ) -> Path:
        """Save predictions in the requested format."""
        # Determine output path
        if request.output_path:
            output_path = request.output_path
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"predictions_{timestamp}.{request.output_format.value}"
            output_path = Path("predictions") / filename
        
        # Ensure directory exists
        await self.storage.create_directory(output_path.parent)
        
        # Save based on format
        if request.output_format == PredictionFormat.JSON:
            await self._save_json(processed_results, output_path)
        elif request.output_format == PredictionFormat.CSV:
            await self._save_csv(processed_results, output_path)
        elif request.output_format == PredictionFormat.PARQUET:
            await self._save_parquet(processed_results, output_path)
        elif request.output_format == PredictionFormat.NUMPY:
            await self._save_numpy(processed_results, output_path)
        
        return output_path
    
    async def _save_json(self, results: Dict[str, Any], path: Path) -> None:
        """Save predictions as JSON."""
        output_data = {
            'predictions': results['predictions'],
            'metadata': {
                'num_samples': results['num_samples'],
                'timestamp': datetime.now().isoformat(),
            }
        }
        
        if results.get('probabilities'):
            output_data['probabilities'] = results['probabilities']
        
        if results.get('top_k_predictions'):
            output_data['top_k_predictions'] = results['top_k_predictions']
        
        await self.storage.save_json(output_data, path)
    
    async def _save_csv(self, results: Dict[str, Any], path: Path) -> None:
        """Save predictions as CSV."""
        # Create rows for CSV
        rows = []
        predictions = results['predictions']
        probabilities = results.get('probabilities', [])
        
        for i, pred in enumerate(predictions):
            row = {
                'index': i,
                'prediction': pred,
            }
            
            if probabilities and i < len(probabilities):
                for j, prob in enumerate(probabilities[i]):
                    row[f'prob_class_{j}'] = prob
            
            rows.append(row)
        
        # Write CSV
        if rows:
            fieldnames = list(rows[0].keys())
            content = []
            content.append(','.join(fieldnames))
            for row in rows:
                content.append(','.join(str(row[f]) for f in fieldnames))
            
            await self.storage.save_text('\n'.join(content), path)
    
    async def _save_parquet(self, results: Dict[str, Any], path: Path) -> None:
        """Save predictions as Parquet."""
        # This would use a parquet library
        # Placeholder for actual implementation
        raise NotImplementedError("Parquet saving to be implemented")
    
    async def _save_numpy(self, results: Dict[str, Any], path: Path) -> None:
        """Save predictions as NumPy arrays."""
        # This would save as .npz file
        # Placeholder for actual implementation
        raise NotImplementedError("NumPy saving to be implemented")
    
    def _compute_statistics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute statistics about predictions."""
        predictions = results['predictions']
        
        # Count prediction distribution
        prediction_distribution = {}
        for pred in predictions:
            pred_str = str(pred)
            prediction_distribution[pred_str] = prediction_distribution.get(pred_str, 0) + 1
        
        # Compute confidence statistics if available
        confidence_stats = {}
        if results.get('probabilities'):
            probabilities = results['probabilities']
            max_probs = [max(p) for p in probabilities]
            
            confidence_stats = {
                'mean_confidence': sum(max_probs) / len(max_probs),
                'min_confidence': min(max_probs),
                'max_confidence': max(max_probs),
                'std_confidence': self._std(max_probs),
            }
        
        return {
            'prediction_distribution': prediction_distribution,
            'confidence_stats': confidence_stats,
        }
    
    def _std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def _create_response(
        self,
        processed_results: Dict[str, Any],
        statistics: Dict[str, Any],
        output_path: Path,
        start_time: datetime,
        end_time: datetime,
        run_id: str,
        request: PredictionRequestDTO,
        model_config: Dict[str, Any]
    ) -> PredictionResponseDTO:
        """Create the response DTO."""
        total_time = (end_time - start_time).total_seconds()
        num_predictions = processed_results['num_samples']
        
        # Get sample predictions for preview
        sample_predictions = []
        for i in range(min(10, num_predictions)):
            sample = {
                'index': i,
                'prediction': processed_results['predictions'][i],
            }
            if processed_results.get('probabilities'):
                sample['probabilities'] = processed_results['probabilities'][i]
            sample_predictions.append(sample)
        
        return PredictionResponseDTO(
            success=True,
            output_path=output_path,
            output_format=request.output_format,
            num_predictions=num_predictions,
            prediction_distribution=statistics['prediction_distribution'],
            confidence_stats=statistics.get('confidence_stats'),
            prediction_time_seconds=total_time,
            samples_per_second=num_predictions / total_time if total_time > 0 else 0.0,
            start_time=start_time,
            end_time=end_time,
            model_path=request.model_path,
            model_config=model_config,
            data_path=request.data_path,
            data_format=request.data_format,
            sample_predictions=sample_predictions,
            run_id=run_id if request.track_predictions else None,
        )