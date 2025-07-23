"""Predict Command - orchestrates batch prediction workflow."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import pandas as pd

from application.dto.prediction import PredictionRequestDTO, PredictionResponseDTO
from domain.entities.model import Model
from domain.entities.dataset import Dataset
from domain.services import TokenizationService
from domain.ports import (
    DataLoaderPort,
    ComputePort,
    MonitoringPort,
    StoragePort
)


@dataclass
class PredictCommand:
    """Command to run predictions on new data.
    
    This command orchestrates the prediction workflow by:
    1. Loading the trained model
    2. Processing input data
    3. Running inference
    4. Post-processing predictions
    5. Saving results in requested format
    """
    
    # Domain services
    tokenization_service: TokenizationService
    
    # Ports
    data_loader_port: DataLoaderPort
    compute_port: ComputePort
    monitoring_port: MonitoringPort
    storage_port: StoragePort
    
    async def execute(self, request: PredictionRequestDTO) -> PredictionResponseDTO:
        """Execute the prediction command.
        
        Args:
            request: Prediction request with model and data information
            
        Returns:
            Response with predictions and metadata
        """
        start_time = datetime.now()
        
        try:
            # 1. Validate request
            errors = request.validate()
            if errors:
                return PredictionResponseDTO(
                    success=False,
                    error_message=f"Validation errors: {', '.join(errors)}"
                )
            
            # 2. Initialize monitoring if requested
            if request.track_predictions:
                await self.monitoring_port.start_run(
                    name=f"predict_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    tags={"type": "prediction", "model": str(request.model_path)}
                )
            
            # 3. Load model
            model = await self._load_model(request.model_path)
            
            # 4. Initialize compute backend
            await self.compute_port.initialize(model)
            
            # 5. Load and prepare data
            if request.data_path:
                # Batch prediction from file
                dataset = await self._load_dataset(request.data_path)
                predictions = await self._batch_predict(
                    model, dataset, request
                )
            else:
                # Single prediction from input data
                predictions = await self._single_predict(
                    model, request.input_data, request
                )
            
            # 6. Post-process predictions
            processed_predictions = await self._post_process_predictions(
                predictions,
                model,
                request
            )
            
            # 7. Apply prediction threshold if specified
            if request.prediction_threshold is not None:
                processed_predictions = self._apply_threshold(
                    processed_predictions,
                    request.prediction_threshold
                )
            
            # 8. Save predictions if requested
            output_path = None
            if request.output_path:
                output_path = await self._save_predictions(
                    processed_predictions,
                    request.output_path,
                    request.output_format
                )
            
            # 9. Calculate prediction statistics
            stats = self._calculate_statistics(processed_predictions)
            
            # 10. Create response
            response = self._create_response(
                processed_predictions,
                stats,
                output_path,
                start_time,
                datetime.now()
            )
            
            # 11. Log if tracking
            if request.track_predictions:
                await self.monitoring_port.log_metrics({
                    "predictions_count": len(processed_predictions["predictions"]),
                    "prediction_time_seconds": response.prediction_time_seconds,
                    "avg_confidence": stats.get("avg_confidence", 0.0)
                })
            
            return response
            
        except Exception as e:
            await self.monitoring_port.log_error(f"Prediction failed: {str(e)}")
            return PredictionResponseDTO.from_error(e)
        finally:
            if request.track_predictions:
                await self.monitoring_port.end_run()
    
    async def _load_model(self, model_path: Path) -> Model:
        """Load the trained model."""
        return await self.storage_port.load_model(model_path)
    
    async def _load_dataset(self, data_path: Path) -> Dataset:
        """Load dataset for batch prediction."""
        raw_data = await self.storage_port.load_data(data_path)
        
        # Check if data needs tokenization
        if self._is_text_data(raw_data):
            # Text data - tokenize it
            tokenized_data = await self.tokenization_service.tokenize_dataset(
                raw_data,
                max_length=512,
                truncation=True,
                padding="max_length"
            )
            return Dataset.from_tokenized(tokenized_data)
        else:
            # Tabular data - convert to text representation
            return Dataset.from_tabular(raw_data)
    
    def _is_text_data(self, data: Any) -> bool:
        """Check if data is text-based."""
        # Implementation would check data format
        if isinstance(data, pd.DataFrame):
            # Check if there are text columns
            return any(data[col].dtype == 'object' for col in data.columns)
        return False
    
    async def _batch_predict(
        self,
        model: Model,
        dataset: Dataset,
        request: PredictionRequestDTO
    ) -> Dict[str, Any]:
        """Run batch predictions on dataset."""
        # Create data loader
        data_loader = await self.data_loader_port.create_loader(
            dataset,
            batch_size=request.batch_size,
            shuffle=False,
            num_workers=request.num_workers
        )
        
        all_predictions = []
        all_probabilities = []
        all_embeddings = []
        
        # Run inference
        with self.compute_port.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                # Forward pass
                outputs = await self.compute_port.forward(model, batch)
                
                # Extract predictions
                if hasattr(outputs, "logits"):
                    # Classification
                    probabilities = await self.compute_port.softmax(outputs.logits, dim=-1)
                    predictions = await self.compute_port.argmax(probabilities, dim=-1)
                    
                    all_predictions.extend(predictions.tolist())
                    all_probabilities.extend(probabilities.tolist())
                else:
                    # Regression
                    all_predictions.extend(outputs.predictions.tolist())
                
                # Extract embeddings if requested
                if request.return_embeddings and hasattr(outputs, "embeddings"):
                    all_embeddings.extend(outputs.embeddings.tolist())
                
                # Log progress
                if batch_idx % 10 == 0:
                    await self.monitoring_port.log_info(
                        f"Processed {batch_idx * request.batch_size} samples"
                    )
        
        return {
            "predictions": all_predictions,
            "probabilities": all_probabilities if all_probabilities else None,
            "embeddings": all_embeddings if all_embeddings else None
        }
    
    async def _single_predict(
        self,
        model: Model,
        input_data: Union[str, Dict[str, Any], List[Any]],
        request: PredictionRequestDTO
    ) -> Dict[str, Any]:
        """Run single prediction on input data."""
        # Prepare input
        if isinstance(input_data, str):
            # Text input
            tokenized = await self.tokenization_service.tokenize_text(
                input_data,
                max_length=512,
                truncation=True,
                padding="max_length"
            )
            prepared_input = tokenized
        else:
            # Structured input (tabular)
            # Convert to text representation
            text_repr = self._convert_to_text(input_data)
            tokenized = await self.tokenization_service.tokenize_text(
                text_repr,
                max_length=512,
                truncation=True,
                padding="max_length"
            )
            prepared_input = tokenized
        
        # Run inference
        with self.compute_port.no_grad():
            outputs = await self.compute_port.forward(model, prepared_input)
            
            if hasattr(outputs, "logits"):
                # Classification
                probabilities = await self.compute_port.softmax(outputs.logits, dim=-1)
                prediction = await self.compute_port.argmax(probabilities, dim=-1)
                
                return {
                    "predictions": [prediction.item()],
                    "probabilities": [probabilities.tolist()],
                    "embeddings": [outputs.embeddings.tolist()] if request.return_embeddings and hasattr(outputs, "embeddings") else None
                }
            else:
                # Regression
                return {
                    "predictions": [outputs.predictions.item()],
                    "probabilities": None,
                    "embeddings": [outputs.embeddings.tolist()] if request.return_embeddings and hasattr(outputs, "embeddings") else None
                }
    
    def _convert_to_text(self, data: Union[Dict[str, Any], List[Any]]) -> str:
        """Convert structured data to text representation."""
        if isinstance(data, dict):
            # Key-value pairs
            parts = []
            for key, value in data.items():
                parts.append(f"{key}: {value}")
            return " | ".join(parts)
        elif isinstance(data, list):
            # List of values
            return " | ".join(str(v) for v in data)
        else:
            return str(data)
    
    async def _post_process_predictions(
        self,
        predictions: Dict[str, Any],
        model: Model,
        request: PredictionRequestDTO
    ) -> Dict[str, Any]:
        """Post-process predictions (e.g., map to labels)."""
        processed = predictions.copy()
        
        # Map predictions to labels if available
        if hasattr(model, "config") and hasattr(model.config, "id2label"):
            label_map = model.config.id2label
            processed["labels"] = [
                label_map.get(pred, f"class_{pred}")
                for pred in predictions["predictions"]
            ]
        
        # Calculate confidence scores for classification
        if predictions["probabilities"]:
            processed["confidence_scores"] = [
                max(probs) for probs in predictions["probabilities"]
            ]
        
        # Apply calibration if requested
        if request.apply_calibration and hasattr(model, "calibration_params"):
            processed = await self._apply_calibration(processed, model.calibration_params)
        
        return processed
    
    def _apply_threshold(
        self,
        predictions: Dict[str, Any],
        threshold: float
    ) -> Dict[str, Any]:
        """Apply prediction threshold for binary classification."""
        if "probabilities" in predictions and predictions["probabilities"]:
            # Assuming binary classification
            thresholded_predictions = []
            for probs in predictions["probabilities"]:
                if len(probs) == 2:  # Binary
                    pred = 1 if probs[1] >= threshold else 0
                    thresholded_predictions.append(pred)
                else:
                    # Multi-class - use original prediction
                    thresholded_predictions.append(
                        predictions["predictions"][len(thresholded_predictions)]
                    )
            
            predictions["predictions"] = thresholded_predictions
            predictions["threshold_applied"] = threshold
        
        return predictions
    
    async def _apply_calibration(
        self,
        predictions: Dict[str, Any],
        calibration_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply probability calibration."""
        # Placeholder for calibration logic
        # Would typically use isotonic regression or Platt scaling
        return predictions
    
    async def _save_predictions(
        self,
        predictions: Dict[str, Any],
        output_path: Path,
        output_format: str
    ) -> Path:
        """Save predictions in requested format."""
        if output_format == "csv":
            # Create DataFrame
            df_data = {
                "prediction": predictions["predictions"]
            }
            
            if "labels" in predictions:
                df_data["label"] = predictions["labels"]
            
            if "confidence_scores" in predictions:
                df_data["confidence"] = predictions["confidence_scores"]
            
            if "probabilities" in predictions and predictions["probabilities"]:
                # Add probability columns
                n_classes = len(predictions["probabilities"][0])
                for i in range(n_classes):
                    df_data[f"prob_class_{i}"] = [
                        probs[i] for probs in predictions["probabilities"]
                    ]
            
            df = pd.DataFrame(df_data)
            await self.storage_port.save_csv(df, output_path)
            
        elif output_format == "json":
            await self.storage_port.save_json(predictions, output_path)
            
        elif output_format == "parquet":
            df = pd.DataFrame(predictions)
            await self.storage_port.save_parquet(df, output_path)
        
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        return output_path
    
    def _calculate_statistics(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate prediction statistics."""
        stats = {
            "total_predictions": len(predictions["predictions"])
        }
        
        # Classification statistics
        if "labels" in predictions:
            from collections import Counter
            label_counts = Counter(predictions["labels"])
            stats["label_distribution"] = dict(label_counts)
            stats["unique_labels"] = len(label_counts)
        
        # Confidence statistics
        if "confidence_scores" in predictions:
            confidence_scores = predictions["confidence_scores"]
            stats["avg_confidence"] = sum(confidence_scores) / len(confidence_scores)
            stats["min_confidence"] = min(confidence_scores)
            stats["max_confidence"] = max(confidence_scores)
            
            # Confidence bins
            bins = [0.0, 0.5, 0.7, 0.9, 1.0]
            import numpy as np
            hist, _ = np.histogram(confidence_scores, bins=bins)
            stats["confidence_distribution"] = {
                f"{bins[i]:.1f}-{bins[i+1]:.1f}": int(hist[i])
                for i in range(len(hist))
            }
        
        # Regression statistics
        if "probabilities" not in predictions or not predictions["probabilities"]:
            predictions_array = predictions["predictions"]
            stats["mean_prediction"] = sum(predictions_array) / len(predictions_array)
            stats["min_prediction"] = min(predictions_array)
            stats["max_prediction"] = max(predictions_array)
        
        return stats
    
    def _create_response(
        self,
        predictions: Dict[str, Any],
        stats: Dict[str, Any],
        output_path: Optional[Path],
        start_time: datetime,
        end_time: datetime
    ) -> PredictionResponseDTO:
        """Create response DTO."""
        return PredictionResponseDTO(
            success=True,
            predictions=predictions.get("predictions", []),
            labels=predictions.get("labels"),
            confidence_scores=predictions.get("confidence_scores"),
            probabilities=predictions.get("probabilities"),
            embeddings=predictions.get("embeddings"),
            statistics=stats,
            output_path=output_path,
            prediction_time_seconds=(end_time - start_time).total_seconds(),
            threshold_applied=predictions.get("threshold_applied")
        )