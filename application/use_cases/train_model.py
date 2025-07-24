"""Refactored Train Model Use Case.

This use case provides a clean interface for training models,
delegating the actual orchestration to the training orchestrator service.
"""

from typing import Optional, Dict, Any
from pathlib import Path
import asyncio

from infrastructure.di import use_case
from application.dto.training import TrainingRequestDTO, TrainingResponseDTO
from application.services.training_orchestrator import TrainingOrchestrator
from domain.entities.model import BertModel, ModelId
from domain.entities.dataset import Dataset, DatasetId
from domain.value_objects.hyperparameters import (
    Hyperparameters,
    OptimizerType,
    LearningRateSchedule
)
from application.ports.secondary.monitoring import MonitoringService
from application.ports.secondary.storage import StorageService


@use_case
class TrainModelUseCase:
    """Clean use case for training a model.
    
    This use case:
    1. Validates the request
    2. Converts DTOs to domain objects
    3. Delegates to the training orchestrator
    4. Converts results back to DTOs
    """
    
    def __init__(
        self,
        training_orchestrator: TrainingOrchestrator,
        monitoring_service: MonitoringService,
        storage_service: StorageService
    ):
        """Initialize with required dependencies."""
        self.orchestrator = training_orchestrator
        self.monitoring = monitoring_service
        self.storage = storage_service
    
    async def execute(self, request: TrainingRequestDTO) -> TrainingResponseDTO:
        """Execute the training use case.
        
        Args:
            request: Training request from the primary adapter
            
        Returns:
            Training response with results
        """
        try:
            # Validate request
            validation_errors = self._validate_request(request)
            if validation_errors:
                return TrainingResponseDTO(
                    success=False,
                    error_message=f"Validation failed: {', '.join(validation_errors)}"
                )
            
            # Start monitoring
            await self.monitoring.start_run(
                name=request.run_name or "training_run",
                experiment=request.experiment_name,
                tags=request.tags or {}
            )
            
            # Convert DTOs to domain objects
            model = await self._load_or_create_model(request)
            dataset = await self._load_dataset(request)
            hyperparameters = self._create_hyperparameters(request)
            
            # Log configuration
            await self._log_configuration(request, hyperparameters)
            
            # Execute training via orchestrator
            result = await self.orchestrator.train_model(
                model=model,
                dataset=dataset,
                hyperparameters=hyperparameters,
                output_dir=str(request.output_dir),
                resume_from=str(request.resume_from_checkpoint) if request.resume_from_checkpoint else None
            )
            
            # Log completion
            await self.monitoring.log_info(
                f"Training completed successfully. "
                f"Best model: {result.best_model_path}"
            )
            
            # End monitoring run
            await self.monitoring.end_run()
            
            # Convert result to response DTO
            return self._create_response(result, request)
            
        except Exception as e:
            # Log error
            await self.monitoring.log_error(f"Training failed: {str(e)}")
            await self.monitoring.end_run(status="FAILED")
            
            # Return error response
            return TrainingResponseDTO(
                success=False,
                error_message=str(e)
            )
    
    def _validate_request(self, request: TrainingRequestDTO) -> list[str]:
        """Validate the training request."""
        errors = []
        
        # Basic validation
        if request.num_epochs <= 0:
            errors.append("num_epochs must be positive")
        if request.batch_size <= 0:
            errors.append("batch_size must be positive")
        if request.learning_rate <= 0:
            errors.append("learning_rate must be positive")
        
        # Path validation
        if not request.train_data_path:
            errors.append("train_data_path is required")
        elif not Path(request.train_data_path).exists():
            errors.append(f"train_data_path does not exist: {request.train_data_path}")
        
        # Output directory
        if not request.output_dir:
            errors.append("output_dir is required")
        
        # Model configuration
        if not request.model_type:
            errors.append("model_type is required")
        
        return errors
    
    async def _load_or_create_model(self, request: TrainingRequestDTO) -> BertModel:
        """Load existing model or create new one."""
        if request.pretrained_model_path:
            # Load pretrained model
            model_data = await self.storage.load_model(
                str(request.pretrained_model_path)
            )
            return BertModel(
                id=ModelId(model_data.get("id", "loaded_model")),
                name=model_data.get("name", request.model_type),
                architecture=model_data.get("architecture"),
                config=model_data.get("config", request.model_config or {})
            )
        else:
            # Create new model
            return BertModel(
                id=ModelId(f"{request.model_type}_model"),
                name=request.model_type,
                architecture={
                    "type": request.model_type,
                    "config": request.model_config or {}
                },
                config=request.model_config or {}
            )
    
    async def _load_dataset(self, request: TrainingRequestDTO) -> Dataset:
        """Load the training dataset."""
        # This is simplified - in practice would use a data factory
        dataset_info = await self.storage.get_file_info(
            str(request.train_data_path)
        )
        
        return Dataset(
            id=DatasetId("training_dataset"),
            name="Training Data",
            size=dataset_info.get("num_samples", 1000),  # Placeholder
            features=dataset_info.get("features", []),
            path=str(request.train_data_path)
        )
    
    def _create_hyperparameters(self, request: TrainingRequestDTO) -> Hyperparameters:
        """Create hyperparameters from request."""
        return Hyperparameters(
            num_epochs=request.num_epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            optimizer_type=OptimizerType(request.optimizer_type.upper()),
            weight_decay=request.weight_decay,
            adam_beta1=0.9,  # Could be in request
            adam_beta2=0.999,  # Could be in request
            adam_epsilon=1e-8,  # Could be in request
            lr_schedule=LearningRateSchedule(request.scheduler_type.upper()),
            warmup_steps=request.warmup_steps,
            warmup_ratio=request.warmup_ratio,
            gradient_accumulation_steps=request.gradient_accumulation_steps,
            gradient_clipping_max_norm=request.max_grad_norm,
            dropout=0.1,  # Could be in model config
            label_smoothing=request.label_smoothing_factor,
            evaluation_strategy=request.eval_strategy,
            eval_steps=request.eval_steps,
            save_strategy=request.save_strategy,
            save_steps=request.save_steps,
            early_stopping_patience=request.early_stopping_patience,
            early_stopping_min_delta=request.early_stopping_threshold,
            mixed_precision=request.use_mixed_precision,
            gradient_checkpointing=request.gradient_checkpointing,
            compile_model=True,  # Could be in request
            seed=42  # Could be in request
        )
    
    async def _log_configuration(
        self,
        request: TrainingRequestDTO,
        hyperparameters: Hyperparameters
    ) -> None:
        """Log training configuration."""
        config_dict = hyperparameters.to_dict()
        config_dict.update({
            "model_type": request.model_type,
            "train_data_path": str(request.train_data_path),
            "val_data_path": str(request.val_data_path) if request.val_data_path else None,
            "output_dir": str(request.output_dir),
            "resume_from": str(request.resume_from_checkpoint) if request.resume_from_checkpoint else None
        })
        
        await self.monitoring.log_params(config_dict)
    
    def _create_response(
        self,
        result: Any,
        request: TrainingRequestDTO
    ) -> TrainingResponseDTO:
        """Create response DTO from training result."""
        return TrainingResponseDTO(
            success=True,
            run_id=result.session.id,
            final_train_loss=result.final_metrics.get("final_loss", 0.0),
            final_val_loss=result.final_metrics.get("final_val_loss"),
            best_val_loss=result.session.state.best_loss,
            best_val_metric=result.session.state.best_metric,
            final_metrics=result.final_metrics,
            train_history=[],  # Could extract from session
            val_history=[],  # Could extract from session
            final_model_path=Path(result.final_model_path),
            best_model_path=Path(result.best_model_path),
            total_epochs=result.session.state.current_epoch,
            total_steps=result.session.state.current_step,
            total_time_seconds=result.training_time_seconds,
            early_stopped=result.session.status.value == "stopped",
            stop_reason="Early stopping" if result.session.status.value == "stopped" else None,
            config_used=hyperparameters.to_dict()
        )