"""Train Model Use Case.

This use case orchestrates the training of a BERT model by coordinating
between domain services, ports, and adapters.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from application.dto.training import TrainingRequestDTO, TrainingResponseDTO
from domain.services.training_service import (
    TrainingConfig, TrainingService, TrainingState,
    OptimizerType, SchedulerType
)
from ports.primary.training import TrainingResult
from ports.secondary.storage import StorageService
from ports.secondary.monitoring import MonitoringService
from ports.secondary.configuration import ConfigurationProvider
from ports.secondary.checkpointing import CheckpointManager
from ports.secondary.metrics import MetricsCollector


class TrainModelUseCase:
    """Use case for training a model.
    
    This orchestrates the entire training workflow by:
    1. Validating the request
    2. Setting up the training environment
    3. Coordinating between domain services and ports
    4. Managing the training lifecycle
    5. Collecting and returning results
    """
    
    def __init__(
        self,
        training_service: TrainingService,
        storage_port: StorageService,
        monitoring_port: MonitoringService,
        config_port: ConfigurationProvider,
        checkpoint_port: CheckpointManager,
        metrics_port: MetricsCollector,
    ):
        """Initialize the use case with required dependencies.
        
        Args:
            training_service: Domain service for training logic
            storage_port: Port for file storage operations
            monitoring_port: Port for logging and monitoring
            config_port: Port for configuration management
            checkpoint_port: Port for checkpoint management
            metrics_port: Port for metrics calculation
        """
        self.training_service = training_service
        self.storage = storage_port
        self.monitoring = monitoring_port
        self.config = config_port
        self.checkpoints = checkpoint_port
        self.metrics = metrics_port
    
    async def execute(self, request: TrainingRequestDTO) -> TrainingResponseDTO:
        """Execute the training use case.
        
        Args:
            request: Training request from external actor
            
        Returns:
            Training response with results
        """
        start_time = datetime.now()
        
        try:
            # Validate request
            errors = request.validate()
            if errors:
                return TrainingResponseDTO(
                    success=False,
                    error_message=f"Validation errors: {', '.join(errors)}"
                )
            
            # Set up monitoring
            run_id = await self._setup_monitoring(request)
            
            # Convert DTO to domain config
            training_config = self._create_training_config(request)
            
            # Load or create model
            model = await self._prepare_model(request)
            
            # Prepare data loaders
            train_loader = await self._create_data_loader(
                request.train_data_path, 
                request.batch_size,
                is_training=True
            )
            
            val_loader = None
            if request.val_data_path:
                val_loader = await self._create_data_loader(
                    request.val_data_path,
                    request.batch_size,
                    is_training=False
                )
            
            # Resume from checkpoint if specified
            if request.resume_from_checkpoint:
                await self._resume_from_checkpoint(
                    model,
                    request.resume_from_checkpoint
                )
            
            # Run training
            training_result = await self._run_training(
                model,
                train_loader,
                val_loader,
                training_config,
                request.output_dir,
                run_id
            )
            
            # Save final artifacts
            final_paths = await self._save_final_artifacts(
                model,
                training_result,
                request.output_dir,
                run_id
            )
            
            # Create response
            end_time = datetime.now()
            response = self._create_response(
                training_result,
                final_paths,
                start_time,
                end_time,
                run_id,
                request
            )
            
            # Log completion
            await self.monitoring.log_info(
                f"Training completed successfully: {run_id}"
            )
            
            return response
            
        except Exception as e:
            await self.monitoring.log_error(
                f"Training failed: {str(e)}"
            )
            return TrainingResponseDTO.from_error(e)
    
    async def _setup_monitoring(self, request: TrainingRequestDTO) -> str:
        """Set up monitoring and tracking.
        
        Returns:
            Run ID for tracking
        """
        # Create run directory
        run_name = request.run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = request.output_dir / run_name
        await self.storage.create_directory(run_dir)
        
        # Initialize monitoring
        await self.monitoring.start_run(
            name=run_name,
            experiment=request.experiment_name,
            tags=request.tags
        )
        
        # Log configuration
        await self.monitoring.log_params({
            "model_type": request.model_type,
            "num_epochs": request.num_epochs,
            "batch_size": request.batch_size,
            "learning_rate": request.learning_rate,
            "optimizer": request.optimizer_type,
            "scheduler": request.scheduler_type,
        })
        
        return run_name
    
    def _create_training_config(self, request: TrainingRequestDTO) -> TrainingConfig:
        """Convert request DTO to domain training config."""
        return TrainingConfig(
            num_epochs=request.num_epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            weight_decay=request.weight_decay,
            max_grad_norm=request.max_grad_norm,
            optimizer_type=OptimizerType(request.optimizer_type.upper()),
            scheduler_type=SchedulerType(request.scheduler_type.upper()),
            warmup_steps=request.warmup_steps,
            warmup_ratio=request.warmup_ratio,
            eval_strategy=request.eval_strategy,
            eval_steps=request.eval_steps,
            save_strategy=request.save_strategy,
            save_steps=request.save_steps,
            early_stopping_patience=request.early_stopping_patience,
            early_stopping_threshold=request.early_stopping_threshold,
            gradient_accumulation_steps=request.gradient_accumulation_steps,
            use_mixed_precision=request.use_mixed_precision,
            save_total_limit=request.save_total_limit,
            load_best_model_at_end=request.load_best_model_at_end,
            metric_for_best_model=request.metric_for_best_model,
            greater_is_better=request.greater_is_better,
            logging_steps=request.logging_steps,
            logging_first_step=request.logging_first_step,
            label_smoothing_factor=request.label_smoothing_factor,
            gradient_checkpointing=request.gradient_checkpointing,
        )
    
    async def _prepare_model(self, request: TrainingRequestDTO) -> Any:
        """Load or create the model.
        
        This would typically use a model factory or builder.
        """
        # This is a placeholder - in real implementation, this would
        # use the model factory to create the appropriate model
        # based on request.model_type and request.model_config
        raise NotImplementedError("Model preparation to be implemented")
    
    async def _create_data_loader(
        self, 
        data_path: Path, 
        batch_size: int,
        is_training: bool
    ) -> Any:
        """Create a data loader.
        
        This would typically use a data factory.
        """
        # This is a placeholder - in real implementation, this would
        # use the data factory to create the appropriate data loader
        raise NotImplementedError("Data loader creation to be implemented")
    
    async def _resume_from_checkpoint(
        self, 
        model: Any, 
        checkpoint_path: Path
    ) -> None:
        """Resume training from a checkpoint."""
        checkpoint_data = await self.checkpoints.load_checkpoint(checkpoint_path)
        # Apply checkpoint to model and training state
        # This would be implemented based on the specific framework
        pass
    
    async def _run_training(
        self,
        model: Any,
        train_loader: Any,
        val_loader: Optional[Any],
        config: TrainingConfig,
        output_dir: Path,
        run_id: str
    ) -> TrainingResult:
        """Run the actual training loop.
        
        This coordinates between the domain training service and
        various ports for checkpointing, monitoring, etc.
        """
        # Initialize training components
        optimizer = self.training_service.create_optimizer(model.parameters())
        scheduler = self.training_service.create_scheduler(optimizer)
        
        # Training state
        state = TrainingState()
        best_metric = float('inf') if not config.greater_is_better else float('-inf')
        best_model_path = None
        
        # Calculate total steps
        total_steps = config.compute_total_steps(len(train_loader.dataset))
        state.total_steps = total_steps
        
        # Training loop
        for epoch in range(config.num_epochs):
            state.epoch = epoch
            state.epoch_start_time = datetime.now()
            
            # Training epoch
            train_metrics = await self._train_epoch(
                model, train_loader, optimizer, scheduler, config, state
            )
            
            # Log metrics
            await self.monitoring.log_metrics(train_metrics.to_dict(), step=state.global_step)
            
            # Validation
            if val_loader and self.training_service.should_evaluate():
                val_metrics = await self._validate(model, val_loader, state)
                await self.monitoring.log_metrics(
                    {f"val_{k}": v for k, v in val_metrics.items()},
                    step=state.global_step
                )
                
                # Check for improvement
                current_metric = val_metrics.get(config.metric_for_best_model, val_metrics['loss'])
                improved = state.check_improvement(
                    current_metric,
                    config.metric_for_best_model,
                    config.greater_is_better
                )
                
                if improved:
                    # Save best model
                    best_model_path = output_dir / f"checkpoint-best"
                    await self._save_checkpoint(
                        model, optimizer, scheduler, state, best_model_path
                    )
            
            # Regular checkpoint
            if self.training_service.should_save():
                checkpoint_path = output_dir / f"checkpoint-{state.global_step}"
                await self._save_checkpoint(
                    model, optimizer, scheduler, state, checkpoint_path
                )
            
            # Early stopping check
            if self.training_service.should_stop():
                state.should_stop = True
                break
        
        # Create training result
        return TrainingResult(
            final_train_loss=state.train_loss,
            final_val_loss=state.eval_loss or 0.0,
            best_val_loss=state.best_metric or 0.0,
            best_val_metric=state.best_metric or 0.0,
            final_metrics=state.metrics,
            train_history=state.train_history,
            val_history=state.eval_history,
            best_model_path=best_model_path,
            total_epochs=state.epoch + 1,
            total_steps=state.global_step,
            early_stopped=state.should_stop,
            stop_reason="Early stopping" if state.should_stop else None,
        )
    
    async def _train_epoch(
        self,
        model: Any,
        train_loader: Any,
        optimizer: Any,
        scheduler: Any,
        config: TrainingConfig,
        state: TrainingState
    ) -> Any:
        """Train for one epoch."""
        # This is a simplified placeholder
        # Real implementation would iterate through batches
        raise NotImplementedError("Training epoch to be implemented")
    
    async def _validate(
        self,
        model: Any,
        val_loader: Any,
        state: TrainingState
    ) -> Dict[str, float]:
        """Validate the model."""
        # This is a simplified placeholder
        # Real implementation would run evaluation
        raise NotImplementedError("Validation to be implemented")
    
    async def _save_checkpoint(
        self,
        model: Any,
        optimizer: Any,
        scheduler: Any,
        state: TrainingState,
        path: Path
    ) -> None:
        """Save a training checkpoint."""
        checkpoint_data = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "training_state": state.to_dict(),
            "timestamp": datetime.now().isoformat(),
        }
        await self.checkpoints.save_checkpoint(checkpoint_data, path)
    
    async def _save_final_artifacts(
        self,
        model: Any,
        result: TrainingResult,
        output_dir: Path,
        run_id: str
    ) -> Dict[str, Path]:
        """Save final model artifacts."""
        paths = {}
        
        # Save final model
        final_path = output_dir / "model_final"
        await self.storage.save_model(model, final_path)
        paths["final_model"] = final_path
        
        # Save training history
        history_path = output_dir / "training_history.json"
        await self.storage.save_json({
            "train_history": result.train_history,
            "val_history": result.val_history,
        }, history_path)
        paths["history"] = history_path
        
        # Save metrics
        metrics_path = output_dir / "final_metrics.json"
        await self.storage.save_json(result.final_metrics, metrics_path)
        paths["metrics"] = metrics_path
        
        return paths
    
    def _create_response(
        self,
        result: TrainingResult,
        paths: Dict[str, Path],
        start_time: datetime,
        end_time: datetime,
        run_id: str,
        request: TrainingRequestDTO
    ) -> TrainingResponseDTO:
        """Create the response DTO."""
        return TrainingResponseDTO(
            success=True,
            final_train_loss=result.final_train_loss,
            final_val_loss=result.final_val_loss,
            best_val_loss=result.best_val_loss,
            best_val_metric=result.best_val_metric,
            final_metrics=result.final_metrics,
            train_history=result.train_history,
            val_history=result.val_history,
            final_model_path=paths.get("final_model"),
            best_model_path=result.best_model_path,
            total_epochs=result.total_epochs,
            total_steps=result.total_steps,
            total_time_seconds=(end_time - start_time).total_seconds(),
            early_stopped=result.early_stopped,
            stop_reason=result.stop_reason,
            run_id=run_id,
            start_time=start_time,
            end_time=end_time,
            config_used=request.__dict__,
        )