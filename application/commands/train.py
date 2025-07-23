"""Train Model Command - orchestrates the training workflow."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from application.dto.training import TrainingRequestDTO, TrainingResponseDTO
from domain.entities.training import TrainingSession
from domain.entities.metrics import TrainingMetrics
from domain.entities.model import BertModel, ModelArchitecture
from domain.entities.dataset import Dataset
from domain.services import ModelTrainingService, TokenizationService, CheckpointingService
from ports.secondary.data import DataLoaderPort
from ports.secondary.compute import ComputeBackend
from ports.secondary.monitoring import MonitoringService
from ports.secondary.storage import StorageService
from ports.secondary.checkpointing import CheckpointManager


@dataclass
class TrainModelCommand:
    """Command to train a model.
    
    This command orchestrates the entire training workflow by:
    1. Validating the request
    2. Setting up the training environment
    3. Loading and preparing data
    4. Creating and configuring the model
    5. Running the training loop
    6. Saving results and artifacts
    """
    
    # Domain services
    training_service: ModelTrainingService
    tokenization_service: TokenizationService
    checkpointing_service: CheckpointingService
    
    # Ports (external dependencies)
    data_loader_port: DataLoaderPort
    compute_port: ComputeBackend
    monitoring_port: MonitoringService
    storage_port: StorageService
    checkpoint_port: CheckpointManager
    
    async def execute(self, request: TrainingRequestDTO) -> TrainingResponseDTO:
        """Execute the training command.
        
        Args:
            request: Training request containing all configuration
            
        Returns:
            Response with training results and metrics
        """
        start_time = datetime.now()
        
        try:
            # 1. Validate request
            errors = request.validate()
            if errors:
                return TrainingResponseDTO(
                    success=False,
                    error_message=f"Validation errors: {', '.join(errors)}"
                )
            
            # 2. Initialize monitoring and tracking
            run_id = await self._initialize_tracking(request)
            
            # 3. Load datasets through port
            train_dataset = await self._load_dataset(
                request.train_data_path,
                is_training=True
            )
            
            val_dataset = None
            if request.val_data_path:
                val_dataset = await self._load_dataset(
                    request.val_data_path,
                    is_training=False
                )
            
            # 4. Create model configuration
            model_config = ModelConfiguration(
                model_type=request.model_type,
                config_dict=request.model_config,
                num_labels=train_dataset.num_labels,
                label_names=train_dataset.label_names
            )
            
            # 5. Create model through domain service
            model = self.training_service.create_model(model_config)
            
            # 6. Initialize compute backend (e.g., MLX)
            await self.compute_port.initialize(model)
            
            # 7. Create data loaders
            train_loader = await self.data_loader_port.create_loader(
                train_dataset,
                batch_size=request.batch_size,
                shuffle=True,
                num_workers=request.num_workers
            )
            
            val_loader = None
            if val_dataset:
                val_loader = await self.data_loader_port.create_loader(
                    val_dataset,
                    batch_size=request.batch_size,
                    shuffle=False,
                    num_workers=request.num_workers
                )
            
            # 8. Resume from checkpoint if specified
            initial_state = None
            if request.resume_from_checkpoint:
                initial_state = await self._resume_from_checkpoint(
                    model,
                    request.resume_from_checkpoint
                )
            
            # 9. Create training session
            session = TrainingSession(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=self._create_training_config(request),
                initial_state=initial_state
            )
            
            # 10. Run training through domain service
            training_result = await self.training_service.train(
                session,
                compute_port=self.compute_port,
                monitoring_port=self.monitoring_port,
                checkpoint_callback=self._create_checkpoint_callback(request.output_dir)
            )
            
            # 11. Save final artifacts
            artifacts = await self._save_artifacts(
                model,
                training_result,
                request.output_dir,
                run_id
            )
            
            # 12. Create response
            response = self._create_response(
                training_result,
                artifacts,
                start_time,
                datetime.now(),
                run_id,
                request
            )
            
            # 13. Finalize tracking
            await self.monitoring_port.log_metrics({
                "final_train_loss": response.final_train_loss,
                "final_val_loss": response.final_val_loss,
                "training_time_seconds": response.total_time_seconds
            })
            
            return response
            
        except Exception as e:
            await self.monitoring_port.log_error(f"Training failed: {str(e)}")
            return TrainingResponseDTO.from_error(e)
        finally:
            await self.monitoring_port.end_run()
    
    async def _initialize_tracking(self, request: TrainingRequestDTO) -> str:
        """Initialize experiment tracking and monitoring."""
        run_name = request.run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        await self.monitoring_port.start_run(
            name=run_name,
            experiment=request.experiment_name,
            tags=request.tags
        )
        
        # Log hyperparameters
        await self.monitoring_port.log_params({
            "model_type": request.model_type,
            "num_epochs": request.num_epochs,
            "batch_size": request.batch_size,
            "learning_rate": request.learning_rate,
            "optimizer": request.optimizer_type,
            "scheduler": request.scheduler_type,
            "weight_decay": request.weight_decay,
            "warmup_ratio": request.warmup_ratio,
        })
        
        return run_name
    
    async def _load_dataset(self, path: Path, is_training: bool) -> Dataset:
        """Load dataset through the data port."""
        # This would typically involve:
        # 1. Reading the raw data
        # 2. Tokenizing if needed (through tokenization service)
        # 3. Creating Dataset entity
        raw_data = await self.storage_port.load_data(path)
        
        # Tokenize if text data
        if self._is_text_data(raw_data):
            tokenized_data = await self.tokenization_service.tokenize_dataset(
                raw_data,
                max_length=512,  # Could be configurable
                truncation=True,
                padding="max_length"
            )
            return Dataset.from_tokenized(tokenized_data)
        else:
            # Assume tabular data, convert to text using templates
            return Dataset.from_tabular(raw_data)
    
    def _is_text_data(self, data: Any) -> bool:
        """Check if data is text-based."""
        # Implementation would check data format
        return False  # Placeholder
    
    def _create_training_config(self, request: TrainingRequestDTO) -> Dict[str, Any]:
        """Create training configuration for domain service."""
        return {
            "num_epochs": request.num_epochs,
            "learning_rate": request.learning_rate,
            "weight_decay": request.weight_decay,
            "max_grad_norm": request.max_grad_norm,
            "optimizer_type": request.optimizer_type,
            "optimizer_params": request.optimizer_params,
            "scheduler_type": request.scheduler_type,
            "warmup_steps": request.warmup_steps,
            "warmup_ratio": request.warmup_ratio,
            "eval_strategy": request.eval_strategy,
            "eval_steps": request.eval_steps,
            "save_strategy": request.save_strategy,
            "save_steps": request.save_steps,
            "early_stopping_patience": request.early_stopping_patience,
            "early_stopping_threshold": request.early_stopping_threshold,
            "metric_for_best_model": request.metric_for_best_model,
            "greater_is_better": request.greater_is_better,
            "gradient_accumulation_steps": request.gradient_accumulation_steps,
            "use_mixed_precision": request.use_mixed_precision,
            "gradient_checkpointing": request.gradient_checkpointing,
            "label_smoothing_factor": request.label_smoothing_factor,
            "logging_steps": request.logging_steps,
            "save_total_limit": request.save_total_limit,
            "load_best_model_at_end": request.load_best_model_at_end,
        }
    
    async def _resume_from_checkpoint(self, model: BertModel, checkpoint_path: Path) -> Dict[str, Any]:
        """Resume training from a checkpoint."""
        checkpoint_data = await self.checkpoint_port.load(checkpoint_path)
        
        # Restore model state
        model.load_state(checkpoint_data["model_state"])
        
        # Return training state
        return checkpoint_data.get("training_state", {})
    
    def _create_checkpoint_callback(self, output_dir: Path):
        """Create a callback function for checkpointing."""
        async def save_checkpoint(model: Model, state: Dict[str, Any], metrics: TrainingMetrics):
            checkpoint_path = output_dir / f"checkpoint-{state['global_step']}"
            
            checkpoint_data = {
                "model_state": model.get_state(),
                "training_state": state,
                "metrics": metrics.to_dict(),
                "timestamp": datetime.now().isoformat()
            }
            
            await self.checkpoint_port.save(checkpoint_data, checkpoint_path)
            await self.monitoring_port.log_info(f"Saved checkpoint at step {state['global_step']}")
        
        return save_checkpoint
    
    async def _save_artifacts(
        self,
        model: Model,
        result: Any,
        output_dir: Path,
        run_id: str
    ) -> Dict[str, Path]:
        """Save training artifacts."""
        artifacts = {}
        
        # Save final model
        final_path = output_dir / "final_model"
        await self.storage_port.save_model(model, final_path)
        artifacts["final_model"] = final_path
        
        # Save best model if different
        if result.best_checkpoint_path:
            best_model = await self.checkpoint_port.load_model(result.best_checkpoint_path)
            best_path = output_dir / "best_model"
            await self.storage_port.save_model(best_model, best_path)
            artifacts["best_model"] = best_path
        
        # Save training history
        history_path = output_dir / "training_history.json"
        await self.storage_port.save_json({
            "train_history": result.train_history,
            "val_history": result.val_history,
            "config": result.config_used,
            "run_id": run_id
        }, history_path)
        artifacts["history"] = history_path
        
        # Save final metrics
        metrics_path = output_dir / "metrics.json"
        await self.storage_port.save_json(result.final_metrics, metrics_path)
        artifacts["metrics"] = metrics_path
        
        return artifacts
    
    def _create_response(
        self,
        result: Any,
        artifacts: Dict[str, Path],
        start_time: datetime,
        end_time: datetime,
        run_id: str,
        request: TrainingRequestDTO
    ) -> TrainingResponseDTO:
        """Create response DTO from training results."""
        return TrainingResponseDTO(
            success=True,
            final_train_loss=result.final_train_loss,
            final_val_loss=result.final_val_loss,
            best_val_loss=result.best_val_loss,
            best_val_metric=result.best_val_metric,
            final_metrics=result.final_metrics,
            train_history=result.train_history,
            val_history=result.val_history,
            final_model_path=artifacts.get("final_model"),
            best_model_path=artifacts.get("best_model"),
            checkpoint_paths=result.checkpoint_paths,
            total_epochs=result.epochs_completed,
            total_steps=result.steps_completed,
            total_time_seconds=(end_time - start_time).total_seconds(),
            samples_seen=result.samples_seen,
            early_stopped=result.early_stopped,
            stop_reason=result.stop_reason,
            stopped_at_epoch=result.stopped_at_epoch,
            stopped_at_step=result.stopped_at_step,
            run_id=run_id,
            experiment_id=request.experiment_name,
            start_time=start_time,
            end_time=end_time,
            config_used=request.__dict__
        )