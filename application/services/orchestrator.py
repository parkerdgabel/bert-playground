"""Training Orchestrator - coordinates complex training workflows."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
import asyncio
from enum import Enum

from domain.entities.training import TrainingSession, TrainingMetrics
from domain.entities.model import Model
from domain.services import ModelTrainingService, EvaluationService
from domain.ports import (
    ComputePort,
    MonitoringPort,
    CheckpointPort,
    MetricsCalculatorPort
)


class TrainingPhase(Enum):
    """Training phases for orchestration."""
    WARMUP = "warmup"
    MAIN_TRAINING = "main_training"
    FINE_TUNING = "fine_tuning"
    EVALUATION = "evaluation"
    OPTIMIZATION = "optimization"


@dataclass
class TrainingPlan:
    """Plan for multi-phase training."""
    phases: List[Dict[str, Any]]
    total_epochs: int
    checkpoints: List[int]  # Epochs to checkpoint
    evaluation_frequency: int
    early_stopping_enabled: bool
    optimization_enabled: bool


@dataclass
class TrainingOrchestrator:
    """Orchestrates complex training workflows.
    
    This service coordinates multi-phase training, hyperparameter
    scheduling, and advanced training strategies.
    """
    
    training_service: ModelTrainingService
    evaluation_service: EvaluationService
    compute_port: ComputePort
    monitoring_port: MonitoringPort
    checkpoint_port: CheckpointPort
    metrics_port: MetricsCalculatorPort
    
    async def execute_training_plan(
        self,
        model: Model,
        training_plan: TrainingPlan,
        train_loader: Any,
        val_loader: Optional[Any] = None,
        callbacks: Optional[List[Callable]] = None
    ) -> Dict[str, Any]:
        """Execute a complex training plan with multiple phases.
        
        Args:
            model: Model to train
            training_plan: Multi-phase training plan
            train_loader: Training data loader
            val_loader: Optional validation data loader
            callbacks: Optional callbacks for custom behavior
            
        Returns:
            Dictionary with training results and metrics
        """
        results = {
            "phases": {},
            "best_metrics": {},
            "final_model_path": None,
            "total_time_seconds": 0,
            "optimization_results": None
        }
        
        start_time = datetime.now()
        best_metric_value = float('inf')
        best_model_state = None
        
        try:
            # Initialize training
            await self.monitoring_port.start_run(
                name=f"orchestrated_training_{start_time.strftime('%Y%m%d_%H%M%S')}",
                tags={"orchestrator": "multi_phase", "phases": len(training_plan.phases)}
            )
            
            # Execute each phase
            for phase_idx, phase_config in enumerate(training_plan.phases):
                phase_name = phase_config["name"]
                phase_type = TrainingPhase(phase_config["type"])
                
                await self.monitoring_port.log_info(f"Starting phase {phase_idx + 1}: {phase_name}")
                
                # Execute phase based on type
                if phase_type == TrainingPhase.WARMUP:
                    phase_result = await self._execute_warmup_phase(
                        model, train_loader, val_loader, phase_config
                    )
                elif phase_type == TrainingPhase.MAIN_TRAINING:
                    phase_result = await self._execute_main_training_phase(
                        model, train_loader, val_loader, phase_config
                    )
                elif phase_type == TrainingPhase.FINE_TUNING:
                    phase_result = await self._execute_fine_tuning_phase(
                        model, train_loader, val_loader, phase_config
                    )
                elif phase_type == TrainingPhase.EVALUATION:
                    phase_result = await self._execute_evaluation_phase(
                        model, val_loader or train_loader, phase_config
                    )
                else:
                    raise ValueError(f"Unknown phase type: {phase_type}")
                
                results["phases"][phase_name] = phase_result
                
                # Track best model
                if "best_metric" in phase_result:
                    metric_value = phase_result["best_metric"]
                    if metric_value < best_metric_value:
                        best_metric_value = metric_value
                        best_model_state = model.get_state()
                        results["best_metrics"] = phase_result.get("metrics", {})
                
                # Run callbacks
                if callbacks:
                    for callback in callbacks:
                        await callback(phase_name, phase_result)
                
                # Check early stopping
                if training_plan.early_stopping_enabled:
                    if await self._should_stop_early(phase_result, phase_config):
                        await self.monitoring_port.log_info("Early stopping triggered")
                        break
            
            # Optimization phase if enabled
            if training_plan.optimization_enabled and best_model_state:
                await self.monitoring_port.log_info("Starting optimization phase")
                
                # Restore best model
                model.load_state(best_model_state)
                
                optimization_result = await self._execute_optimization_phase(
                    model, val_loader or train_loader
                )
                results["optimization_results"] = optimization_result
            
            # Save final model
            final_path = await self._save_final_model(model, best_model_state)
            results["final_model_path"] = final_path
            
            # Calculate total time
            end_time = datetime.now()
            results["total_time_seconds"] = (end_time - start_time).total_seconds()
            
            # Log summary
            await self._log_training_summary(results)
            
            return results
            
        except Exception as e:
            await self.monitoring_port.log_error(f"Training orchestration failed: {str(e)}")
            raise
        finally:
            await self.monitoring_port.end_run()
    
    async def _execute_warmup_phase(
        self,
        model: Model,
        train_loader: Any,
        val_loader: Optional[Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute warmup training phase with lower learning rate."""
        # Freeze certain layers if specified
        if config.get("freeze_layers"):
            self._freeze_layers(model, config["freeze_layers"])
        
        # Create warmup session
        session = TrainingSession(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config={
                "num_epochs": config.get("epochs", 1),
                "learning_rate": config.get("learning_rate", 1e-5),
                "weight_decay": config.get("weight_decay", 0.01),
                "optimizer_type": "adamw",
                "scheduler_type": "constant",
                "eval_strategy": "epoch",
                "logging_steps": 50
            }
        )
        
        # Run training
        result = await self.training_service.train(
            session,
            compute_port=self.compute_port,
            monitoring_port=self.monitoring_port
        )
        
        # Unfreeze layers
        if config.get("freeze_layers"):
            self._unfreeze_layers(model)
        
        return {
            "epochs_completed": result.epochs_completed,
            "final_loss": result.final_train_loss,
            "best_metric": result.best_val_loss or result.final_train_loss,
            "metrics": result.final_metrics
        }
    
    async def _execute_main_training_phase(
        self,
        model: Model,
        train_loader: Any,
        val_loader: Optional[Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute main training phase."""
        # Apply training strategies
        strategies = []
        
        if config.get("use_gradient_accumulation"):
            strategies.append("gradient_accumulation")
        
        if config.get("use_mixed_precision"):
            strategies.append("mixed_precision")
        
        if config.get("use_gradient_clipping"):
            strategies.append("gradient_clipping")
        
        # Create training session
        session = TrainingSession(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config={
                "num_epochs": config.get("epochs", 10),
                "learning_rate": config.get("learning_rate", 5e-5),
                "weight_decay": config.get("weight_decay", 0.01),
                "optimizer_type": config.get("optimizer", "adamw"),
                "scheduler_type": config.get("scheduler", "cosine"),
                "warmup_ratio": config.get("warmup_ratio", 0.1),
                "eval_strategy": "epoch",
                "save_strategy": "epoch",
                "gradient_accumulation_steps": config.get("gradient_accumulation_steps", 1),
                "use_mixed_precision": config.get("use_mixed_precision", False),
                "max_grad_norm": config.get("max_grad_norm", 1.0),
                "early_stopping_patience": config.get("early_stopping_patience", 3),
                "metric_for_best_model": config.get("metric_for_best_model", "eval_loss"),
                "logging_steps": 100
            }
        )
        
        # Create checkpoint callback
        checkpoint_callback = self._create_checkpoint_callback(config.get("checkpoint_dir"))
        
        # Run training
        result = await self.training_service.train(
            session,
            compute_port=self.compute_port,
            monitoring_port=self.monitoring_port,
            checkpoint_callback=checkpoint_callback
        )
        
        return {
            "epochs_completed": result.epochs_completed,
            "steps_completed": result.steps_completed,
            "final_loss": result.final_train_loss,
            "best_metric": result.best_val_loss or result.final_train_loss,
            "metrics": result.final_metrics,
            "early_stopped": result.early_stopped,
            "best_checkpoint": result.best_checkpoint_path
        }
    
    async def _execute_fine_tuning_phase(
        self,
        model: Model,
        train_loader: Any,
        val_loader: Optional[Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute fine-tuning phase with careful hyperparameters."""
        # Apply LoRA or other parameter-efficient methods if specified
        if config.get("use_lora"):
            model = await self._apply_lora(model, config["lora_config"])
        
        # Fine-tuning typically uses lower learning rate
        session = TrainingSession(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config={
                "num_epochs": config.get("epochs", 3),
                "learning_rate": config.get("learning_rate", 2e-5),
                "weight_decay": config.get("weight_decay", 0.01),
                "optimizer_type": "adamw",
                "scheduler_type": "linear",
                "warmup_ratio": 0.06,
                "eval_strategy": "steps",
                "eval_steps": config.get("eval_steps", 500),
                "save_strategy": "steps",
                "save_steps": config.get("save_steps", 500),
                "gradient_accumulation_steps": config.get("gradient_accumulation_steps", 4),
                "label_smoothing_factor": config.get("label_smoothing", 0.1),
                "logging_steps": 50
            }
        )
        
        # Run training
        result = await self.training_service.train(
            session,
            compute_port=self.compute_port,
            monitoring_port=self.monitoring_port
        )
        
        return {
            "epochs_completed": result.epochs_completed,
            "final_loss": result.final_train_loss,
            "best_metric": result.best_val_loss or result.final_train_loss,
            "metrics": result.final_metrics,
            "lora_applied": config.get("use_lora", False)
        }
    
    async def _execute_evaluation_phase(
        self,
        model: Model,
        data_loader: Any,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute evaluation phase."""
        # Run comprehensive evaluation
        eval_results = await self.evaluation_service.evaluate(
            model=model,
            data_loader=data_loader,
            compute_port=self.compute_port,
            metrics_to_compute=config.get("metrics", ["accuracy", "f1", "loss"])
        )
        
        # Additional analysis if requested
        analysis_results = {}
        
        if config.get("compute_confusion_matrix"):
            analysis_results["confusion_matrix"] = await self._compute_confusion_matrix(
                model, data_loader
            )
        
        if config.get("compute_per_class_metrics"):
            analysis_results["per_class_metrics"] = await self._compute_per_class_metrics(
                eval_results
            )
        
        if config.get("error_analysis"):
            analysis_results["error_analysis"] = await self._perform_error_analysis(
                model, data_loader, eval_results
            )
        
        return {
            "metrics": eval_results.metrics,
            "loss": eval_results.loss,
            "analysis": analysis_results,
            "samples_evaluated": eval_results.num_samples
        }
    
    async def _execute_optimization_phase(
        self,
        model: Model,
        data_loader: Any
    ) -> Dict[str, Any]:
        """Execute model optimization phase."""
        optimization_results = {}
        
        # Quantization
        if await self._should_quantize(model):
            quantized_model = await self.compute_port.quantize(model, bits=8)
            quant_metrics = await self._evaluate_optimized_model(
                quantized_model, data_loader, "quantized"
            )
            optimization_results["quantization"] = quant_metrics
        
        # Pruning
        if await self._should_prune(model):
            pruned_model = await self.compute_port.prune(model, prune_ratio=0.1)
            prune_metrics = await self._evaluate_optimized_model(
                pruned_model, data_loader, "pruned"
            )
            optimization_results["pruning"] = prune_metrics
        
        # Knowledge distillation
        if await self._should_distill(model):
            distilled_model = await self._perform_distillation(model, data_loader)
            distill_metrics = await self._evaluate_optimized_model(
                distilled_model, data_loader, "distilled"
            )
            optimization_results["distillation"] = distill_metrics
        
        return optimization_results
    
    def _freeze_layers(self, model: Model, layers_to_freeze: List[str]) -> None:
        """Freeze specified layers in the model."""
        for layer_name in layers_to_freeze:
            if hasattr(model, layer_name):
                layer = getattr(model, layer_name)
                for param in layer.parameters():
                    param.requires_grad = False
    
    def _unfreeze_layers(self, model: Model) -> None:
        """Unfreeze all layers in the model."""
        for param in model.parameters():
            param.requires_grad = True
    
    async def _apply_lora(self, model: Model, lora_config: Dict[str, Any]) -> Model:
        """Apply LoRA to the model."""
        # This would integrate with LoRA implementation
        # Placeholder for now
        return model
    
    def _create_checkpoint_callback(self, checkpoint_dir: Optional[Path]) -> Callable:
        """Create callback for saving checkpoints."""
        checkpoint_dir = checkpoint_dir or Path("checkpoints")
        
        async def save_checkpoint(model: Model, state: Dict[str, Any], metrics: TrainingMetrics):
            checkpoint_path = checkpoint_dir / f"checkpoint-{state['global_step']}"
            await self.checkpoint_port.save({
                "model_state": model.get_state(),
                "training_state": state,
                "metrics": metrics.to_dict(),
                "timestamp": datetime.now().isoformat()
            }, checkpoint_path)
        
        return save_checkpoint
    
    async def _should_stop_early(
        self,
        phase_result: Dict[str, Any],
        phase_config: Dict[str, Any]
    ) -> bool:
        """Check if early stopping criteria are met."""
        if not phase_config.get("early_stopping_enabled", True):
            return False
        
        # Check if loss is not improving
        if "loss_not_improving" in phase_result:
            return phase_result["loss_not_improving"]
        
        # Check if target metric is reached
        target_metric = phase_config.get("target_metric")
        if target_metric:
            current_value = phase_result.get("best_metric", float('inf'))
            if current_value <= target_metric:
                return True
        
        return False
    
    async def _save_final_model(
        self,
        model: Model,
        best_model_state: Optional[Dict[str, Any]]
    ) -> Path:
        """Save the final model."""
        output_dir = Path("output") / f"orchestrated_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save best model if available
        if best_model_state:
            model.load_state(best_model_state)
        
        # Save model
        model_path = output_dir / "final_model"
        await self.checkpoint_port.save({
            "model_state": model.get_state(),
            "model_config": model.config.to_dict(),
            "metadata": {
                "training_completed": datetime.now().isoformat(),
                "orchestrator_version": "1.0.0"
            }
        }, model_path)
        
        return model_path
    
    async def _log_training_summary(self, results: Dict[str, Any]) -> None:
        """Log comprehensive training summary."""
        summary = {
            "total_phases": len(results["phases"]),
            "total_time_hours": results["total_time_seconds"] / 3600,
            "best_metrics": results["best_metrics"],
            "phases_summary": {}
        }
        
        for phase_name, phase_result in results["phases"].items():
            summary["phases_summary"][phase_name] = {
                "epochs": phase_result.get("epochs_completed", 0),
                "final_loss": phase_result.get("final_loss", 0),
                "best_metric": phase_result.get("best_metric", 0)
            }
        
        if results.get("optimization_results"):
            summary["optimization"] = {
                method: result.get("improvement", 0)
                for method, result in results["optimization_results"].items()
            }
        
        await self.monitoring_port.log_metrics(summary)
        await self.monitoring_port.log_info(f"Training completed: {summary}")
    
    async def _compute_confusion_matrix(self, model: Model, data_loader: Any) -> Dict[str, Any]:
        """Compute confusion matrix for classification tasks."""
        # Placeholder implementation
        return {"computed": True}
    
    async def _compute_per_class_metrics(self, eval_results: Any) -> Dict[str, Any]:
        """Compute per-class metrics."""
        # Placeholder implementation
        return {"computed": True}
    
    async def _perform_error_analysis(
        self,
        model: Model,
        data_loader: Any,
        eval_results: Any
    ) -> Dict[str, Any]:
        """Perform detailed error analysis."""
        # Placeholder implementation
        return {"computed": True}
    
    async def _should_quantize(self, model: Model) -> bool:
        """Determine if model should be quantized."""
        # Check model size and target deployment
        return model.num_parameters > 100_000_000
    
    async def _should_prune(self, model: Model) -> bool:
        """Determine if model should be pruned."""
        # Check sparsity potential
        return model.num_parameters > 50_000_000
    
    async def _should_distill(self, model: Model) -> bool:
        """Determine if model should be distilled."""
        # Check if distillation would be beneficial
        return model.num_parameters > 200_000_000
    
    async def _evaluate_optimized_model(
        self,
        model: Model,
        data_loader: Any,
        optimization_type: str
    ) -> Dict[str, Any]:
        """Evaluate an optimized model."""
        eval_results = await self.evaluation_service.evaluate(
            model=model,
            data_loader=data_loader,
            compute_port=self.compute_port
        )
        
        return {
            "type": optimization_type,
            "metrics": eval_results.metrics,
            "size_reduction": 0.0,  # Would calculate actual reduction
            "speed_improvement": 0.0  # Would benchmark actual improvement
        }
    
    async def _perform_distillation(self, model: Model, data_loader: Any) -> Model:
        """Perform knowledge distillation."""
        # Placeholder - would implement actual distillation
        return model