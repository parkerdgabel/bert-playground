"""Next-generation MLX trainer with comprehensive features.

This trainer implements the complete MLX training system using the new
configuration architecture and universal dataloader integration.
"""

import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from loguru import logger
from rich.console import Console
from training.rich_display_manager import RichDisplayManager
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
import mlflow
import pandas as pd

# Import protocols for flexible typing
from training.protocols import DataLoaderProtocol, OptimizerProtocol, ModelProtocol
from training.config import TrainingConfig
from training.memory_manager import (
    AppleSiliconMemoryManager,
    MemoryOptimizer,
    MemoryThresholds,
)
from training.monitoring import ComprehensiveMonitor
from training.performance_profiler import AppleSiliconProfiler, ProfilerConfig
from utils.logging_config import ExperimentLogger, LoggingConfig
from utils.model_registry import ModelRegistry, register_mlx_model
from utils.mlflow_evaluation import ModelEvaluator, TitanicMetrics, create_evaluation_dataset

console = Console()


class MLXTrainer:
    """Production-ready MLX trainer with comprehensive monitoring.

    This trainer combines all the lessons learned from previous implementations
    with the new configuration system, universal dataloader, and comprehensive
    monitoring including MLflow, rich console, and performance profiling.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        optimizer: optim.Optimizer | None = None,
        display_manager: RichDisplayManager | None = None,
    ):
        """Initialize the production MLX trainer.

        Args:
            model: MLX model to train
            config: Comprehensive training configuration
            optimizer: Optional optimizer (created if not provided)
            display_manager: Optional shared display manager for Rich console
        """
        self.model = model
        self.config = config
        self.display_manager = display_manager

        # Initialize logging
        self.logging_config = LoggingConfig(
            log_dir=Path(config.output_dir) / "logs",
            log_level=config.monitoring.log_level,
            log_to_file=config.monitoring.log_to_file,
            log_to_console=True,
            experiment_name=config.experiment_name or "mlx_training",
        )

        # Create optimizer if not provided
        if optimizer is None:
            self.optimizer = self._create_optimizer()
        else:
            self.optimizer = optimizer

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = -float("inf")
        self.best_metric_step = 0
        self.effective_batch_size = config.get_effective_batch_size()

        # MLX optimization state
        self.accumulated_loss = None
        self.accumulated_grads = None
        self.accumulation_step = 0
        self.steps_since_eval = 0
        self.total_steps = 0
        

        # Advanced memory management
        memory_thresholds = MemoryThresholds(
            critical_memory=config.memory.unified_memory_fraction
            * 1.1,  # Allow slight overage
            high_memory=config.memory.unified_memory_fraction,
            optimal_memory=config.memory.unified_memory_fraction * 0.8,
            low_memory=config.memory.unified_memory_fraction * 0.5,
            min_batch_size=config.memory.min_batch_size,
            max_batch_size=config.memory.max_batch_size,
        )
        self.memory_manager = AppleSiliconMemoryManager(memory_thresholds)
        self.memory_optimizer = MemoryOptimizer(self.memory_manager)
        self.current_memory_usage = 0.0
        self.memory_history = []
        self.dynamic_batch_size = config.batch_size

        # Performance profiling
        profiler_config = ProfilerConfig(
            metrics_collection_interval=config.monitoring.log_frequency,
            detailed_profiling_interval=config.monitoring.log_frequency * 10,
            thermal_monitoring_interval=config.memory.memory_check_interval,
            enable_neural_engine_monitoring=config.mlx_optimization.enable_jit,
            enable_thermal_monitoring=True,
            enable_power_monitoring=True,
            save_detailed_logs=config.monitoring.log_to_file,
            log_to_console=config.monitoring.enable_rich_console,
        )
        self.profiler = AppleSiliconProfiler(profiler_config)

        # Early stopping
        self.early_stopping_counter = 0
        self.patience_counter = 0

        # Performance tracking
        self.step_times = []
        self.throughput_history = []

        # Comprehensive monitoring system
        self.monitor = ComprehensiveMonitor(
            config=config, memory_manager=self.memory_manager, profiler=self.profiler,
            display_manager=self.display_manager
        )

        # Apply Apple Silicon optimizations if available
        self._apply_apple_silicon_optimizations()

        logger.info(
            f"Initialized Production MLX Trainer:\n"
            f"  Model: {model.__class__.__name__}\n"
            f"  Optimization Level: {config.optimization_level.value}\n"
            f"  Batch Size: {config.batch_size} (effective: {self.effective_batch_size})\n"
            f"  Learning Rate: {config.learning_rate}\n"
            f"  Epochs: {config.epochs}\n"
            f"  Apple Silicon: {self.memory_manager.is_apple_silicon}\n"
            f"  MLflow Enabled: {config.monitoring.enable_mlflow}\n"
            f"  Rich Console: {config.monitoring.enable_rich_console}\n"
            f"  Output Dir: {config.output_dir}"
        )

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        from training.config import OptimizerType

        # Base parameters
        lr = self.config.learning_rate
        weight_decay = self.config.advanced.weight_decay

        if self.config.optimizer == OptimizerType.ADAMW:
            return optim.AdamW(learning_rate=lr, weight_decay=weight_decay)
        elif self.config.optimizer == OptimizerType.ADAM:
            # Adam doesn't support weight_decay in MLX
            return optim.Adam(learning_rate=lr)
        elif self.config.optimizer == OptimizerType.SGD:
            return optim.SGD(learning_rate=lr, weight_decay=weight_decay)
        elif self.config.optimizer == OptimizerType.RMSPROP:
            # RMSprop doesn't support weight_decay in MLX
            return optim.RMSprop(learning_rate=lr)
        elif self.config.optimizer == OptimizerType.ADAGRAD:
            # Adagrad doesn't support weight_decay in MLX
            return optim.Adagrad(learning_rate=lr)
        else:
            logger.warning(f"Unknown optimizer {self.config.optimizer}, using AdamW")
            return optim.AdamW(learning_rate=lr, weight_decay=weight_decay)

    def get_memory_usage(self) -> float:
        """Get current memory usage as a fraction of available memory."""
        metrics = self.memory_manager.get_current_metrics()
        self.current_memory_usage = metrics.memory_percentage
        return metrics.memory_percentage

    def adjust_batch_size_dynamically(self) -> int:
        """Dynamically adjust batch size based on memory usage and performance."""
        if not self.config.memory.dynamic_batch_sizing:
            return self.dynamic_batch_size

        # Use advanced memory optimizer
        new_batch_size, optimization_info = (
            self.memory_optimizer.optimize_training_memory(self.dynamic_batch_size)
        )

        if new_batch_size != self.dynamic_batch_size:
            logger.info(
                f"Memory optimizer adjusted batch size: {self.dynamic_batch_size} -> {new_batch_size}"
            )
            if optimization_info.get("optimizations_applied"):
                logger.debug(
                    f"Applied optimizations: {optimization_info['optimizations_applied']}"
                )

            self.dynamic_batch_size = new_batch_size

        return self.dynamic_batch_size

    def get_learning_rate(self) -> float:
        """Get current learning rate based on schedule and step."""
        from training.config import LearningRateSchedule

        if self.total_steps == 0:
            return self.config.learning_rate

        base_lr = self.config.learning_rate
        warmup_steps = self.config.warmup_steps
        current_step = self.global_step

        # Handle warmup
        if current_step < warmup_steps:
            warmup_factor = current_step / max(1, warmup_steps)
            return base_lr * warmup_factor

        # Apply schedule after warmup
        effective_step = current_step - warmup_steps
        effective_total = max(1, self.total_steps - warmup_steps)
        progress = effective_step / effective_total

        if self.config.lr_schedule == LearningRateSchedule.CONSTANT:
            return base_lr
        elif self.config.lr_schedule == LearningRateSchedule.LINEAR_WARMUP:
            return base_lr  # Already handled warmup above
        elif (
            self.config.lr_schedule == LearningRateSchedule.COSINE
            or self.config.lr_schedule == LearningRateSchedule.COSINE_WARMUP
        ):
            return base_lr * 0.5 * (1 + np.cos(np.pi * progress))
        elif self.config.lr_schedule == LearningRateSchedule.POLYNOMIAL:
            return base_lr * (1 - progress) ** 2
        elif self.config.lr_schedule == LearningRateSchedule.EXPONENTIAL:
            return base_lr * (0.96 ** (effective_step // 100))
        else:
            return base_lr

    def compute_loss(
        self, batch: dict[str, mx.array], apply_label_smoothing: bool = True
    ) -> mx.array:
        """Compute loss for a batch with optional label smoothing."""
        # Forward pass
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch.get("labels"),
        )

        loss = outputs["loss"]

        # Apply label smoothing if configured
        if (
            apply_label_smoothing
            and self.config.advanced.label_smoothing > 0
            and "labels" in batch
        ):
            logits = outputs["logits"]
            num_classes = logits.shape[-1]

            # Convert to probabilities
            log_probs = mx.log(
                mx.softmax(logits, axis=-1) + 1e-8
            )  # Add small epsilon for numerical stability

            # Compute smoothed loss
            smoothing = self.config.advanced.label_smoothing
            confidence = 1.0 - smoothing

            # Simplified label smoothing using gather operation
            labels = batch["labels"].squeeze()

            # Get log probabilities for true labels
            true_log_probs = mx.take_along_axis(
                log_probs, labels[:, None], axis=-1
            ).squeeze()

            # Uniform distribution log probability
            uniform_log_prob = mx.log(mx.array(1.0 / num_classes))

            # Smooth loss
            smooth_loss = -(
                confidence * true_log_probs + smoothing * uniform_log_prob
            ).mean()

            # Blend with original loss
            loss = confidence * loss + smoothing * smooth_loss

        return loss

    def train_step(self, batch: dict[str, mx.array]) -> tuple[float, dict[str, Any]]:
        """Execute a training step with gradient accumulation."""

        # Start performance profiling
        self.profiler.start_step_timer(self.global_step)

        def loss_fn(model):
            return self.compute_loss(batch)

        # Compute gradients with profiling
        loss, grads = self.profiler.profile_mlx_operation(
            "gradient_computation",
            lambda: nn.value_and_grad(self.model, loss_fn)(self.model),
        )[0]

        # Scale for gradient accumulation
        if self.config.mlx_optimization.gradient_accumulation_steps > 1:
            loss = loss / self.config.mlx_optimization.gradient_accumulation_steps

        # Accumulate gradients
        if self.accumulated_grads is None:
            self.accumulated_grads = grads
            self.accumulated_loss = loss
        else:
            # Manual tree accumulation
            def accumulate_recursive(acc_grads, new_grads):
                if isinstance(new_grads, dict):
                    return {
                        k: accumulate_recursive(acc_grads[k], v)
                        for k, v in new_grads.items()
                    }
                elif hasattr(new_grads, "shape"):  # MLX array
                    return acc_grads + new_grads
                else:
                    return new_grads

            self.accumulated_grads = accumulate_recursive(self.accumulated_grads, grads)
            self.accumulated_loss = self.accumulated_loss + loss

        self.accumulation_step += 1
        self.steps_since_eval += 1

        # Update when accumulation is complete
        should_update = (
            self.accumulation_step
            % self.config.mlx_optimization.gradient_accumulation_steps
            == 0
        )

        if should_update:
            # Force evaluation of accumulated gradients
            mx.eval(self.accumulated_grads, self.accumulated_loss)

            # Gradient clipping
            if self.config.mlx_optimization.max_grad_norm > 0:
                self._clip_gradients()

            # Update model parameters
            self.optimizer.update(self.model, self.accumulated_grads)

            # Update learning rate
            new_lr = self.get_learning_rate()
            if hasattr(self.optimizer, "learning_rate"):
                self.optimizer.learning_rate = new_lr

            # Get metrics
            loss_value = float(self.accumulated_loss)

            # Reset accumulation
            self.accumulated_grads = None
            self.accumulated_loss = None
            self.accumulation_step = 0

            # Advanced memory management
            if (
                self.config.memory.force_garbage_collection
                and self.global_step % self.config.memory.gc_interval == 0
            ):
                self.memory_manager.force_garbage_collection(
                    aggressive=self.current_memory_usage
                    > self.memory_manager.thresholds.high_memory
                )

            # End performance profiling and get metrics
            batch_size = batch["input_ids"].shape[0]
            sequence_length = (
                batch["input_ids"].shape[1]
                if len(batch["input_ids"].shape) > 1
                else None
            )
            performance_metrics = self.profiler.end_step_timer(
                self.global_step, batch_size, sequence_length
            )

            metrics = {
                "loss": loss_value,
                "learning_rate": new_lr,
                "batch_size": self.dynamic_batch_size,
                "memory_usage": self.current_memory_usage,
                "epoch": self.current_epoch,
                "step": self.global_step,
                "step_time": performance_metrics.step_time_seconds,
                "samples_per_second": performance_metrics.samples_per_second,
                "tokens_per_second": performance_metrics.tokens_per_second,
            }

            # Log step metrics to comprehensive monitoring
            self.monitor.log_step(
                step=self.global_step,
                epoch=self.current_epoch,
                train_loss=loss_value,
                train_accuracy=None,  # Will be set by evaluation
                learning_rate=new_lr,
                batch_size=self.dynamic_batch_size,
            )
            
            # Log additional metrics to MLflow if enabled
            if self.config.monitoring.enable_mlflow:
                try:
                    mlflow.log_metrics({
                        "step_time": performance_metrics.step_time_seconds,
                        "samples_per_second": performance_metrics.samples_per_second,
                        "tokens_per_second": performance_metrics.tokens_per_second,
                        "memory_usage": self.current_memory_usage,
                    }, step=self.global_step)
                except Exception as e:
                    logger.debug(f"Failed to log additional metrics to MLflow: {e}")

            return loss_value, metrics
        else:
            # Still accumulating - but need to update progress
            # Call monitor to update progress display even during accumulation
            self.monitor.log_step(
                step=self.global_step,
                epoch=self.current_epoch,
                train_loss=float(self.accumulated_loss / self.accumulation_step),  # Average loss so far
                train_accuracy=None,
                learning_rate=self.get_learning_rate(),
                batch_size=self.dynamic_batch_size,
            )
            
            return 0.0, {
                "accumulating": True,
                "accumulation_step": self.accumulation_step,
            }

    def _clip_gradients(self) -> None:
        """Clip gradients by global norm."""
        max_norm = self.config.mlx_optimization.max_grad_norm

        # Compute global norm manually
        total_norm_squared = mx.array(0.0)

        def compute_norm_recursive(grads):
            nonlocal total_norm_squared
            if isinstance(grads, dict):
                for value in grads.values():
                    compute_norm_recursive(value)
            elif hasattr(grads, "shape"):  # MLX array
                total_norm_squared = total_norm_squared + mx.sum(grads * grads)

        compute_norm_recursive(self.accumulated_grads)
        grad_norm = mx.sqrt(total_norm_squared)

        # Clip if necessary
        clip_coeff = max_norm / (grad_norm + 1e-6)
        if clip_coeff < 1.0:
            # Apply clipping recursively
            def clip_recursive(grads):
                if isinstance(grads, dict):
                    return {k: clip_recursive(v) for k, v in grads.items()}
                elif hasattr(grads, "shape"):  # MLX array
                    return grads * clip_coeff
                else:
                    return grads

            self.accumulated_grads = clip_recursive(self.accumulated_grads)

    def force_evaluation_if_needed(self) -> None:
        """Force evaluation periodically to prevent memory buildup."""
        eval_freq = self.config.mlx_optimization.eval_frequency

        if self.steps_since_eval >= eval_freq:
            if self.accumulated_loss is not None:
                mx.eval(self.accumulated_loss)

            # Force evaluation of model parameters
            mx.eval(self.model.parameters())
            self.steps_since_eval = 0

    def evaluate(
        self,
        dataloader: DataLoaderProtocol,
        phase: str = "validation",
        max_batches: int | None = None,
    ) -> dict[str, float]:
        """Evaluate the model on the given dataset."""
        logger.info(f"Starting {phase} evaluation")

        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        num_batches = 0

        # Progress tracking
        eval_task_id = None
        if self.display_manager:
            eval_task_id = self.display_manager.create_progress_task(
                f"eval_{phase}", f"Evaluating {phase}", max_batches or len(dataloader)
            )

        try:
            for batch_idx, batch in enumerate(dataloader):
                if max_batches and batch_idx >= max_batches:
                    break

                # Forward pass (evaluation mode)
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch.get("labels"),
                )

                # Accumulate loss
                if "labels" in batch:
                    total_loss += float(outputs["loss"])
                    all_labels.extend(batch["labels"].squeeze().tolist())

                # Get predictions and probabilities
                logits = outputs["logits"]
                predictions = mx.argmax(logits, axis=-1)
                probabilities = mx.softmax(logits, axis=-1)

                # Force evaluation periodically
                if batch_idx % 10 == 0:
                    mx.eval(predictions, probabilities)

                all_predictions.extend(predictions.tolist())

                # Store probabilities for binary classification
                if probabilities.shape[-1] == 2:
                    all_probabilities.extend(probabilities[:, 1].tolist())

                num_batches += 1
                
                # Update progress
                if eval_task_id:
                    self.display_manager.update_progress_task(eval_task_id, advance=1)

        finally:
            # Clean up progress task
            if eval_task_id:
                self.display_manager.remove_progress_task(eval_task_id)

        # Force final evaluation
        if total_loss > 0:
            mx.eval(total_loss)

        # Compute metrics
        metrics = {}

        if all_labels:
            avg_loss = total_loss / num_batches
            accuracy = accuracy_score(all_labels, all_predictions)

            # Precision, recall, F1
            num_classes = len(set(all_labels))
            average_type = "binary" if num_classes == 2 else "weighted"

            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_predictions, average=average_type, zero_division=0
            )

            metrics = {
                f"{phase}_loss": avg_loss,
                f"{phase}_accuracy": accuracy,
                f"{phase}_precision": precision,
                f"{phase}_recall": recall,
                f"{phase}_f1": f1,
            }

            # AUC for binary classification
            if all_probabilities and num_classes == 2:
                try:
                    auc = roc_auc_score(all_labels, all_probabilities)
                    metrics[f"{phase}_auc"] = auc
                except Exception as e:
                    logger.warning(f"Could not compute AUC: {e}")

            logger.info(
                f"{phase.capitalize()} Results: "
                f"Loss={avg_loss:.4f}, Acc={accuracy:.4f}, F1={f1:.4f}"
            )

            # Log validation metrics to comprehensive monitoring
            if phase == "validation":
                improved = self.monitor.log_validation(
                    step=self.global_step,
                    val_loss=avg_loss,
                    val_accuracy=accuracy,
                    additional_metrics={
                        "val_precision": precision,
                        "val_recall": recall,
                        "val_f1": f1,
                    },
                )
                if improved:
                    logger.info(
                        f"New best validation metrics at step {self.global_step}"
                    )

        return metrics

    def should_stop_early(self, current_metric: float) -> bool:
        """Check if training should stop early based on validation metric."""
        if not self.config.evaluation.enable_early_stopping:
            return False

        patience = self.config.evaluation.early_stopping_patience
        threshold = self.config.evaluation.early_stopping_threshold
        mode = self.config.evaluation.early_stopping_mode

        # Check improvement
        if mode == "max":
            improved = current_metric > (self.best_metric + threshold)
        else:  # mode == "min"
            improved = current_metric < (self.best_metric - threshold)

        if improved:
            self.best_metric = current_metric
            self.best_metric_step = self.global_step
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        should_stop = self.patience_counter >= patience

        if should_stop:
            logger.info(
                f"Early stopping triggered: "
                f"No improvement for {self.patience_counter} evaluations"
            )

        return should_stop

    def train(
        self,
        train_loader: DataLoaderProtocol,
        val_loader: DataLoaderProtocol | None = None,
        test_loader: DataLoaderProtocol | None = None,
        resume_from_checkpoint: str | None = None,
    ) -> dict[str, Any]:
        """Main training loop with comprehensive features."""

        # Setup experiment tracking
        experiment_name = self.config.experiment_name or "mlx_training"

        with ExperimentLogger(experiment_name, self.config.to_dict()):
            # Resume from checkpoint if specified
            if resume_from_checkpoint:
                self._load_checkpoint(resume_from_checkpoint)

            # Update config based on dataset
            if hasattr(train_loader, "dataset_spec"):
                self.config.update_from_dataset(
                    {"dataset_spec": train_loader.dataset_spec.__dict__}
                )

            # Calculate total steps
            steps_per_epoch = len(train_loader)
            self.total_steps = self.config.get_total_steps(
                len(train_loader) * self.config.batch_size
            )

            logger.info(
                f"Training Configuration:\n"
                f"  Steps per epoch: {steps_per_epoch}\n"
                f"  Total steps: {self.total_steps}\n"
                f"  Effective batch size: {self.effective_batch_size}\n"
                f"  Warmup steps: {self.config.warmup_steps}\n"
                f"  Checkpoint mode: {'Best model only' if self.config.checkpoint.save_best_only else 'Regular + best model'}"
            )

            # Start comprehensive monitoring
            self.monitor.start_training(self.config.epochs, steps_per_epoch)
            
            # Log training parameters to MLflow if enabled
            if self.config.monitoring.enable_mlflow:
                try:
                    mlflow.log_params({
                        "model_name": self.model.__class__.__name__,
                        "batch_size": self.config.batch_size,
                        "effective_batch_size": self.effective_batch_size,
                        "learning_rate": self.config.learning_rate,
                        "optimizer": self.config.optimizer.value,
                        "epochs": self.config.epochs,
                        "warmup_steps": self.config.warmup_steps,
                        "total_steps": self.total_steps,
                        "gradient_accumulation_steps": self.config.mlx_optimization.gradient_accumulation_steps,
                        "max_grad_norm": self.config.mlx_optimization.max_grad_norm,
                        "weight_decay": self.config.advanced.weight_decay,
                        "label_smoothing": self.config.advanced.label_smoothing,
                        "dynamic_batch_sizing": self.config.memory.dynamic_batch_sizing,
                        "apple_silicon": self.memory_manager.is_apple_silicon,
                    })
                    logger.info("Training parameters logged to MLflow")
                except Exception as e:
                    logger.warning(f"Failed to log parameters to MLflow: {e}")

            # Training history
            history = {
                "train_loss": [],
                "validation_metrics": [],
                "learning_rates": [],
                "memory_usage": [],
                "batch_sizes": [],
            }

            start_time = time.time()

            try:
                # Main training loop
                for epoch in range(self.current_epoch, self.config.epochs):
                    self.current_epoch = epoch
                    epoch_start_time = time.time()
                    
                    # Reset epoch progress in monitoring system
                    self.monitor.reset_epoch_progress(epoch, self.config.epochs)

                    # Dynamic batch size adjustment
                    if self.config.memory.dynamic_batch_sizing:
                        self.adjust_batch_size_dynamically()

                    epoch_metrics = self._train_epoch(train_loader, val_loader, history)
                    
                    # Advance epoch progress in monitoring system
                    self.monitor.advance_epoch_progress()

                    # Log epoch summary
                    epoch_time = time.time() - epoch_start_time
                    logger.info(
                        f"Epoch {epoch + 1}/{self.config.epochs} completed: "
                        f"Loss={epoch_metrics['avg_loss']:.4f}, "
                        f"Time={epoch_time:.1f}s"
                    )

                    # Early stopping check
                    if (
                        val_loader
                        and epoch_metrics.get("val_metric")
                        and self.should_stop_early(epoch_metrics["val_metric"])
                    ):
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                        break

                # Final evaluation
                final_metrics = self._final_evaluation(val_loader, test_loader)
                
                # Run MLflow evaluation if enabled
                mlflow_eval_results = {}
                if self.config.monitoring.enable_mlflow and val_loader:
                    try:
                        mlflow_eval_results = self.run_mlflow_evaluation(
                            val_loader, phase="final_validation"
                        )
                        logger.info("MLflow evaluation completed successfully")
                    except Exception as e:
                        logger.warning(f"MLflow evaluation failed: {e}")

                # Training summary
                total_time = time.time() - start_time

                results = {
                    "best_metric": self.best_metric,
                    "best_step": self.best_metric_step,
                    "final_metrics": final_metrics,
                    "mlflow_evaluation": mlflow_eval_results,
                    "total_time": total_time,
                    "training_history": history,
                }

                # Save final checkpoint (only if not save_best_only or no best model saved)
                final_checkpoint = None
                if not self.config.checkpoint.save_best_only or self.best_metric_step == 0:
                    self._save_checkpoint("final")
                    final_checkpoint = Path(self.config.checkpoint.checkpoint_dir) / "final"
                else:
                    # Use best model as final checkpoint when save_best_only is enabled
                    final_checkpoint = Path(self.config.checkpoint.checkpoint_dir) / "best_model"

                # Save monitoring artifacts
                self.monitor.save_checkpoint_artifacts(
                    final_checkpoint,
                    {
                        "training_history": str(
                            Path(self.config.output_dir) / "training_history.json"
                        )
                    },
                )
                
                # Log model to MLflow if enabled
                if self.config.monitoring.enable_mlflow:
                    try:
                        self._log_model_to_mlflow(final_checkpoint)
                        logger.info("Model logged to MLflow successfully")
                    except Exception as e:
                        logger.warning(f"Failed to log model to MLflow: {e}")
                
                # Register model if MLflow is enabled
                if self.config.monitoring.enable_mlflow and hasattr(self.monitor, "get_current_run_id"):
                    try:
                        run_id = self.monitor.get_current_run_id()
                        if run_id:
                            model_name = f"{self.config.experiment_name or 'mlx'}_model"
                            model_version = self._register_model(
                                run_id=run_id,
                                model_name=model_name,
                                metrics=final_metrics,
                                stage="Production" if self.best_metric > 0.9 else "Staging",
                            )
                            logger.info(
                                f"Model registered as {model_name} version {model_version.version}"
                            )
                            results["registered_model"] = {
                                "name": model_name,
                                "version": model_version.version,
                                "stage": model_version.current_stage,
                            }
                    except Exception as e:
                        logger.warning(f"Model registration failed: {e}")

                # End monitoring with success
                training_summary = self.monitor.end_training("FINISHED")
                results.update({"monitoring_summary": training_summary})

            except Exception as e:
                logger.error(f"Training failed: {e}")
                self.monitor.end_training("FAILED")
                raise
            except KeyboardInterrupt:
                logger.info("Training interrupted by user")
                self.monitor.end_training("KILLED")
                raise

            # Save advanced profiling reports
            self._save_advanced_reports()

            logger.info(
                f"Training completed in {total_time:.1f}s\n"
                f"Best metric: {self.best_metric:.4f} at step {self.best_metric_step}"
            )

            return results

    def _train_epoch(
        self,
        train_loader: DataLoaderProtocol,
        val_loader: DataLoaderProtocol | None,
        history: dict[str, list],
    ) -> dict[str, Any]:
        """Train for one epoch."""

        epoch_loss = 0.0
        num_updates = 0
        step_times = []

        # Progress tracking is handled by the monitoring system
        # No need to create additional progress tasks here

        try:
            for _batch_idx, batch in enumerate(train_loader):
                step_start = time.time()

                self.global_step += 1

                # Training step
                loss, metrics = self.train_step(batch)

                # Force evaluation if needed
                self.force_evaluation_if_needed()

                # Track metrics for actual updates (not accumulation steps)
                if "loss" in metrics:
                    epoch_loss += metrics["loss"]
                    num_updates += 1

                # Progress updates are handled by the monitoring system
                # which gets called via monitor.log_step() below

                # Add to history only when we have actual metrics
                if "loss" in metrics:
                    history["learning_rates"].append(metrics["learning_rate"])
                    history["memory_usage"].append(metrics["memory_usage"])
                    history["batch_sizes"].append(metrics["batch_size"])

                # Track step timing
                step_time = time.time() - step_start
                step_times.append(step_time)

                # Validation evaluation
                if (
                    val_loader
                    and self.global_step % self.config.evaluation.eval_steps == 0
                ):
                    val_metrics = self.evaluate(val_loader, "validation")
                    history["validation_metrics"].append(val_metrics)

                    # Check for best model
                    primary_metric = self.config.evaluation.primary_metric
                    val_key = f"validation_{primary_metric}"

                    if val_key in val_metrics:
                        current_val_metric = val_metrics[val_key]

                        if current_val_metric > self.best_metric:
                            self.best_metric = current_val_metric
                            self.best_metric_step = self.global_step

                            if self.config.checkpoint.save_best_model:
                                self._save_checkpoint("best_model")
                                
                                # Log best model to MLflow if enabled
                                if self.config.monitoring.enable_mlflow:
                                    try:
                                        best_model_path = Path(self.config.checkpoint.checkpoint_dir) / "best_model"
                                        self._log_model_to_mlflow(best_model_path)
                                        logger.info(f"Best model logged to MLflow at step {self.global_step}")
                                    except Exception as e:
                                        logger.warning(f"Failed to log best model to MLflow: {e}")


                # Checkpoint saving (only if not save_best_only)
                if (
                    self.config.checkpoint.enable_checkpointing
                    and not self.config.checkpoint.save_best_only
                    and self.global_step % self.config.checkpoint.checkpoint_frequency
                    == 0
                ):
                    checkpoint_name = f"checkpoint-step-{self.global_step}"
                    self._save_checkpoint(checkpoint_name)
                    
                    # Log model to MLflow if enabled
                    if self.config.monitoring.enable_mlflow:
                        try:
                            checkpoint_path = Path(self.config.checkpoint.checkpoint_dir) / checkpoint_name
                            self._log_model_to_mlflow(checkpoint_path)
                            logger.info(f"Model logged to MLflow at step {self.global_step}")
                        except Exception as e:
                            logger.warning(f"Failed to log model to MLflow: {e}")

        finally:
            # Progress cleanup is handled by the monitoring system
            pass

        # Epoch metrics
        avg_loss = epoch_loss / max(1, num_updates)
        avg_step_time = np.mean(step_times) if step_times else 0
        throughput = self.dynamic_batch_size / avg_step_time if avg_step_time > 0 else 0

        history["train_loss"].append(avg_loss)

        epoch_metrics = {
            "avg_loss": avg_loss,
            "avg_step_time": avg_step_time,
            "throughput": throughput,
        }

        # Add validation metric if available
        if history["validation_metrics"]:
            latest_val = history["validation_metrics"][-1]
            primary_metric = self.config.evaluation.primary_metric
            val_key = f"validation_{primary_metric}"
            if val_key in latest_val:
                epoch_metrics["val_metric"] = latest_val[val_key]

        return epoch_metrics

    def _final_evaluation(
        self,
        val_loader: DataLoaderProtocol | None,
        test_loader: DataLoaderProtocol | None,
    ) -> dict[str, float]:
        """Run final evaluation on validation and test sets."""
        final_metrics = {}

        if val_loader:
            logger.info("Running final validation evaluation")
            val_metrics = self.evaluate(val_loader, "final_validation")
            final_metrics.update(val_metrics)

        if test_loader:
            logger.info("Running test evaluation")
            test_metrics = self.evaluate(test_loader, "test")
            final_metrics.update(test_metrics)

        return final_metrics
    
    def run_mlflow_evaluation(
        self,
        data_loader: DataLoaderProtocol,
        phase: str = "test",
        model_uri: str | None = None,
    ) -> dict[str, Any]:
        """Run comprehensive MLflow evaluation with custom metrics.
        
        Args:
            data_loader: Data loader for evaluation
            phase: Evaluation phase name
            model_uri: Optional model URI for evaluation
            
        Returns:
            Evaluation results dictionary
        """
        logger.info(f"Running MLflow evaluation for {phase}")
        
        # Collect all data for evaluation
        all_features = []
        all_labels = []
        all_predictions = []
        all_probabilities = []
        
        # MLX doesn't have eval() mode or no_grad context
        # Models are automatically in evaluation mode when not training
        for batch in data_loader:
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            
            if "labels" in batch:
                all_labels.extend(batch["labels"].squeeze().tolist())
            
            logits = outputs["logits"]
            predictions = mx.argmax(logits, axis=-1)
            probabilities = mx.softmax(logits, axis=-1)
            
            all_predictions.extend(predictions.tolist())
            if probabilities.shape[-1] == 2:
                all_probabilities.extend(probabilities[:, 1].tolist())
            
            # Store features for evaluation dataset
            # For BERT, we might want to store input_ids or other features
            batch_size = batch["input_ids"].shape[0]
            for i in range(batch_size):
                # Simple feature: sequence length
                seq_len = (batch["attention_mask"][i] == 1).sum().item()
                all_features.append([seq_len])
        
        # Create evaluation dataset
        eval_df = create_evaluation_dataset(
            X=np.array(all_features),
            y=np.array(all_labels),
            predictions=np.array(all_predictions),
            prediction_proba=np.array(all_probabilities) if all_probabilities else None,
            feature_names=["sequence_length"],
        )
        
        # Initialize custom metrics for Titanic
        custom_metrics = [
            TitanicMetrics.survival_rate_error(),
            TitanicMetrics.class_weighted_accuracy(),
            TitanicMetrics.false_negative_rate_survived(),
        ]
        
        # Run MLflow evaluation
        evaluator = ModelEvaluator(model_uri=model_uri)
        results = evaluator.evaluate_model(
            data=eval_df,
            model_type="classifier",
            custom_metrics=custom_metrics,
            plot_results=True,
        )
        
        logger.info(f"MLflow evaluation complete for {phase}")
        return results
    
    def _log_model_to_mlflow(self, checkpoint_path: Path) -> None:
        """Log the trained model to MLflow for registration.
        
        Args:
            checkpoint_path: Path to the saved model checkpoint
        """
        import mlflow
        
        try:
            # Log the raw model files as artifacts
            mlflow.log_artifacts(str(checkpoint_path), "model")
            
            # Log model configuration
            mlflow.log_dict(self.config.to_dict(), "model_config.json")
            
            # Tag the model for easier identification
            mlflow.set_tags({
                "model_type": "mlx_bert",
                "framework": "mlx",
                "task": "classification",
                "dataset": self.config.experiment_name or "unknown",
                "apple_silicon": str(self.memory_manager.is_apple_silicon),
                "training_stage": "final" if "final" in str(checkpoint_path) else "best",
            })
            
            logger.info(f"Model artifacts logged to MLflow from {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to log model artifacts to MLflow: {e}")
            raise

    def _register_model(
        self,
        run_id: str,
        model_name: str,
        metrics: dict[str, float],
        stage: str | None = None,
    ) -> Any:
        """Register model in MLflow Model Registry.
        
        Args:
            run_id: MLflow run ID
            model_name: Name for the registered model
            metrics: Model performance metrics
            stage: Optional stage to transition to
            
        Returns:
            ModelVersion object
        """
        return register_mlx_model(
            run_id=run_id,
            model_path="model",  # Standard MLflow model path
            model_name=model_name,
            config=self.config,
            metrics=metrics,
            stage=stage,
            await_registration=True,
        )


    def _save_checkpoint(self, checkpoint_name: str) -> None:
        """Save model checkpoint with comprehensive state."""
        if not self.config.checkpoint.enable_checkpointing:
            return

        checkpoint_dir = Path(self.config.checkpoint.checkpoint_dir)
        checkpoint_path = checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        try:
            # Save model weights
            if self.config.checkpoint.save_model_weights:
                self.model.save_pretrained(str(checkpoint_path))

            # Save optimizer state
            if self.config.checkpoint.save_optimizer_state:
                optimizer_path = checkpoint_path / "optimizer.safetensors"
                try:
                    from mlx.utils import tree_flatten

                    state_flat = dict(tree_flatten(self.optimizer.state))
                    mx.save_safetensors(str(optimizer_path), state_flat)
                except Exception as e:
                    logger.warning(f"Failed to save optimizer state: {e}")

            # Save trainer state
            trainer_state = {
                "global_step": self.global_step,
                "current_epoch": self.current_epoch,
                "best_metric": self.best_metric,
                "best_metric_step": self.best_metric_step,
                "dynamic_batch_size": self.dynamic_batch_size,
                "early_stopping_counter": self.early_stopping_counter,
                "patience_counter": self.patience_counter,
            }

            # Save random state if configured
            if self.config.checkpoint.save_random_state:
                trainer_state["random_state"] = {
                    "numpy_state": np.random.get_state(),
                }

            import json

            with open(checkpoint_path / "trainer_state.json", "w") as f:
                json.dump(trainer_state, f, indent=2, default=str)

            logger.info(f"Checkpoint saved: {checkpoint_path}")

            # Cleanup old checkpoints
            self._cleanup_old_checkpoints()

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def _load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load checkpoint and restore training state."""
        checkpoint_dir = Path(checkpoint_path)

        if not checkpoint_dir.exists():
            logger.error(f"Checkpoint not found: {checkpoint_dir}")
            return False

        try:
            # Load trainer state
            state_file = checkpoint_dir / "trainer_state.json"
            if state_file.exists():
                import json

                with open(state_file) as f:
                    trainer_state = json.load(f)

                self.global_step = trainer_state["global_step"]
                self.current_epoch = trainer_state["current_epoch"]
                self.best_metric = trainer_state["best_metric"]
                self.best_metric_step = trainer_state["best_metric_step"]
                self.dynamic_batch_size = trainer_state.get(
                    "dynamic_batch_size", self.config.batch_size
                )
                self.early_stopping_counter = trainer_state.get(
                    "early_stopping_counter", 0
                )
                self.patience_counter = trainer_state.get("patience_counter", 0)

                # Restore random state if available
                if "random_state" in trainer_state:
                    np.random.set_state(trainer_state["random_state"]["numpy_state"])

            # Load optimizer state
            optimizer_file = checkpoint_dir / "optimizer.safetensors"
            if optimizer_file.exists() and self.config.checkpoint.save_optimizer_state:
                try:
                    from mlx.utils import tree_unflatten

                    state_flat = mx.load(str(optimizer_file))
                    self.optimizer.state = tree_unflatten(list(state_flat.items()))
                except Exception as e:
                    logger.warning(f"Failed to load optimizer state: {e}")

            logger.info(f"Checkpoint loaded from: {checkpoint_dir}")
            return True

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints beyond the maximum limit."""
        if self.config.checkpoint.max_checkpoints_to_keep <= 0:
            return

        checkpoint_dir = Path(self.config.checkpoint.checkpoint_dir)

        # Find all step-based checkpoints
        checkpoints = []
        for path in checkpoint_dir.iterdir():
            if path.is_dir() and path.name.startswith("checkpoint-step-"):
                try:
                    step_num = int(path.name.split("-")[-1])
                    checkpoints.append((step_num, path))
                except (ValueError, IndexError):
                    continue

        # Sort by step number and keep only the latest N
        checkpoints.sort(key=lambda x: x[0], reverse=True)

        for _, old_checkpoint in checkpoints[
            self.config.checkpoint.max_checkpoints_to_keep :
        ]:
            try:
                import shutil

                shutil.rmtree(old_checkpoint)
                logger.info(f"Removed old checkpoint: {old_checkpoint}")
            except Exception as e:
                logger.warning(f"Failed to remove old checkpoint {old_checkpoint}: {e}")

    def _apply_apple_silicon_optimizations(self) -> None:
        """Apply Apple Silicon specific optimizations."""
        if self.memory_manager.is_apple_silicon:
            logger.info("Applying Apple Silicon optimizations...")

            # Apply memory optimizations
            optimizations = self.memory_manager.optimize_for_apple_silicon()

            if optimizations.get("unified_memory"):
                logger.debug("Unified memory optimization applied")
            if optimizations.get("neural_engine"):
                logger.debug("Neural Engine optimization applied")
            if optimizations.get("cache_optimization"):
                logger.debug("Cache optimization applied")

            # Log memory recommendations
            recommendations = self.memory_manager.get_memory_recommendations()
            if recommendations["actions"]:
                logger.info(
                    f"Memory recommendations: {', '.join(recommendations['actions'])}"
                )

    def _save_advanced_reports(self) -> None:
        """Save advanced memory and performance reports."""
        output_dir = Path(self.config.output_dir)

        try:
            # Save memory report
            memory_report_path = output_dir / "memory_report.json"
            self.memory_manager.save_memory_report(memory_report_path)

            # Save performance report
            performance_report_path = output_dir / "performance_report.json"
            self.profiler.save_performance_report(performance_report_path)

            # Save performance summary
            performance_summary = self.profiler.get_performance_summary()
            summary_path = output_dir / "performance_summary.json"

            import json

            with open(summary_path, "w") as f:
                json.dump(performance_summary, f, indent=2, default=str)

            logger.info("Advanced profiling reports saved successfully")

        except Exception as e:
            logger.warning(f"Failed to save advanced reports: {e}")

    def get_system_status(self) -> dict[str, Any]:
        """Get comprehensive system status for monitoring."""
        return {
            "memory": self.memory_manager.get_current_metrics().__dict__,
            "performance": self.profiler.get_performance_summary(),
            "training_state": {
                "global_step": self.global_step,
                "current_epoch": self.current_epoch,
                "best_metric": self.best_metric,
                "dynamic_batch_size": self.dynamic_batch_size,
            },
            "recommendations": self.memory_manager.get_memory_recommendations(),
        }
