"""
Base trainer implementation with MLX optimizations for Apple Silicon.
"""

import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn
from loguru import logger

from .protocols import (
    Model, DataLoader, Trainer, TrainerConfig,
    TrainingState, TrainingResult, TrainingHook
)
from .config import BaseTrainerConfig
from .optimization import (
    create_optimizer, create_lr_scheduler, GradientAccumulator,
    clip_gradients, compute_gradient_stats
)
from .state import TrainingStateManager, CheckpointManager


class BaseTrainer:
    """
    Base trainer implementation with MLX optimizations.
    
    This trainer provides:
    - Efficient MLX-based training loop
    - Gradient accumulation
    - Mixed precision training (automatic in MLX)
    - Checkpoint management
    - Hook/callback system
    - Comprehensive logging
    """
    
    def __init__(
        self,
        model: Model,
        config: BaseTrainerConfig,
        callbacks: Optional[List[TrainingHook]] = None,
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Model to train
            config: Training configuration
            callbacks: Optional list of training callbacks
        """
        self._model = model  # Set internal attributes directly
        self._config = config
        self.callbacks = callbacks or []
        
        # Create output directory
        self.config.environment.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_path = self.config.environment.output_dir / "trainer_config.yaml"
        self.config.save(config_path)
        
        # Initialize components
        self.optimizer = create_optimizer(self.model, self.config.optimizer)
        self.lr_scheduler = create_lr_scheduler(self.optimizer, self.config.scheduler)
        self.gradient_accumulator = GradientAccumulator(self.config.training.gradient_accumulation_steps)
        
        # Initialize state management
        self._state = TrainingState()  # Set the internal attribute directly
        self.state_manager = TrainingStateManager(self.config.environment.output_dir)
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.config.environment.output_dir / "checkpoints",
            save_total_limit=self.config.training.save_total_limit,
        )
        
        # Training function
        self._train_step = self._create_train_step()
        self._eval_step = self._create_eval_step()
        
        # Initialize best metric tracking
        self._best_metric_value = float('inf') if self.config.training.best_metric_mode == "min" else float('-inf')
        self._is_better = self._create_metric_comparator()
        
        logger.info(f"Initialized BaseTrainer with config: {self.config.training}")
    
    def _create_metric_comparator(self) -> Callable[[float, float], bool]:
        """Create function to compare metrics based on mode."""
        if self.config.training.best_metric_mode == "min":
            return lambda new, best: new < best
        else:
            return lambda new, best: new > best
    
    def _create_train_step(self) -> Callable:
        """Create the training step function."""
        def loss_fn(model, batch):
            logger.debug("train_step: Entering loss_fn")
            # Forward pass - handle different model calling conventions
            # Remove metadata if present as it's not needed for model forward
            model_inputs = {k: v for k, v in batch.items() 
                          if k not in ['metadata'] and v is not None}
            
            logger.debug(f"train_step: Model inputs keys: {list(model_inputs.keys())}")
            
            try:
                # Try unpacked arguments first (for BERT models)
                logger.debug("train_step: Attempting model forward pass with unpacked arguments")
                outputs = model(**model_inputs)
            except TypeError:
                # Fall back to batch dictionary (for simple test models)
                logger.debug("train_step: Falling back to batch dictionary for model forward")
                outputs = model(batch)
            
            logger.debug(f"train_step: Model outputs keys: {list(outputs.keys()) if isinstance(outputs, dict) else 'not a dict'}")
            
            # Extract loss (assuming model returns dict with 'loss' key)
            loss = outputs.get("loss")
            if loss is None:
                raise ValueError("Model must return a dictionary with 'loss' key")
            
            logger.debug(f"train_step: Loss shape: {loss.shape if hasattr(loss, 'shape') else 'scalar'}")
            
            # Apply label smoothing if configured
            if self.config.training.label_smoothing > 0:
                # This is a simplified version - actual implementation depends on task
                loss = loss * (1 - self.config.training.label_smoothing)
            
            logger.debug("train_step: Exiting loss_fn")
            return loss, outputs
        
        # Create value and grad function
        value_and_grad_fn = mx.value_and_grad(loss_fn)
        
        def train_step(batch: Dict[str, mx.array]) -> Tuple[float, Dict[str, mx.array]]:
            """Single training step."""
            logger.debug("train_step: Starting training step")
            
            logger.debug("train_step: Computing loss and gradients")
            (loss, outputs), grads = value_and_grad_fn(self.model, batch)
            logger.debug("train_step: Loss and gradients computed")
            
            # Gradient clipping
            logger.debug("train_step: Starting gradient clipping")
            if self.config.optimizer.max_grad_norm > 0:
                grads, grad_norm = clip_gradients(grads, self.config.optimizer.max_grad_norm)
                logger.debug(f"train_step: Gradient norm after clipping: {grad_norm}")
            else:
                # Skip detailed gradient stats computation during training for performance
                grad_norm = 0.0  # Placeholder value
                logger.debug("train_step: Skipping gradient norm computation (no clipping)")
            
            # Accumulate gradients
            logger.debug("train_step: Accumulating gradients")
            should_update = self.gradient_accumulator.accumulate(grads)
            logger.debug(f"train_step: Should update: {should_update}")
            
            if should_update:
                # Get accumulated gradients
                logger.debug("train_step: Getting accumulated gradients")
                accumulated_grads = self.gradient_accumulator.get_gradients()
                
                # Update model
                logger.debug("train_step: Updating model parameters")
                self.optimizer.update(self.model, accumulated_grads)
                logger.debug("train_step: Model parameters updated")
                
                # Update learning rate
                if self.lr_scheduler is not None:
                    current_lr = self.lr_scheduler.step()
                else:
                    current_lr = self.optimizer.learning_rate
            else:
                current_lr = self.optimizer.learning_rate
            
            # Ensure computation is executed
            logger.debug("train_step: Calling mx.eval to synchronize")
            mx.eval(loss, self.model.parameters())
            logger.debug("train_step: mx.eval completed")
            
            # Only convert scalar values to Python scalars
            logger.debug("train_step: Converting loss to Python scalar")
            loss_value = loss.item()
            logger.debug(f"train_step: Loss value: {loss_value}")
            
            metrics = {
                "grad_norm": grad_norm,
                "learning_rate": current_lr,
            }
            
            # Add other outputs, converting scalars only
            for k, v in outputs.items():
                if k != "loss":
                    if hasattr(v, 'item') and v.size == 1:
                        metrics[k] = v.item()
                    elif not hasattr(v, 'shape'):  # Already a Python scalar
                        metrics[k] = v
                    # Skip tensors with multiple elements
            
            logger.debug("train_step: Training step completed successfully")
            return loss_value, metrics
        
        return train_step
    
    def _create_eval_step(self) -> Callable:
        """Create the evaluation step function."""
        def eval_step(batch: Dict[str, mx.array]) -> Tuple[float, Dict[str, mx.array]]:
            """Single evaluation step."""
            # Forward pass (no gradients) - handle different model calling conventions
            # Remove metadata if present as it's not needed for model forward
            model_inputs = {k: v for k, v in batch.items() 
                          if k not in ['metadata'] and v is not None}
            
            try:
                # Try unpacked arguments first (for BERT models)
                outputs = self.model(**model_inputs)
            except TypeError:
                # Fall back to batch dictionary (for simple test models)
                outputs = self.model(batch)
            
            # Extract loss
            loss = outputs.get("loss")
            if loss is None:
                raise ValueError("Model must return a dictionary with 'loss' key")
            
            # Ensure computation is executed
            mx.eval(loss)
            
            # Only convert scalar values to Python scalars
            metrics = {}
            for k, v in outputs.items():
                if k != "loss":
                    if hasattr(v, 'item') and v.size == 1:
                        metrics[k] = v.item()
                    elif not hasattr(v, 'shape'):  # Already a Python scalar
                        metrics[k] = v
                    # Skip tensors with multiple elements
            
            return loss.item(), metrics
        
        return eval_step
    
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        resume_from: Optional[Path] = None,
    ) -> TrainingResult:
        """
        Run the training loop.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Optional validation data loader
            resume_from: Optional checkpoint path to resume from
            
        Returns:
            TrainingResult with final metrics and paths
        """
        # Resume from checkpoint if specified
        if resume_from:
            self._load_checkpoint(resume_from)
            logger.info(f"Resumed from checkpoint: {resume_from}")
        
        # Calculate total steps
        steps_per_epoch = len(train_dataloader)
        total_steps = steps_per_epoch * self.config.training.num_epochs
        
        # Update scheduler config if needed
        if self.config.scheduler.num_training_steps is None:
            self.config.scheduler.num_training_steps = total_steps
            if self.lr_scheduler:
                self.lr_scheduler.config.num_training_steps = total_steps
        
        # Initialize training
        self.state.training_start_time = time.time()
        self._call_hooks("on_train_begin", self.state)
        
        logger.info(f"Starting training for {self.config.training.num_epochs} epochs")
        logger.info(f"Total steps: {total_steps}, Steps per epoch: {steps_per_epoch}")
        
        # Training loop
        for epoch in range(self.state.epoch, self.config.training.num_epochs):
            self.state.epoch = epoch
            self.state.epoch_start_time = time.time()
            
            # Train epoch
            train_metrics = self._train_epoch(train_dataloader, epoch)
            self.state.train_loss = train_metrics["loss"]
            self.state.train_history.append(train_metrics)
            
            # Evaluate if needed
            should_evaluate = (
                val_dataloader is not None and
                (self.config.training.eval_strategy == "epoch" or
                 (epoch + 1) == self.config.training.num_epochs)
            )
            
            if should_evaluate:
                val_metrics = self.evaluate(val_dataloader)
                self.state.val_loss = val_metrics.get("eval_loss", val_metrics.get("loss", 0.0))
                self.state.val_history.append(val_metrics)
                self.state.metrics.update(val_metrics)
                
                # Check for best model
                metric_name = self.config.training.best_metric
                if metric_name in val_metrics:
                    metric_value = val_metrics[metric_name]
                    if self._is_better(metric_value, self._best_metric_value):
                        self._best_metric_value = metric_value
                        self.state.best_val_metric = metric_value
                        self.state.improvement_streak += 1
                        self.state.no_improvement_count = 0
                        
                        # Save best model
                        if self.config.training.save_strategy in ["best", "all"]:
                            best_path = self._save_checkpoint(is_best=True)
                            logger.info(f"New best model saved: {best_path}")
                    else:
                        self.state.improvement_streak = 0
                        self.state.no_improvement_count += 1
                
                # Check early stopping
                if self.config.training.early_stopping:
                    if self.state.no_improvement_count >= self.config.training.early_stopping_patience:
                        self.state.should_stop = True
                        logger.info(f"Early stopping triggered after {self.state.no_improvement_count} epochs without improvement")
            
            # Save checkpoint if needed
            should_save = (
                self.config.training.save_strategy == "epoch" or
                (self.config.training.save_strategy == "all")
            )
            if should_save and not self.config.training.save_best_only:
                self._save_checkpoint(is_best=False)
            
            # Call epoch end hooks
            self._call_hooks("on_epoch_end", self.state)
            
            # Check if should stop
            if self.state.should_stop:
                break
        
        # Training complete
        training_time = time.time() - self.state.training_start_time
        
        # Save final model
        final_path = self._save_checkpoint(is_best=False, is_final=True)
        
        # Create result
        result = TrainingResult(
            final_train_loss=self.state.train_loss,
            final_val_loss=self.state.val_loss,
            best_val_loss=self.state.best_val_loss,
            best_val_metric=self.state.best_val_metric,
            final_metrics=self.state.metrics,
            train_history=self.state.train_history,
            val_history=self.state.val_history,
            final_model_path=final_path,
            best_model_path=self.checkpoint_manager.get_best_checkpoint(),
            total_epochs=self.state.epoch + 1,
            total_steps=self.state.global_step,
            total_time=training_time,
            early_stopped=self.state.should_stop and self.config.training.early_stopping,
            stop_reason="early_stopping" if self.state.should_stop else "completed",
        )
        
        # Call training end hooks
        self._call_hooks("on_train_end", self.state, result)
        
        # Save final result
        result_path = self.config.environment.output_dir / "training_result.json"
        import json
        
        # Convert result to dict and ensure all values are JSON serializable
        def make_json_serializable(obj):
            """Convert MLX arrays and other non-serializable objects to JSON-serializable format."""
            import mlx.core as mx
            
            # Check if it's an MLX array directly
            if isinstance(obj, mx.array):
                try:
                    return obj.item() if obj.size == 1 else obj.tolist()
                except:
                    return float(obj)
            # Check if it's an MLX array by module
            elif hasattr(obj, '__module__') and obj.__module__ and 'mlx' in obj.__module__:
                if hasattr(obj, 'item') and hasattr(obj, 'size'):
                    try:
                        return obj.item() if obj.size == 1 else obj.tolist()
                    except:
                        return float(obj)
                else:
                    # For other MLX objects, convert to string
                    return str(obj)
            elif hasattr(obj, 'item') and hasattr(obj, 'size'):
                # Handle numpy arrays or similar
                try:
                    return obj.item() if obj.size == 1 else obj.tolist()
                except:
                    return float(obj)
            elif isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_serializable(v) for v in obj]
            elif isinstance(obj, tuple):
                return [make_json_serializable(v) for v in obj]
            elif isinstance(obj, Path):
                return str(obj)
            else:
                return obj
        
        result_dict = make_json_serializable(result.to_dict())
        
        with open(result_path, "w") as f:
            json.dump(result_dict, f, indent=2)
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Final model saved to: {final_path}")
        
        return result
    
    def _train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self._call_hooks("on_epoch_begin", self.state)
        
        # Initialize metrics
        epoch_loss = 0.0
        epoch_metrics = {}
        num_batches = 0
        
        # Get total batches for progress tracking
        total_batches = len(dataloader)
        
        for batch_idx, batch in enumerate(dataloader):
            logger.debug(f"_train_epoch: Processing batch {batch_idx}")
            # Manual progress tracking
            if batch_idx % max(1, total_batches // 10) == 0 or batch_idx == 0:
                progress_pct = (batch_idx / total_batches) * 100
                print(f"Epoch {epoch} - Batch {batch_idx}/{total_batches} ({progress_pct:.1f}%)")
            
            self.state.global_step += 1
            self.state.samples_seen += self.config.data.batch_size
            
            # Call batch begin hooks
            logger.debug("_train_epoch: Calling batch begin hooks")
            self._call_hooks("on_batch_begin", self.state, batch)
            
            # Training step
            logger.debug("_train_epoch: Calling _train_step")
            loss, metrics = self._train_step(batch)
            logger.debug(f"_train_epoch: _train_step completed with loss: {loss}")
            
            # Update metrics
            epoch_loss += loss
            for k, v in metrics.items():
                if k not in epoch_metrics:
                    epoch_metrics[k] = 0.0
                epoch_metrics[k] += v
            num_batches += 1
            
            # Call batch end hooks
            logger.debug("_train_epoch: Calling batch end hooks")
            self._call_hooks("on_batch_end", self.state, loss)
            
            # Log batch metrics occasionally
            if batch_idx % self.config.training.logging_steps == 0:
                logger.info(f"Step {self.state.global_step} - Loss: {loss:.4f}, LR: {metrics.get('learning_rate', 0):.2e}")
            
            # Evaluate during training if needed
            if (self.config.training.eval_strategy == "steps" and 
                self.state.global_step % self.config.training.eval_steps == 0):
                if hasattr(self, '_val_dataloader') and self._val_dataloader is not None:
                    val_metrics = self.evaluate(self._val_dataloader)
                    self.state.val_loss = val_metrics.get("eval_loss", val_metrics.get("loss", 0.0))
                    self.state.metrics.update(val_metrics)
            
            # Save checkpoint if needed
            if (self.config.training.save_strategy == "steps" and
                self.state.global_step % self.config.training.save_steps == 0):
                self._save_checkpoint(is_best=False)
        
        # Average metrics
        avg_metrics = {
            "loss": epoch_loss / num_batches,
            **{k: v / num_batches for k, v in epoch_metrics.items()}
        }
        
        return avg_metrics
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on a dataset.
        
        Args:
            dataloader: Data loader for evaluation
            
        Returns:
            Dictionary of metrics
        """
        self._call_hooks("on_evaluate_begin", self.state)
        
        # Initialize metrics
        total_loss = 0.0
        total_metrics = {}
        num_batches = 0
        
        # Evaluation loop
        total_batches = len(dataloader)
        
        for batch_idx, batch in enumerate(dataloader):
            # Progress tracking
            if batch_idx % max(1, total_batches // 10) == 0 or batch_idx == 0:
                progress_pct = (batch_idx / total_batches) * 100
                print(f"Evaluating - Batch {batch_idx}/{total_batches} ({progress_pct:.1f}%)")
            # Evaluation step
            loss, metrics = self._eval_step(batch)
            
            # Update metrics
            total_loss += loss
            for k, v in metrics.items():
                if k not in total_metrics:
                    total_metrics[k] = 0.0
                total_metrics[k] += v
            num_batches += 1
            
            # Progress tracking is handled above, no need for pbar update
        
        # Average metrics
        avg_metrics = {
            "loss": total_loss / num_batches,
            **{k: v / num_batches for k, v in total_metrics.items()}
        }
        
        # Prefix with eval_
        eval_metrics = {f"eval_{k}": v for k, v in avg_metrics.items()}
        
        self._call_hooks("on_evaluate_end", self.state, eval_metrics)
        
        return eval_metrics
    
    def predict(self, dataloader: DataLoader) -> mx.array:
        """
        Generate predictions for a dataset.
        
        Args:
            dataloader: Data loader for prediction
            
        Returns:
            Predictions as MLX array
        """
        predictions = []
        
        # Avoid tqdm due to MLX threading issues
        # pbar = tqdm(dataloader, desc="Predicting", leave=False, dynamic_ncols=True)
        pbar = dataloader
        
        for batch in pbar:
            # Forward pass - handle different model calling conventions
            # Remove metadata if present as it's not needed for model forward
            model_inputs = {k: v for k, v in batch.items() 
                          if k not in ['metadata'] and v is not None}
            
            try:
                # Try unpacked arguments first (for BERT models)
                outputs = self.model(**model_inputs)
            except TypeError:
                # Fall back to batch dictionary (for simple test models)
                outputs = self.model(batch)
            
            # Extract predictions (assuming 'logits' key)
            if "logits" in outputs:
                preds = outputs["logits"]
            elif "predictions" in outputs:
                preds = outputs["predictions"]
            else:
                raise ValueError("Model must return 'logits' or 'predictions' in output dict")
            
            predictions.append(preds)
        
        # Concatenate all predictions
        return mx.concatenate(predictions, axis=0)
    
    def _save_checkpoint(self, is_best: bool = False, is_final: bool = False) -> Path:
        """Save training checkpoint."""
        # Determine checkpoint name
        if is_final:
            name = "final"
        elif is_best:
            name = "best"
        else:
            name = f"checkpoint-{self.state.global_step}"
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            state=self.state,
            metrics=self.state.metrics,
            is_best=is_best,
            name=name,
        )
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        return checkpoint_path
    
    def _load_checkpoint(self, path: Path) -> None:
        """Load training checkpoint."""
        self.state = self.checkpoint_manager.load_checkpoint(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
        )
        
        # Update scheduler state if exists
        if self.lr_scheduler is not None and hasattr(self.state, "scheduler_state"):
            self.lr_scheduler.load_state_dict(self.state.scheduler_state)
    
    def _call_hooks(self, method: str, *args, **kwargs) -> None:
        """Call all registered hooks."""
        for callback in self.callbacks:
            if hasattr(callback, method):
                getattr(callback, method)(self, *args, **kwargs)
    
    def save_checkpoint(self, path: Path) -> None:
        """Public method to save checkpoint."""
        self._save_checkpoint(is_best=False, is_final=False)
    
    def load_checkpoint(self, path: Path) -> None:
        """Public method to load checkpoint."""
        self._load_checkpoint(path)
    
    @property
    def model(self) -> Model:
        """Get the model being trained."""
        return self._model
    
    @model.setter
    def model(self, value: Model):
        """Set the model."""
        self._model = value
    
    @property
    def config(self) -> BaseTrainerConfig:
        """Get trainer configuration."""
        return self._config
    
    @config.setter
    def config(self, value: BaseTrainerConfig):
        """Set trainer configuration."""
        # Validate configuration
        errors = value.validate()
        if errors:
            raise ValueError(f"Invalid configuration: {errors}")
        self._config = value
    
    @property
    def state(self) -> TrainingState:
        """Get current training state."""
        return self._state
    
    @state.setter
    def state(self, value: TrainingState):
        """Set training state."""
        self._state = value