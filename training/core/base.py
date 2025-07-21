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
import sys

# Configure loguru to not buffer output
logger.remove()
logger.add(sys.stderr, level="INFO")

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
from .memory_pool import create_memory_pools, ArrayPool, GradientPool


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
        # Optimizer will be created later in train() when we know the total steps
        self.optimizer = None
        
        # LR scheduler will be initialized in fit() when we know the total steps
        self.lr_scheduler = None
        self._lr_schedule_fn = None  # MLX native schedule function
        
        # Initialize memory pools if enabled
        memory_pool_config = self.config.custom.get("memory_pool", {})
        self.array_pool, self.gradient_pool = create_memory_pools(
            self.model, memory_pool_config
        )
        
        self.gradient_accumulator = GradientAccumulator(
            self.config.training.gradient_accumulation_steps
        )
        
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
        
        # Setup compilation if enabled
        self._setup_compilation()
        
        # Initialize best metric tracking
        self._best_metric_value = float('inf') if self.config.training.best_metric_mode == "min" else float('-inf')
        self._is_better = self._create_metric_comparator()
        
        logger.info(f"Initialized BaseTrainer with config: {self.config.training}")
    
    def _setup_compilation(self):
        """Setup MLX compilation for training if available and beneficial."""
        self._use_compiled = False
        self._compiled_train_step = self._train_step
        self._compiled_eval_step = self._eval_step
        self._compilation_state = None
        
        # Check if compilation is available and should be used
        try:
            from .compiled import (
                create_compiled_train_step,
                create_compiled_eval_step,
                should_compile_model
            )
            
            # Check if we should compile this model
            if not should_compile_model(self.model, self.config):
                logger.info("Model/config not suitable for compilation")
                return
            
            # Check if explicitly disabled
            if hasattr(self.config.training, 'use_compilation') and not self.config.training.use_compilation:
                logger.info("Compilation disabled by config")
                return
            
            logger.info("Setting up MLX compilation for training")
            
            # We'll create compiled functions after optimizer is initialized
            # Store the flag to compile later
            self._should_compile = True
            
        except ImportError:
            logger.info("MLX compilation module not available")
        except Exception as e:
            logger.warning(f"Failed to setup compilation: {e}")
    
    def _create_metric_comparator(self) -> Callable[[float, float], bool]:
        """Create function to compare metrics based on mode."""
        if self.config.training.best_metric_mode == "min":
            return lambda new, best: new < best
        else:
            return lambda new, best: new > best
    
    def _create_train_step(self) -> Callable:
        """Create the training step function."""
        def loss_fn(model, batch):
            # Forward pass - handle different model calling conventions
            # Remove metadata if present as it's not needed for model forward
            model_inputs = {k: v for k, v in batch.items() 
                          if k not in ['metadata'] and v is not None}
            
            # Apply mixed precision if enabled
            if self.config.training.mixed_precision:
                # Cast float inputs to bfloat16 for computation
                model_inputs = {
                    k: v.astype(mx.bfloat16) if v.dtype == mx.float32 else v
                    for k, v in model_inputs.items()
                }
            
            try:
                # Try unpacked arguments first (for BERT models)
                outputs = model(**model_inputs)
            except TypeError:
                # Fall back to batch dictionary (for simple test models)
                outputs = model(batch)
            
            # Extract loss (assuming model returns dict with 'loss' key)
            loss = outputs.get("loss")
            if loss is None:
                raise ValueError("Model must return a dictionary with 'loss' key")
            
            # Apply label smoothing if configured
            if self.config.training.label_smoothing > 0:
                # This is a simplified version - actual implementation depends on task
                loss = loss * (1 - self.config.training.label_smoothing)
            return loss, outputs
        
        # Create value and grad function
        value_and_grad_fn = mx.value_and_grad(loss_fn)
        
        def train_step(batch: Dict[str, mx.array]) -> Tuple[float, Dict[str, mx.array]]:
            """Single training step - optimized for MLX lazy evaluation."""
            logger.debug("Starting train_step")
            (loss, outputs), grads = value_and_grad_fn(self.model, batch)
            logger.debug(f"Got loss and grads, loss shape: {loss.shape if hasattr(loss, 'shape') else 'scalar'}")
            
            # Critical: Force evaluation of gradients to prevent hanging
            # This ensures the computation graph is evaluated immediately
            # rather than building up a large lazy computation graph
            mx.eval(grads)
            
            # Gradient clipping (keep lazy)
            if self.config.optimizer.max_grad_norm > 0:
                grads, grad_norm = clip_gradients(grads, self.config.optimizer.max_grad_norm)
            else:
                grad_norm = None  # Don't compute if not needed
            
            # Accumulate gradients
            should_update = self.gradient_accumulator.accumulate(grads)
            logger.debug(f"Should update: {should_update}")
            
            if should_update:
                # Get accumulated gradients
                accumulated_grads = self.gradient_accumulator.get_gradients()
                
                # Update model
                logger.debug("Updating model with optimizer")
                self.optimizer.update(self.model, accumulated_grads)
                logger.debug("Model updated")
                
            # Get current learning rate - MLX schedules update automatically
            # Only convert to float when needed for logging
            current_lr = self.optimizer.learning_rate
            
            # Build metrics dict - keep arrays lazy
            metrics = {
                "learning_rate": current_lr,
                "loss": loss,  # Keep as MLX array
            }
            
            # Only add grad_norm if it was computed
            if grad_norm is not None:
                metrics["grad_norm"] = grad_norm
            
            # Add other outputs without conversion
            for k, v in outputs.items():
                if k != "loss":
                    metrics[k] = v  # Keep as MLX arrays
            
            return loss, metrics
        
        return train_step
    
    def _create_eval_step(self) -> Callable:
        """Create the evaluation step function."""
        def eval_step(batch: Dict[str, mx.array]) -> Tuple[mx.array, Dict[str, mx.array]]:
            """Single evaluation step - optimized for lazy evaluation."""
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
            
            # Extract loss - handle both dictionary and direct outputs
            if isinstance(outputs, dict) and "loss" in outputs:
                loss = outputs["loss"]
            elif isinstance(outputs, dict) and "logits" in outputs:
                # Model returned logits but no loss - compute loss manually for evaluation
                import mlx.nn as nn
                logits = outputs["logits"]
                if "labels" in model_inputs:
                    loss = nn.losses.cross_entropy(
                        logits, 
                        model_inputs["labels"], 
                        reduction="mean"
                    )
                else:
                    # No labels available for evaluation - use dummy loss
                    loss = mx.array(0.0)
            else:
                # Handle case where outputs is not a dict or missing expected keys
                if hasattr(outputs, "loss"):
                    loss = outputs.loss
                elif hasattr(outputs, "logits"):
                    # Compute loss if we have logits and labels
                    import mlx.nn as nn
                    if "labels" in model_inputs:
                        loss = nn.losses.cross_entropy(
                            outputs.logits, 
                            model_inputs["labels"], 
                            reduction="mean"
                        )
                    else:
                        loss = mx.array(0.0)
                else:
                    # Fallback: assume outputs is the loss value
                    loss = outputs if isinstance(outputs, mx.array) else mx.array(float(outputs))
            
            # Keep metrics as MLX arrays - no conversion
            if isinstance(outputs, dict):
                metrics = {k: v for k, v in outputs.items() if k != "loss"}
            else:
                # If outputs is not a dict, create minimal metrics
                metrics = {}
            
            # Return loss and metrics as MLX arrays
            return loss, metrics
        
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
        
        # Create learning rate schedule if needed
        learning_rate = self.config.optimizer.learning_rate
        if self.config.scheduler.type != "none":
            from .optimization import create_mlx_lr_schedule
            
            # Create MLX native schedule
            learning_rate = create_mlx_lr_schedule(
                config=self.config.scheduler,
                base_lr=self.config.optimizer.learning_rate,
                num_training_steps=total_steps
            )
            self._lr_schedule_fn = learning_rate if callable(learning_rate) else None
            
            logger.info(f"Initialized MLX learning rate scheduler with {total_steps} total steps")
        
        # Create optimizer with the learning rate (schedule or constant)
        from .optimization import create_optimizer
        original_lr = self.config.optimizer.learning_rate
        self.config.optimizer.learning_rate = learning_rate  # Temporarily set for optimizer creation
        self.optimizer = create_optimizer(self.model, self.config.optimizer)
        self.config.optimizer.learning_rate = original_lr  # Restore original value
        
        # Now create compiled functions if requested
        if hasattr(self, '_should_compile') and self._should_compile:
            try:
                from .compiled import create_compiled_train_step, create_compiled_eval_step
                
                # Create compiled training step
                self._compiled_train_step, self._compilation_state = create_compiled_train_step(
                    self.model,
                    self.optimizer,
                    self.config,
                    self.gradient_accumulator
                )
                
                # Create compiled eval step
                self._compiled_eval_step = create_compiled_eval_step(self.model)
                
                self._use_compiled = True
                logger.info("Successfully created compiled training functions")
                
            except Exception as e:
                logger.warning(f"Failed to create compiled functions: {e}")
                self._use_compiled = False
        
        # Initialize training
        self.state.training_start_time = time.time()
        self._call_hooks("on_train_begin", self.state)
        
        logger.info(f"Starting training for {self.config.training.num_epochs} epochs")
        logger.info(f"Total steps: {total_steps}, Steps per epoch: {steps_per_epoch}")
        
        # Ensure output is flushed
        # Note: Commented out due to hanging issue with loguru
        # import sys
        # logger.debug("About to flush stdout/stderr")
        # sys.stdout.flush()
        # sys.stderr.flush()
        # logger.debug("Flushed stdout/stderr")
        
        # Training loop
        logger.debug(f"About to start training loop for {self.config.training.num_epochs} epochs")
        for epoch in range(self.state.epoch, self.config.training.num_epochs):
            logger.debug(f"Starting epoch {epoch}")
            self.state.epoch = epoch
            self.state.epoch_start_time = time.time()
            
            # Train epoch
            logger.debug(f"About to call _train_epoch for epoch {epoch}")
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
            import numpy as np
            
            # Check if it's an MLX array directly
            if isinstance(obj, mx.array):
                try:
                    return obj.item() if obj.size == 1 else obj.tolist()
                except:
                    return float(obj)
            # Check if it's a numpy array
            elif isinstance(obj, np.ndarray):
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
            elif isinstance(obj, (np.integer, np.floating)):
                # Handle numpy scalar types
                return obj.item()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            else:
                # For any other type, try to convert to string if not already serializable
                try:
                    json.dumps(obj)
                    return obj
                except:
                    return str(obj)
        
        result_dict = make_json_serializable(result.to_dict())
        
        with open(result_path, "w") as f:
            json.dump(result_dict, f, indent=2)
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Final model saved to: {final_path}")
        
        return result
    
    def _train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        logger.debug(f"Entered _train_epoch for epoch {epoch}")
        self._call_hooks("on_epoch_begin", self.state)
        
        # Initialize metrics
        epoch_loss = 0.0
        epoch_metrics = {}
        num_batches = 0
        
        # Get total batches for progress tracking
        total_batches = len(dataloader)
        logger.debug(f"Total batches in epoch: {total_batches}")
        
        logger.debug(f"About to start enumerate loop over dataloader")
        logger.debug(f"Creating dataloader iterator")
        dataloader_iter = enumerate(dataloader)
        logger.debug(f"Iterator created, starting loop")
        for batch_idx, batch in dataloader_iter:
            try:
                logger.debug(f"Got batch {batch_idx}")
                logger.debug(f"Batch keys: {batch.keys() if isinstance(batch, dict) else 'Not a dict'}")
                
                # Manual progress tracking
                if batch_idx % max(1, total_batches // 10) == 0 or batch_idx == 0:
                    progress_pct = (batch_idx / total_batches) * 100
                    logger.info(f"Epoch {epoch} - Batch {batch_idx}/{total_batches} ({progress_pct:.1f}%)")
                
                self.state.global_step += 1
                self.state.samples_seen += self.config.data.batch_size
                
                # Call batch begin hooks
                self._call_hooks("on_batch_begin", self.state, batch)
                
                # Training step - use compiled version if available
                logger.debug(f"About to call train step, use_compiled: {self._use_compiled}")
                if self._use_compiled:
                    loss, metrics = self._compiled_train_step(batch)
                else:
                    loss, metrics = self._train_step(batch)
                
                # CRITICAL FIX: Force evaluation after gradient computation to prevent graph buildup
                # This prevents MLX lazy evaluation from accumulating large computation graphs
                # which cause hanging during gradient computation with complex models like ModernBERT
                # See: https://github.com/ml-explore/mlx/issues/451 for MLX gradient computation performance issues
                mx.eval(loss, self.model.parameters(), self.optimizer.state)
                
                logger.debug("Train step completed")
            
                # Update state with current batch metrics (for progress callback)
                if 'grad_norm' in metrics and metrics['grad_norm'] is not None:
                    # Convert to float for state storage (used by progress callbacks)
                    if hasattr(metrics['grad_norm'], 'item'):
                        self.state.grad_norm = float(metrics['grad_norm'].item())
                    else:
                        self.state.grad_norm = float(metrics['grad_norm'])
                
                # Accumulate loss (keep as MLX array for now)
                if epoch_loss == 0.0:
                    epoch_loss = loss
                else:
                    epoch_loss = epoch_loss + loss
                
                # Accumulate metrics lazily
                for k, v in metrics.items():
                    if k == "loss" or v is None:
                        continue
                        
                    # Skip non-scalar metrics (like logits)
                    if hasattr(v, 'shape') and v.size > 1:
                        continue
                        
                    if k not in epoch_metrics:
                        epoch_metrics[k] = v
                    else:
                        epoch_metrics[k] = epoch_metrics[k] + v
                
                num_batches += 1
            
                # Call batch end hooks with MLX array
                self._call_hooks("on_batch_end", self.state, loss)
                
                # Log batch metrics occasionally - only convert for logging
                if batch_idx % self.config.training.logging_steps == 0:
                    # Force evaluation only for logging
                    loss_val = float(loss.item()) if hasattr(loss, 'item') else float(loss)
                    logger.info(f"Step {self.state.global_step} - Loss: {loss_val:.4f}, LR: {metrics.get('learning_rate', 0):.2e}")
                
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
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                raise
        
        # Average metrics - evaluate only at the end
        mx.eval(epoch_loss)  # Force evaluation before division
        avg_loss = float(epoch_loss.item()) / num_batches if hasattr(epoch_loss, 'item') else float(epoch_loss) / num_batches
        
        avg_metrics = {"loss": avg_loss}
        
        # Average other metrics
        for k, v in epoch_metrics.items():
            if hasattr(v, 'item'):
                mx.eval(v)  # Evaluate before conversion
                avg_metrics[k] = float(v.item()) / num_batches
            else:
                avg_metrics[k] = float(v) / num_batches
        
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
        
        # Set model to eval mode
        self.model.eval()
        
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
                logger.info(f"Evaluating - Batch {batch_idx}/{total_batches} ({progress_pct:.1f}%)")
            # Evaluation step - don't use compiled version for now due to train/eval mode issues
            # TODO: Fix compiled evaluation to handle train/eval mode changes properly
            loss, metrics = self._eval_step(batch)
            
            # Accumulate metrics lazily
            if total_loss == 0.0:
                total_loss = loss
            else:
                total_loss = total_loss + loss
                
            for k, v in metrics.items():
                if v is None or (hasattr(v, 'shape') and v.size > 1):
                    continue
                if k not in total_metrics:
                    total_metrics[k] = v
                else:
                    total_metrics[k] = total_metrics[k] + v
            num_batches += 1
            
            # Progress tracking is handled above, no need for pbar update
        
        # Force evaluation and average metrics at the end
        mx.eval(total_loss)
        avg_loss = float(total_loss.item()) / num_batches if hasattr(total_loss, 'item') else float(total_loss) / num_batches
        
        avg_metrics = {"loss": avg_loss}
        
        # Average other metrics
        for k, v in total_metrics.items():
            if hasattr(v, 'item'):
                mx.eval(v)
                avg_metrics[k] = float(v.item()) / num_batches
            else:
                avg_metrics[k] = float(v) / num_batches
        
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
        if method == "on_batch_begin":
            logger.debug(f"Calling hooks for {method}, num callbacks: {len(self.callbacks)}")
        for callback in self.callbacks:
            if hasattr(callback, method):
                logger.debug(f"Calling {method} on {callback.__class__.__name__}")
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