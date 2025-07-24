"""MLX implementation of the training executor port.

This adapter handles the actual training execution using MLX framework,
implementing the TrainingExecutor port interface.
"""

from typing import Tuple, Any, Optional, Iterator, Dict, List
from dataclasses import dataclass
import time
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from infrastructure.di import adapter
from application.ports.secondary.training import (
    TrainingExecutor,
    TrainingBatch,
    TrainingStepResult,
    EvaluationResult
)
from domain.entities.model import BertModel
from domain.entities.dataset import Dataset
from domain.entities.training import TrainingState
from domain.value_objects.hyperparameters import (
    Hyperparameters,
    OptimizerType,
    LearningRateSchedule
)


@adapter(port=TrainingExecutor, priority=100)
class MLXTrainingExecutor:
    """MLX implementation of training executor.
    
    This adapter handles:
    - Model compilation and optimization
    - Training step execution
    - Gradient computation and updates
    - Learning rate scheduling
    - Checkpoint management
    """
    
    def __init__(self):
        """Initialize the MLX training executor."""
        self._compiled_models = {}
        self._loss_fn = None
        self._eval_fn = None
    
    def initialize_training(
        self,
        model: BertModel,
        hyperparameters: Hyperparameters,
        device: Optional[str] = None
    ) -> Tuple[Any, Any, Any]:
        """Initialize training components.
        
        Returns:
            Tuple of (compiled_model, optimizer, scheduler)
        """
        # Convert domain model to MLX model
        mlx_model = self._create_mlx_model(model)
        
        # Compile if requested
        if hyperparameters.compile_model:
            mlx_model = self._compile_model(mlx_model)
        
        # Create optimizer
        optimizer = self._create_optimizer(mlx_model, hyperparameters)
        
        # Create scheduler
        scheduler = self._create_scheduler(hyperparameters)
        
        # Setup loss function
        self._setup_loss_function(hyperparameters)
        
        return mlx_model, optimizer, scheduler
    
    def create_data_iterator(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0
    ) -> Iterator[TrainingBatch]:
        """Create iterator for training data."""
        # This is a simplified implementation
        # In practice, this would use the MLX data loading utilities
        
        indices = mx.arange(dataset.size)
        if shuffle:
            indices = mx.random.permutation(indices)
        
        num_batches = (dataset.size + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, dataset.size)
            batch_indices = indices[start_idx:end_idx]
            
            # Create batch (simplified - would load actual data)
            batch = TrainingBatch(
                input_ids=mx.zeros((end_idx - start_idx, 512), dtype=mx.int32),
                attention_mask=mx.ones((end_idx - start_idx, 512), dtype=mx.int32),
                labels=mx.zeros((end_idx - start_idx,), dtype=mx.int32),
                batch_idx=i,
                total_batches=num_batches
            )
            
            yield batch
    
    def training_step(
        self,
        model: Any,
        batch: TrainingBatch,
        optimizer: Any,
        accumulation_steps: int = 1,
        clip_grad_norm: Optional[float] = None
    ) -> TrainingStepResult:
        """Execute a single training step."""
        start_time = time.time()
        
        # Define loss function for this batch
        def loss_fn(model):
            # Forward pass
            logits = model(
                batch.input_ids,
                attention_mask=batch.attention_mask
            )
            
            # Compute loss
            loss = self._compute_loss(logits, batch.labels)
            
            # Scale for gradient accumulation
            if accumulation_steps > 1:
                loss = loss / accumulation_steps
            
            return loss
        
        # Compute gradients
        loss_value, grads = mx.value_and_grad(loss_fn)(model)
        
        # Clip gradients if requested
        grad_norm = None
        if clip_grad_norm is not None:
            grads, grad_norm = self._clip_gradients(grads, clip_grad_norm)
        
        # Update model
        optimizer.update(model, grads)
        
        # Compute metrics
        step_time = time.time() - start_time
        batch_size = batch.input_ids.shape[0]
        
        # Get memory stats
        memory_stats = self.get_memory_stats()
        
        return TrainingStepResult(
            loss=float(loss_value),
            gradients_norm=float(grad_norm) if grad_norm is not None else None,
            learning_rate=optimizer.learning_rate,
            throughput_samples_per_sec=batch_size / step_time,
            memory_used_mb=memory_stats.get("allocated_mb", 0),
            additional_metrics={
                "step_time": step_time,
                "batch_progress": batch.progress
            }
        )
    
    def evaluation_step(
        self,
        model: Any,
        batch: TrainingBatch
    ) -> Tuple[float, Dict[str, float]]:
        """Execute a single evaluation step."""
        # Forward pass without gradients
        with mx.no_grad():
            logits = model(
                batch.input_ids,
                attention_mask=batch.attention_mask
            )
            
            # Compute loss
            loss = self._compute_loss(logits, batch.labels)
            
            # Compute metrics
            predictions = mx.argmax(logits, axis=-1)
            accuracy = mx.mean(predictions == batch.labels)
            
            metrics = {
                "accuracy": float(accuracy),
                "perplexity": float(mx.exp(loss))
            }
        
        return float(loss), metrics
    
    def update_learning_rate(
        self,
        scheduler: Any,
        step: int
    ) -> float:
        """Update and return current learning rate."""
        if hasattr(scheduler, "step"):
            scheduler.step()
            return scheduler.get_lr()
        else:
            # Simple scheduler
            return scheduler(step)
    
    def save_checkpoint(
        self,
        model: Any,
        optimizer: Any,
        scheduler: Any,
        training_state: TrainingState,
        path: str
    ) -> None:
        """Save training checkpoint."""
        checkpoint_dir = Path(path)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        model_path = checkpoint_dir / "model.safetensors"
        mx.save_safetensors(str(model_path), dict(tree_flatten(model.parameters())))
        
        # Save optimizer state
        optimizer_path = checkpoint_dir / "optimizer.safetensors"
        mx.save_safetensors(str(optimizer_path), dict(tree_flatten(optimizer.state)))
        
        # Save training state
        state_path = checkpoint_dir / "training_state.json"
        state_dict = {
            "current_epoch": training_state.current_epoch,
            "current_step": training_state.current_step,
            "current_loss": training_state.current_loss,
            "best_loss": training_state.best_loss,
            "loss_history": training_state.loss_history[-100:],  # Keep last 100
            "epochs_without_improvement": training_state.epochs_without_improvement
        }
        with open(state_path, "w") as f:
            json.dump(state_dict, f, indent=2)
        
        # Save scheduler state if available
        if hasattr(scheduler, "state_dict"):
            scheduler_path = checkpoint_dir / "scheduler.json"
            with open(scheduler_path, "w") as f:
                json.dump(scheduler.state_dict(), f, indent=2)
    
    def load_checkpoint(
        self,
        path: str
    ) -> Tuple[Any, Any, Any, TrainingState]:
        """Load training checkpoint."""
        checkpoint_dir = Path(path)
        
        # Load model weights
        model_path = checkpoint_dir / "model.safetensors"
        model_weights = mx.load(str(model_path))
        # Note: Model reconstruction would happen here
        model = None  # Placeholder
        
        # Load optimizer state
        optimizer_path = checkpoint_dir / "optimizer.safetensors"
        optimizer_state = mx.load(str(optimizer_path))
        # Note: Optimizer reconstruction would happen here
        optimizer = None  # Placeholder
        
        # Load training state
        state_path = checkpoint_dir / "training_state.json"
        with open(state_path, "r") as f:
            state_dict = json.load(f)
        
        training_state = TrainingState()
        training_state.current_epoch = state_dict["current_epoch"]
        training_state.current_step = state_dict["current_step"]
        training_state.current_loss = state_dict["current_loss"]
        training_state.best_loss = state_dict["best_loss"]
        training_state.loss_history = state_dict["loss_history"]
        training_state.epochs_without_improvement = state_dict["epochs_without_improvement"]
        
        # Load scheduler state if available
        scheduler = None  # Placeholder
        scheduler_path = checkpoint_dir / "scheduler.json"
        if scheduler_path.exists():
            with open(scheduler_path, "r") as f:
                scheduler_state = json.load(f)
                # Note: Scheduler reconstruction would happen here
        
        return model, optimizer, scheduler, training_state
    
    def compile_model(
        self,
        model: Any,
        compile_options: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Compile model for optimized execution."""
        # MLX compilation is automatic for most operations
        # This is a placeholder for any specific compilation needs
        return model
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics."""
        # MLX memory tracking
        return {
            "allocated_mb": mx.metal.get_active_memory() / (1024 * 1024),
            "cached_mb": mx.metal.get_cache_memory() / (1024 * 1024),
            "peak_mb": mx.metal.get_peak_memory() / (1024 * 1024)
        }
    
    def cleanup(self) -> None:
        """Clean up training resources."""
        # Clear compiled models cache
        self._compiled_models.clear()
        
        # Clear MLX cache
        mx.metal.clear_cache()
    
    def _create_mlx_model(self, model: BertModel) -> Any:
        """Convert domain model to MLX model."""
        # This would use the MLX model factory
        # For now, return a placeholder
        return nn.Module()
    
    def _compile_model(self, model: Any) -> Any:
        """Compile model for better performance."""
        # MLX handles most compilation automatically
        # This could add specific optimizations
        return model
    
    def _create_optimizer(
        self,
        model: Any,
        hyperparameters: Hyperparameters
    ) -> Any:
        """Create optimizer instance."""
        # Get model parameters
        parameters = model.parameters()
        
        # Create optimizer based on type
        if hyperparameters.optimizer_type == OptimizerType.ADAM:
            return optim.Adam(
                learning_rate=hyperparameters.learning_rate,
                betas=(hyperparameters.adam_beta1, hyperparameters.adam_beta2),
                eps=hyperparameters.adam_epsilon
            )
        elif hyperparameters.optimizer_type == OptimizerType.ADAMW:
            return optim.AdamW(
                learning_rate=hyperparameters.learning_rate,
                betas=(hyperparameters.adam_beta1, hyperparameters.adam_beta2),
                eps=hyperparameters.adam_epsilon,
                weight_decay=hyperparameters.weight_decay
            )
        elif hyperparameters.optimizer_type == OptimizerType.SGD:
            return optim.SGD(
                learning_rate=hyperparameters.learning_rate,
                momentum=0.9,
                weight_decay=hyperparameters.weight_decay
            )
        else:
            # Default to AdamW
            return optim.AdamW(
                learning_rate=hyperparameters.learning_rate,
                weight_decay=hyperparameters.weight_decay
            )
    
    def _create_scheduler(
        self,
        hyperparameters: Hyperparameters
    ) -> Any:
        """Create learning rate scheduler."""
        # Simple scheduler implementation
        # In practice, this would be more sophisticated
        
        base_lr = hyperparameters.learning_rate
        
        if hyperparameters.lr_schedule == LearningRateSchedule.CONSTANT:
            return lambda step: base_lr
        elif hyperparameters.lr_schedule == LearningRateSchedule.LINEAR:
            def linear_schedule(step):
                return base_lr * (1 - step / 1000)  # Placeholder
            return linear_schedule
        else:
            return lambda step: base_lr
    
    def _setup_loss_function(
        self,
        hyperparameters: Hyperparameters
    ) -> None:
        """Setup loss function based on hyperparameters."""
        # This would configure the loss function
        # For now, use cross entropy
        self._loss_fn = nn.losses.cross_entropy
    
    def _compute_loss(
        self,
        logits: mx.array,
        labels: mx.array
    ) -> mx.array:
        """Compute loss value."""
        # Reshape for loss computation
        vocab_size = logits.shape[-1]
        logits = logits.reshape(-1, vocab_size)
        labels = labels.reshape(-1)
        
        # Compute cross entropy loss
        return self._loss_fn(logits, labels)
    
    def _clip_gradients(
        self,
        grads: Dict[str, Any],
        max_norm: float
    ) -> Tuple[Dict[str, Any], float]:
        """Clip gradients by norm."""
        # Compute gradient norm
        grad_norm = mx.sqrt(
            sum(mx.sum(g ** 2) for g in tree_flatten(grads))
        )
        
        # Clip if needed
        if grad_norm > max_norm:
            scale = max_norm / grad_norm
            grads = tree_unflatten(
                [g * scale for g in tree_flatten(grads)]
            )
        
        return grads, grad_norm