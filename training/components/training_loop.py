"""Training loop component for handling core training iteration logic.

This component is responsible for:
- Executing training steps on batches
- Managing gradient accumulation
- Applying gradient clipping
- Updating model parameters
"""

from typing import Callable, Protocol, Tuple, Dict, Any
import mlx.core as mx
from loguru import logger

from core.protocols.training import Optimizer, TrainingState
from core.protocols.data import DataLoader
from core.protocols.models import Model
from training.core.optimization import GradientAccumulator, clip_gradients
from training.core.config import TrainingConfig


class TrainingStepFunction(Protocol):
    """Protocol for training step functions."""
    
    def __call__(self, batch: dict[str, mx.array]) -> tuple[float, dict[str, mx.array]]:
        """Execute a single training step."""
        ...


class TrainingLoop:
    """Handles core training iteration logic.
    
    This component manages the inner training loop including:
    - Batch processing
    - Gradient computation and accumulation
    - Parameter updates
    - Loss tracking
    """
    
    def __init__(
        self,
        model: Model,
        optimizer: Optimizer,
        config: TrainingConfig,
        gradient_accumulator: GradientAccumulator | None = None,
        max_grad_norm: float = 0.0,
        batch_size: int = 32,
    ):
        """Initialize the training loop.
        
        Args:
            model: Model to train
            optimizer: Optimizer for parameter updates
            config: Training configuration
            gradient_accumulator: Optional gradient accumulator
            max_grad_norm: Maximum gradient norm for clipping
            batch_size: Batch size for sample counting
        """
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.gradient_accumulator = gradient_accumulator or GradientAccumulator(
            config.gradient_accumulation_steps
        )
        
        # Create training step function
        self._train_step = self._create_train_step()
        
        # Compiled version (if applicable)
        self._compiled_train_step = None
        self._use_compiled = False
        
        logger.debug("Initialized TrainingLoop")
        
    def _create_train_step(self) -> TrainingStepFunction:
        """Create the training step function."""
        
        def loss_fn(model, batch):
            # Remove metadata if present
            model_inputs = {
                k: v
                for k, v in batch.items()
                if k not in ["metadata"] and v is not None
            }
            
            # Apply mixed precision if enabled
            if self.config.mixed_precision:
                model_inputs = {
                    k: v.astype(mx.bfloat16) if v.dtype == mx.float32 else v
                    for k, v in model_inputs.items()
                }
            
            # Forward pass
            try:
                outputs = model(**model_inputs)
            except TypeError:
                outputs = model(batch)
            
            # Extract loss
            loss = outputs.get("loss")
            if loss is None:
                raise ValueError("Model must return a dictionary with 'loss' key")
            
            # Apply label smoothing if configured
            if self.config.label_smoothing > 0:
                loss = loss * (1 - self.config.label_smoothing)
                
            return loss, outputs
        
        # Create value and grad function
        value_and_grad_fn = mx.value_and_grad(loss_fn)
        
        def train_step(batch: dict[str, mx.array]) -> tuple[float, dict[str, mx.array]]:
            """Single training step."""
            (loss, outputs), grads = value_and_grad_fn(self.model, batch)
            
            # Force evaluation of gradients
            mx.eval(grads)
            
            # Gradient clipping
            grad_norm = None
            if self.max_grad_norm > 0:
                grads, grad_norm = clip_gradients(grads, self.max_grad_norm)
            
            # Accumulate gradients
            should_update = self.gradient_accumulator.accumulate(grads)
            
            if should_update:
                # Get accumulated gradients and update
                accumulated_grads = self.gradient_accumulator.get_gradients()
                self.optimizer.update(self.model, accumulated_grads)
            
            # Build metrics
            metrics = {
                "learning_rate": self.optimizer.learning_rate,
                "loss": loss,
            }
            
            if grad_norm is not None:
                metrics["grad_norm"] = grad_norm
                
            # Add other outputs
            for k, v in outputs.items():
                if k != "loss":
                    metrics[k] = v
                    
            return loss, metrics
        
        return train_step
    
    def train_batch(self, batch: dict[str, mx.array]) -> tuple[mx.array, dict[str, mx.array]]:
        """Process a single training batch.
        
        Args:
            batch: Input batch
            
        Returns:
            Tuple of (loss, metrics)
        """
        if self._use_compiled and self._compiled_train_step:
            return self._compiled_train_step(batch)
        else:
            return self._train_step(batch)
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        state: TrainingState,
        callbacks: list[Callable] | None = None,
    ) -> dict[str, float]:
        """Train for one epoch.
        
        Args:
            dataloader: Training data loader
            state: Current training state
            callbacks: Optional list of callback functions
            
        Returns:
            Dictionary of average metrics for the epoch
        """
        # Initialize metrics
        epoch_loss = 0.0
        epoch_metrics: dict[str, mx.array] = {}
        num_batches = 0
        
        # Process batches
        for batch_idx, batch in enumerate(dataloader):
            # Update state
            state.global_step += 1
            state.samples_seen += self.batch_size
            
            # Execute callbacks
            if callbacks:
                for callback in callbacks:
                    if hasattr(callback, "on_batch_begin"):
                        callback(state, batch)
            
            # Training step
            loss, metrics = self.train_batch(batch)
            
            # Force evaluation to prevent graph buildup
            mx.eval(loss, self.model.parameters(), self.optimizer.state)
            
            # Update state
            if "grad_norm" in metrics and metrics["grad_norm"] is not None:
                if hasattr(metrics["grad_norm"], "item"):
                    state.grad_norm = float(metrics["grad_norm"].item())
                else:
                    state.grad_norm = float(metrics["grad_norm"])
            
            # Accumulate metrics
            if epoch_loss == 0.0:
                epoch_loss = loss
            else:
                epoch_loss = epoch_loss + loss
                
            for k, v in metrics.items():
                if k == "loss" or v is None:
                    continue
                    
                # Skip non-scalar metrics
                if hasattr(v, "shape") and v.size > 1:
                    continue
                    
                if k not in epoch_metrics:
                    epoch_metrics[k] = v
                else:
                    epoch_metrics[k] = epoch_metrics[k] + v
                    
            num_batches += 1
            
            # Execute callbacks
            if callbacks:
                for callback in callbacks:
                    if hasattr(callback, "on_batch_end"):
                        callback(state, loss)
        
        # Average metrics
        mx.eval(epoch_loss)
        avg_loss = float(epoch_loss.item()) / num_batches if hasattr(epoch_loss, "item") else float(epoch_loss) / num_batches
        
        avg_metrics = {"loss": avg_loss}
        
        # Average other metrics
        for k, v in epoch_metrics.items():
            if hasattr(v, "item"):
                mx.eval(v)
                avg_metrics[k] = float(v.item()) / num_batches
            else:
                avg_metrics[k] = float(v) / num_batches
                
        return avg_metrics
    
    def set_compiled_step(self, compiled_step: TrainingStepFunction) -> None:
        """Set a compiled training step function.
        
        Args:
            compiled_step: Compiled training step function
        """
        self._compiled_train_step = compiled_step
        self._use_compiled = True
        logger.info("Using compiled training step")
        
    def reset_gradients(self) -> None:
        """Reset gradient accumulator."""
        self.gradient_accumulator.reset()