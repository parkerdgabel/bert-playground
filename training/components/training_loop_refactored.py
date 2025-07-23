"""Training loop component using hexagonal architecture.

This component handles core training iteration logic using framework adapters
instead of direct MLX imports, following the hexagonal architecture pattern.
"""

from typing import Callable, Protocol, Tuple, Dict, Any, Optional
from loguru import logger

from core.protocols.training import (
    Optimizer, 
    TrainingState, 
    Model,
    FrameworkAdapter as IFrameworkAdapter
)
from core.protocols.data import DataLoader
from training.core.config import TrainingConfig
from training.adapters import get_framework_adapter


class TrainingStepFunction(Protocol):
    """Protocol for training step functions."""
    
    def __call__(self, batch: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        """Execute a single training step."""
        ...


class GradientAccumulator:
    """Gradient accumulator using framework adapter."""
    
    def __init__(self, steps: int, adapter: IFrameworkAdapter):
        """Initialize gradient accumulator.
        
        Args:
            steps: Number of accumulation steps
            adapter: Framework adapter
        """
        self.steps = steps
        self.adapter = adapter
        self.accumulated_grads: Optional[dict[str, Any]] = None
        self.current_step = 0
    
    def accumulate(self, gradients: dict[str, Any]) -> bool:
        """Accumulate gradients.
        
        Args:
            gradients: Current gradients
            
        Returns:
            True if should update weights
        """
        if self.accumulated_grads is None:
            self.accumulated_grads = gradients
        else:
            self.accumulated_grads = self.adapter.accumulate_gradients(
                self.accumulated_grads, gradients
            )
        
        self.current_step += 1
        
        if self.current_step >= self.steps:
            # Scale accumulated gradients
            if self.steps > 1:
                self.accumulated_grads = self.adapter.scale_gradients(
                    self.accumulated_grads, 1.0 / self.steps
                )
            return True
        
        return False
    
    def get_gradients(self) -> dict[str, Any]:
        """Get accumulated gradients and reset."""
        grads = self.accumulated_grads
        self.accumulated_grads = None
        self.current_step = 0
        return grads
    
    def reset(self) -> None:
        """Reset accumulator state."""
        self.accumulated_grads = None
        self.current_step = 0


class TrainingLoop:
    """Training loop using hexagonal architecture.
    
    This component manages the inner training loop using framework adapters
    for all ML operations, making it framework-agnostic.
    """
    
    def __init__(
        self,
        model: Model,
        optimizer: Optimizer,
        config: TrainingConfig,
        framework: str = "mlx",
        gradient_accumulator: Optional[GradientAccumulator] = None,
        max_grad_norm: float = 0.0,
        batch_size: int = 32,
    ):
        """Initialize the training loop.
        
        Args:
            model: Model to train
            optimizer: Optimizer for parameter updates
            config: Training configuration
            framework: Framework to use ("mlx", "pytorch", etc.)
            gradient_accumulator: Optional gradient accumulator
            max_grad_norm: Maximum gradient norm for clipping
            batch_size: Batch size for sample counting
        """
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        
        # Get framework adapter
        self.adapter = get_framework_adapter(framework)
        logger.info(f"Using {self.adapter.name} framework adapter")
        
        # Initialize gradient accumulator
        self.gradient_accumulator = gradient_accumulator or GradientAccumulator(
            config.gradient_accumulation_steps, self.adapter
        )
        
        # Create training step function
        self._train_step = self._create_train_step()
        
        # Compiled version (if applicable)
        self._compiled_train_step = None
        self._use_compiled = False
        
        logger.debug("Initialized TrainingLoop with framework adapter")
    
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
                model_inputs = self.adapter.apply_mixed_precision(model_inputs)
            
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
                loss = self.adapter.tensor_multiply(
                    loss, 1 - self.config.label_smoothing
                )
                
            return loss, outputs
        
        # Create value and grad function using adapter
        value_and_grad_fn = self.adapter.create_value_and_grad_fn(
            self.model, loss_fn
        )
        
        def train_step(batch: dict[str, Any]) -> tuple[float, dict[str, Any]]:
            """Single training step."""
            (loss, outputs), grads = value_and_grad_fn(self.model, batch)
            
            # Force evaluation of gradients
            self.adapter.evaluate_tensors(grads)
            
            # Gradient clipping
            grad_norm = None
            if self.max_grad_norm > 0:
                grads, grad_norm = self.adapter.clip_gradients_by_norm(
                    grads, self.max_grad_norm
                )
            
            # Accumulate gradients
            should_update = self.gradient_accumulator.accumulate(grads)
            
            if should_update:
                # Get accumulated gradients and update
                accumulated_grads = self.gradient_accumulator.get_gradients()
                self.adapter.update_model_parameters(
                    self.model, self.optimizer, accumulated_grads
                )
            
            # Build metrics
            metrics = {
                "learning_rate": self.adapter.get_learning_rate(self.optimizer),
                "loss": self.adapter.to_python(loss),
            }
            
            if grad_norm is not None:
                metrics["grad_norm"] = grad_norm
                
            # Add other outputs
            for k, v in outputs.items():
                if k != "loss":
                    metrics[k] = self.adapter.to_python(v) if v is not None else None
                    
            return loss, metrics
        
        return train_step
    
    def train_batch(self, batch: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
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
        epoch_metrics: dict[str, float] = {}
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
            model_params = self.adapter.get_model_parameters(self.model)
            optimizer_state = getattr(self.optimizer, 'state', {})
            self.adapter.evaluate_tensors(loss, model_params, optimizer_state)
            
            # Update state
            if "grad_norm" in metrics and metrics["grad_norm"] is not None:
                state.grad_norm = float(metrics["grad_norm"])
            
            # Accumulate metrics
            loss_value = self.adapter.to_python(loss)
            epoch_loss += loss_value
                
            for k, v in metrics.items():
                if k == "loss" or v is None:
                    continue
                    
                # Skip non-scalar metrics
                if isinstance(v, (list, dict)):
                    continue
                    
                if k not in epoch_metrics:
                    epoch_metrics[k] = v
                else:
                    epoch_metrics[k] += v
                    
            num_batches += 1
            
            # Execute callbacks
            if callbacks:
                for callback in callbacks:
                    if hasattr(callback, "on_batch_end"):
                        callback(state, loss_value)
        
        # Average metrics
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        avg_metrics = {"loss": avg_loss}
        
        # Average other metrics
        for k, v in epoch_metrics.items():
            avg_metrics[k] = v / num_batches if num_batches > 0 else 0.0
                
        return avg_metrics
    
    def set_compiled_step(self, compiled_step: TrainingStepFunction) -> None:
        """Set a compiled training step function.
        
        Args:
            compiled_step: Compiled training step function
        """
        self._compiled_train_step = compiled_step
        self._use_compiled = True
        logger.info("Using compiled training step")
        
    def compile(self) -> None:
        """Compile the training step if supported by framework."""
        if self.adapter.supports_compilation:
            self._compiled_train_step = self.adapter.compile_function(
                self._train_step
            )
            self._use_compiled = True
            logger.info(f"Compiled training step using {self.adapter.name}")
        else:
            logger.info(f"{self.adapter.name} does not support compilation")
        
    def reset_gradients(self) -> None:
        """Reset gradient accumulator."""
        self.gradient_accumulator.reset()