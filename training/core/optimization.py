"""
Optimization utilities for MLX training including optimizers, schedulers, and gradient handling.
"""

from typing import Dict, Any, Optional, Callable, Tuple
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from loguru import logger

from .config import OptimizerType, SchedulerType, OptimizerConfig, SchedulerConfig


def create_optimizer(
    model: nn.Module,
    config: OptimizerConfig,
) -> optim.Optimizer:
    """
    Create an MLX optimizer from configuration.
    
    Args:
        model: Model to optimize
        config: Optimizer configuration
        
    Returns:
        MLX optimizer instance
    """
    # Get trainable parameters
    trainable_params = model.trainable_parameters()
    
    # Create optimizer based on type
    if config.type == OptimizerType.ADAM:
        optimizer = optim.Adam(
            learning_rate=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.epsilon,
        )
    elif config.type == OptimizerType.ADAMW:
        optimizer = optim.AdamW(
            learning_rate=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.epsilon,
            weight_decay=config.weight_decay,
        )
    elif config.type == OptimizerType.SGD:
        optimizer = optim.SGD(
            learning_rate=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            nesterov=config.nesterov,
        )
    elif config.type == OptimizerType.LION:
        optimizer = optim.Lion(
            learning_rate=config.learning_rate,
            betas=(config.lion_beta1, config.lion_beta2),
            weight_decay=config.weight_decay,
        )
    elif config.type == OptimizerType.ADAFACTOR:
        optimizer = optim.Adafactor(
            learning_rate=config.learning_rate,
            eps=(1e-30, config.epsilon),
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {config.type}")
    
    logger.info(f"Created {config.type.value} optimizer with lr={config.learning_rate}")
    return optimizer


class LearningRateScheduler:
    """Base class for learning rate schedulers."""
    
    def __init__(self, optimizer: optim.Optimizer, config: SchedulerConfig):
        self.optimizer = optimizer
        self.config = config
        self.base_lr = optimizer.learning_rate
        self.current_step = 0
        self.current_lr = self.base_lr
        
        # Warmup settings
        self.warmup_steps = config.warmup_steps
        if config.warmup_ratio > 0 and config.num_training_steps:
            self.warmup_steps = int(config.warmup_ratio * config.num_training_steps)
    
    def step(self, metrics: Optional[Dict[str, float]] = None) -> float:
        """Update learning rate and return new value."""
        self.current_step += 1
        
        # Handle warmup
        if self.current_step <= self.warmup_steps:
            self.current_lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            self.current_lr = self._compute_lr(self.current_step - self.warmup_steps)
        
        # Update optimizer learning rate
        self.optimizer.learning_rate = self.current_lr
        
        return self.current_lr
    
    def _compute_lr(self, step: int) -> float:
        """Compute learning rate for given step (after warmup)."""
        return self.base_lr
    
    def get_last_lr(self) -> float:
        """Get the last computed learning rate."""
        return self.current_lr
    
    @property
    def state_dict(self) -> Dict[str, Any]:
        """Get scheduler state."""
        return {
            "current_step": self.current_step,
            "current_lr": self.current_lr,
            "base_lr": self.base_lr,
        }
    
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load scheduler state."""
        self.current_step = state["current_step"]
        self.current_lr = state["current_lr"]
        self.base_lr = state.get("base_lr", self.base_lr)
        self.optimizer.learning_rate = self.current_lr


class LinearScheduler(LearningRateScheduler):
    """Linear learning rate decay."""
    
    def _compute_lr(self, step: int) -> float:
        if self.config.num_training_steps is None:
            return self.base_lr
        
        remaining_steps = self.config.num_training_steps - self.warmup_steps
        if remaining_steps <= 0:
            return self.base_lr
        
        return self.base_lr * max(0.0, 1.0 - step / remaining_steps)


class CosineScheduler(LearningRateScheduler):
    """Cosine learning rate decay."""
    
    def _compute_lr(self, step: int) -> float:
        if self.config.num_training_steps is None:
            return self.base_lr
        
        import math
        remaining_steps = self.config.num_training_steps - self.warmup_steps
        if remaining_steps <= 0:
            return self.base_lr
        
        progress = min(step / remaining_steps, 1.0)
        return self.base_lr * 0.5 * (1.0 + math.cos(math.pi * self.config.num_cycles * 2.0 * progress))


class CosineWithRestartsScheduler(LearningRateScheduler):
    """Cosine learning rate decay with restarts."""
    
    def _compute_lr(self, step: int) -> float:
        if self.config.num_training_steps is None:
            return self.base_lr
        
        import math
        remaining_steps = self.config.num_training_steps - self.warmup_steps
        if remaining_steps <= 0:
            return self.base_lr
        
        # Calculate cycle length
        cycle_length = remaining_steps // (self.config.num_restarts + 1)
        if cycle_length <= 0:
            return self.base_lr
        
        # Find current position in cycle
        cycle_position = step % cycle_length
        progress = cycle_position / cycle_length
        
        return self.base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


class ExponentialScheduler(LearningRateScheduler):
    """Exponential learning rate decay."""
    
    def _compute_lr(self, step: int) -> float:
        return self.base_lr * (self.config.gamma ** step)


class ReduceOnPlateauScheduler(LearningRateScheduler):
    """Reduce learning rate when metric plateaus."""
    
    def __init__(self, optimizer: optim.Optimizer, config: SchedulerConfig):
        super().__init__(optimizer, config)
        self.best_metric = None
        self.patience_counter = 0
        self.num_reductions = 0
    
    def step(self, metrics: Optional[Dict[str, float]] = None) -> float:
        """Update learning rate based on metric."""
        self.current_step += 1
        
        # Handle warmup
        if self.current_step <= self.warmup_steps:
            self.current_lr = self.base_lr * (self.current_step / self.warmup_steps)
            self.optimizer.learning_rate = self.current_lr
            return self.current_lr
        
        # Check if we have metrics
        if metrics is None or "eval_loss" not in metrics:
            return self.current_lr
        
        metric_value = metrics["eval_loss"]
        
        # Initialize best metric
        if self.best_metric is None:
            self.best_metric = metric_value
            return self.current_lr
        
        # Check for improvement
        if metric_value < self.best_metric - self.config.min_lr:
            self.best_metric = metric_value
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        # Reduce LR if patience exceeded
        if self.patience_counter >= self.config.patience:
            self.current_lr = max(
                self.current_lr * self.config.factor,
                self.config.min_lr
            )
            self.optimizer.learning_rate = self.current_lr
            self.patience_counter = 0
            self.num_reductions += 1
            logger.info(f"Reduced learning rate to {self.current_lr}")
        
        return self.current_lr


def create_lr_scheduler(
    optimizer: optim.Optimizer,
    config: SchedulerConfig,
) -> Optional[LearningRateScheduler]:
    """
    Create a learning rate scheduler from configuration.
    
    Args:
        optimizer: MLX optimizer
        config: Scheduler configuration
        
    Returns:
        Learning rate scheduler or None
    """
    # Handle string types that weren't converted to enum
    scheduler_type = config.type
    if isinstance(scheduler_type, str):
        try:
            scheduler_type = SchedulerType(scheduler_type)
        except ValueError:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    if scheduler_type == SchedulerType.NONE:
        return None
    elif scheduler_type == SchedulerType.CONSTANT:
        return LearningRateScheduler(optimizer, config)  # Base class is constant
    elif scheduler_type == SchedulerType.LINEAR:
        return LinearScheduler(optimizer, config)
    elif scheduler_type == SchedulerType.COSINE:
        return CosineScheduler(optimizer, config)
    elif scheduler_type == SchedulerType.COSINE_WITH_RESTARTS:
        return CosineWithRestartsScheduler(optimizer, config)
    elif scheduler_type == SchedulerType.EXPONENTIAL:
        return ExponentialScheduler(optimizer, config)
    elif scheduler_type == SchedulerType.REDUCE_ON_PLATEAU:
        return ReduceOnPlateauScheduler(optimizer, config)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


class GradientAccumulator:
    """Handles gradient accumulation for larger effective batch sizes."""
    
    def __init__(self, accumulation_steps: int = 1):
        self.accumulation_steps = accumulation_steps
        self.accumulated_grads: Optional[Dict[str, Any]] = None
        self.step_count = 0
    
    def accumulate(self, gradients: Dict[str, Any]) -> bool:
        """
        Accumulate gradients.
        
        Args:
            gradients: Current batch gradients (may be nested)
            
        Returns:
            True if should update weights, False otherwise
        """
        self.step_count += 1
        
        def accumulate_recursive(acc_grads, new_grads, scale):
            """Recursively accumulate gradients."""
            if acc_grads is None:
                acc_grads = {}
            
            for k, v in new_grads.items():
                if isinstance(v, dict):
                    # Recursive case for nested dict
                    if k not in acc_grads:
                        acc_grads[k] = {}
                    acc_grads[k] = accumulate_recursive(acc_grads[k], v, scale)
                elif isinstance(v, list):
                    # Handle list of gradients (e.g., from layers)
                    if k not in acc_grads:
                        acc_grads[k] = []
                        for i, item in enumerate(v):
                            if isinstance(item, dict):
                                acc_grads[k].append(accumulate_recursive({}, item, scale))
                            elif item is not None:
                                acc_grads[k].append(item * scale)
                            else:
                                acc_grads[k].append(item)
                    else:
                        for i, item in enumerate(v):
                            if isinstance(item, dict):
                                acc_grads[k][i] = accumulate_recursive(acc_grads[k][i], item, scale)
                            elif item is not None:
                                acc_grads[k][i] += item * scale
                elif v is not None:
                    # Base case for arrays
                    if k in acc_grads:
                        acc_grads[k] += v * scale
                    else:
                        acc_grads[k] = v * scale
                else:
                    acc_grads[k] = v
            return acc_grads
        
        scale = 1.0  # Don't scale during accumulation
        self.accumulated_grads = accumulate_recursive(self.accumulated_grads, gradients, scale)
        
        # Check if we should update
        should_update = (self.step_count % self.accumulation_steps) == 0
        
        return should_update
    
    def get_gradients(self, average: bool = False) -> Dict[str, Any]:
        """Get accumulated gradients and reset."""
        grads = self.accumulated_grads
        if average and grads is not None:
            # Average the gradients by the number of accumulation steps
            def average_grads(g):
                if isinstance(g, dict):
                    return {k: average_grads(v) for k, v in g.items()}
                else:
                    return g / self.accumulation_steps
            grads = average_grads(grads)
        self.accumulated_grads = None
        return grads
    
    def reset(self) -> None:
        """Reset accumulator."""
        self.accumulated_grads = None
        self.step_count = 0


def clip_gradients(
    gradients: Dict[str, Any],
    max_norm: float,
) -> Tuple[Dict[str, Any], float]:
    """
    Clip gradients by global norm.
    
    Args:
        gradients: Dictionary of gradients (may be nested)
        max_norm: Maximum gradient norm
        
    Returns:
        Clipped gradients and original norm
    """
    # Flatten gradients to compute norm
    def flatten_grads(grads):
        """Recursively flatten gradient dictionary."""
        flat = []
        for k, v in grads.items():
            if isinstance(v, dict):
                flat.extend(flatten_grads(v))
            elif isinstance(v, list):
                # Handle list of gradients (e.g., from layers)
                for item in v:
                    if isinstance(item, dict):
                        flat.extend(flatten_grads(item))
                    elif item is not None:
                        flat.append(item)
            elif v is not None:
                flat.append(v)
        return flat
    
    # Compute global norm
    flat_grads = flatten_grads(gradients)
    total_norm = 0.0
    for grad in flat_grads:
        if grad is not None:
            total_norm += mx.sum(grad ** 2).item()
    total_norm = total_norm ** 0.5
    
    # Clip if needed
    if total_norm > max_norm:
        scale = max_norm / total_norm
        
        def scale_grads(grads):
            """Recursively scale gradients."""
            scaled = {}
            for k, v in grads.items():
                if isinstance(v, dict):
                    scaled[k] = scale_grads(v)
                elif isinstance(v, list):
                    # Handle list of gradients (e.g., from layers)
                    scaled[k] = []
                    for item in v:
                        if isinstance(item, dict):
                            scaled[k].append(scale_grads(item))
                        elif item is not None:
                            scaled[k].append(item * scale)
                        else:
                            scaled[k].append(item)
                elif v is not None:
                    scaled[k] = v * scale
                else:
                    scaled[k] = v
            return scaled
        
        clipped_grads = scale_grads(gradients)
    else:
        clipped_grads = gradients
    
    return clipped_grads, total_norm


def compute_gradient_stats(gradients: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute statistics about gradients for monitoring.
    
    Args:
        gradients: Dictionary of gradients (may be nested)
        
    Returns:
        Dictionary of statistics
    """
    # Flatten gradients for statistics
    def flatten_grads(grads):
        """Recursively flatten gradient dictionary."""
        flat = []
        for k, v in grads.items():
            if isinstance(v, dict):
                flat.extend(flatten_grads(v))
            elif isinstance(v, list):
                # Handle list of gradients (e.g., from layers)
                for item in v:
                    if isinstance(item, dict):
                        flat.extend(flatten_grads(item))
                    elif item is not None:
                        flat.append(item)
            elif v is not None:
                flat.append(v)
        return flat
    
    flat_grads = flatten_grads(gradients)
    
    stats = {
        "grad_norm": 0.0,
        "grad_max": 0.0,
        "grad_min": float('inf'),
        "grad_mean": 0.0,
        "grad_std": 0.0,
    }
    
    total_elements = 0
    total_sum = 0.0
    all_values = []
    
    for grad in flat_grads:
        if grad is not None:
            stats["grad_norm"] += mx.sum(grad ** 2).item()
            stats["grad_max"] = max(stats["grad_max"], mx.max(mx.abs(grad)).item())
            stats["grad_min"] = min(stats["grad_min"], mx.min(mx.abs(grad)).item())
            
            total_sum += mx.sum(mx.abs(grad)).item()
            total_elements += grad.size
            
            # Collect values for std calculation
            all_values.append(mx.abs(grad).flatten())
    
    stats["grad_norm"] = stats["grad_norm"] ** 0.5
    
    if total_elements > 0:
        stats["grad_mean"] = total_sum / total_elements
    else:
        # No gradients processed, keep defaults
        stats["grad_min"] = float('inf')
        stats["grad_mean"] = 0.0
    
    # Calculate standard deviation
    if all_values:
        all_values = mx.concatenate(all_values)
        stats["grad_std"] = mx.std(all_values).item()
    else:
        stats["grad_std"] = 0.0
    
    return stats