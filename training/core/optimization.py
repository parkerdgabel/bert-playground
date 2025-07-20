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
        self.base_lr = float(optimizer.learning_rate)
        self.current_step = 0
        self.current_lr = float(self.base_lr)
        
        # Warmup settings
        self.warmup_steps = config.warmup_steps
        if config.warmup_ratio > 0 and config.num_training_steps:
            self.warmup_steps = int(config.warmup_ratio * config.num_training_steps)
    
    def step(self, metrics: Optional[Dict[str, float]] = None) -> float:
        """Update learning rate and return new value."""
        self.current_step += 1
        
        # Handle warmup
        if self.current_step <= self.warmup_steps:
            self.current_lr = float(self.base_lr * (self.current_step / self.warmup_steps))
        else:
            self.current_lr = float(self._compute_lr(self.current_step - self.warmup_steps))
        
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
            self.current_lr = float(self.base_lr * (self.current_step / self.warmup_steps))
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


def create_mlx_lr_schedule(
    config: SchedulerConfig,
    base_lr: float,
    num_training_steps: int,
) -> Callable:
    """
    Create an MLX-native learning rate schedule from configuration.
    
    Args:
        config: Scheduler configuration
        base_lr: Base learning rate
        num_training_steps: Total number of training steps
        
    Returns:
        MLX learning rate schedule function
    """
    # Handle string types that weren't converted to enum
    scheduler_type = config.type
    if isinstance(scheduler_type, str):
        try:
            scheduler_type = SchedulerType(scheduler_type)
        except ValueError:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    # Calculate warmup steps
    warmup_steps = config.warmup_steps
    if warmup_steps == 0 and config.warmup_ratio > 0:
        warmup_steps = int(num_training_steps * config.warmup_ratio)
    
    # Create the main schedule
    if scheduler_type == SchedulerType.NONE or scheduler_type == SchedulerType.CONSTANT:
        # For constant LR, return a scalar array directly
        return mx.array(base_lr)
    elif scheduler_type == SchedulerType.LINEAR:
        # Linear decay from base_lr to min_lr
        decay_steps = num_training_steps - warmup_steps
        main_schedule = optim.schedulers.linear_schedule(
            base_lr, config.min_lr, decay_steps
        )
    elif scheduler_type == SchedulerType.COSINE:
        # Cosine decay
        decay_steps = num_training_steps - warmup_steps
        main_schedule = optim.schedulers.cosine_decay(
            base_lr, decay_steps, end=config.min_lr
        )
    elif scheduler_type == SchedulerType.EXPONENTIAL:
        # Exponential decay
        main_schedule = optim.schedulers.exponential_decay(
            base_lr, config.gamma
        )
    else:
        # For unsupported schedulers, fall back to constant
        logger.warning(f"Scheduler {scheduler_type} not supported with MLX native schedulers, using constant LR")
        main_schedule = lambda step: base_lr
    
    # Add warmup if needed
    if warmup_steps > 0:
        warmup_schedule = optim.schedulers.linear_schedule(
            0.0, base_lr, warmup_steps
        )
        # Join warmup and main schedule
        schedule = optim.schedulers.join_schedules(
            [warmup_schedule, main_schedule],
            [warmup_steps]
        )
    else:
        schedule = main_schedule
    
    logger.info(
        f"Created MLX {scheduler_type} schedule with warmup_steps={warmup_steps}, "
        f"num_training_steps={num_training_steps}, base_lr={base_lr}"
    )
    
    return schedule


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
        Accumulate gradients using MLX tree_map for efficiency.
        
        Args:
            gradients: Current batch gradients (may be nested)
            
        Returns:
            True if should update weights, False otherwise
        """
        from mlx.utils import tree_map
        
        self.step_count += 1
        
        if self.accumulated_grads is None:
            # First accumulation - just store the gradients
            self.accumulated_grads = gradients
        else:
            # Accumulate using tree_map for efficiency
            self.accumulated_grads = tree_map(
                lambda acc, new: acc + new if acc is not None and new is not None else (acc or new),
                self.accumulated_grads,
                gradients
            )
        
        # Check if we should update
        should_update = (self.step_count % self.accumulation_steps) == 0
        
        return should_update
    
    def get_gradients(self, average: bool = True) -> Dict[str, Any]:
        """Get accumulated gradients and reset."""
        from mlx.utils import tree_map
        
        grads = self.accumulated_grads
        
        if average and grads is not None and self.accumulation_steps > 1:
            # Average the gradients by the number of accumulation steps
            scale = 1.0 / self.accumulation_steps
            grads = tree_map(lambda g: g * scale if g is not None else None, grads)
        
        # Reset for next accumulation
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
    Clip gradients by global norm using MLX's native implementation.
    
    Args:
        gradients: Dictionary of gradients (may be nested)
        max_norm: Maximum gradient norm
        
    Returns:
        Clipped gradients and original norm
    """
    # Use MLX's native clip_grad_norm function
    import mlx.optimizers as mlx_opt
    
    # MLX's clip_grad_norm returns (clipped_grads, total_norm)
    clipped_grads, total_norm = mlx_opt.clip_grad_norm(gradients, max_norm)
    
    return clipped_grads, total_norm


def compute_gradient_stats(gradients: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute statistics about gradients for monitoring.
    
    Optimized version that minimizes .item() calls and leverages MLX's lazy evaluation.
    
    Args:
        gradients: Dictionary of gradients (may be nested)
        
    Returns:
        Dictionary of statistics
    """
    # Use tree_flatten for efficient gradient collection
    from mlx.utils import tree_flatten
    
    flat_grads = tree_flatten(gradients)
    
    if not flat_grads:
        return {
            "grad_norm": 0.0,
            "grad_max": 0.0,
            "grad_min": 0.0,
            "grad_mean": 0.0,
            "grad_std": 0.0,
        }
    
    # Filter out None values and get just the arrays
    grad_arrays = [v for k, v in flat_grads if v is not None]
    
    if not grad_arrays:
        return {
            "grad_norm": 0.0,
            "grad_max": 0.0,
            "grad_min": 0.0,
            "grad_mean": 0.0,
            "grad_std": 0.0,
        }
    
    # Compute norm efficiently
    norm_sq = mx.sum(mx.stack([mx.sum(g * g) for g in grad_arrays]))
    grad_norm = mx.sqrt(norm_sq)
    
    # For other stats, only compute if really needed (expensive)
    # Most training loops only need grad_norm
    return {
        "grad_norm": grad_norm,  # Keep as MLX array
        "grad_max": mx.array(0.0),  # Placeholder
        "grad_min": mx.array(0.0),  # Placeholder
        "grad_mean": mx.array(0.0),  # Placeholder
        "grad_std": mx.array(0.0),  # Placeholder
    }


def compute_gradient_stats_detailed(gradients: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute detailed gradient statistics (use sparingly - expensive).
    
    Args:
        gradients: Dictionary of gradients (may be nested)
        
    Returns:
        Dictionary of detailed statistics with Python floats
    """
    from mlx.utils import tree_flatten
    
    flat_grads = tree_flatten(gradients)
    grad_arrays = [v for k, v in flat_grads if v is not None]
    
    if not grad_arrays:
        return {
            "grad_norm": 0.0,
            "grad_max": 0.0,
            "grad_min": 0.0,
            "grad_mean": 0.0,
            "grad_std": 0.0,
        }
    
    # Compute all statistics
    norm_sq = mx.sum(mx.stack([mx.sum(g * g) for g in grad_arrays]))
    grad_norm = mx.sqrt(norm_sq)
    
    # Concatenate for detailed stats
    all_grads = mx.concatenate([g.flatten() for g in grad_arrays])
    all_abs_grads = mx.abs(all_grads)
    
    grad_max = mx.max(all_abs_grads)
    grad_min = mx.min(all_abs_grads)
    grad_mean = mx.mean(all_abs_grads)
    grad_std = mx.std(all_abs_grads)
    
    # Single evaluation
    mx.eval(grad_norm, grad_max, grad_min, grad_mean, grad_std)
    
    return {
        "grad_norm": float(grad_norm.item()),
        "grad_max": float(grad_max.item()),
        "grad_min": float(grad_min.item()),
        "grad_mean": float(grad_mean.item()),
        "grad_std": float(grad_std.item()),
    }