"""
Framework-agnostic optimization utilities for training.

This module provides optimizers, schedulers, and gradient handling
using the FrameworkAdapter abstraction to support multiple backends.
"""

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from loguru import logger

from core.protocols.training import Optimizer as IOptimizer, LRScheduler as ILRScheduler
from training.adapters.framework_adapter import FrameworkAdapter
from .config import OptimizerConfig, SchedulerConfig, SchedulerType
from .optimizer_factory import create_optimizer as create_optimizer_impl


def create_optimizer(
    model: Any,
    config: OptimizerConfig,
    framework: FrameworkAdapter
) -> IOptimizer:
    """
    Create an optimizer from configuration.
    
    Args:
        model: Model to optimize
        config: Optimizer configuration
        framework: Framework adapter
        
    Returns:
        Optimizer instance
    """
    return create_optimizer_impl(model, config, framework)


class LearningRateScheduler:
    """Base class for learning rate schedulers."""

    def __init__(
        self, 
        optimizer: IOptimizer, 
        config: SchedulerConfig,
        framework: FrameworkAdapter
    ):
        self.optimizer = optimizer
        self.config = config
        self.framework = framework
        self.base_lr = float(framework.get_learning_rate(optimizer))
        self.current_step = 0
        self.current_lr = float(self.base_lr)

        # Warmup settings
        self.warmup_steps = config.warmup_steps
        if config.warmup_ratio > 0 and config.num_training_steps:
            self.warmup_steps = int(config.warmup_ratio * config.num_training_steps)

    def step(self, metrics: dict[str, float] | None = None) -> float:
        """Update learning rate and return new value."""
        self.current_step += 1

        # Handle warmup
        if self.current_step <= self.warmup_steps:
            self.current_lr = float(
                self.base_lr * (self.current_step / self.warmup_steps)
            )
        else:
            self.current_lr = float(
                self._compute_lr(self.current_step - self.warmup_steps)
            )

        # Update optimizer learning rate
        self._update_optimizer_lr(self.current_lr)

        return self.current_lr

    def _compute_lr(self, step: int) -> float:
        """Compute learning rate for given step (after warmup)."""
        return self.base_lr
    
    def _update_optimizer_lr(self, lr: float) -> None:
        """Update optimizer learning rate in framework-agnostic way."""
        # For MLX optimizers, we can directly set the learning_rate attribute
        if hasattr(self.optimizer, 'learning_rate'):
            self.optimizer.learning_rate = lr
        else:
            # For other frameworks, may need different approach
            logger.warning(f"Cannot update learning rate for optimizer type: {type(self.optimizer)}")

    def get_last_lr(self) -> float:
        """Get the last computed learning rate."""
        return self.current_lr

    @property
    def state_dict(self) -> dict[str, Any]:
        """Get scheduler state."""
        return {
            "current_step": self.current_step,
            "current_lr": self.current_lr,
            "base_lr": self.base_lr,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load scheduler state."""
        self.current_step = state["current_step"]
        self.current_lr = state["current_lr"]
        self.base_lr = state.get("base_lr", self.base_lr)
        self._update_optimizer_lr(self.current_lr)


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
        return (
            self.base_lr
            * 0.5
            * (1.0 + math.cos(math.pi * self.config.num_cycles * 2.0 * progress))
        )


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
        return self.base_lr * (self.config.gamma**step)


class ReduceOnPlateauScheduler(LearningRateScheduler):
    """Reduce learning rate when metric plateaus."""

    def __init__(
        self,
        optimizer: IOptimizer,
        config: SchedulerConfig,
        framework: FrameworkAdapter
    ):
        super().__init__(optimizer, config, framework)
        self.best_metric = None
        self.patience_counter = 0
        self.num_reductions = 0

    def step(self, metrics: dict[str, float] | None = None) -> float:
        """Update learning rate based on metric."""
        self.current_step += 1

        # Handle warmup
        if self.current_step <= self.warmup_steps:
            self.current_lr = float(
                self.base_lr * (self.current_step / self.warmup_steps)
            )
            self._update_optimizer_lr(self.current_lr)
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
                self.current_lr * self.config.factor, self.config.min_lr
            )
            self._update_optimizer_lr(self.current_lr)
            self.patience_counter = 0
            self.num_reductions += 1
            logger.info(f"Reduced learning rate to {self.current_lr}")

        return self.current_lr


def create_lr_scheduler(
    optimizer: IOptimizer,
    config: SchedulerConfig,
    framework: FrameworkAdapter
) -> ILRScheduler | None:
    """
    Create a learning rate scheduler from configuration.

    Args:
        optimizer: Optimizer instance
        config: Scheduler configuration
        framework: Framework adapter

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
        return LearningRateScheduler(optimizer, config, framework)  # Base class is constant
    elif scheduler_type == SchedulerType.LINEAR:
        return LinearScheduler(optimizer, config, framework)
    elif scheduler_type == SchedulerType.COSINE:
        return CosineScheduler(optimizer, config, framework)
    elif scheduler_type == SchedulerType.COSINE_WITH_RESTARTS:
        return CosineWithRestartsScheduler(optimizer, config, framework)
    elif scheduler_type == SchedulerType.EXPONENTIAL:
        return ExponentialScheduler(optimizer, config, framework)
    elif scheduler_type == SchedulerType.REDUCE_ON_PLATEAU:
        return ReduceOnPlateauScheduler(optimizer, config, framework)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


class GradientAccumulator:
    """Handles gradient accumulation for larger effective batch sizes."""

    def __init__(self, accumulation_steps: int = 1, framework: FrameworkAdapter = None):
        self.accumulation_steps = accumulation_steps
        self.framework = framework or FrameworkAdapter()
        self.accumulated_grads: dict[str, Any] | None = None
        self.step_count = 0

    def accumulate(self, gradients: dict[str, Any]) -> bool:
        """
        Accumulate gradients using framework adapter.

        Args:
            gradients: Current batch gradients (may be nested)

        Returns:
            True if should update weights, False otherwise
        """
        self.step_count += 1

        if self.accumulated_grads is None:
            # First accumulation - just store the gradients
            self.accumulated_grads = gradients
        else:
            # Accumulate using framework adapter
            self.accumulated_grads = self.framework.accumulate_gradients(
                self.accumulated_grads, gradients
            )

        # Check if we should update
        should_update = (self.step_count % self.accumulation_steps) == 0

        return should_update

    def get_gradients(self, average: bool = True) -> dict[str, Any]:
        """Get accumulated gradients and reset."""
        grads = self.accumulated_grads

        if average and grads is not None and self.accumulation_steps > 1:
            # Average the gradients by the number of accumulation steps
            scale = 1.0 / self.accumulation_steps
            grads = self.framework.scale_gradients(grads, scale)

        # Reset for next accumulation
        self.accumulated_grads = None

        return grads

    def reset(self) -> None:
        """Reset accumulator."""
        self.accumulated_grads = None
        self.step_count = 0


def clip_gradients(
    gradients: dict[str, Any],
    max_norm: float,
    framework: FrameworkAdapter
) -> tuple[dict[str, Any], float]:
    """
    Clip gradients by global norm using framework adapter.

    Args:
        gradients: Dictionary of gradients (may be nested)
        max_norm: Maximum gradient norm
        framework: Framework adapter

    Returns:
        Clipped gradients and original norm
    """
    return framework.clip_gradients_by_norm(gradients, max_norm)


def compute_gradient_stats(
    gradients: dict[str, Any],
    framework: FrameworkAdapter,
    detailed: bool = False
) -> dict[str, float]:
    """
    Compute statistics about gradients for monitoring.

    Args:
        gradients: Dictionary of gradients (may be nested)
        framework: Framework adapter
        detailed: Whether to compute detailed stats (expensive)

    Returns:
        Dictionary of statistics
    """
    if not gradients:
        return {
            "grad_norm": 0.0,
            "grad_max": 0.0,
            "grad_min": 0.0,
            "grad_mean": 0.0,
            "grad_std": 0.0,
        }

    # Always compute gradient norm
    grad_norm = framework.compute_gradient_norm(gradients)

    if detailed:
        # Detailed stats would require framework-specific implementation
        # For now, just return norm
        return {
            "grad_norm": grad_norm,
            "grad_max": 0.0,  # Placeholder
            "grad_min": 0.0,  # Placeholder
            "grad_mean": 0.0,  # Placeholder
            "grad_std": 0.0,  # Placeholder
        }
    else:
        # Basic stats - just norm
        return {
            "grad_norm": grad_norm,
            "grad_max": 0.0,
            "grad_min": 0.0,
            "grad_mean": 0.0,
            "grad_std": 0.0,
        }


@runtime_checkable
class NativeLRScheduleFactory(Protocol):
    """Protocol for creating native framework LR schedules."""
    
    def create_schedule(
        self,
        config: SchedulerConfig,
        base_lr: float,
        num_training_steps: int
    ) -> Any:
        """Create a native framework learning rate schedule.
        
        Args:
            config: Scheduler configuration
            base_lr: Base learning rate
            num_training_steps: Total number of training steps
            
        Returns:
            Framework-specific schedule object
        """
        ...


class MLXLRScheduleFactory:
    """Factory for creating MLX-native learning rate schedules."""
    
    def create_schedule(
        self,
        config: SchedulerConfig,
        base_lr: float,
        num_training_steps: int
    ) -> Any:
        """Create MLX-native learning rate schedule.
        
        Args:
            config: Scheduler configuration
            base_lr: Base learning rate
            num_training_steps: Total number of training steps
            
        Returns:
            MLX learning rate schedule
        """
        import mlx.core as mx
        import mlx.optimizers as optim
        
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
            main_schedule = optim.schedulers.exponential_decay(base_lr, config.gamma)
        else:
            # For unsupported schedulers, fall back to constant
            logger.warning(
                f"Scheduler {scheduler_type} not supported with MLX native schedulers, using constant LR"
            )
            main_schedule = lambda step: base_lr
        
        # Add warmup if needed
        if warmup_steps > 0:
            warmup_schedule = optim.schedulers.linear_schedule(0.0, base_lr, warmup_steps)
            # Join warmup and main schedule
            schedule = optim.schedulers.join_schedules(
                [warmup_schedule, main_schedule], [warmup_steps]
            )
        else:
            schedule = main_schedule
        
        logger.info(
            f"Created MLX {scheduler_type} schedule with warmup_steps={warmup_steps}, "
            f"num_training_steps={num_training_steps}, base_lr={base_lr}"
        )
        
        return schedule


# Registry for native schedule factories
_schedule_factories: dict[str, NativeLRScheduleFactory] = {
    "mlx": MLXLRScheduleFactory()
}


def create_native_lr_schedule(
    config: SchedulerConfig,
    base_lr: float,
    num_training_steps: int,
    framework: FrameworkAdapter
) -> Any:
    """
    Create a native framework learning rate schedule.
    
    Args:
        config: Scheduler configuration
        base_lr: Base learning rate
        num_training_steps: Total number of training steps
        framework: Framework adapter
        
    Returns:
        Framework-specific schedule object
    """
    factory_key = framework.name.lower()
    factory = _schedule_factories.get(factory_key)
    
    if factory is None:
        logger.warning(
            f"No native LR schedule factory for framework: {framework.name}. "
            f"Use create_lr_scheduler for a framework-agnostic scheduler."
        )
        return None
    
    return factory.create_schedule(config, base_lr, num_training_steps)