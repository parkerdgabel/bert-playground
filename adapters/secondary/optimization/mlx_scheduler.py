"""MLX implementation of LRScheduler port."""

from typing import Any, Optional
import math

from infrastructure.di import adapter, Scope
from ports.secondary.optimization import LRScheduler, SchedulerConfig


@adapter(LRScheduler, scope=Scope.SINGLETON)
class MLXSchedulerAdapter:
    """MLX implementation of the LRScheduler port."""
    
    def __init__(
        self,
        optimizer: "MLXOptimizerAdapter",
        config: SchedulerConfig,
    ):
        """Initialize MLX scheduler adapter.
        
        Args:
            optimizer: The optimizer to schedule
            config: Scheduler configuration
        """
        self.optimizer = optimizer
        self.config = config
        self._step_count = 0
        self._base_lr = optimizer.learning_rate
        self._current_lr = self._base_lr
        self._last_lr = [self._base_lr]
        
        # Calculate warmup steps
        if config.warmup_ratio > 0 and config.num_training_steps:
            self.warmup_steps = int(config.warmup_ratio * config.num_training_steps)
        else:
            self.warmup_steps = config.warmup_steps
    
    def step(self, metrics: Optional[dict[str, float]] = None) -> float:
        """Update learning rate and return new value.
        
        Args:
            metrics: Optional metrics for schedulers that use them
            
        Returns:
            New learning rate
        """
        self._step_count += 1
        
        # Calculate new learning rate based on scheduler type
        if self.config.scheduler_type == "constant":
            new_lr = self._base_lr
        elif self.config.scheduler_type == "linear":
            new_lr = self._linear_schedule()
        elif self.config.scheduler_type == "cosine":
            new_lr = self._cosine_schedule()
        elif self.config.scheduler_type == "cosine_with_restarts":
            new_lr = self._cosine_with_restarts_schedule()
        elif self.config.scheduler_type == "polynomial":
            new_lr = self._polynomial_schedule()
        elif self.config.scheduler_type == "exponential":
            new_lr = self._exponential_schedule()
        elif self.config.scheduler_type == "reduce_on_plateau":
            new_lr = self._reduce_on_plateau_schedule(metrics)
        else:
            raise ValueError(f"Unknown scheduler type: {self.config.scheduler_type}")
        
        # Apply warmup if in warmup phase
        if self._step_count <= self.warmup_steps and self.warmup_steps > 0:
            warmup_factor = self._step_count / self.warmup_steps
            new_lr = self._base_lr * warmup_factor
        
        # Apply min_lr constraint
        new_lr = max(new_lr, self.config.min_lr)
        
        # Update optimizer learning rate
        self.optimizer.learning_rate = new_lr
        self._current_lr = new_lr
        self._last_lr = [new_lr]
        
        return new_lr
    
    @property
    def current_lr(self) -> float:
        """Current learning rate."""
        return self._current_lr
    
    def state_dict(self) -> dict[str, Any]:
        """Get scheduler state.
        
        Returns:
            State dictionary
        """
        return {
            "step_count": self._step_count,
            "base_lr": self._base_lr,
            "current_lr": self._current_lr,
            "last_lr": self._last_lr,
            "config": self.config.to_dict(),
        }
    
    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load scheduler state.
        
        Args:
            state: State dictionary to load
        """
        self._step_count = state.get("step_count", 0)
        self._base_lr = state.get("base_lr", self._base_lr)
        self._current_lr = state.get("current_lr", self._current_lr)
        self._last_lr = state.get("last_lr", [self._current_lr])
        
        if "config" in state:
            self.config = SchedulerConfig(**state["config"])
    
    def get_last_lr(self) -> list[float]:
        """Get learning rates from last step.
        
        Returns:
            List of learning rates (one per param group)
        """
        return self._last_lr.copy()
    
    def get_lr(self) -> list[float]:
        """Compute learning rates for current step.
        
        Returns:
            List of learning rates (one per param group)
        """
        # For now, we use a single learning rate for all param groups
        return [self._current_lr]
    
    def print_lr(self, is_verbose: bool = False) -> None:
        """Print current learning rates.
        
        Args:
            is_verbose: Whether to print verbose info
        """
        if is_verbose:
            print(f"Step {self._step_count}: Learning rate = {self._current_lr:.6e}")
        else:
            print(f"LR: {self._current_lr:.6e}")
    
    # Schedule implementations
    
    def _linear_schedule(self) -> float:
        """Linear decay schedule."""
        if not self.config.num_training_steps:
            return self._base_lr
        
        # Skip warmup steps in decay calculation
        effective_step = max(0, self._step_count - self.warmup_steps)
        effective_total = self.config.num_training_steps - self.warmup_steps
        
        if effective_total <= 0:
            return self._base_lr
        
        progress = min(effective_step / effective_total, 1.0)
        return self._base_lr * (1.0 - progress)
    
    def _cosine_schedule(self) -> float:
        """Cosine annealing schedule."""
        if not self.config.num_training_steps:
            return self._base_lr
        
        # Skip warmup steps in decay calculation
        effective_step = max(0, self._step_count - self.warmup_steps)
        effective_total = self.config.num_training_steps - self.warmup_steps
        
        if effective_total <= 0:
            return self._base_lr
        
        progress = min(effective_step / effective_total, 1.0)
        return self.config.min_lr + (self._base_lr - self.config.min_lr) * \
               0.5 * (1.0 + math.cos(math.pi * progress))
    
    def _cosine_with_restarts_schedule(self) -> float:
        """Cosine schedule with warm restarts."""
        if not self.config.num_training_steps:
            return self._base_lr
        
        # Skip warmup steps
        effective_step = max(0, self._step_count - self.warmup_steps)
        
        # Calculate cycle length
        cycle_length = int(self.config.num_training_steps * self.config.num_cycles)
        if cycle_length <= 0:
            return self._base_lr
        
        # Current position in cycle
        cycle_progress = (effective_step % cycle_length) / cycle_length
        
        return self.config.min_lr + (self._base_lr - self.config.min_lr) * \
               0.5 * (1.0 + math.cos(math.pi * cycle_progress))
    
    def _polynomial_schedule(self) -> float:
        """Polynomial decay schedule."""
        if not self.config.num_training_steps:
            return self._base_lr
        
        # Skip warmup steps
        effective_step = max(0, self._step_count - self.warmup_steps)
        effective_total = self.config.num_training_steps - self.warmup_steps
        
        if effective_total <= 0:
            return self._base_lr
        
        progress = min(effective_step / effective_total, 1.0)
        return self._base_lr * ((1.0 - progress) ** self.config.power)
    
    def _exponential_schedule(self) -> float:
        """Exponential decay schedule."""
        # Decay rate calculated to reach min_lr at the end of training
        if not self.config.num_training_steps or self._base_lr <= self.config.min_lr:
            return self._base_lr
        
        decay_rate = (self.config.min_lr / self._base_lr) ** (1.0 / self.config.num_training_steps)
        return self._base_lr * (decay_rate ** self._step_count)
    
    def _reduce_on_plateau_schedule(self, metrics: Optional[dict[str, float]]) -> float:
        """Reduce learning rate when metric plateaus."""
        # This is a simplified version - full implementation would track
        # metric history and patience
        return self._current_lr
    
    def set_lr(self, lr: float) -> None:
        """Manually set learning rate.
        
        Args:
            lr: New learning rate
        """
        self._current_lr = lr
        self.optimizer.learning_rate = lr
        self._last_lr = [lr]