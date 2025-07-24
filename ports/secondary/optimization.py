"""Secondary optimization port - Optimization services that the application depends on.

This port defines the optimization interface that the application core uses
for model optimization. It's a driven port implemented by adapters for
different optimization backends.
"""

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable, Callable, Optional

from infrastructure.di import port
from .compute import Array, Module


@dataclass
class OptimizerConfig:
    """Configuration for optimizers."""
    
    learning_rate: float
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    amsgrad: bool = False
    momentum: float = 0.0
    nesterov: bool = False
    centered: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "amsgrad": self.amsgrad,
            "momentum": self.momentum,
            "nesterov": self.nesterov,
            "centered": self.centered,
        }


@dataclass
class SchedulerConfig:
    """Configuration for learning rate schedulers."""
    
    scheduler_type: str = "constant"
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    num_training_steps: Optional[int] = None
    num_cycles: float = 0.5
    power: float = 1.0
    min_lr: float = 0.0
    max_lr: Optional[float] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scheduler_type": self.scheduler_type,
            "warmup_steps": self.warmup_steps,
            "warmup_ratio": self.warmup_ratio,
            "num_training_steps": self.num_training_steps,
            "num_cycles": self.num_cycles,
            "power": self.power,
            "min_lr": self.min_lr,
            "max_lr": self.max_lr,
        }


@port()
@runtime_checkable
class Optimizer(Protocol):
    """Secondary port for optimization operations.
    
    This interface is implemented by adapters for specific optimization
    backends. The application core depends on this for parameter updates.
    """

    def update(self, model: Module, gradients: dict[str, Array]) -> None:
        """Update model parameters with gradients.
        
        Args:
            model: Model to update
            gradients: Gradients for each parameter
        """
        ...

    @property
    def learning_rate(self) -> float:
        """Current learning rate."""
        ...

    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        """Set learning rate."""
        ...

    @property
    def state(self) -> dict[str, Any]:
        """Optimizer state (momentum buffers, etc.)."""
        ...

    def state_dict(self) -> dict[str, Any]:
        """Get optimizer state dictionary.
        
        Returns:
            State dictionary for serialization
        """
        ...

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load optimizer state.
        
        Args:
            state_dict: State dictionary to load
        """
        ...

    def zero_grad(self) -> None:
        """Zero out gradients."""
        ...

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Perform optimization step.
        
        Args:
            closure: Optional closure for computing loss
            
        Returns:
            Loss value if closure provided
        """
        ...

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        """Add a parameter group with its own settings.
        
        Args:
            param_group: Parameter group configuration
        """
        ...

    def get_param_groups(self) -> list[dict[str, Any]]:
        """Get all parameter groups.
        
        Returns:
            List of parameter groups
        """
        ...


@port()
@runtime_checkable
class LRScheduler(Protocol):
    """Secondary port for learning rate scheduling.
    
    This interface is implemented by adapters for different scheduling
    strategies. The application core depends on this for LR scheduling.
    """

    def step(self, metrics: Optional[dict[str, float]] = None) -> float:
        """Update learning rate and return new value.
        
        Args:
            metrics: Optional metrics for schedulers that use them
            
        Returns:
            New learning rate
        """
        ...

    @property
    def current_lr(self) -> float:
        """Current learning rate."""
        ...

    def state_dict(self) -> dict[str, Any]:
        """Get scheduler state.
        
        Returns:
            State dictionary
        """
        ...

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load scheduler state.
        
        Args:
            state: State dictionary to load
        """
        ...

    def get_last_lr(self) -> list[float]:
        """Get learning rates from last step.
        
        Returns:
            List of learning rates (one per param group)
        """
        ...

    def get_lr(self) -> list[float]:
        """Compute learning rates for current step.
        
        Returns:
            List of learning rates (one per param group)
        """
        ...

    def print_lr(self, is_verbose: bool = False) -> None:
        """Print current learning rates.
        
        Args:
            is_verbose: Whether to print verbose info
        """
        ...