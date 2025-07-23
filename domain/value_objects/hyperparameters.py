"""Hyperparameter value objects.

These represent training hyperparameters as immutable value objects
with validation and business logic.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LearningRate:
    """Learning rate value object with validation."""
    value: float
    
    def __post_init__(self):
        if self.value <= 0:
            raise ValueError("Learning rate must be positive")
        if self.value > 1.0:
            raise ValueError("Learning rate should typically be less than 1.0")
    
    def with_warmup(self, current_step: int, warmup_steps: int) -> float:
        """Calculate learning rate with linear warmup."""
        if current_step < warmup_steps:
            return self.value * (current_step / warmup_steps)
        return self.value
    
    def with_decay(self, decay_factor: float) -> 'LearningRate':
        """Create new learning rate with decay applied."""
        return LearningRate(self.value * decay_factor)


@dataclass(frozen=True)
class BatchSize:
    """Batch size value object with validation."""
    value: int
    gradient_accumulation_steps: int = 1
    
    def __post_init__(self):
        if self.value <= 0:
            raise ValueError("Batch size must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("Gradient accumulation steps must be positive")
    
    @property
    def effective_batch_size(self) -> int:
        """Get effective batch size after gradient accumulation."""
        return self.value * self.gradient_accumulation_steps
    
    def is_divisible_by(self, dataset_size: int) -> bool:
        """Check if dataset size is divisible by batch size."""
        return dataset_size % self.value == 0


@dataclass(frozen=True)
class Epochs:
    """Training epochs value object."""
    value: int
    
    def __post_init__(self):
        if self.value <= 0:
            raise ValueError("Number of epochs must be positive")
    
    def to_steps(self, steps_per_epoch: int) -> int:
        """Convert epochs to total training steps."""
        return self.value * steps_per_epoch
    
    def progress(self, current_epoch: int) -> float:
        """Calculate training progress as percentage."""
        return min(current_epoch / self.value, 1.0)


@dataclass(frozen=True)
class WarmupSteps:
    """Warmup steps value object."""
    value: Optional[int] = None
    ratio: Optional[float] = None
    
    def __post_init__(self):
        if self.value is not None and self.ratio is not None:
            raise ValueError("Cannot specify both warmup steps and ratio")
        if self.value is not None and self.value < 0:
            raise ValueError("Warmup steps must be non-negative")
        if self.ratio is not None and not (0 <= self.ratio <= 1):
            raise ValueError("Warmup ratio must be between 0 and 1")
    
    def calculate_steps(self, total_steps: int) -> int:
        """Calculate actual warmup steps."""
        if self.value is not None:
            return min(self.value, total_steps)
        elif self.ratio is not None:
            return int(total_steps * self.ratio)
        else:
            return 0
    
    @property
    def is_enabled(self) -> bool:
        """Check if warmup is enabled."""
        return self.value is not None or self.ratio is not None


@dataclass(frozen=True)
class GradientClipping:
    """Gradient clipping value object."""
    max_norm: Optional[float] = None
    max_value: Optional[float] = None
    
    def __post_init__(self):
        if self.max_norm is not None and self.max_norm <= 0:
            raise ValueError("Max gradient norm must be positive")
        if self.max_value is not None and self.max_value <= 0:
            raise ValueError("Max gradient value must be positive")
    
    @property
    def is_enabled(self) -> bool:
        """Check if gradient clipping is enabled."""
        return self.max_norm is not None or self.max_value is not None
    
    @property
    def clip_type(self) -> Optional[str]:
        """Get type of clipping."""
        if self.max_norm is not None:
            return "norm"
        elif self.max_value is not None:
            return "value"
        return None


@dataclass(frozen=True)
class WeightDecay:
    """Weight decay (L2 regularization) value object."""
    value: float = 0.0
    exclude_bias: bool = True
    exclude_layer_norm: bool = True
    
    def __post_init__(self):
        if self.value < 0:
            raise ValueError("Weight decay must be non-negative")
        if self.value > 1:
            raise ValueError("Weight decay should typically be less than 1")
    
    @property
    def is_enabled(self) -> bool:
        """Check if weight decay is enabled."""
        return self.value > 0
    
    def should_decay(self, param_name: str) -> bool:
        """Check if parameter should have weight decay applied."""
        if not self.is_enabled:
            return False
        
        # Exclude bias terms
        if self.exclude_bias and "bias" in param_name:
            return False
        
        # Exclude layer norm parameters
        if self.exclude_layer_norm and ("layer_norm" in param_name or "ln" in param_name):
            return False
        
        return True


@dataclass(frozen=True)
class Dropout:
    """Dropout probability value object."""
    value: float
    
    def __post_init__(self):
        if not (0 <= self.value < 1):
            raise ValueError("Dropout probability must be in [0, 1)")
    
    @property
    def keep_prob(self) -> float:
        """Get keep probability (1 - dropout)."""
        return 1.0 - self.value
    
    @property
    def is_enabled(self) -> bool:
        """Check if dropout is enabled."""
        return self.value > 0
    
    def scale_factor(self) -> float:
        """Get scaling factor for dropout compensation."""
        return 1.0 / self.keep_prob if self.is_enabled else 1.0