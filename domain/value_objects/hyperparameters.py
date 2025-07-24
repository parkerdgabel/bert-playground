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


from enum import Enum


class OptimizerType(Enum):
    """Types of optimizers."""
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    LAMB = "lamb"
    LION = "lion"


class LearningRateSchedule(Enum):
    """Learning rate schedule types."""
    CONSTANT = "constant"
    LINEAR = "linear"
    COSINE = "cosine"
    POLYNOMIAL = "polynomial"
    EXPONENTIAL = "exponential"
    WARMUP_CONSTANT = "warmup_constant"
    WARMUP_LINEAR = "warmup_linear"
    WARMUP_COSINE = "warmup_cosine"


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """Early stopping configuration."""
    patience: Optional[int] = None
    min_delta: float = 0.0
    monitor_metric: str = "loss"
    mode: str = "min"  # "min" or "max"
    
    def __post_init__(self):
        if self.patience is not None and self.patience < 1:
            raise ValueError("Early stopping patience must be at least 1")
        if self.min_delta < 0:
            raise ValueError("Min delta must be non-negative")
        if self.mode not in ["min", "max"]:
            raise ValueError("Mode must be 'min' or 'max'")
    
    @property
    def is_enabled(self) -> bool:
        """Check if early stopping is enabled."""
        return self.patience is not None
    
    def is_improvement(self, current: float, best: float) -> bool:
        """Check if current value is an improvement over best."""
        if self.mode == "min":
            return current < best - self.min_delta
        else:
            return current > best + self.min_delta


@dataclass(frozen=True)
class Hyperparameters:
    """Complete hyperparameters for training."""
    # Basic training parameters
    num_epochs: int
    batch_size: int
    learning_rate: float
    
    # Optimizer configuration
    optimizer_type: OptimizerType = OptimizerType.ADAMW
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # Learning rate schedule
    lr_schedule: LearningRateSchedule = LearningRateSchedule.WARMUP_LINEAR
    warmup_steps: Optional[int] = None
    warmup_ratio: float = 0.1
    
    # Training dynamics
    gradient_accumulation_steps: int = 1
    gradient_clipping_max_norm: Optional[float] = 1.0
    
    # Regularization
    dropout: float = 0.1
    label_smoothing: float = 0.0
    
    # Evaluation and checkpointing
    evaluation_strategy: str = "epoch"  # "epoch", "steps", "no"
    eval_steps: Optional[int] = None
    save_strategy: str = "epoch"  # "epoch", "steps", "best"
    save_steps: Optional[int] = None
    
    # Early stopping
    early_stopping_patience: Optional[int] = None
    early_stopping_min_delta: float = 0.0
    
    # Advanced options
    mixed_precision: bool = False
    gradient_checkpointing: bool = False
    compile_model: bool = True
    seed: int = 42
    
    def __post_init__(self):
        """Validate hyperparameters."""
        if self.num_epochs <= 0:
            raise ValueError("Number of epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        
        # Validate evaluation strategy
        if self.evaluation_strategy == "steps" and self.eval_steps is None:
            raise ValueError("eval_steps must be specified when evaluation_strategy='steps'")
        if self.save_strategy == "steps" and self.save_steps is None:
            raise ValueError("save_steps must be specified when save_strategy='steps'")
    
    @property
    def effective_batch_size(self) -> int:
        """Get effective batch size with gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps
    
    def get_warmup_steps(self, total_steps: int) -> int:
        """Calculate warmup steps."""
        if self.warmup_steps is not None:
            return min(self.warmup_steps, total_steps)
        return int(total_steps * self.warmup_ratio)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "optimizer_type": self.optimizer_type.value,
            "weight_decay": self.weight_decay,
            "lr_schedule": self.lr_schedule.value,
            "warmup_steps": self.warmup_steps,
            "warmup_ratio": self.warmup_ratio,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "gradient_clipping_max_norm": self.gradient_clipping_max_norm,
            "dropout": self.dropout,
            "label_smoothing": self.label_smoothing,
            "evaluation_strategy": self.evaluation_strategy,
            "eval_steps": self.eval_steps,
            "save_strategy": self.save_strategy,
            "save_steps": self.save_steps,
            "early_stopping_patience": self.early_stopping_patience,
            "mixed_precision": self.mixed_precision,
            "gradient_checkpointing": self.gradient_checkpointing,
            "compile_model": self.compile_model,
            "seed": self.seed
        }