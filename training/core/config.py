"""
Trainer configuration with comprehensive validation and YAML/JSON support.
Following the same patterns as data/core modules.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import json
import yaml
from loguru import logger


class OptimizerType(Enum):
    """Supported optimizer types."""
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    LION = "lion"
    ADAFACTOR = "adafactor"


class SchedulerType(Enum):
    """Supported learning rate scheduler types."""
    NONE = "none"
    CONSTANT = "constant"
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    EXPONENTIAL = "exponential"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"


class EvalStrategy(Enum):
    """Evaluation strategy during training."""
    STEPS = "steps"
    EPOCH = "epoch"
    NO = "no"


class LogLevel(Enum):
    """Logging levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class CheckpointStrategy(Enum):
    """Checkpoint saving strategy."""
    STEPS = "steps"
    EPOCH = "epoch"
    BEST = "best"
    ALL = "all"


@dataclass
class OptimizerConfig:
    """Configuration for optimizer."""
    
    type: OptimizerType = OptimizerType.ADAMW
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    
    # SGD specific
    momentum: float = 0.9
    nesterov: bool = False
    
    # Lion specific
    lion_beta1: float = 0.9
    lion_beta2: float = 0.99
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    gradient_clip_val: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        config = asdict(self)
        config["type"] = self.type.value
        return config
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "OptimizerConfig":
        """Create from dictionary."""
        config = config.copy()
        if "type" in config:
            config["type"] = OptimizerType(config["type"])
        return cls(**config)


@dataclass
class SchedulerConfig:
    """Configuration for learning rate scheduler."""
    
    type: Union[SchedulerType, str] = SchedulerType.COSINE
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    
    # Linear schedule
    num_training_steps: Optional[int] = None
    
    # Cosine schedule
    num_cycles: float = 0.5
    
    # Cosine with restarts
    num_restarts: int = 1
    
    # Polynomial
    power: float = 1.0
    
    # Exponential
    gamma: float = 0.95
    
    def __post_init__(self):
        """Convert string type to enum if needed."""
        if isinstance(self.type, str):
            try:
                self.type = SchedulerType(self.type)
            except ValueError:
                # Leave as string to be caught later in create_lr_scheduler
                pass
    
    # Reduce on plateau
    patience: int = 5
    factor: float = 0.5
    min_lr: float = 1e-7
    mode: str = "min"  # "min" or "max"
    threshold: float = 0.01
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        config = asdict(self)
        config["type"] = self.type.value
        return config
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "SchedulerConfig":
        """Create from dictionary."""
        config = config.copy()
        if "type" in config:
            config["type"] = SchedulerType(config["type"])
        return cls(**config)


@dataclass
class DataConfig:
    """Configuration for data loading."""
    
    batch_size: int = 32
    eval_batch_size: Optional[int] = None
    num_workers: int = 4
    prefetch_size: int = 2
    shuffle_train: bool = True
    drop_last: bool = False
    pin_memory: bool = True
    
    # Data augmentation
    augment_train: bool = False
    augmentation_prob: float = 0.5
    
    def __post_init__(self):
        if self.eval_batch_size is None:
            self.eval_batch_size = self.batch_size
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "DataConfig":
        """Create from dictionary."""
        return cls(**config)


@dataclass
class TrainingConfig:
    """Configuration for training loop."""
    
    num_epochs: int = 10
    max_steps: int = -1
    gradient_accumulation_steps: int = 1
    
    # Mixed precision
    mixed_precision: bool = True
    
    # Evaluation
    eval_strategy: EvalStrategy = EvalStrategy.EPOCH
    eval_steps: int = 500
    eval_delay: int = 0
    
    # Logging
    logging_steps: int = 100
    log_level: LogLevel = LogLevel.INFO
    report_to: List[str] = field(default_factory=lambda: ["mlflow"])
    
    # Checkpointing
    save_strategy: CheckpointStrategy = CheckpointStrategy.EPOCH
    save_steps: int = 500
    save_total_limit: Optional[int] = 3
    save_best_only: bool = False
    best_metric: str = "eval_loss"
    best_metric_mode: str = "min"  # "min" or "max"
    
    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.0001
    
    # Regularization
    label_smoothing: float = 0.0
    dropout_rate: float = 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        config = asdict(self)
        # Handle enum fields - check if they're already strings or enum objects
        config["eval_strategy"] = self.eval_strategy.value if hasattr(self.eval_strategy, 'value') else self.eval_strategy
        config["save_strategy"] = self.save_strategy.value if hasattr(self.save_strategy, 'value') else self.save_strategy
        config["log_level"] = self.log_level.value if hasattr(self.log_level, 'value') else self.log_level
        return config
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "TrainingConfig":
        """Create from dictionary."""
        config = config.copy()
        if "eval_strategy" in config:
            config["eval_strategy"] = EvalStrategy(config["eval_strategy"])
        if "save_strategy" in config:
            config["save_strategy"] = CheckpointStrategy(config["save_strategy"])
        if "log_level" in config:
            config["log_level"] = LogLevel(config["log_level"])
        return cls(**config)


@dataclass
class EnvironmentConfig:
    """Configuration for training environment."""
    
    output_dir: Path = Path("output")
    experiment_name: Optional[str] = None
    run_name: Optional[str] = None
    seed: int = 42
    
    # Device settings
    device: str = "gpu"  # MLX uses unified memory
    
    # Memory optimization
    gradient_checkpointing: bool = False
    memory_efficient_attention: bool = True
    
    # MLflow settings
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment_id: Optional[str] = None
    mlflow_tags: Dict[str, str] = field(default_factory=dict)
    
    # Debugging
    debug_mode: bool = False
    profile: bool = False
    
    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        config = asdict(self)
        config["output_dir"] = str(self.output_dir)
        return config
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "EnvironmentConfig":
        """Create from dictionary."""
        config = config.copy()
        if "output_dir" in config:
            config["output_dir"] = Path(config["output_dir"])
        return cls(**config)


@dataclass
class BaseTrainerConfig:
    """Base configuration for trainers combining all sub-configs."""
    
    # Sub-configurations
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    
    # Additional custom configs can be added by subclasses
    custom: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Ensure all sub-configs are properly typed
        if isinstance(self.optimizer, dict):
            self.optimizer = OptimizerConfig.from_dict(self.optimizer)
        if isinstance(self.scheduler, dict):
            self.scheduler = SchedulerConfig.from_dict(self.scheduler)
        if isinstance(self.data, dict):
            self.data = DataConfig.from_dict(self.data)
        if isinstance(self.training, dict):
            self.training = TrainingConfig.from_dict(self.training)
        if isinstance(self.environment, dict):
            self.environment = EnvironmentConfig.from_dict(self.environment)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "optimizer": self.optimizer.to_dict(),
            "scheduler": self.scheduler.to_dict(),
            "data": self.data.to_dict(),
            "training": self.training.to_dict(),
            "environment": self.environment.to_dict(),
            "custom": self.custom,
        }
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "BaseTrainerConfig":
        """Create from dictionary."""
        return cls(
            optimizer=OptimizerConfig.from_dict(config.get("optimizer", {})),
            scheduler=SchedulerConfig.from_dict(config.get("scheduler", {})),
            data=DataConfig.from_dict(config.get("data", {})),
            training=TrainingConfig.from_dict(config.get("training", {})),
            environment=EnvironmentConfig.from_dict(config.get("environment", {})),
            custom=config.get("custom", {}),
        )
    
    def save(self, path: Path) -> None:
        """Save configuration to file."""
        path = Path(path)
        config_dict = self.to_dict()
        
        if path.suffix == ".yaml" or path.suffix == ".yml":
            with open(path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            with open(path, "w") as f:
                json.dump(config_dict, f, indent=2)
        
        logger.info(f"Saved configuration to {path}")
    
    @classmethod
    def load(cls, path: Path) -> "BaseTrainerConfig":
        """Load configuration from file."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        if path.suffix == ".yaml" or path.suffix == ".yml":
            with open(path) as f:
                config_dict = yaml.safe_load(f)
        else:
            with open(path) as f:
                config_dict = json.load(f)
        
        logger.info(f"Loaded configuration from {path}")
        return cls.from_dict(config_dict)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Validate optimizer config
        if self.optimizer.learning_rate <= 0:
            errors.append("Learning rate must be positive")
        if self.optimizer.weight_decay < 0:
            errors.append("Weight decay must be non-negative")
        
        # Validate scheduler config
        if self.scheduler.warmup_ratio < 0 or self.scheduler.warmup_ratio > 1:
            errors.append("Warmup ratio must be between 0 and 1")
        
        # Validate data config
        if self.data.batch_size <= 0:
            errors.append("Batch size must be positive")
        if self.data.num_workers < 0:
            errors.append("Number of workers must be non-negative")
        
        # Validate training config
        if self.training.num_epochs <= 0 and self.training.max_steps <= 0:
            errors.append("Either num_epochs or max_steps must be positive")
        if self.training.gradient_accumulation_steps <= 0:
            errors.append("Gradient accumulation steps must be positive")
        
        # Validate paths
        if not self.environment.output_dir.parent.exists():
            errors.append(f"Parent directory of output_dir does not exist: {self.environment.output_dir.parent}")
        
        return errors
    
    @property
    def effective_batch_size(self) -> int:
        """Calculate effective batch size including gradient accumulation."""
        return self.data.batch_size * self.training.gradient_accumulation_steps
    
    @property
    def num_epochs(self) -> int:
        """Convenience property for number of epochs."""
        return self.training.num_epochs
    
    @property
    def learning_rate(self) -> float:
        """Convenience property for learning rate."""
        return self.optimizer.learning_rate
    
    @property
    def batch_size(self) -> int:
        """Convenience property for batch size."""
        return self.data.batch_size
    
    @property
    def gradient_accumulation_steps(self) -> int:
        """Convenience property for gradient accumulation steps."""
        return self.training.gradient_accumulation_steps
    
    @property
    def eval_steps(self) -> int:
        """Convenience property for evaluation steps."""
        return self.training.eval_steps
    
    @property
    def save_steps(self) -> int:
        """Convenience property for save steps."""
        return self.training.save_steps
    
    @property
    def output_dir(self) -> Path:
        """Convenience property for output directory."""
        return self.environment.output_dir


# Preset configurations for common scenarios
def get_quick_test_config() -> BaseTrainerConfig:
    """Get configuration for quick testing."""
    return BaseTrainerConfig(
        optimizer=OptimizerConfig(
            type=OptimizerType.ADAMW,
            learning_rate=5e-5,
        ),
        scheduler=SchedulerConfig(
            type=SchedulerType.NONE,
        ),
        data=DataConfig(
            batch_size=8,
            num_workers=2,
        ),
        training=TrainingConfig(
            num_epochs=1,
            eval_strategy=EvalStrategy.NO,
            save_strategy=CheckpointStrategy.EPOCH,
            logging_steps=10,
            early_stopping=False,
            save_best_only=False,
        ),
        environment=EnvironmentConfig(
            output_dir=Path("output/test"),
        ),
    )


def get_development_config() -> BaseTrainerConfig:
    """Get configuration for development."""
    return BaseTrainerConfig(
        optimizer=OptimizerConfig(
            type=OptimizerType.ADAMW,
            learning_rate=2e-5,
        ),
        scheduler=SchedulerConfig(
            type=SchedulerType.LINEAR,
            warmup_ratio=0.1,
        ),
        data=DataConfig(
            batch_size=16,
            num_workers=4,
        ),
        training=TrainingConfig(
            num_epochs=5,
            eval_strategy=EvalStrategy.EPOCH,
            save_strategy=CheckpointStrategy.EPOCH,
            early_stopping=True,
            early_stopping_patience=2,
        ),
        environment=EnvironmentConfig(
            output_dir=Path("output/dev"),
            debug_mode=True,
        ),
    )


def get_production_config() -> BaseTrainerConfig:
    """Get configuration for production training."""
    return BaseTrainerConfig(
        optimizer=OptimizerConfig(
            type=OptimizerType.ADAMW,
            learning_rate=2e-5,
            weight_decay=0.01,
        ),
        scheduler=SchedulerConfig(
            type=SchedulerType.COSINE,
            warmup_ratio=0.05,
        ),
        data=DataConfig(
            batch_size=32,
            num_workers=8,
            prefetch_size=4,
        ),
        training=TrainingConfig(
            num_epochs=10,
            eval_strategy=EvalStrategy.STEPS,
            eval_steps=500,
            save_strategy=CheckpointStrategy.BEST,
            save_best_only=True,
            early_stopping=True,
            early_stopping_patience=3,
            mixed_precision=True,
        ),
        environment=EnvironmentConfig(
            output_dir=Path("output/production"),
            gradient_checkpointing=True,
            memory_efficient_attention=True,
        ),
    )


def get_kaggle_competition_config() -> BaseTrainerConfig:
    """Get configuration optimized for Kaggle competitions."""
    return BaseTrainerConfig(
        optimizer=OptimizerConfig(
            type=OptimizerType.ADAMW,
            learning_rate=1e-5,
            weight_decay=0.01,
            max_grad_norm=1.0,
        ),
        scheduler=SchedulerConfig(
            type=SchedulerType.COSINE_WITH_RESTARTS,
            warmup_ratio=0.1,
            num_restarts=2,
        ),
        data=DataConfig(
            batch_size=64,
            num_workers=8,
            prefetch_size=8,
            augment_train=True,
            augmentation_prob=0.5,
        ),
        training=TrainingConfig(
            num_epochs=15,
            gradient_accumulation_steps=2,
            eval_strategy=EvalStrategy.STEPS,
            eval_steps=250,
            save_strategy=CheckpointStrategy.BEST,
            save_best_only=True,
            best_metric="eval_auc",
            best_metric_mode="max",
            early_stopping=True,
            early_stopping_patience=5,
            label_smoothing=0.1,
            mixed_precision=True,
            report_to=["mlflow", "tensorboard"],
        ),
        environment=EnvironmentConfig(
            output_dir=Path("output/kaggle"),
            gradient_checkpointing=True,
            memory_efficient_attention=True,
            mlflow_tags={"competition": "kaggle", "framework": "mlx"},
        ),
    )