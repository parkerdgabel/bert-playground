"""Training configuration system for next-generation MLX trainer.

This module provides comprehensive configuration management for the unified
MLX training system, supporting all features from the previous implementations
plus new advanced capabilities.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union

from loguru import logger

# Import the new config loader
from utils.config_loader import ConfigLoader


class OptimizationLevel(Enum):
    """Training optimization levels."""

    DEVELOPMENT = "development"  # Fast iteration, minimal optimization
    STANDARD = "standard"  # Balanced performance and features
    PRODUCTION = "production"  # Maximum performance optimization
    COMPETITION = "competition"  # Ultra-optimized for competitions
    AUTO = "auto"  # Automatically choose based on dataset


class LearningRateSchedule(Enum):
    """Learning rate scheduling strategies."""

    CONSTANT = "constant"
    LINEAR_WARMUP = "linear_warmup"
    COSINE = "cosine"
    COSINE_WARMUP = "cosine_warmup"
    POLYNOMIAL = "polynomial"
    EXPONENTIAL = "exponential"
    PLATEAU = "plateau"
    CUSTOM = "custom"


class OptimizerType(Enum):
    """Supported optimizers."""

    ADAMW = "adamw"
    ADAM = "adam"
    SGD = "sgd"
    RMSPROP = "rmsprop"
    ADAGRAD = "adagrad"


class LossFunction(Enum):
    """Supported loss functions."""

    CROSS_ENTROPY = "cross_entropy"
    FOCAL_LOSS = "focal_loss"
    WEIGHTED_CROSS_ENTROPY = "weighted_cross_entropy"
    LABEL_SMOOTHING = "label_smoothing"
    ADAPTIVE_LOSS = "adaptive_loss"


@dataclass
class MemoryConfig:
    """Memory management configuration."""

    # Memory optimization settings
    enable_memory_profiling: bool = True
    memory_limit_gb: float | None = None  # Auto-detect if None
    dynamic_batch_sizing: bool = True
    min_batch_size: int = 4
    max_batch_size: int = 128
    memory_check_interval: int = 100  # Steps between memory checks

    # MLX-specific memory settings
    unified_memory_fraction: float = 0.8  # Fraction of unified memory to use
    enable_memory_pool: bool = True
    force_garbage_collection: bool = True
    gc_interval: int = 500  # Steps between forced GC


@dataclass
class MLXOptimizationConfig:
    """MLX-specific optimization settings."""

    # Lazy computation management
    enable_lazy_evaluation: bool = True
    eval_frequency: int = 10  # Force eval every N steps

    # Gradient computation
    enable_gradient_checkpointing: bool = False
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Device and parallelization
    device_placement_strategy: str = "auto"  # "auto", "cpu", "gpu"
    enable_multi_device: bool = False

    # Precision settings
    mixed_precision: bool = False
    precision_dtype: str = "float32"  # "float32", "float16", "bfloat16"

    # Compilation and optimization
    enable_jit: bool = True
    optimize_memory_layout: bool = True


@dataclass
class MonitoringConfig:
    """Monitoring and logging configuration."""

    # MLflow integration
    enable_mlflow: bool = True
    experiment_name: str = "mlx_training"
    run_name: str | None = None
    tracking_uri: str | None = None

    # Logging settings
    log_level: str = "INFO"
    log_to_file: bool = True
    log_file_path: str | None = None
    enable_rich_console: bool = True

    # Metrics tracking
    log_frequency: int = 10  # Log every N steps
    eval_frequency: int = 500  # Evaluate every N steps
    save_frequency: int = 1000  # Save checkpoint every N steps

    # Progress tracking
    enable_progress_bar: bool = True
    progress_bar_style: str = "rich"  # "rich", "tqdm", "simple"

    # Metric collection
    track_gradients: bool = False
    track_weights: bool = False
    track_memory: bool = True
    track_performance: bool = True


@dataclass
class CheckpointConfig:
    """Checkpointing and state management configuration."""

    # Basic settings
    enable_checkpointing: bool = True
    checkpoint_dir: str = "checkpoints"
    checkpoint_frequency: int = 1000  # Save every N steps

    # State preservation
    save_optimizer_state: bool = True
    save_scheduler_state: bool = True
    save_random_state: bool = True
    save_model_weights: bool = True

    # Checkpoint management
    max_checkpoints_to_keep: int = 5
    save_best_model: bool = True
    best_model_metric: str = "val_accuracy"
    best_model_mode: str = "max"  # "max" or "min"

    # Resume settings
    auto_resume: bool = True
    resume_from_checkpoint: str | None = None

    # File format and compression
    use_safetensors: bool = True
    compress_checkpoints: bool = False


@dataclass
class EvaluationConfig:
    """Evaluation and validation configuration."""

    # Basic evaluation settings
    eval_during_training: bool = True
    eval_steps: int = 500
    eval_strategy: str = "steps"  # "steps", "epoch", "no"

    # Metrics configuration
    primary_metric: str = "accuracy"
    metrics_to_compute: list[str] = field(
        default_factory=lambda: ["accuracy", "precision", "recall", "f1", "auc"]
    )

    # Early stopping
    enable_early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_threshold: float = 0.001
    early_stopping_metric: str = "val_loss"
    early_stopping_mode: str = "min"  # "min" or "max"

    # Validation settings
    validation_split: float = 0.2  # If no separate validation set
    validation_batch_size: int | None = None  # Use training batch size if None

    # Test evaluation
    test_at_end: bool = True
    generate_predictions: bool = True
    save_predictions: bool = True


@dataclass
class AdvancedFeatures:
    """Advanced training features configuration."""

    # Regularization
    label_smoothing: float = 0.0
    dropout_rate: float = 0.1
    weight_decay: float = 0.01

    # Data augmentation
    enable_augmentation: bool = True
    augmentation_probability: float = 0.5

    # Curriculum learning
    enable_curriculum_learning: bool = False
    curriculum_strategy: str = "difficulty"  # "difficulty", "length", "custom"
    curriculum_pace: float = 0.1

    # Model ensembling
    enable_ensembling: bool = False
    ensemble_size: int = 1
    ensemble_strategy: str = "averaging"  # "averaging", "voting", "stacking"

    # Knowledge distillation
    enable_distillation: bool = False
    teacher_model_path: str | None = None
    distillation_temperature: float = 3.0
    distillation_alpha: float = 0.5

    # Hyperparameter optimization
    enable_hpo: bool = False
    hpo_backend: str = "optuna"  # "optuna", "ray_tune"
    hpo_trials: int = 50
    hpo_metric: str = "val_accuracy"


@dataclass
class TrainingConfig:
    """Comprehensive training configuration for MLX trainer.

    This configuration class unifies all settings from previous trainer
    implementations and adds new advanced capabilities.
    """

    # Basic training parameters
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 2e-5
    warmup_steps: int = 500
    max_steps: int | None = None

    # Model configuration
    model_name: str = "answerdotai/ModernBERT-base"
    model_type: str = "modernbert"  # "modernbert", "cnn_hybrid"
    max_length: int = 256
    num_labels: int | None = None  # Auto-detected from data

    # Data configuration
    train_path: str | None = None
    val_path: str | None = None
    test_path: str | None = None
    target_column: str | None = None

    # Optimizer and scheduler
    optimizer: OptimizerType = OptimizerType.ADAMW
    lr_schedule: LearningRateSchedule = LearningRateSchedule.COSINE_WARMUP
    loss_function: LossFunction = LossFunction.CROSS_ENTROPY

    # Optimization level
    optimization_level: OptimizationLevel = OptimizationLevel.AUTO

    # Sub-configurations
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    mlx_optimization: MLXOptimizationConfig = field(
        default_factory=MLXOptimizationConfig
    )
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    advanced: AdvancedFeatures = field(default_factory=AdvancedFeatures)

    # Runtime settings
    seed: int = 42
    deterministic: bool = True
    output_dir: str = "output"
    experiment_name: str | None = None
    run_name: str | None = None

    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Store original optimization level for later auto-configuration
        self._original_optimization_level = self.optimization_level

        # Apply optimization level settings (but keep AUTO as-is for later)
        if self.optimization_level == OptimizationLevel.AUTO:
            # Use standard as default for now, will be updated when dataset is available
            self._apply_optimization_level_settings(OptimizationLevel.STANDARD)
        else:
            self._apply_optimization_level()

        # Validate configuration
        self._validate_config()

        # Setup paths
        self._setup_paths()

    def _auto_configure_optimization(self, dataset_size: int):
        """Automatically configure optimization based on dataset characteristics."""
        logger.info("Auto-configuring optimization level...")

        if dataset_size < 5000:
            self.optimization_level = OptimizationLevel.DEVELOPMENT
        elif dataset_size < 50000:
            self.optimization_level = OptimizationLevel.STANDARD
        elif dataset_size < 500000:
            self.optimization_level = OptimizationLevel.PRODUCTION
        else:
            self.optimization_level = OptimizationLevel.COMPETITION

        self._apply_optimization_level()

    def _apply_optimization_level(self):
        """Apply optimization level specific settings."""
        self._apply_optimization_level_settings(self.optimization_level)

    def _apply_optimization_level_settings(self, level: OptimizationLevel):
        """Apply optimization level specific settings for a given level."""
        if level == OptimizationLevel.DEVELOPMENT:
            # Fast iteration settings
            self.memory.dynamic_batch_sizing = False
            self.memory.enable_memory_profiling = False
            self.mlx_optimization.eval_frequency = 5
            self.monitoring.log_frequency = 5
            self.checkpoint.checkpoint_frequency = 100

        elif level == OptimizationLevel.STANDARD:
            # Balanced settings (defaults are already good)
            pass

        elif level == OptimizationLevel.PRODUCTION:
            # Performance-focused settings
            self.memory.dynamic_batch_sizing = True
            self.memory.max_batch_size = 64
            self.mlx_optimization.enable_jit = True
            self.mlx_optimization.optimize_memory_layout = True
            self.monitoring.log_frequency = 50

        elif level == OptimizationLevel.COMPETITION:
            # Ultra-optimized settings
            self.memory.dynamic_batch_sizing = True
            self.memory.max_batch_size = 128
            self.mlx_optimization.enable_jit = True
            self.mlx_optimization.mixed_precision = True
            self.mlx_optimization.gradient_accumulation_steps = 2
            self.advanced.enable_ensembling = True
            self.advanced.ensemble_size = 3

    def _validate_config(self):
        """Validate configuration parameters."""
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        if self.memory.min_batch_size > self.memory.max_batch_size:
            raise ValueError("min_batch_size cannot be greater than max_batch_size")

        if self.max_length <= 0:
            raise ValueError("max_length must be positive")

        logger.info("Configuration validation passed")

    def _setup_paths(self):
        """Setup and create necessary directories."""
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Setup checkpoint directory
        checkpoint_path = Path(self.checkpoint.checkpoint_dir)
        if not checkpoint_path.is_absolute():
            checkpoint_path = output_path / checkpoint_path
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.checkpoint.checkpoint_dir = str(checkpoint_path)

        # Setup log file path if not specified
        if self.monitoring.log_to_file and self.monitoring.log_file_path is None:
            self.monitoring.log_file_path = str(output_path / "training.log")

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary for serialization."""

        def convert_value(value):
            if isinstance(value, Enum):
                return value.value
            elif hasattr(value, "to_dict"):
                return value.to_dict()
            elif hasattr(value, "__dict__"):
                return {k: convert_value(v) for k, v in value.__dict__.items()}
            elif isinstance(value, list | tuple):
                return [convert_value(item) for item in value]
            else:
                return value

        # Exclude internal attributes from serialization
        return {
            key: convert_value(value)
            for key, value in self.__dict__.items()
            if not key.startswith("_")
        }

    def save(self, config_path: str | Path) -> None:
        """Save configuration to file."""
        config_path = Path(config_path)
        config_dict = self.to_dict()

        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)

        logger.info(f"Configuration saved to: {config_path}")

    @classmethod
    def load(cls, config_path: str | Path) -> "TrainingConfig":
        """Load configuration from file (auto-detects format)."""
        config_path = Path(config_path)
        
        # Auto-detect format based on extension
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            return cls.load_yaml(config_path)
        elif config_path.suffix.lower() == '.json':
            with open(config_path) as f:
                config_dict = json.load(f)
        else:
            # Try JSON by default for backward compatibility
            try:
                with open(config_path) as f:
                    config_dict = json.load(f)
            except json.JSONDecodeError:
                raise ValueError(
                    f"Could not load configuration from {config_path}. "
                    "Please use .json or .yaml/.yml extension."
                )

        # Convert enum strings back to enums
        def restore_enums(data: dict[str, Any]) -> dict[str, Any]:
            enum_mappings = {
                "optimization_level": OptimizationLevel,
                "optimizer": OptimizerType,
                "lr_schedule": LearningRateSchedule,
                "loss_function": LossFunction,
            }

            for key, enum_class in enum_mappings.items():
                if key in data and isinstance(data[key], str):
                    data[key] = enum_class(data[key])

            return data

        config_dict = restore_enums(config_dict)

        # Reconstruct sub-configurations
        if "memory" in config_dict:
            config_dict["memory"] = MemoryConfig(**config_dict["memory"])
        if "mlx_optimization" in config_dict:
            config_dict["mlx_optimization"] = MLXOptimizationConfig(
                **config_dict["mlx_optimization"]
            )
        if "monitoring" in config_dict:
            config_dict["monitoring"] = MonitoringConfig(**config_dict["monitoring"])
        if "checkpoint" in config_dict:
            config_dict["checkpoint"] = CheckpointConfig(**config_dict["checkpoint"])
        if "evaluation" in config_dict:
            config_dict["evaluation"] = EvaluationConfig(**config_dict["evaluation"])
        if "advanced" in config_dict:
            config_dict["advanced"] = AdvancedFeatures(**config_dict["advanced"])

        logger.info(f"Configuration loaded from: {config_path}")
        return cls(**config_dict)

    def save_yaml(self, config_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        config_dict = self.to_dict()
        ConfigLoader.save(config_dict, config_path, format='yaml')

    @classmethod
    def load_yaml(cls, config_path: Union[str, Path]) -> "TrainingConfig":
        """Load configuration from YAML file."""
        config_dict = ConfigLoader.load(config_path, format='yaml')
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingConfig":
        """Create TrainingConfig from dictionary.
        
        This method handles the conversion of dictionaries to dataclass instances
        and enum strings to enum values.
        """
        # Convert enum strings back to enums
        def restore_enums(data: Dict[str, Any]) -> Dict[str, Any]:
            enum_mappings = {
                "optimization_level": OptimizationLevel,
                "optimizer": OptimizerType,
                "lr_schedule": LearningRateSchedule,
                "loss_function": LossFunction,
            }
            
            for key, enum_class in enum_mappings.items():
                if key in data and isinstance(data[key], str):
                    try:
                        data[key] = enum_class(data[key])
                    except ValueError:
                        logger.warning(f"Invalid enum value for {key}: {data[key]}")
            
            return data
        
        config_dict = restore_enums(config_dict)
        
        # Reconstruct sub-configurations
        if "memory" in config_dict and isinstance(config_dict["memory"], dict):
            config_dict["memory"] = MemoryConfig(**config_dict["memory"])
        if "mlx_optimization" in config_dict and isinstance(config_dict["mlx_optimization"], dict):
            config_dict["mlx_optimization"] = MLXOptimizationConfig(
                **config_dict["mlx_optimization"]
            )
        if "monitoring" in config_dict and isinstance(config_dict["monitoring"], dict):
            # Handle None values in monitoring config
            monitoring_dict = config_dict["monitoring"].copy()
            if monitoring_dict.get("log_file_path") is None:
                monitoring_dict.pop("log_file_path", None)
            if monitoring_dict.get("tracking_uri") is None:
                monitoring_dict.pop("tracking_uri", None)
            config_dict["monitoring"] = MonitoringConfig(**monitoring_dict)
        if "checkpoint" in config_dict and isinstance(config_dict["checkpoint"], dict):
            # Handle None values in checkpoint config
            checkpoint_dict = config_dict["checkpoint"].copy()
            if checkpoint_dict.get("checkpoint_dir") is None:
                checkpoint_dict["checkpoint_dir"] = "checkpoints"  # Use default
            if checkpoint_dict.get("resume_from_checkpoint") is None:
                checkpoint_dict.pop("resume_from_checkpoint", None)
            config_dict["checkpoint"] = CheckpointConfig(**checkpoint_dict)
        if "evaluation" in config_dict and isinstance(config_dict["evaluation"], dict):
            config_dict["evaluation"] = EvaluationConfig(**config_dict["evaluation"])
        if "advanced" in config_dict and isinstance(config_dict["advanced"], dict):
            config_dict["advanced"] = AdvancedFeatures(**config_dict["advanced"])
        
        # Filter out any extra fields not defined in TrainingConfig
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        
        # Log any fields that were filtered out
        extra_fields = set(config_dict.keys()) - valid_fields
        if extra_fields:
            logger.debug(f"Filtered out extra fields from config: {extra_fields}")
        
        # Type conversions for numeric fields that might come as strings
        numeric_fields = {
            "learning_rate": float,
            "epochs": int,
            "batch_size": int,
            "warmup_steps": int,
            "max_length": int,
            "num_labels": int,
            "seed": int,
        }
        
        for field, converter in numeric_fields.items():
            if field in filtered_dict and isinstance(filtered_dict[field], str):
                try:
                    filtered_dict[field] = converter(filtered_dict[field])
                except ValueError:
                    logger.warning(f"Failed to convert {field} value: {filtered_dict[field]}")
        
        # Remove None values for optional fields that shouldn't be None
        optional_fields = ["train_path", "val_path", "test_path", "target_column", 
                          "max_steps", "experiment_name", "run_name"]
        for field in optional_fields:
            if field in filtered_dict and filtered_dict[field] is None:
                filtered_dict.pop(field)
        
        return cls(**filtered_dict)

    def get_effective_batch_size(self) -> int:
        """Get the effective batch size considering gradient accumulation."""
        return self.batch_size * self.mlx_optimization.gradient_accumulation_steps

    def get_total_steps(self, dataset_size: int) -> int:
        """Calculate total training steps."""
        steps_per_epoch = dataset_size // self.get_effective_batch_size()
        if self.max_steps is not None:
            return min(self.max_steps, self.epochs * steps_per_epoch)
        return self.epochs * steps_per_epoch

    def update_from_dataset(self, dataset_info: dict[str, Any]) -> None:
        """Update configuration based on dataset characteristics."""
        # Auto-detect number of labels
        if self.num_labels is None:
            problem_type = dataset_info.get("dataset_spec", {}).get("problem_type")
            if problem_type == "binary_classification":
                self.num_labels = 2
            elif problem_type == "multiclass_classification":
                # This would need to be detected from actual data
                self.num_labels = dataset_info.get("num_classes", 2)
            else:
                self.num_labels = 1  # Regression

        # Auto-configure optimization level based on dataset size
        if self._original_optimization_level == OptimizationLevel.AUTO:
            dataset_size = dataset_info.get("dataset_spec", {}).get(
                "expected_size", 1000
            )
            self._auto_configure_optimization(dataset_size)
            logger.info(
                f"Auto-configured optimization level: {self.optimization_level}"
            )


# Predefined configurations for common use cases
def get_development_config(**overrides) -> TrainingConfig:
    """Get a configuration optimized for development and experimentation."""
    defaults = {
        "epochs": 1,
        "batch_size": 16,
        "optimization_level": OptimizationLevel.DEVELOPMENT,
    }
    defaults.update(overrides)
    return TrainingConfig(**defaults)


def get_production_config(**overrides) -> TrainingConfig:
    """Get a configuration optimized for production training."""
    defaults = {
        "epochs": 5,
        "batch_size": 64,
        "optimization_level": OptimizationLevel.PRODUCTION,
    }
    defaults.update(overrides)
    return TrainingConfig(**defaults)


def get_competition_config(**overrides) -> TrainingConfig:
    """Get a configuration optimized for competition performance."""
    defaults = {
        "epochs": 10,
        "batch_size": 32,
        "optimization_level": OptimizationLevel.COMPETITION,
    }
    defaults.update(overrides)
    config = TrainingConfig(**defaults)

    # Apply competition-specific advanced settings if not overridden
    if "advanced" not in overrides:
        config.advanced.enable_ensembling = True
        config.advanced.ensemble_size = 5
        config.advanced.label_smoothing = 0.1

    return config
