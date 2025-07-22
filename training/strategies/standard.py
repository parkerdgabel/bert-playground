"""Standard training strategy implementations.

This module provides commonly used training strategies like standard training,
gradient accumulation, and mixed precision training.
"""

from training.commands import (
    ForwardCommand,
    BackwardCommand,
    OptimizerStepCommand,
    GradientAccumulationCommand,
    EvaluationCommand,
    CheckpointCommand,
    LoggingCommand,
    MLXBackwardCommand,
    MLXGradientAccumulationCommand,
    MLXOptimizerStepCommand,
)
from training.commands.base import CommandContext
from training.pipeline.base import Pipeline, PipelineBuilder
from training.pipeline.middleware import (
    TimingMiddleware,
    ErrorHandlingMiddleware,
    MetricsMiddleware,
    ValidationMiddleware,
)

from .base import BaseTrainingStrategy, StrategyConfig


class StandardTraining(BaseTrainingStrategy):
    """Standard training strategy with basic forward/backward passes."""
    
    def __init__(self, config: dict | None = None):
        """Initialize standard training strategy.
        
        Args:
            config: Strategy configuration
        """
        super().__init__(
            name="StandardTraining",
            description="Basic training with forward/backward passes",
            config=config
        )
    
    def create_pipeline(self, context: CommandContext) -> Pipeline:
        """Create standard training pipeline."""
        # Create commands
        forward_cmd = ForwardCommand(
            compute_loss=True,
            return_outputs=True,
            mixed_precision=self.config.get("mixed_precision", False),
            label_smoothing=self.config.get("label_smoothing", 0.0),
        )
        
        backward_cmd = BackwardCommand(
            grad_clip_norm=self.config.get("max_grad_norm", 1.0),
            compute_grad_norm=True,
        )
        
        optimizer_cmd = OptimizerStepCommand(
            update_lr_scheduler=True,
            zero_grad_after_step=True,
        )
        
        logging_cmd = LoggingCommand(
            log_interval=self.config.get("log_interval", 10),
            log_to_console=True,
            log_to_mlflow=True,
        )
        
        # Build pipeline
        pipeline = (
            PipelineBuilder("StandardTrainingPipeline")
            .add_commands(forward_cmd, backward_cmd, optimizer_cmd, logging_cmd)
            .add_middlewares(
                TimingMiddleware(log_timings=False),
                ErrorHandlingMiddleware(max_retries=2),
                MetricsMiddleware(rolling_window=50),
                ValidationMiddleware(strict=False),
            )
            .continue_on_error(False)
            .build()
        )
        
        return pipeline
    
    def get_default_config(self) -> dict:
        """Get default configuration for standard training."""
        return {
            **super().get_default_config(),
            "mixed_precision": False,
            "label_smoothing": 0.0,
            "max_grad_norm": 1.0,
            "log_interval": 10,
        }


class GradientAccumulationTraining(BaseTrainingStrategy):
    """Training strategy with gradient accumulation for larger effective batch sizes."""
    
    def __init__(self, config: dict | None = None):
        """Initialize gradient accumulation training.
        
        Args:
            config: Strategy configuration
        """
        super().__init__(
            name="GradientAccumulationTraining",
            description="Training with gradient accumulation for memory efficiency",
            config=config
        )
    
    def create_pipeline(self, context: CommandContext) -> Pipeline:
        """Create gradient accumulation training pipeline."""
        accumulation_steps = self.config.get("gradient_accumulation_steps", 4)
        
        # Create commands
        forward_cmd = ForwardCommand(
            compute_loss=True,
            return_outputs=True,
            mixed_precision=self.config.get("mixed_precision", False),
        )
        
        backward_cmd = BackwardCommand(
            grad_clip_norm=self.config.get("max_grad_norm", 1.0),
            compute_grad_norm=True,
        )
        
        accumulation_cmd = GradientAccumulationCommand(
            accumulation_steps=accumulation_steps,
            normalize_accumulated_gradients=True,
        )
        
        optimizer_cmd = OptimizerStepCommand(
            scale_learning_rate=True,
            update_lr_scheduler=True,
            zero_grad_after_step=True,
        )
        
        logging_cmd = LoggingCommand(
            log_interval=self.config.get("log_interval", 10),
            log_to_console=True,
        )
        
        # Build pipeline
        pipeline = (
            PipelineBuilder("GradientAccumulationPipeline")
            .add_commands(
                forward_cmd, backward_cmd, accumulation_cmd, optimizer_cmd, logging_cmd
            )
            .add_middlewares(
                TimingMiddleware(log_timings=True),
                ErrorHandlingMiddleware(max_retries=1),
                MetricsMiddleware(rolling_window=100),
            )
            .continue_on_error(False)
            .build()
        )
        
        return pipeline
    
    def get_default_config(self) -> dict:
        """Get default configuration for gradient accumulation."""
        return {
            **super().get_default_config(),
            "gradient_accumulation_steps": 4,
            "mixed_precision": False,
            "max_grad_norm": 1.0,
            "log_interval": 10,
        }


class MixedPrecisionTraining(BaseTrainingStrategy):
    """Training strategy with mixed precision for faster training."""
    
    def __init__(self, config: dict | None = None):
        """Initialize mixed precision training.
        
        Args:
            config: Strategy configuration
        """
        super().__init__(
            name="MixedPrecisionTraining",
            description="Training with mixed precision (bfloat16/float32)",
            config=config
        )
    
    def create_pipeline(self, context: CommandContext) -> Pipeline:
        """Create mixed precision training pipeline."""
        # Create commands with mixed precision enabled
        forward_cmd = ForwardCommand(
            compute_loss=True,
            return_outputs=True,
            mixed_precision=True,  # Enable mixed precision
            label_smoothing=self.config.get("label_smoothing", 0.0),
        )
        
        backward_cmd = BackwardCommand(
            grad_clip_norm=self.config.get("max_grad_norm", 1.0),
            compute_grad_norm=True,
            loss_scale=self.config.get("loss_scale", 1.0),
        )
        
        optimizer_cmd = OptimizerStepCommand(
            update_lr_scheduler=True,
            zero_grad_after_step=True,
        )
        
        logging_cmd = LoggingCommand(
            log_interval=self.config.get("log_interval", 10),
            verbose=True,
        )
        
        # Build pipeline
        pipeline = (
            PipelineBuilder("MixedPrecisionPipeline")
            .add_commands(forward_cmd, backward_cmd, optimizer_cmd, logging_cmd)
            .add_middlewares(
                TimingMiddleware(log_timings=True),
                MetricsMiddleware(rolling_window=50),
                ValidationMiddleware(strict=False),
            )
            .continue_on_error(False)
            .build()
        )
        
        return pipeline
    
    def get_default_config(self) -> dict:
        """Get default configuration for mixed precision."""
        return {
            **super().get_default_config(),
            "mixed_precision": True,
            "loss_scale": 1.0,
            "label_smoothing": 0.0,
            "max_grad_norm": 1.0,
            "log_interval": 10,
        }


class MLXOptimizedTraining(BaseTrainingStrategy):
    """MLX-optimized training strategy using framework-specific optimizations."""
    
    def __init__(self, config: dict | None = None):
        """Initialize MLX-optimized training.
        
        Args:
            config: Strategy configuration
        """
        super().__init__(
            name="MLXOptimizedTraining",
            description="Training optimized for Apple MLX framework",
            config=config
        )
    
    def create_pipeline(self, context: CommandContext) -> Pipeline:
        """Create MLX-optimized training pipeline."""
        # Use MLX-specific commands
        forward_cmd = ForwardCommand(
            compute_loss=True,
            return_outputs=True,
            mixed_precision=self.config.get("mixed_precision", True),  # MLX handles this well
        )
        
        backward_cmd = MLXBackwardCommand(
            grad_clip_norm=self.config.get("max_grad_norm", 1.0),
            compute_grad_norm=True,
        )
        
        # Use gradient accumulation if configured
        accumulation_steps = self.config.get("gradient_accumulation_steps", 1)
        commands = [forward_cmd, backward_cmd]
        
        if accumulation_steps > 1:
            accumulation_cmd = MLXGradientAccumulationCommand(
                accumulation_steps=accumulation_steps,
                normalize_accumulated_gradients=True,
            )
            commands.append(accumulation_cmd)
        
        optimizer_cmd = MLXOptimizerStepCommand(
            scale_learning_rate=accumulation_steps > 1,
            update_lr_scheduler=True,
            zero_grad_after_step=True,
        )
        
        logging_cmd = LoggingCommand(
            log_interval=self.config.get("log_interval", 10),
            verbose=False,
        )
        
        commands.extend([optimizer_cmd, logging_cmd])
        
        # Build pipeline with minimal middleware for maximum performance
        pipeline = (
            PipelineBuilder("MLXOptimizedPipeline")
            .add_commands(*commands)
            .add_middlewares(
                ErrorHandlingMiddleware(max_retries=1),
                MetricsMiddleware(rolling_window=20),
            )
            .continue_on_error(False)
            .build()
        )
        
        return pipeline
    
    def validate_requirements(self, context: CommandContext) -> list[str]:
        """Validate MLX-specific requirements."""
        errors = super().validate_requirements(context)
        
        # Check for MLX availability
        try:
            import mlx.core as mx
        except ImportError:
            errors.append("MLX framework not available")
        
        return errors
    
    def get_default_config(self) -> dict:
        """Get default configuration for MLX-optimized training."""
        return {
            **super().get_default_config(),
            "mixed_precision": True,  # MLX handles mixed precision well
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0,
            "log_interval": 50,  # Less frequent logging for performance
            "use_compilation": True,
        }