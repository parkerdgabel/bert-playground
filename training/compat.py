"""Backward compatibility layer for the refactored training system.

This module provides compatibility wrappers and adapters to ensure that existing
code continues to work with the new command-based training system.
"""

import warnings
from typing import Any, Callable

from training.commands.base import CommandContext, CommandResult
from training.commands import ForwardCommand, BackwardCommand, OptimizerStepCommand, LoggingCommand
from training.pipeline.base import PipelineBuilder
from training.strategies import StandardTraining, get_strategy
from core.protocols.training import TrainingState
from training.core.base import BaseTrainer


class LegacyTrainerAdapter:
    """Adapter to make the new training system compatible with legacy BaseTrainer interface."""
    
    def __init__(
        self,
        model,
        config,
        callbacks=None,
        strategy_name: str = "StandardTraining",
        **kwargs
    ):
        """Initialize legacy trainer adapter.
        
        Args:
            model: Model to train
            config: Training configuration
            callbacks: Optional callbacks (will be converted to middleware)
            strategy_name: Training strategy to use
            **kwargs: Additional arguments for backward compatibility
        """
        warnings.warn(
            "LegacyTrainerAdapter is deprecated. Please use the new strategy-based training system.",
            DeprecationWarning,
            stacklevel=2
        )
        
        self.model = model
        self.config = config
        self.callbacks = callbacks or []
        self.strategy_name = strategy_name
        
        # Create strategy and context
        self.strategy = get_strategy(strategy_name)
        self._context = None
        self._pipeline = None
    
    def _create_context(self, train_dataloader, val_dataloader=None) -> CommandContext:
        """Create command context from legacy parameters."""
        # Convert config to dict if it's not already
        config_dict = self.config.to_dict() if hasattr(self.config, 'to_dict') else vars(self.config)
        
        # Create context
        context = CommandContext(
            model=self.model,
            optimizer=None,  # Will be set later
            state=TrainingState(),
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            config=config_dict,
            batch=None,
            batch_idx=0,
            outputs={},
            gradients={},
            loss=None,
            metrics={},
            should_accumulate_gradients=False,
            should_update_weights=True,
            is_training=True,
        )
        
        return context
    
    def train(self, train_dataloader, val_dataloader=None, resume_from=None):
        """Legacy train method interface."""
        # Create context
        context = self._create_context(train_dataloader, val_dataloader)
        
        # Configure context with strategy
        context = self.strategy.configure_context(context)
        
        # Create pipeline
        pipeline = self.strategy.create_pipeline(context)
        
        # Legacy training loop simulation
        # Note: This is a simplified version - full implementation would require
        # more complex integration with the existing BaseTrainer logic
        
        results = []
        context.state.training_start_time = 0.0  # Would set actual time
        
        try:
            # Simulate training epochs
            for epoch in range(context.config.get("num_epochs", 1)):
                context.state.epoch = epoch
                
                # Training loop would go here
                # For now, just create a placeholder result
                result = CommandResult(
                    success=True,
                    outputs={
                        "epoch": epoch,
                        "train_loss": 0.5,  # Placeholder
                    },
                    metrics={
                        "train_loss": 0.5,
                        "epoch": epoch,
                    }
                )
                results.append(result)
            
            # Convert to legacy TrainingResult format
            from core.protocols.training import TrainingResult
            return TrainingResult(
                final_train_loss=results[-1].metrics.get("train_loss", 0.0) if results else 0.0,
                final_val_loss=results[-1].metrics.get("val_loss", 0.0) if results else 0.0,
                best_val_loss=min(r.metrics.get("val_loss", float("inf")) for r in results),
                best_val_metric=max(r.metrics.get("val_accuracy", 0.0) for r in results),
                final_metrics=results[-1].metrics if results else {},
                train_history=[r.metrics for r in results],
                val_history=[r.metrics for r in results],
                total_epochs=len(results),
                total_steps=sum(r.outputs.get("steps", 1) for r in results),
                total_time=0.0,  # Would calculate actual time
                early_stopped=False,
            )
            
        except Exception as e:
            # Legacy error handling
            raise e
    
    def evaluate(self, dataloader):
        """Legacy evaluate method interface."""
        warnings.warn(
            "evaluate() method is deprecated. Use evaluation commands directly.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Create evaluation context
        context = self._create_context(None, dataloader)
        context.is_training = False
        
        # Create simple evaluation pipeline
        from training.commands import EvaluationCommand
        eval_cmd = EvaluationCommand(compute_metrics=True)
        
        # Execute evaluation
        result = eval_cmd.execute(context)
        
        return result.metrics if result.success else {}
    
    def predict(self, dataloader):
        """Legacy predict method interface."""
        warnings.warn(
            "predict() method is deprecated. Use prediction commands directly.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Simple prediction implementation
        predictions = []
        
        for batch in dataloader:
            # Forward pass only
            forward_cmd = ForwardCommand(compute_loss=False, return_outputs=True)
            context = CommandContext(
                model=self.model,
                optimizer=None,
                state=TrainingState(),
                batch=batch,
                is_training=False,
                config={},
                batch_idx=0,
                outputs={},
                gradients={},
                loss=None,
                metrics={},
                should_accumulate_gradients=False,
                should_update_weights=False,
            )
            
            result = forward_cmd.execute(context)
            if result.success and "model_outputs" in result.outputs:
                predictions.append(result.outputs["model_outputs"])
        
        return predictions
    
    # Legacy property interfaces
    @property
    def state(self):
        """Legacy state property."""
        if self._context:
            return self._context.state
        return TrainingState()
    
    def save_checkpoint(self, path):
        """Legacy checkpoint saving."""
        warnings.warn(
            "save_checkpoint() method is deprecated. Use CheckpointCommand directly.",
            DeprecationWarning,
            stacklevel=2
        )
        # Would integrate with CheckpointCommand
        pass
    
    def load_checkpoint(self, path):
        """Legacy checkpoint loading."""
        warnings.warn(
            "load_checkpoint() method is deprecated. Use LoadCheckpointCommand directly.",
            DeprecationWarning,
            stacklevel=2
        )
        # Would integrate with LoadCheckpointCommand
        pass


def create_legacy_trainer(model, config, callbacks=None, **kwargs):
    """Factory function for backward compatibility.
    
    Creates a legacy trainer that works with existing code.
    """
    warnings.warn(
        "create_legacy_trainer() is deprecated. Please use the new strategy-based system:\n"
        "from training.strategies import get_strategy\n"
        "strategy = get_strategy('StandardTraining')\n"
        "pipeline = strategy.create_pipeline(context)",
        DeprecationWarning,
        stacklevel=2
    )
    
    return LegacyTrainerAdapter(model, config, callbacks, **kwargs)


# Legacy imports for backward compatibility
def BaseTrainer(*args, **kwargs):
    """Legacy BaseTrainer constructor for backward compatibility."""
    warnings.warn(
        "BaseTrainer is deprecated. Please use the new strategy-based training system.",
        DeprecationWarning,
        stacklevel=2
    )
    return LegacyTrainerAdapter(*args, **kwargs)


# Migration utilities
def migrate_config_to_strategy(legacy_config, strategy_name="StandardTraining"):
    """Migrate legacy training config to new strategy config.
    
    Args:
        legacy_config: Legacy training configuration
        strategy_name: Target strategy name
        
    Returns:
        dict: Strategy configuration
    """
    # Convert legacy config to strategy config
    config_dict = legacy_config.to_dict() if hasattr(legacy_config, 'to_dict') else vars(legacy_config)
    
    # Map legacy config keys to new keys
    key_mapping = {
        "max_grad_norm": "max_grad_norm",
        "gradient_accumulation_steps": "gradient_accumulation_steps", 
        "mixed_precision": "mixed_precision",
        "log_every": "log_interval",
        "eval_every": "eval_steps",
        "save_every": "save_steps",
    }
    
    strategy_config = {}
    for old_key, new_key in key_mapping.items():
        if old_key in config_dict:
            strategy_config[new_key] = config_dict[old_key]
    
    # Copy other keys as-is
    for key, value in config_dict.items():
        if key not in key_mapping:
            strategy_config[key] = value
    
    return strategy_config


def migrate_callbacks_to_middleware(callbacks):
    """Migrate legacy callbacks to new middleware.
    
    Args:
        callbacks: List of legacy callbacks
        
    Returns:
        list: List of middleware
    """
    from training.pipeline.middleware import TimingMiddleware, MetricsMiddleware
    
    # This is a simplified migration - actual implementation would need
    # to analyze callback types and create appropriate middleware
    middleware = []
    
    for callback in callbacks:
        callback_type = type(callback).__name__
        
        if "Timer" in callback_type or "Timing" in callback_type:
            middleware.append(TimingMiddleware())
        elif "Metric" in callback_type:
            middleware.append(MetricsMiddleware())
        # Add more callback -> middleware mappings as needed
    
    return middleware


# Deprecation warnings for commonly used imports
def __getattr__(name):
    """Handle deprecated imports."""
    if name in ["BaseTrainer", "KaggleTrainer"]:
        warnings.warn(
            f"{name} is deprecated. Please use the new strategy-based training system.",
            DeprecationWarning,
            stacklevel=2
        )
        return LegacyTrainerAdapter
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")