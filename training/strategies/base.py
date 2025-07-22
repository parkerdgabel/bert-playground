"""Base training strategy protocol and implementations.

This module defines the strategy pattern for training algorithms,
allowing different training approaches to be plugged in easily.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from training.commands.base import Command, CommandContext, CommandResult
from training.pipeline.base import Pipeline


@runtime_checkable
class TrainingStrategy(Protocol):
    """Protocol for training strategies."""
    
    @property
    def name(self) -> str:
        """Strategy name."""
        ...
    
    @property
    def description(self) -> str:
        """Strategy description."""
        ...
    
    def create_pipeline(self, context: CommandContext) -> Pipeline:
        """Create training pipeline for this strategy.
        
        Args:
            context: Training context
            
        Returns:
            Configured pipeline
        """
        ...
    
    def configure_context(self, context: CommandContext) -> CommandContext:
        """Configure context for this strategy.
        
        Args:
            context: Base context
            
        Returns:
            Modified context
        """
        ...
    
    def validate_requirements(self, context: CommandContext) -> list[str]:
        """Validate that context meets strategy requirements.
        
        Args:
            context: Training context
            
        Returns:
            List of validation errors (empty if valid)
        """
        ...
    
    def get_default_config(self) -> dict[str, Any]:
        """Get default configuration for this strategy.
        
        Returns:
            Default configuration dictionary
        """
        ...


class BaseTrainingStrategy(ABC):
    """Base implementation of TrainingStrategy with common functionality."""
    
    def __init__(
        self,
        name: str | None = None,
        description: str | None = None,
        config: dict[str, Any] | None = None,
    ):
        """Initialize strategy.
        
        Args:
            name: Strategy name
            description: Strategy description
            config: Strategy configuration
        """
        self._name = name or self.__class__.__name__
        self._description = description or f"Training strategy: {self._name}"
        self.config = {**self.get_default_config(), **(config or {})}
    
    @property
    def name(self) -> str:
        """Strategy name."""
        return self._name
    
    @property
    def description(self) -> str:
        """Strategy description."""
        return self._description
    
    def configure_context(self, context: CommandContext) -> CommandContext:
        """Default context configuration - merge strategy config."""
        # Add strategy config to context
        context.config.update(self.config)
        return context
    
    def validate_requirements(self, context: CommandContext) -> list[str]:
        """Default validation - check basic requirements."""
        errors = []
        
        if context.model is None:
            errors.append("Model is required")
        
        if context.optimizer is None and context.is_training:
            errors.append("Optimizer is required for training")
        
        if context.train_dataloader is None and context.is_training:
            errors.append("Training dataloader is required")
        
        return errors
    
    def get_default_config(self) -> dict[str, Any]:
        """Default configuration."""
        return {
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0,
            "log_interval": 10,
        }
    
    @abstractmethod
    def create_pipeline(self, context: CommandContext) -> Pipeline:
        """Create training pipeline - must be implemented by subclasses."""
        pass


@dataclass
class StrategyConfig:
    """Configuration for training strategies."""
    
    # Basic training parameters
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    mixed_precision: bool = False
    
    # Learning rate and optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 0
    
    # Evaluation and checkpointing
    eval_steps: int = 500
    save_steps: int = 1000
    log_interval: int = 10
    
    # Strategy-specific parameters
    extra_config: dict[str, Any] = None
    
    def __post_init__(self):
        if self.extra_config is None:
            self.extra_config = {}
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_grad_norm": self.max_grad_norm,
            "mixed_precision": self.mixed_precision,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_steps": self.warmup_steps,
            "eval_steps": self.eval_steps,
            "save_steps": self.save_steps,
            "log_interval": self.log_interval,
            **self.extra_config,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StrategyConfig":
        """Create from dictionary."""
        known_fields = {
            "gradient_accumulation_steps",
            "max_grad_norm", 
            "mixed_precision",
            "learning_rate",
            "weight_decay",
            "warmup_steps",
            "eval_steps",
            "save_steps",
            "log_interval",
        }
        
        # Separate known fields from extra config
        known_data = {k: v for k, v in data.items() if k in known_fields}
        extra_data = {k: v for k, v in data.items() if k not in known_fields}
        
        return cls(extra_config=extra_data, **known_data)


class StrategyManager:
    """Manager for training strategies."""
    
    def __init__(self):
        """Initialize strategy manager."""
        self._strategies: dict[str, TrainingStrategy] = {}
        self._default_strategy: str | None = None
    
    def register_strategy(
        self,
        strategy: TrainingStrategy,
        set_as_default: bool = False
    ) -> None:
        """Register a training strategy.
        
        Args:
            strategy: Strategy to register
            set_as_default: Whether to set as default strategy
        """
        self._strategies[strategy.name] = strategy
        
        if set_as_default or self._default_strategy is None:
            self._default_strategy = strategy.name
    
    def get_strategy(self, name: str) -> TrainingStrategy:
        """Get strategy by name.
        
        Args:
            name: Strategy name
            
        Returns:
            Training strategy
            
        Raises:
            KeyError: If strategy not found
        """
        if name not in self._strategies:
            raise KeyError(f"Strategy '{name}' not found. Available: {list(self._strategies.keys())}")
        
        return self._strategies[name]
    
    def get_default_strategy(self) -> TrainingStrategy:
        """Get default strategy.
        
        Returns:
            Default training strategy
            
        Raises:
            RuntimeError: If no default strategy set
        """
        if self._default_strategy is None:
            raise RuntimeError("No default strategy set")
        
        return self.get_strategy(self._default_strategy)
    
    def list_strategies(self) -> dict[str, str]:
        """List available strategies.
        
        Returns:
            Dictionary of strategy name -> description
        """
        return {name: strategy.description for name, strategy in self._strategies.items()}
    
    def validate_strategy(
        self,
        strategy_name: str,
        context: CommandContext
    ) -> list[str]:
        """Validate strategy against context.
        
        Args:
            strategy_name: Name of strategy to validate
            context: Training context
            
        Returns:
            List of validation errors
        """
        try:
            strategy = self.get_strategy(strategy_name)
            return strategy.validate_requirements(context)
        except KeyError:
            return [f"Strategy '{strategy_name}' not found"]


# Global strategy manager instance
_strategy_manager = StrategyManager()

def register_strategy(strategy: TrainingStrategy, set_as_default: bool = False) -> None:
    """Register a training strategy globally."""
    _strategy_manager.register_strategy(strategy, set_as_default)

def get_strategy(name: str) -> TrainingStrategy:
    """Get strategy by name."""
    return _strategy_manager.get_strategy(name)

def get_default_strategy() -> TrainingStrategy:
    """Get default strategy."""
    return _strategy_manager.get_default_strategy()

def list_strategies() -> dict[str, str]:
    """List available strategies."""
    return _strategy_manager.list_strategies()