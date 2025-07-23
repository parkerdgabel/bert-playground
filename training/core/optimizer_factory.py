"""Framework-agnostic optimizer factory for k-bert.

This module provides optimizer creation without direct framework dependencies,
using the FrameworkAdapter to handle framework-specific operations.
"""

from typing import Any, Protocol, runtime_checkable

from loguru import logger

from core.protocols.training import Optimizer as IOptimizer
from training.adapters.framework_adapter import FrameworkAdapter
from .config import OptimizerConfig, OptimizerType


@runtime_checkable
class OptimizerFactory(Protocol):
    """Protocol for optimizer factories."""
    
    def create_optimizer(
        self, 
        model: Any,
        config: OptimizerConfig,
        framework: FrameworkAdapter
    ) -> IOptimizer:
        """Create an optimizer instance.
        
        Args:
            model: Model to optimize
            config: Optimizer configuration
            framework: Framework adapter for backend operations
            
        Returns:
            Optimizer instance
        """
        ...


class MLXOptimizerFactory:
    """MLX-specific optimizer factory."""
    
    def create_optimizer(
        self,
        model: Any,
        config: OptimizerConfig,
        framework: FrameworkAdapter
    ) -> IOptimizer:
        """Create MLX optimizer.
        
        Args:
            model: MLX model to optimize
            config: Optimizer configuration
            framework: Framework adapter (must be MLX)
            
        Returns:
            MLX optimizer instance
            
        Raises:
            ValueError: If framework is not MLX or optimizer type unknown
        """
        if framework.name != "MLX":
            raise ValueError(f"MLX optimizer factory requires MLX framework, got {framework.name}")
        
        # Import MLX optimizers only when needed
        import mlx.optimizers as optim
        
        # Create optimizer based on type
        if config.type == OptimizerType.ADAM:
            optimizer = optim.Adam(
                learning_rate=config.learning_rate,
                betas=(config.beta1, config.beta2),
                eps=config.epsilon,
            )
        elif config.type == OptimizerType.ADAMW:
            optimizer = optim.AdamW(
                learning_rate=config.learning_rate,
                betas=(config.beta1, config.beta2),
                eps=config.epsilon,
                weight_decay=config.weight_decay,
            )
        elif config.type == OptimizerType.SGD:
            optimizer = optim.SGD(
                learning_rate=config.learning_rate,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
                nesterov=config.nesterov,
            )
        elif config.type == OptimizerType.LION:
            optimizer = optim.Lion(
                learning_rate=config.learning_rate,
                betas=(config.lion_beta1, config.lion_beta2),
                weight_decay=config.weight_decay,
            )
        elif config.type == OptimizerType.ADAFACTOR:
            optimizer = optim.Adafactor(
                learning_rate=config.learning_rate,
                eps=(1e-30, config.epsilon),
                weight_decay=config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer type: {config.type}")
        
        logger.info(f"Created {config.type.value} optimizer with lr={config.learning_rate}")
        return optimizer


class OptimizerRegistry:
    """Registry for optimizer factories."""
    
    def __init__(self):
        self._factories: dict[str, OptimizerFactory] = {}
        self._register_defaults()
    
    def _register_defaults(self):
        """Register default optimizer factories."""
        self._factories["mlx"] = MLXOptimizerFactory()
    
    def register(self, framework: str, factory: OptimizerFactory) -> None:
        """Register an optimizer factory.
        
        Args:
            framework: Framework name
            factory: Optimizer factory instance
        """
        self._factories[framework.lower()] = factory
        logger.debug(f"Registered optimizer factory for {framework}")
    
    def create_optimizer(
        self,
        model: Any,
        config: OptimizerConfig,
        framework: FrameworkAdapter
    ) -> IOptimizer:
        """Create optimizer using appropriate factory.
        
        Args:
            model: Model to optimize
            config: Optimizer configuration
            framework: Framework adapter
            
        Returns:
            Optimizer instance
            
        Raises:
            ValueError: If no factory registered for framework
        """
        factory_key = framework.name.lower()
        factory = self._factories.get(factory_key)
        
        if factory is None:
            raise ValueError(
                f"No optimizer factory registered for framework: {framework.name}. "
                f"Available: {list(self._factories.keys())}"
            )
        
        return factory.create_optimizer(model, config, framework)


# Global registry instance
_optimizer_registry = OptimizerRegistry()


def create_optimizer(
    model: Any,
    config: OptimizerConfig,
    framework: FrameworkAdapter
) -> IOptimizer:
    """Create an optimizer using the global registry.
    
    Args:
        model: Model to optimize
        config: Optimizer configuration
        framework: Framework adapter
        
    Returns:
        Optimizer instance
    """
    return _optimizer_registry.create_optimizer(model, config, framework)


def register_optimizer_factory(framework: str, factory: OptimizerFactory) -> None:
    """Register a custom optimizer factory.
    
    Args:
        framework: Framework name
        factory: Optimizer factory instance
    """
    _optimizer_registry.register(framework, factory)