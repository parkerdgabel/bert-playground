"""Dependency injection setup for model creation.

This module sets up the dependency injection container for model builders,
allowing for clean separation of concerns and testability.
"""

from dataclasses import dataclass
from typing import Any, Optional

from loguru import logger

from .builders import (
    ConfigResolver,
    HeadFactory,
    ModelBuilder,
    ModelRegistry,
    ValidationService,
)


@dataclass
class ModelDIContainer:
    """Dependency injection container for model creation."""
    
    config_resolver: ConfigResolver
    validation_service: ValidationService
    head_factory: HeadFactory
    model_builder: ModelBuilder
    model_registry: ModelRegistry
    
    @classmethod
    def create_default(cls) -> "ModelDIContainer":
        """Create a default DI container with all components."""
        # Create base services
        config_resolver = ConfigResolver()
        validation_service = ValidationService()
        
        # Create factories with dependencies
        head_factory = HeadFactory(config_resolver=config_resolver)
        
        # Create model builder with all dependencies
        model_builder = ModelBuilder(
            config_resolver=config_resolver,
            head_factory=head_factory,
            validation_service=validation_service,
        )
        
        # Create registry
        model_registry = ModelRegistry()
        
        # Register default models
        model_registry.register_defaults(model_builder)
        
        logger.info("Created default model DI container")
        
        return cls(
            config_resolver=config_resolver,
            validation_service=validation_service,
            head_factory=head_factory,
            model_builder=model_builder,
            model_registry=model_registry,
        )
    
    def get_model_builder(self) -> ModelBuilder:
        """Get the model builder instance."""
        return self.model_builder
    
    def get_registry(self) -> ModelRegistry:
        """Get the model registry instance."""
        return self.model_registry
    
    def get_head_factory(self) -> HeadFactory:
        """Get the head factory instance."""
        return self.head_factory
    
    def register_custom_model(
        self,
        name: str,
        factory: Any,
        **kwargs,
    ) -> None:
        """Register a custom model type.
        
        Args:
            name: Model name
            factory: Factory function
            **kwargs: Additional registration parameters
        """
        self.model_registry.register(name, factory, **kwargs)
    
    def create_model(self, name: str, **kwargs) -> Any:
        """Create a model by name using the registry.
        
        Args:
            name: Model name
            **kwargs: Model creation parameters
            
        Returns:
            Created model instance
        """
        return self.model_registry.create(name, **kwargs)


# Global container instance
_container: Optional[ModelDIContainer] = None


def get_container() -> ModelDIContainer:
    """Get the global DI container, creating it if needed.
    
    Returns:
        ModelDIContainer instance
    """
    global _container
    if _container is None:
        _container = ModelDIContainer.create_default()
    return _container


def set_container(container: ModelDIContainer) -> None:
    """Set the global DI container.
    
    Args:
        container: Container to use globally
    """
    global _container
    _container = container
    logger.info("Set global model DI container")


def reset_container() -> None:
    """Reset the global DI container."""
    global _container
    _container = None
    logger.info("Reset global model DI container")