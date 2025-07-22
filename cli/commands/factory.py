"""Command factory for creating commands with dependency injection.

This module provides a factory for creating command instances with
proper dependency injection and initialization.
"""

from typing import Dict, Optional, Type

from loguru import logger

from core.di import Container
from .base import BaseCommand


class CommandFactory:
    """Factory for creating command instances with DI."""
    
    def __init__(self, container: Optional[Container] = None):
        """Initialize command factory.
        
        Args:
            container: DI container to use for commands
        """
        self.container = container
        self._command_registry: Dict[str, Type[BaseCommand]] = {}
    
    def register_command(self, name: str, command_class: Type[BaseCommand]) -> None:
        """Register a command class.
        
        Args:
            name: Command name
            command_class: Command class to register
        """
        self._command_registry[name] = command_class
        logger.debug(f"Registered command: {name}")
    
    def create_command(
        self,
        name: str,
        container: Optional[Container] = None
    ) -> BaseCommand:
        """Create a command instance.
        
        Args:
            name: Command name
            container: Optional container override
            
        Returns:
            Initialized command instance
            
        Raises:
            KeyError: If command not registered
        """
        if name not in self._command_registry:
            raise KeyError(f"Command not registered: {name}")
        
        # Get command class
        command_class = self._command_registry[name]
        
        # Use provided container or factory's container
        command_container = container or self.container
        if command_container is None:
            raise RuntimeError("No container available for command creation")
        
        # Create command instance
        if command_container.has(command_class):
            # Let container create it if registered
            command = command_container.resolve(command_class)
        else:
            # Create manually and initialize
            command = command_class()
            command.initialize(command_container)
        
        logger.debug(f"Created command instance: {name}")
        return command
    
    def get_registered_commands(self) -> Dict[str, Type[BaseCommand]]:
        """Get all registered commands.
        
        Returns:
            Dictionary of command names to classes
        """
        return self._command_registry.copy()
    
    def has_command(self, name: str) -> bool:
        """Check if a command is registered.
        
        Args:
            name: Command name
            
        Returns:
            True if command is registered
        """
        return name in self._command_registry


# Global command registry
_global_factory = CommandFactory()


def register_command(name: str):
    """Decorator to register a command class.
    
    Args:
        name: Command name
        
    Returns:
        Decorator function
    """
    def decorator(command_class: Type[BaseCommand]) -> Type[BaseCommand]:
        _global_factory.register_command(name, command_class)
        return command_class
    
    return decorator


def get_command_factory() -> CommandFactory:
    """Get the global command factory.
    
    Returns:
        Global command factory instance
    """
    return _global_factory


def create_command(name: str, container: Container) -> BaseCommand:
    """Create a command instance using the global factory.
    
    Args:
        name: Command name
        container: DI container
        
    Returns:
        Initialized command instance
    """
    return _global_factory.create_command(name, container)