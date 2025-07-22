"""Command factory with dependency injection support."""

import inspect
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, get_type_hints

from loguru import logger

from core.di import Container as DIContainer
from cli.middleware.base import MiddlewarePipeline
from cli.pipeline.base import CommandPipeline
from cli.plugins.cli_plugin import CLIPluginLoader


T = TypeVar("T")


class CommandFactory:
    """Factory for creating commands with dependency injection."""
    
    def __init__(self, container: Optional[DIContainer] = None):
        """Initialize command factory."""
        self.container = container or DIContainer()
        self.middleware_pipeline = MiddlewarePipeline()
        self.command_pipeline = CommandPipeline()
        self.plugin_loader = CLIPluginLoader()
        
        # Register core services
        self._register_core_services()
    
    def _register_core_services(self) -> None:
        """Register core services."""
        self.container.register(DIContainer, self.container, instance=True)
        self.container.register(MiddlewarePipeline, self.middleware_pipeline, instance=True)
        self.container.register(CommandPipeline, self.command_pipeline, instance=True)
        self.container.register(CLIPluginLoader, self.plugin_loader, instance=True)
        self.container.register(CommandFactory, self, instance=True)
    
    def register_service(
        self,
        service_type: Type[T],
        implementation: Optional[Type[T]] = None,
        instance: Optional[T] = None,
        singleton: bool = True
    ) -> "CommandFactory":
        """Register a service."""
        if instance is not None:
            self.container.register(service_type, instance, instance=True)
        elif implementation is not None:
            self.container.register(service_type, implementation, singleton=singleton)
        else:
            self.container.register(service_type, service_type, singleton=singleton)
        
        return self
    
    def create_command(
        self,
        command_func: Callable,
        inject_dependencies: bool = True
    ) -> Callable:
        """Create command with dependency injection.
        
        Args:
            command_func: Original command function
            inject_dependencies: Whether to inject dependencies
            
        Returns:
            Enhanced command function
        """
        if not inject_dependencies:
            return command_func
        
        # Get function signature and type hints
        sig = inspect.signature(command_func)
        hints = get_type_hints(command_func)
        
        # Identify injectable parameters
        injectable_params = {}
        for param_name, param in sig.parameters.items():
            if param_name in hints:
                param_type = hints[param_name]
                if self.container.has(param_type):
                    injectable_params[param_name] = param_type
        
        if not injectable_params:
            return command_func
        
        def wrapper(*args, **kwargs):
            """Wrapper function with dependency injection."""
            # Inject dependencies
            for param_name, param_type in injectable_params.items():
                if param_name not in kwargs:
                    try:
                        kwargs[param_name] = self.container.resolve(param_type)
                    except Exception as e:
                        logger.warning(f"Failed to inject {param_name}: {e}")
            
            return command_func(*args, **kwargs)
        
        # Preserve metadata
        wrapper.__name__ = command_func.__name__
        wrapper.__doc__ = command_func.__doc__
        wrapper.__annotations__ = command_func.__annotations__
        
        # Copy Typer metadata if present
        if hasattr(command_func, "__typer_meta__"):
            wrapper.__typer_meta__ = command_func.__typer_meta__
        
        return wrapper
    
    def create_middleware_command(
        self,
        command_func: Callable,
        middleware: Optional[List[str]] = None
    ) -> Callable:
        """Create command with middleware support.
        
        Args:
            command_func: Original command function
            middleware: List of middleware names to apply
            
        Returns:
            Command with middleware
        """
        def wrapper(*args, **kwargs):
            """Wrapper with middleware execution."""
            from cli.middleware.base import CommandContext
            
            # Create context
            context = CommandContext(
                command_name=command_func.__name__,
                args=args,
                kwargs=kwargs
            )
            
            # Create handler that properly extracts args/kwargs
            def handler(ctx):
                return command_func(*ctx.args, **ctx.kwargs)
            
            # Execute through middleware
            result = self.middleware_pipeline.execute(context, handler)
            
            if result.success:
                return result.data
            else:
                if result.error:
                    raise result.error
                raise RuntimeError("Command failed")
        
        # Preserve metadata
        wrapper.__name__ = command_func.__name__
        wrapper.__doc__ = command_func.__doc__
        wrapper.__annotations__ = command_func.__annotations__
        
        if hasattr(command_func, "__typer_meta__"):
            wrapper.__typer_meta__ = command_func.__typer_meta__
        
        return wrapper
    
    async def create_pipeline_command(
        self,
        command_func: Callable,
        hooks: Optional[List[str]] = None
    ) -> Callable:
        """Create command with pipeline support.
        
        Args:
            command_func: Original command function
            hooks: List of hook names to apply
            
        Returns:
            Command with pipeline
        """
        async def wrapper(*args, **kwargs):
            """Wrapper with pipeline execution."""
            return await self.command_pipeline.execute(
                command_func.__name__,
                command_func,
                *args,
                **kwargs
            )
        
        # Preserve metadata
        wrapper.__name__ = command_func.__name__
        wrapper.__doc__ = command_func.__doc__
        wrapper.__annotations__ = command_func.__annotations__
        
        if hasattr(command_func, "__typer_meta__"):
            wrapper.__typer_meta__ = command_func.__typer_meta__
        
        return wrapper
    
    def create_full_command(
        self,
        command_func: Callable,
        inject_dependencies: bool = True,
        use_middleware: bool = True,
        use_pipeline: bool = False,
        middleware_names: Optional[List[str]] = None,
        hook_names: Optional[List[str]] = None
    ) -> Callable:
        """Create command with full feature set.
        
        Args:
            command_func: Original command function
            inject_dependencies: Whether to inject dependencies
            use_middleware: Whether to use middleware
            use_pipeline: Whether to use command pipeline
            middleware_names: Specific middleware to use
            hook_names: Specific hooks to use
            
        Returns:
            Fully enhanced command
        """
        enhanced_func = command_func
        
        # Apply dependency injection
        if inject_dependencies:
            enhanced_func = self.create_command(enhanced_func, inject_dependencies=True)
        
        # Apply middleware
        if use_middleware:
            enhanced_func = self.create_middleware_command(
                enhanced_func,
                middleware=middleware_names
            )
        
        # Apply pipeline
        if use_pipeline:
            enhanced_func = self.create_pipeline_command(
                enhanced_func,
                hooks=hook_names
            )
        
        return enhanced_func
    
    def load_plugins(self) -> None:
        """Load and register plugins."""
        self.plugin_loader.load_all_plugins()
        
        # Register plugin middleware
        for middleware in self.plugin_loader.middleware:
            self.middleware_pipeline.add(middleware)
        
        # Register plugin hooks
        for hook in self.plugin_loader.hooks:
            self.command_pipeline.add_hook(hook)
    
    def get_plugin_commands(self) -> Dict[str, Callable]:
        """Get all plugin commands."""
        return self.plugin_loader.commands


# Decorators for easy command creation
def command(
    factory: Optional[CommandFactory] = None,
    inject: bool = True,
    middleware: bool = True,
    pipeline: bool = False
):
    """Decorator for creating enhanced commands."""
    if factory is None:
        factory = CommandFactory()
    
    def decorator(func: Callable) -> Callable:
        return factory.create_full_command(
            func,
            inject_dependencies=inject,
            use_middleware=middleware,
            use_pipeline=pipeline
        )
    
    return decorator


def injectable(service_type: Type[T]) -> Callable[[Callable], Callable]:
    """Decorator to mark a service as injectable."""
    def decorator(cls: Type[T]) -> Type[T]:
        # Add metadata to class
        cls._injectable_type = service_type
        return cls
    
    return decorator


# Global factory instance
_global_factory: Optional[CommandFactory] = None


def get_command_factory() -> CommandFactory:
    """Get global command factory instance."""
    global _global_factory
    if _global_factory is None:
        _global_factory = CommandFactory()
    return _global_factory


def set_command_factory(factory: CommandFactory) -> None:
    """Set global command factory instance."""
    global _global_factory
    _global_factory = factory