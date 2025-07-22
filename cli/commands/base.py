"""Base command class with dependency injection and middleware support.

This module provides the base class for all CLI commands with:
- Dependency injection support
- Command middleware/hooks
- Common error handling
- Configuration management
"""

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

import typer
from loguru import logger
from rich.console import Console

from core.di import Container


T = TypeVar("T")


class CommandContext:
    """Context object passed through command execution."""
    
    def __init__(
        self,
        command_name: str,
        args: Dict[str, Any],
        container: Container,
        console: Console,
    ):
        """Initialize command context.
        
        Args:
            command_name: Name of the command being executed
            args: Command arguments
            container: DI container
            console: Rich console for output
        """
        self.command_name = command_name
        self.args = args
        self.container = container
        self.console = console
        self.metadata: Dict[str, Any] = {}
        self.result: Any = None
        self.error: Optional[Exception] = None


class CommandMiddleware(ABC):
    """Base class for command middleware."""
    
    @abstractmethod
    async def __call__(
        self,
        context: CommandContext,
        next_middleware: Callable[[CommandContext], Any]
    ) -> Any:
        """Execute middleware.
        
        Args:
            context: Command context
            next_middleware: Next middleware in chain
            
        Returns:
            Command result
        """
        pass


class BaseCommand(ABC):
    """Base class for all CLI commands with DI support."""
    
    def __init__(self, container: Optional[Container] = None):
        """Initialize command with optional container.
        
        Args:
            container: DI container (will be injected if not provided)
        """
        self._container = container
        self._middleware: List[CommandMiddleware] = []
        self._console: Optional[Console] = None
        self._initialized = False
    
    @property
    def container(self) -> Container:
        """Get the DI container."""
        if self._container is None:
            raise RuntimeError("Container not initialized. Call initialize() first.")
        return self._container
    
    @property
    def console(self) -> Console:
        """Get the console for output."""
        if self._console is None:
            self._console = self.container.resolve(Console)
        return self._console
    
    def initialize(self, container: Container) -> None:
        """Initialize command with DI container.
        
        Args:
            container: DI container
        """
        self._container = container
        self._setup_logging()
        self._setup_middleware()
        self._initialized = True
    
    def add_middleware(self, middleware: CommandMiddleware) -> None:
        """Add middleware to the command.
        
        Args:
            middleware: Middleware instance
        """
        self._middleware.append(middleware)
    
    def inject(self, service_type: Type[T]) -> T:
        """Inject a service from the container.
        
        Args:
            service_type: Type of service to inject
            
        Returns:
            Service instance
        """
        return self.container.resolve(service_type)
    
    def run(self, **kwargs) -> Any:
        """Run the command with middleware support.
        
        Args:
            **kwargs: Command arguments
            
        Returns:
            Command result
        """
        if not self._initialized:
            raise RuntimeError("Command not initialized. Call initialize() first.")
        
        # Create context
        context = CommandContext(
            command_name=self.__class__.__name__,
            args=kwargs,
            container=self.container,
            console=self.console,
        )
        
        # Build middleware chain
        def execute_command(ctx: CommandContext) -> Any:
            return self.execute(**ctx.args)
        
        # Apply middleware in reverse order  
        chain = execute_command
        for middleware in reversed(self._middleware):
            # Create closure to capture middleware and next_chain
            def make_chain(mw, next_fn):
                def chain_fn(ctx):
                    # Handle both sync and async middleware
                    import asyncio
                    import inspect
                    result = mw(ctx, next_fn)
                    if inspect.iscoroutine(result):
                        return asyncio.run(result)
                    return result
                return chain_fn
            chain = make_chain(middleware, chain)
        
        try:
            # Execute the chain
            result = chain(context)
            context.result = result
            return result
        except Exception as e:
            context.error = e
            self._handle_error(e)
    
    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Execute the command logic.
        
        Must be implemented by subclasses.
        
        Args:
            **kwargs: Command arguments
            
        Returns:
            Command result
        """
        pass
    
    def _setup_logging(self) -> None:
        """Setup logging for the command."""
        # Remove default logger
        logger.remove()
        
        # Get logging configuration from container
        log_level = self.container.get_config("logging.level", "INFO")
        log_format = self.container.get_config("logging.format", "simple")
        
        # Add console logger
        if log_format == "simple":
            format_string = "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
        else:
            format_string = "{time} | {level} | {name}:{function}:{line} | {message}"
        
        logger.add(
            sys.stderr,
            level=log_level,
            format=format_string,
            colorize=True,
        )
    
    def _setup_middleware(self) -> None:
        """Setup default middleware."""
        # Add default middleware from container if available
        if self.container.has(List[CommandMiddleware]):
            default_middleware = self.container.resolve(List[CommandMiddleware])
            for middleware in default_middleware:
                self.add_middleware(middleware)
    
    def _handle_error(self, error: Exception) -> None:
        """Handle command errors.
        
        Args:
            error: Exception that occurred
        """
        logger.error(f"Command failed: {str(error)}")
        
        if self.container.get_config("debug", False):
            logger.exception(error)
        
        self.console.print(f"[red]Error:[/red] {str(error)}")
        raise typer.Exit(1)
    
    def resolve_path(self, path: Path) -> Path:
        """Resolve path relative to project root.
        
        Args:
            path: Path to resolve
            
        Returns:
            Resolved path
        """
        if path.is_absolute():
            return path
        
        # Get project root from container
        project_root = self.container.get_config("project_root", Path.cwd())
        return project_root / path
    
    def validate_path(
        self,
        path: Path,
        must_exist: bool = False,
        must_be_file: bool = False,
        must_be_dir: bool = False,
    ) -> None:
        """Validate a path with error handling.
        
        Args:
            path: Path to validate
            must_exist: Whether path must exist
            must_be_file: Whether path must be a file
            must_be_dir: Whether path must be a directory
            
        Raises:
            typer.Exit: If validation fails
        """
        if must_exist and not path.exists():
            self.console.print(f"[red]Error:[/red] Path does not exist: {path}")
            raise typer.Exit(1)
        
        if path.exists():
            if must_be_file and not path.is_file():
                self.console.print(f"[red]Error:[/red] Path is not a file: {path}")
                raise typer.Exit(1)
            
            if must_be_dir and not path.is_dir():
                self.console.print(f"[red]Error:[/red] Path is not a directory: {path}")
                raise typer.Exit(1)


class AsyncBaseCommand(BaseCommand):
    """Base class for asynchronous commands."""
    
    async def run_async(self, **kwargs) -> Any:
        """Run the command asynchronously.
        
        Args:
            **kwargs: Command arguments
            
        Returns:
            Command result
        """
        if not self._initialized:
            raise RuntimeError("Command not initialized. Call initialize() first.")
        
        # Create context
        context = CommandContext(
            command_name=self.__class__.__name__,
            args=kwargs,
            container=self.container,
            console=self.console,
        )
        
        # Build async middleware chain
        async def execute_command(ctx: CommandContext) -> Any:
            return await self.execute_async(**ctx.args)
        
        # Apply middleware in reverse order
        chain = execute_command
        for middleware in reversed(self._middleware):
            next_chain = chain
            chain = lambda ctx: middleware(ctx, next_chain)
        
        try:
            # Execute the chain
            result = await chain(context)
            context.result = result
            return result
        except Exception as e:
            context.error = e
            self._handle_error(e)
    
    def run(self, **kwargs) -> Any:
        """Run the async command synchronously.
        
        Args:
            **kwargs: Command arguments
            
        Returns:
            Command result
        """
        import asyncio
        return asyncio.run(self.run_async(**kwargs))
    
    @abstractmethod
    async def execute_async(self, **kwargs) -> Any:
        """Execute the async command logic.
        
        Must be implemented by subclasses.
        
        Args:
            **kwargs: Command arguments
            
        Returns:
            Command result
        """
        pass
    
    def execute(self, **kwargs) -> Any:
        """Synchronous execute (not used for async commands)."""
        raise NotImplementedError("Use execute_async for asynchronous commands")