"""Error handling middleware for CLI commands."""

import asyncio
import sys
import traceback
from typing import Any, Callable, Dict, List, Optional, Type

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from cli.middleware.base import CommandContext, Middleware, MiddlewareResult


class ErrorMiddleware(Middleware):
    """Middleware for comprehensive error handling."""
    
    def __init__(
        self,
        name: str = "ErrorMiddleware",
        handlers: Optional[Dict[Type[Exception], Callable]] = None,
        fallback_handler: Optional[Callable] = None,
        show_traceback: bool = False,
        console: Optional[Console] = None
    ):
        """Initialize error middleware.
        
        Args:
            name: Middleware name
            handlers: Exception type to handler mapping
            fallback_handler: Handler for unhandled exceptions
            show_traceback: Whether to show full traceback
            console: Rich console for error display
        """
        super().__init__(name)
        self.handlers = handlers or {}
        self.fallback_handler = fallback_handler or self._default_handler
        self.show_traceback = show_traceback
        self.console = console or Console()
    
    def _default_handler(self, context: CommandContext, error: Exception) -> MiddlewareResult:
        """Default error handler."""
        error_msg = f"Command '{context.command_name}' failed: {str(error)}"
        
        if self.show_traceback:
            tb = traceback.format_exc()
            logger.error(f"{error_msg}\n{tb}")
            
            # Rich error display
            error_panel = Panel(
                Text(tb, style="red"),
                title=f"Error in {context.command_name}",
                border_style="red"
            )
            self.console.print(error_panel)
        else:
            logger.error(error_msg)
            self.console.print(f"[red]Error:[/red] {str(error)}")
        
        return MiddlewareResult.fail(error, handled=True)
    
    def _find_handler(self, error: Exception) -> Optional[Callable]:
        """Find appropriate error handler."""
        # Exact match
        error_type = type(error)
        if error_type in self.handlers:
            return self.handlers[error_type]
        
        # Check inheritance
        for exc_type, handler in self.handlers.items():
            if isinstance(error, exc_type):
                return handler
        
        return None
    
    async def process(
        self,
        context: CommandContext,
        next_handler: Callable[[CommandContext], Any]
    ) -> MiddlewareResult:
        """Process error middleware."""
        try:
            # Execute next handler
            if asyncio.iscoroutinefunction(next_handler):
                result = await next_handler(context)
            else:
                result = next_handler(context)
            
            # Check for errors in result
            if not result.success and result.error:
                handler = self._find_handler(result.error) or self.fallback_handler
                return handler(context, result.error)
            
            return result
            
        except Exception as e:
            # Handle uncaught exceptions
            logger.exception(f"Uncaught exception in {context.command_name}")
            
            handler = self._find_handler(e) or self.fallback_handler
            return handler(context, e)


class RetryMiddleware(ErrorMiddleware):
    """Middleware with retry capability."""
    
    def __init__(
        self,
        name: str = "RetryMiddleware",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        retry_exceptions: Optional[List[Type[Exception]]] = None,
        **kwargs
    ):
        """Initialize retry middleware.
        
        Args:
            name: Middleware name
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries in seconds
            retry_exceptions: Exceptions to retry on
        """
        super().__init__(name=name, **kwargs)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_exceptions = retry_exceptions or [Exception]
    
    def _should_retry(self, error: Exception) -> bool:
        """Check if error should trigger retry."""
        return any(isinstance(error, exc_type) for exc_type in self.retry_exceptions)
    
    async def process(
        self,
        context: CommandContext,
        next_handler: Callable[[CommandContext], Any]
    ) -> MiddlewareResult:
        """Process with retry logic."""
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(f"Retry attempt {attempt}/{self.max_retries} for {context.command_name}")
                    await asyncio.sleep(self.retry_delay)
                
                # Execute handler
                if asyncio.iscoroutinefunction(next_handler):
                    result = await next_handler(context)
                else:
                    result = next_handler(context)
                
                # Check result
                if result.success or (result.error and not self._should_retry(result.error)):
                    return result
                
                last_error = result.error
                
            except Exception as e:
                last_error = e
                if not self._should_retry(e) or attempt == self.max_retries:
                    raise
        
        # All retries exhausted
        return MiddlewareResult.fail(
            last_error or Exception("Max retries exceeded"),
            retries=self.max_retries
        )


class FallbackMiddleware(ErrorMiddleware):
    """Middleware with fallback command support."""
    
    def __init__(
        self,
        name: str = "FallbackMiddleware",
        fallback_commands: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """Initialize fallback middleware.
        
        Args:
            name: Middleware name
            fallback_commands: Command to fallback command mapping
        """
        super().__init__(name=name, **kwargs)
        self.fallback_commands = fallback_commands or {}
    
    async def process(
        self,
        context: CommandContext,
        next_handler: Callable[[CommandContext], Any]
    ) -> MiddlewareResult:
        """Process with fallback logic."""
        try:
            result = await super().process(context, next_handler)
            
            if not result.success and context.command_name in self.fallback_commands:
                fallback = self.fallback_commands[context.command_name]
                logger.warning(f"Falling back from {context.command_name} to {fallback}")
                
                # Update context for fallback
                context.command_name = fallback
                context.set("is_fallback", True)
                context.set("original_command", context.command_name)
                
                # Execute fallback
                return await super().process(context, next_handler)
            
            return result
            
        except Exception as e:
            if context.command_name in self.fallback_commands:
                fallback = self.fallback_commands[context.command_name]
                logger.warning(f"Error in {context.command_name}, trying fallback: {fallback}")
                
                try:
                    context.command_name = fallback
                    context.set("is_fallback", True)
                    return await super().process(context, next_handler)
                except Exception:
                    pass
            
            raise