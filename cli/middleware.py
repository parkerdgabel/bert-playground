"""Command middleware implementations for k-bert CLI.

This module provides various middleware for cross-cutting concerns like
logging, error handling, validation, and performance tracking.
"""

import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .commands.base import CommandContext, CommandMiddleware


class LoggingMiddleware(CommandMiddleware):
    """Middleware for logging command execution."""
    
    def __init__(self, log_level: str = "INFO"):
        """Initialize logging middleware.
        
        Args:
            log_level: Minimum log level to capture
        """
        self.log_level = log_level
    
    async def __call__(
        self,
        context: CommandContext,
        next_middleware: Callable[[CommandContext], Any]
    ) -> Any:
        """Log command execution.
        
        Args:
            context: Command context
            next_middleware: Next middleware in chain
            
        Returns:
            Command result
        """
        # Log command start
        logger.info(
            f"Executing command: {context.command_name}",
            command=context.command_name,
            args=self._sanitize_args(context.args),
        )
        
        try:
            # Execute next middleware
            result = await next_middleware(context)
            
            # Log success
            logger.info(
                f"Command completed: {context.command_name}",
                command=context.command_name,
                success=True,
            )
            
            return result
            
        except Exception as e:
            # Log failure
            logger.error(
                f"Command failed: {context.command_name}",
                command=context.command_name,
                success=False,
                error=str(e),
            )
            raise
    
    def _sanitize_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize sensitive arguments for logging.
        
        Args:
            args: Command arguments
            
        Returns:
            Sanitized arguments
        """
        sensitive_keys = ["password", "key", "token", "secret"]
        sanitized = {}
        
        for key, value in args.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                sanitized[key] = "***"
            else:
                sanitized[key] = value
        
        return sanitized


class ErrorHandlingMiddleware(CommandMiddleware):
    """Middleware for consistent error handling."""
    
    def __init__(self, debug: bool = False):
        """Initialize error handling middleware.
        
        Args:
            debug: Whether to show full stack traces
        """
        self.debug = debug
    
    async def __call__(
        self,
        context: CommandContext,
        next_middleware: Callable[[CommandContext], Any]
    ) -> Any:
        """Handle errors during command execution.
        
        Args:
            context: Command context
            next_middleware: Next middleware in chain
            
        Returns:
            Command result
        """
        try:
            return await next_middleware(context)
            
        except KeyboardInterrupt:
            # Handle user interruption
            context.console.print("\n[yellow]Command interrupted by user[/yellow]")
            raise
            
        except Exception as e:
            # Store error in context
            context.error = e
            
            # Format error message
            error_panel = self._format_error(e, context)
            context.console.print(error_panel)
            
            # Log full error if debug
            if self.debug:
                logger.exception("Command error details:")
            
            # Re-raise to maintain error flow
            raise
    
    def _format_error(self, error: Exception, context: CommandContext) -> Panel:
        """Format error for display.
        
        Args:
            error: Exception that occurred
            context: Command context
            
        Returns:
            Formatted error panel
        """
        error_type = type(error).__name__
        error_msg = str(error)
        
        # Create error content
        content = f"[bold red]{error_type}:[/bold red] {error_msg}"
        
        # Add context if available
        if hasattr(error, "errors") and error.errors:
            content += "\n\n[yellow]Details:[/yellow]"
            for detail in error.errors:
                content += f"\n  • {detail}"
        
        # Add suggestions if available
        suggestions = self._get_error_suggestions(error, context)
        if suggestions:
            content += "\n\n[green]Suggestions:[/green]"
            for suggestion in suggestions:
                content += f"\n  → {suggestion}"
        
        return Panel(
            content,
            title=f"[red]Command Failed: {context.command_name}[/red]",
            border_style="red",
            padding=(1, 2),
        )
    
    def _get_error_suggestions(
        self,
        error: Exception,
        context: CommandContext
    ) -> list[str]:
        """Get suggestions for common errors.
        
        Args:
            error: Exception that occurred
            context: Command context
            
        Returns:
            List of suggestions
        """
        suggestions = []
        
        error_msg = str(error).lower()
        
        if "permission" in error_msg:
            suggestions.append("Check file permissions")
            suggestions.append("Try running with appropriate privileges")
        elif "not found" in error_msg:
            suggestions.append("Verify the file or resource exists")
            suggestions.append("Check the path is correct")
        elif "connection" in error_msg or "network" in error_msg:
            suggestions.append("Check your internet connection")
            suggestions.append("Verify any API endpoints or servers are accessible")
        elif "config" in error_msg:
            suggestions.append("Run 'k-bert config validate' to check configuration")
            suggestions.append("Run 'k-bert config init' to create default configuration")
        
        return suggestions


class ValidationMiddleware(CommandMiddleware):
    """Middleware for validating command inputs."""
    
    async def __call__(
        self,
        context: CommandContext,
        next_middleware: Callable[[CommandContext], Any]
    ) -> Any:
        """Validate command inputs before execution.
        
        Args:
            context: Command context
            next_middleware: Next middleware in chain
            
        Returns:
            Command result
        """
        # Validate required arguments
        validation_errors = self._validate_args(context)
        
        if validation_errors:
            # Display validation errors
            self._display_validation_errors(validation_errors, context)
            
            # Create validation exception
            from .config.validators import ConfigValidationError
            raise ConfigValidationError(validation_errors)
        
        # Continue with execution
        return await next_middleware(context)
    
    def _validate_args(self, context: CommandContext) -> list[str]:
        """Validate command arguments.
        
        Args:
            context: Command context
            
        Returns:
            List of validation errors
        """
        errors = []
        args = context.args
        
        # Common validations
        if "config" in args and args["config"]:
            config_path = args["config"]
            if not config_path.exists():
                errors.append(f"Configuration file not found: {config_path}")
        
        if "output_dir" in args and args["output_dir"]:
            output_dir = args["output_dir"]
            if output_dir.exists() and not output_dir.is_dir():
                errors.append(f"Output path exists but is not a directory: {output_dir}")
        
        # Numeric validations
        numeric_args = {
            "batch_size": (1, 1024),
            "epochs": (1, 1000),
            "max_length": (1, 8192),
        }
        
        for arg_name, (min_val, max_val) in numeric_args.items():
            if arg_name in args and args[arg_name] is not None:
                value = args[arg_name]
                if not isinstance(value, (int, float)) or value < min_val or value > max_val:
                    errors.append(
                        f"{arg_name} must be between {min_val} and {max_val}, got {value}"
                    )
        
        return errors
    
    def _display_validation_errors(
        self,
        errors: list[str],
        context: CommandContext
    ) -> None:
        """Display validation errors.
        
        Args:
            errors: List of validation errors
            context: Command context
        """
        console = context.console
        
        console.print(
            Panel(
                "\n".join(f"• {error}" for error in errors),
                title="[red]Validation Failed[/red]",
                border_style="red",
            )
        )


class PerformanceMiddleware(CommandMiddleware):
    """Middleware for tracking command performance."""
    
    def __init__(self, threshold_seconds: float = 5.0):
        """Initialize performance middleware.
        
        Args:
            threshold_seconds: Threshold for slow command warning
        """
        self.threshold_seconds = threshold_seconds
    
    async def __call__(
        self,
        context: CommandContext,
        next_middleware: Callable[[CommandContext], Any]
    ) -> Any:
        """Track command performance.
        
        Args:
            context: Command context
            next_middleware: Next middleware in chain
            
        Returns:
            Command result
        """
        start_time = time.time()
        
        try:
            # Execute command
            result = await next_middleware(context)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Store in context
            context.metadata["duration_seconds"] = duration
            
            # Log performance
            logger.debug(
                f"Command {context.command_name} completed in {duration:.2f}s",
                command=context.command_name,
                duration_seconds=duration,
            )
            
            # Warn if slow
            if duration > self.threshold_seconds:
                context.console.print(
                    f"[yellow]Note: Command took {duration:.1f} seconds to complete[/yellow]"
                )
            
            return result
            
        except Exception:
            # Still track duration for failed commands
            duration = time.time() - start_time
            context.metadata["duration_seconds"] = duration
            raise


class CachingMiddleware(CommandMiddleware):
    """Middleware for caching command results."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize caching middleware.
        
        Args:
            cache_dir: Directory for cache storage
        """
        from pathlib import Path
        self.cache_dir = cache_dir or Path.home() / ".k-bert" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Any] = {}
    
    async def __call__(
        self,
        context: CommandContext,
        next_middleware: Callable[[CommandContext], Any]
    ) -> Any:
        """Cache command results when appropriate.
        
        Args:
            context: Command context
            next_middleware: Next middleware in chain
            
        Returns:
            Command result
        """
        # Check if command is cacheable
        if not self._is_cacheable(context):
            return await next_middleware(context)
        
        # Generate cache key
        cache_key = self._generate_cache_key(context)
        
        # Check cache
        if cache_key in self._cache:
            logger.debug(f"Cache hit for command: {context.command_name}")
            return self._cache[cache_key]
        
        # Execute command
        result = await next_middleware(context)
        
        # Cache result
        self._cache[cache_key] = result
        logger.debug(f"Cached result for command: {context.command_name}")
        
        return result
    
    def _is_cacheable(self, context: CommandContext) -> bool:
        """Check if command result can be cached.
        
        Args:
            context: Command context
            
        Returns:
            True if cacheable
        """
        # Only cache read-only commands
        cacheable_commands = ["info", "list", "validate", "inspect"]
        
        return any(
            cmd in context.command_name.lower()
            for cmd in cacheable_commands
        )
    
    def _generate_cache_key(self, context: CommandContext) -> str:
        """Generate cache key for command.
        
        Args:
            context: Command context
            
        Returns:
            Cache key
        """
        import hashlib
        import json
        
        # Create key from command name and args
        key_data = {
            "command": context.command_name,
            "args": {k: str(v) for k, v in context.args.items()},
        }
        
        key_json = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_json.encode()).hexdigest()


class ProfilingMiddleware(CommandMiddleware):
    """Middleware for detailed performance profiling."""
    
    def __init__(self, profile_dir: Optional[Path] = None):
        """Initialize profiling middleware.
        
        Args:
            profile_dir: Directory to save profiles
        """
        from pathlib import Path
        self.profile_dir = profile_dir or Path.home() / ".k-bert" / "profiles"
        self.profile_dir.mkdir(parents=True, exist_ok=True)
    
    async def __call__(
        self,
        context: CommandContext,
        next_middleware: Callable[[CommandContext], Any]
    ) -> Any:
        """Profile command execution.
        
        Args:
            context: Command context
            next_middleware: Next middleware in chain
            
        Returns:
            Command result
        """
        import cProfile
        import pstats
        from datetime import datetime
        
        # Create profiler
        profiler = cProfile.Profile()
        
        try:
            # Profile command execution
            profiler.enable()
            result = await next_middleware(context)
            profiler.disable()
            
            # Save profile
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            profile_file = self.profile_dir / f"{context.command_name}_{timestamp}.prof"
            profiler.dump_stats(str(profile_file))
            
            # Log top functions
            stats = pstats.Stats(profiler)
            stats.sort_stats("cumulative")
            
            logger.debug(f"Profile saved to: {profile_file}")
            
            return result
            
        except Exception:
            profiler.disable()
            raise