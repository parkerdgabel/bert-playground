"""Base classes for training pipeline architecture.

This module defines the core pipeline and middleware abstractions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

from training.commands.base import Command, CommandContext, CommandResult


@runtime_checkable
class Middleware(Protocol):
    """Protocol for pipeline middleware."""
    
    @property
    def name(self) -> str:
        """Middleware name for debugging."""
        ...
    
    @property
    def enabled(self) -> bool:
        """Whether middleware is enabled."""
        ...
    
    def before_pipeline(self, context: CommandContext) -> CommandContext:
        """Called before pipeline execution starts.
        
        Args:
            context: Pipeline context
            
        Returns:
            Modified context
        """
        ...
    
    def after_pipeline(
        self,
        context: CommandContext,
        result: CommandResult
    ) -> CommandResult:
        """Called after pipeline execution completes.
        
        Args:
            context: Pipeline context
            result: Pipeline execution result
            
        Returns:
            Modified result
        """
        ...
    
    def before_command(
        self,
        command: Command,
        context: CommandContext
    ) -> tuple[Command, CommandContext]:
        """Called before each command execution.
        
        Args:
            command: Command to be executed
            context: Current context
            
        Returns:
            Tuple of (possibly modified command, possibly modified context)
        """
        ...
    
    def after_command(
        self,
        command: Command,
        context: CommandContext,
        result: CommandResult
    ) -> CommandResult:
        """Called after each command execution.
        
        Args:
            command: Command that was executed
            context: Current context
            result: Command execution result
            
        Returns:
            Modified result
        """
        ...
    
    def on_error(
        self,
        command: Command,
        context: CommandContext,
        error: Exception
    ) -> CommandResult | None:
        """Called when a command raises an error.
        
        Args:
            command: Command that failed
            context: Current context
            error: The exception that was raised
            
        Returns:
            CommandResult to use instead of error, or None to propagate error
        """
        ...


class BaseMiddleware(ABC):
    """Base implementation of Middleware with sensible defaults."""
    
    def __init__(self, name: str | None = None, enabled: bool = True):
        """Initialize middleware.
        
        Args:
            name: Middleware name
            enabled: Whether middleware is enabled
        """
        self._name = name or self.__class__.__name__
        self._enabled = enabled
    
    @property
    def name(self) -> str:
        """Middleware name."""
        return self._name
    
    @property
    def enabled(self) -> bool:
        """Whether middleware is enabled."""
        return self._enabled
    
    def before_pipeline(self, context: CommandContext) -> CommandContext:
        """Default implementation - no modification."""
        return context
    
    def after_pipeline(
        self,
        context: CommandContext,
        result: CommandResult
    ) -> CommandResult:
        """Default implementation - no modification."""
        return result
    
    def before_command(
        self,
        command: Command,
        context: CommandContext
    ) -> tuple[Command, CommandContext]:
        """Default implementation - no modification."""
        return command, context
    
    def after_command(
        self,
        command: Command,
        context: CommandContext,
        result: CommandResult
    ) -> CommandResult:
        """Default implementation - no modification."""
        return result
    
    def on_error(
        self,
        command: Command,
        context: CommandContext,
        error: Exception
    ) -> CommandResult | None:
        """Default implementation - propagate error."""
        return None


@dataclass
class Pipeline:
    """Training pipeline that executes commands with middleware support."""
    
    commands: list[Command]
    middleware: list[Middleware]
    name: str = "TrainingPipeline"
    stop_on_error: bool = True
    
    def add_command(self, command: Command) -> "Pipeline":
        """Add a command to the pipeline."""
        self.commands.append(command)
        return self
    
    def add_middleware(self, middleware: Middleware) -> "Pipeline":
        """Add middleware to the pipeline."""
        self.middleware.append(middleware)
        return self
    
    def execute(self, context: CommandContext) -> CommandResult:
        """Execute the pipeline with all middleware."""
        # Apply before_pipeline middleware
        for mw in self.middleware:
            if mw.enabled:
                context = mw.before_pipeline(context)
        
        # Initialize result
        result = CommandResult(success=True)
        
        # Execute commands
        for command in self.commands:
            # Check if command can execute
            if not command.can_execute(context):
                continue
            
            # Apply before_command middleware
            cmd_to_execute = command
            ctx_to_use = context
            for mw in self.middleware:
                if mw.enabled:
                    cmd_to_execute, ctx_to_use = mw.before_command(
                        cmd_to_execute, ctx_to_use
                    )
            
            try:
                # Execute command
                cmd_result = cmd_to_execute.execute(ctx_to_use)
                
                # Apply after_command middleware
                for mw in self.middleware:
                    if mw.enabled:
                        cmd_result = mw.after_command(
                            cmd_to_execute, ctx_to_use, cmd_result
                        )
                
                # Merge results
                result.merge(cmd_result)
                
                # Check if we should stop
                if not cmd_result.success and self.stop_on_error:
                    break
                
                if cmd_result.should_skip_remaining:
                    break
                    
            except Exception as e:
                # Apply error middleware
                error_handled = False
                for mw in self.middleware:
                    if mw.enabled:
                        error_result = mw.on_error(cmd_to_execute, ctx_to_use, e)
                        if error_result is not None:
                            result.merge(error_result)
                            error_handled = True
                            break
                
                if not error_handled:
                    # Create error result
                    result = CommandResult(
                        success=False,
                        error=e,
                        should_continue=not self.stop_on_error
                    )
                    if self.stop_on_error:
                        break
        
        # Apply after_pipeline middleware
        for mw in self.middleware:
            if mw.enabled:
                result = mw.after_pipeline(context, result)
        
        return result


class PipelineBuilder:
    """Builder for creating pipelines with fluent interface."""
    
    def __init__(self, name: str = "Pipeline"):
        """Initialize pipeline builder."""
        self._name = name
        self._commands: list[Command] = []
        self._middleware: list[Middleware] = []
        self._stop_on_error = True
    
    def with_name(self, name: str) -> "PipelineBuilder":
        """Set pipeline name."""
        self._name = name
        return self
    
    def add_command(self, command: Command) -> "PipelineBuilder":
        """Add a command."""
        self._commands.append(command)
        return self
    
    def add_commands(self, *commands: Command) -> "PipelineBuilder":
        """Add multiple commands."""
        self._commands.extend(commands)
        return self
    
    def add_middleware(self, middleware: Middleware) -> "PipelineBuilder":
        """Add middleware."""
        self._middleware.append(middleware)
        return self
    
    def add_middlewares(self, *middlewares: Middleware) -> "PipelineBuilder":
        """Add multiple middleware."""
        self._middleware.extend(middlewares)
        return self
    
    def continue_on_error(self, value: bool = True) -> "PipelineBuilder":
        """Set whether to continue on error."""
        self._stop_on_error = not value
        return self
    
    def build(self) -> Pipeline:
        """Build the pipeline."""
        return Pipeline(
            commands=self._commands.copy(),
            middleware=self._middleware.copy(),
            name=self._name,
            stop_on_error=self._stop_on_error
        )