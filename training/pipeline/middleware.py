"""Common middleware implementations for training pipelines.

This module provides reusable middleware for common training concerns
like timing, error handling, metrics collection, and validation.
"""

import time
from collections import defaultdict
from typing import Any

from loguru import logger

from training.commands.base import Command, CommandContext, CommandResult
from .base import BaseMiddleware


class TimingMiddleware(BaseMiddleware):
    """Middleware that tracks command execution times."""
    
    def __init__(self, log_timings: bool = True):
        """Initialize timing middleware.
        
        Args:
            log_timings: Whether to log timing information
        """
        super().__init__("TimingMiddleware")
        self.log_timings = log_timings
        self.timings: dict[str, list[float]] = defaultdict(list)
        self._start_times: dict[str, float] = {}
    
    def before_command(
        self,
        command: Command,
        context: CommandContext
    ) -> tuple[Command, CommandContext]:
        """Record command start time."""
        self._start_times[command.name] = time.time()
        return command, context
    
    def after_command(
        self,
        command: Command,
        context: CommandContext,
        result: CommandResult
    ) -> CommandResult:
        """Record command execution time."""
        if command.name in self._start_times:
            elapsed = time.time() - self._start_times[command.name]
            self.timings[command.name].append(elapsed)
            
            if self.log_timings:
                logger.debug(f"{command.name} took {elapsed:.3f}s")
            
            # Add timing to result
            result.metrics[f"{command.name}_time"] = elapsed
            
            del self._start_times[command.name]
        
        return result
    
    def get_summary(self) -> dict[str, dict[str, float]]:
        """Get timing summary statistics."""
        summary = {}
        for command_name, times in self.timings.items():
            if times:
                summary[command_name] = {
                    "count": len(times),
                    "total": sum(times),
                    "mean": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times),
                }
        return summary


class ErrorHandlingMiddleware(BaseMiddleware):
    """Middleware that provides error handling and recovery."""
    
    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        recoverable_errors: tuple[type[Exception], ...] = (RuntimeError,),
        log_errors: bool = True,
    ):
        """Initialize error handling middleware.
        
        Args:
            max_retries: Maximum number of retries for recoverable errors
            retry_delay: Delay between retries in seconds
            recoverable_errors: Tuple of exception types that are recoverable
            log_errors: Whether to log errors
        """
        super().__init__("ErrorHandlingMiddleware")
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.recoverable_errors = recoverable_errors
        self.log_errors = log_errors
        self._retry_counts: dict[str, int] = defaultdict(int)
    
    def on_error(
        self,
        command: Command,
        context: CommandContext,
        error: Exception
    ) -> CommandResult | None:
        """Handle command errors with retry logic."""
        # Check if error is recoverable
        if not isinstance(error, self.recoverable_errors):
            if self.log_errors:
                logger.error(f"Non-recoverable error in {command.name}: {error}")
            return None
        
        # Check retry count
        retry_key = f"{command.name}_{context.state.global_step}"
        self._retry_counts[retry_key] += 1
        
        if self._retry_counts[retry_key] > self.max_retries:
            if self.log_errors:
                logger.error(
                    f"Max retries exceeded for {command.name} at step "
                    f"{context.state.global_step}: {error}"
                )
            return None
        
        # Log retry attempt
        if self.log_errors:
            logger.warning(
                f"Retrying {command.name} (attempt {self._retry_counts[retry_key]}"
                f"/{self.max_retries}) after error: {error}"
            )
        
        # Wait before retry
        if self.retry_delay > 0:
            time.sleep(self.retry_delay)
        
        # Try to execute command again
        try:
            result = command.execute(context)
            # Clear retry count on success
            del self._retry_counts[retry_key]
            return result
        except Exception as e:
            # Recursive retry through on_error
            return self.on_error(command, context, e)


class MetricsMiddleware(BaseMiddleware):
    """Middleware that collects and aggregates metrics."""
    
    def __init__(self, rolling_window: int = 100):
        """Initialize metrics middleware.
        
        Args:
            rolling_window: Size of rolling window for metrics
        """
        super().__init__("MetricsMiddleware")
        self.rolling_window = rolling_window
        self._metrics_history: dict[str, list[float]] = defaultdict(list)
        self._step_metrics: dict[int, dict[str, float]] = {}
    
    def after_command(
        self,
        command: Command,
        context: CommandContext,
        result: CommandResult
    ) -> CommandResult:
        """Collect metrics from command execution."""
        # Store metrics for this step
        step = context.state.global_step
        if step not in self._step_metrics:
            self._step_metrics[step] = {}
        
        # Collect metrics from result
        for key, value in result.metrics.items():
            self._step_metrics[step][key] = value
            
            # Update rolling history
            self._metrics_history[key].append(value)
            if len(self._metrics_history[key]) > self.rolling_window:
                self._metrics_history[key].pop(0)
        
        # Add rolling averages to context
        for key, values in self._metrics_history.items():
            if values:
                context.metrics[f"{key}_rolling_avg"] = sum(values) / len(values)
        
        return result
    
    def get_metrics_summary(self) -> dict[str, Any]:
        """Get summary of collected metrics."""
        summary = {}
        
        # Compute statistics for each metric
        for metric_name, values in self._metrics_history.items():
            if values:
                summary[metric_name] = {
                    "current": values[-1],
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "window_size": len(values),
                }
        
        return summary


class CachingMiddleware(BaseMiddleware):
    """Middleware that caches command results."""
    
    def __init__(self, cache_size: int = 100):
        """Initialize caching middleware.
        
        Args:
            cache_size: Maximum number of cached results
        """
        super().__init__("CachingMiddleware")
        self.cache_size = cache_size
        self._cache: dict[str, CommandResult] = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def before_command(
        self,
        command: Command,
        context: CommandContext
    ) -> tuple[Command, CommandContext]:
        """Check cache before executing command."""
        # Generate cache key
        cache_key = self._generate_cache_key(command, context)
        
        # Check if result is cached
        if cache_key in self._cache:
            self._cache_hits += 1
            # Create a wrapper command that returns cached result
            cached_result = self._cache[cache_key]
            
            class CachedCommand:
                def __init__(self, result):
                    self.name = command.name
                    self.requires_grad = command.requires_grad
                    self._result = result
                
                def can_execute(self, ctx):
                    return True
                
                def execute(self, ctx):
                    return self._result
                
                def rollback(self, ctx):
                    pass
            
            return CachedCommand(cached_result), context
        
        self._cache_misses += 1
        return command, context
    
    def after_command(
        self,
        command: Command,
        context: CommandContext,
        result: CommandResult
    ) -> CommandResult:
        """Cache successful command results."""
        if result.success:
            cache_key = self._generate_cache_key(command, context)
            self._cache[cache_key] = result
            
            # Evict oldest entries if cache is full
            if len(self._cache) > self.cache_size:
                # Simple FIFO eviction
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
        
        return result
    
    def _generate_cache_key(self, command: Command, context: CommandContext) -> str:
        """Generate cache key for command and context."""
        # Simple key based on command name and step
        # In practice, would need more sophisticated key generation
        return f"{command.name}_{context.state.global_step}_{context.state.epoch}"
    
    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
        }


class ValidationMiddleware(BaseMiddleware):
    """Middleware that validates command inputs and outputs."""
    
    def __init__(self, strict: bool = False):
        """Initialize validation middleware.
        
        Args:
            strict: Whether to fail on validation errors
        """
        super().__init__("ValidationMiddleware")
        self.strict = strict
        self._validation_errors: list[str] = []
    
    def before_command(
        self,
        command: Command,
        context: CommandContext
    ) -> tuple[Command, CommandContext]:
        """Validate command inputs."""
        errors = []
        
        # Validate context state
        if context.model is None:
            errors.append(f"{command.name}: Model is None")
        
        if command.requires_grad and context.optimizer is None:
            errors.append(f"{command.name}: Optimizer is None but gradients required")
        
        if context.is_training and context.train_dataloader is None:
            errors.append(f"{command.name}: Training but no train dataloader")
        
        # Log errors
        if errors:
            for error in errors:
                logger.warning(f"Validation warning: {error}")
                self._validation_errors.append(error)
            
            if self.strict:
                raise ValueError(f"Validation failed: {'; '.join(errors)}")
        
        return command, context
    
    def after_command(
        self,
        command: Command,
        context: CommandContext,
        result: CommandResult
    ) -> CommandResult:
        """Validate command outputs."""
        errors = []
        
        # Validate result
        if result.success and result.error is not None:
            errors.append(f"{command.name}: Success=True but error is set")
        
        # Validate metrics
        for key, value in result.metrics.items():
            if not isinstance(value, (int, float)):
                errors.append(f"{command.name}: Metric '{key}' is not numeric: {type(value)}")
            elif value != value:  # NaN check
                errors.append(f"{command.name}: Metric '{key}' is NaN")
        
        # Log errors
        if errors:
            for error in errors:
                logger.warning(f"Validation warning: {error}")
                self._validation_errors.append(error)
            
            if self.strict:
                result.success = False
                result.error = ValueError(f"Validation failed: {'; '.join(errors)}")
        
        return result
    
    def get_validation_report(self) -> dict[str, Any]:
        """Get validation error report."""
        return {
            "total_errors": len(self._validation_errors),
            "errors": self._validation_errors.copy(),
        }