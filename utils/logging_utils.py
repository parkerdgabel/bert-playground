"""Logging utilities for standardized logging across the application."""

import functools
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from loguru import logger


def configure_logging(
    level: str = "INFO",
    format: Optional[str] = None,
    enqueue: bool = False,
    backtrace: bool = False,
    diagnose: bool = False,
) -> None:
    """
    Configure loguru logging with standardized settings.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Custom log format string (uses default if None)
        enqueue: Whether to enqueue logs (helps with threading issues)
        backtrace: Whether to show full traceback on errors
        diagnose: Whether to show variable values in tracebacks
    """
    # Remove default handler
    logger.remove()
    
    # Default format - concise and informative
    if format is None:
        format = "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    
    # Add new handler with configuration
    logger.add(
        sys.stderr,
        format=format,
        level=level,
        enqueue=enqueue,
        backtrace=backtrace,
        diagnose=diagnose,
    )


def get_module_logger(name: str, level: Optional[str] = None) -> "logger":
    """
    Get a logger instance for a specific module with optional level override.
    
    Args:
        name: Module name (e.g., "training.core", "data.loaders")
        level: Optional level override for this module
        
    Returns:
        Configured logger instance
    """
    module_logger = logger.bind(name=name)
    
    # Check for environment variable overrides
    env_key = f"LOG_LEVEL_{name.upper().replace('.', '_')}"
    env_level = os.environ.get(env_key, level)
    
    if env_level:
        # Create a filtered logger for this module
        module_logger = module_logger.opt(depth=1)
        # Note: In production, you might want to add per-module filtering
    
    return module_logger


def log_once(message: str, level: str = "INFO") -> None:
    """
    Log a message only once during the application lifetime.
    
    Useful for warnings or info that shouldn't be repeated.
    
    Args:
        message: Message to log
        level: Log level
    """
    # Use a set to track logged messages
    if not hasattr(log_once, "_logged_messages"):
        log_once._logged_messages = set()
    
    if message not in log_once._logged_messages:
        log_once._logged_messages.add(message)
        logger.log(level, message)


def progress_logger(total: int, desc: str = "Processing", update_freq: int = 10):
    """
    Create a progress logger that logs at intervals instead of every iteration.
    
    Args:
        total: Total number of items
        desc: Description of the task
        update_freq: How often to log progress (as percentage)
        
    Returns:
        Progress logger context manager
    """
    class ProgressLogger:
        def __init__(self, total: int, desc: str, update_freq: int):
            self.total = total
            self.desc = desc
            self.update_freq = update_freq
            self.current = 0
            self.last_logged_pct = -1
            
        def update(self, n: int = 1):
            self.current += n
            pct = int((self.current / self.total) * 100)
            
            # Log at intervals
            if pct >= self.last_logged_pct + self.update_freq:
                logger.info(f"{self.desc}: {pct}% ({self.current}/{self.total})")
                self.last_logged_pct = pct
                
        def __enter__(self):
            logger.info(f"{self.desc}: Starting ({self.total} items)")
            return self
            
        def __exit__(self, *args):
            if self.current < self.total:
                logger.warning(f"{self.desc}: Incomplete ({self.current}/{self.total})")
            else:
                logger.info(f"{self.desc}: Complete")
    
    return ProgressLogger(total, desc, update_freq)


# Logging level utilities
def is_debug_enabled() -> bool:
    """Check if debug logging is enabled."""
    return logger._core.min_level <= 10  # DEBUG level


def is_verbose_enabled() -> bool:
    """Check if verbose (TRACE) logging is enabled."""
    return logger._core.min_level <= 5  # TRACE level


# Performance logging utilities
def log_performance(func):
    """
    Decorator to log function performance (only in debug mode).
    
    Usage:
        @log_performance
        def slow_function():
            ...
    """
    import time
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not is_debug_enabled():
            return func(*args, **kwargs)
            
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        
        if elapsed > 1.0:  # Only log if > 1 second
            logger.debug(f"{func.__name__} took {elapsed:.2f}s")
            
        return result
    
    return wrapper


# Structured logging helpers
def log_config(config: dict, level: str = "DEBUG") -> None:
    """Log configuration in a structured way."""
    if level == "DEBUG" and not is_debug_enabled():
        return
        
    logger.log(level, "Configuration:")
    for key, value in config.items():
        if isinstance(value, dict):
            logger.log(level, f"  {key}:")
            for sub_key, sub_value in value.items():
                logger.log(level, f"    {sub_key}: {sub_value}")
        else:
            logger.log(level, f"  {key}: {value}")


def log_metrics(metrics: dict, step: Optional[int] = None, level: str = "INFO") -> None:
    """Log metrics in a concise format."""
    if not metrics:
        return
        
    # Format metrics into a single line
    parts = []
    for key, value in metrics.items():
        if isinstance(value, float):
            parts.append(f"{key}: {value:.4f}")
        else:
            parts.append(f"{key}: {value}")
    
    message = " | ".join(parts)
    if step is not None:
        message = f"Step {step} - {message}"
        
    logger.log(level, message)


# Module-specific log level configuration
def configure_module_levels(levels: dict[str, str]) -> None:
    """
    Configure log levels for specific modules.
    
    Args:
        levels: Dict mapping module names to log levels
        
    Example:
        configure_module_levels({
            "training.core": "INFO",
            "data.loaders": "WARNING",
            "models": "DEBUG",
        })
    """
    for module, level in levels.items():
        env_key = f"LOG_LEVEL_{module.upper().replace('.', '_')}"
        os.environ[env_key] = level
        logger.info(f"Set {module} log level to {level}")


# Common logging patterns
def log_training_start(epochs: int, batch_size: int, learning_rate: float) -> None:
    """Log training start with key parameters."""
    logger.info(
        f"Training: {epochs} epochs | Batch: {batch_size} | LR: {learning_rate}"
    )


def log_epoch_metrics(epoch: int, train_loss: float, val_loss: Optional[float] = None,
                     train_acc: Optional[float] = None, val_acc: Optional[float] = None) -> None:
    """Log epoch metrics in a standardized format."""
    parts = [f"Epoch {epoch}", f"Train Loss: {train_loss:.4f}"]
    
    if train_acc is not None:
        parts.append(f"Train Acc: {train_acc:.4f}")
    if val_loss is not None:
        parts.append(f"Val Loss: {val_loss:.4f}")
    if val_acc is not None:
        parts.append(f"Val Acc: {val_acc:.4f}")
    
    logger.info(" | ".join(parts))


def add_file_logger(
    file_path: str | Path,
    level: str = "INFO",
    format: Optional[str] = None,
    rotation: str = "500 MB",
    retention: str = "30 days",
    compression: str = "zip",
) -> None:
    """
    Add a file logger handler to save logs to a file.
    
    Args:
        file_path: Path to the log file
        level: Logging level for the file handler
        format: Custom log format (uses default if None)
        rotation: When to rotate the log file (e.g., "500 MB", "1 day")
        retention: How long to keep old logs (e.g., "30 days", "10 files")
        compression: Compression method for rotated logs ("zip", "gz", "bz2", "xz", "tar", "tar.gz", "tar.bz2", "tar.xz")
    """
    if format is None:
        format = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}"
    
    logger.add(
        file_path,
        level=level,
        format=format,
        rotation=rotation,
        retention=retention,
        compression=compression,
        enqueue=True,  # Always use enqueue for file handlers to prevent I/O blocking
        backtrace=True,  # Include backtrace in file logs
        diagnose=True,   # Include variable values in tracebacks
    )


# Advanced logging features

def bind_context(**kwargs) -> "logger":
    """
    Create a logger with bound context values.
    
    Example:
        log = bind_context(epoch=1, fold=0, model="bert")
        log.info("Training started")  # Will include epoch, fold, model in context
    """
    return logger.bind(**kwargs)


def bind_training_context(
    epoch: int,
    step: Optional[int] = None,
    fold: Optional[int] = None,
    phase: str = "train",
    **extra
) -> "logger":
    """
    Create a logger with training-specific context.
    
    Args:
        epoch: Current epoch number
        step: Current step number (optional)
        fold: Current fold number for cross-validation (optional)
        phase: Training phase ("train", "val", "test")
        **extra: Additional context values
    """
    context = {"epoch": epoch, "phase": phase}
    if step is not None:
        context["step"] = step
    if fold is not None:
        context["fold"] = fold
    context.update(extra)
    return logger.bind(**context)


@contextmanager
def log_timing(
    operation: str,
    level: str = "INFO",
    include_memory: bool = False,
    **context
):
    """
    Context manager for timing operations with automatic logging.
    
    Example:
        with log_timing("data_loading", batch_size=32):
            # Load data
            pass
        # Automatically logs: "data_loading completed in 1.23s"
    """
    start_time = time.time()
    start_memory = None
    
    if include_memory:
        try:
            import psutil
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            pass
    
    # Bind context and log start
    bound_logger = logger.bind(**context) if context else logger
    bound_logger.debug(f"{operation} started")
    
    try:
        yield bound_logger
    finally:
        elapsed = time.time() - start_time
        
        # Build completion message
        message_parts = [f"{operation} completed in {elapsed:.2f}s"]
        
        if include_memory and start_memory is not None:
            try:
                end_memory = process.memory_info().rss / 1024 / 1024
                memory_delta = end_memory - start_memory
                message_parts.append(f"memory Δ: {memory_delta:+.1f}MB")
            except:
                pass
        
        bound_logger.log(level, " | ".join(message_parts))


def lazy_debug(message: str, computation: Callable[[], Any], **context):
    """
    Log debug message with lazy evaluation of expensive computation.
    
    Example:
        lazy_debug("Gradient stats", lambda: compute_gradient_statistics(model))
    """
    if logger._core.min_level <= 10:  # DEBUG level
        bound_logger = logger.bind(**context) if context else logger
        result = computation()
        bound_logger.debug(f"{message}: {result}")


class MetricsLogger:
    """Specialized logger for structured metrics output."""
    
    def __init__(self, sink_path: Optional[Union[str, Path]] = None):
        """
        Initialize metrics logger.
        
        Args:
            sink_path: Optional path to save metrics in JSON format
        """
        self.sink_path = sink_path
        self._logger = logger.bind(metrics=True)
        
        if sink_path:
            # Add JSON sink for metrics
            logger.add(
                sink_path,
                format="{message}",
                filter=lambda record: record["extra"].get("metrics", False),
                serialize=True,
                rotation="100 MB",
            )
    
    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        **context
    ):
        """Log metrics in structured format."""
        record = {
            "timestamp": time.time(),
            "metrics": metrics
        }
        
        if step is not None:
            record["step"] = step
        if epoch is not None:
            record["epoch"] = epoch
        
        record.update(context)
        
        # Log as JSON for the metrics sink
        self._logger.info(json.dumps(record))
        
        # Also log human-readable version to console
        parts = []
        if epoch is not None:
            parts.append(f"Epoch {epoch}")
        if step is not None:
            parts.append(f"Step {step}")
        
        metric_parts = [f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                       for k, v in metrics.items()]
        
        message = " | ".join(parts + metric_parts) if parts else " | ".join(metric_parts)
        logger.info(message)


def catch_and_log(
    exception: type = Exception,
    message: str = "Operation failed",
    reraise: bool = True,
    default: Any = None,
    **context
):
    """
    Decorator to catch and log exceptions with context.
    
    Example:
        @catch_and_log(ValueError, "Model loading failed", model_path=path)
        def load_model(path):
            # Implementation
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exception as e:
                bound_logger = logger.bind(**context) if context else logger
                bound_logger.opt(exception=True).error(f"{message}: {str(e)}")
                
                if reraise:
                    raise
                return default
        return wrapper
    return decorator


class FrequencyLogger:
    """Logger that only logs every N occurrences."""
    
    def __init__(self, frequency: int = 100):
        self.frequency = frequency
        self._counts = {}
    
    def log(self, key: str, message: str, level: str = "INFO", **context):
        """Log message only every N times for the given key."""
        self._counts[key] = self._counts.get(key, 0) + 1
        
        if self._counts[key] % self.frequency == 0:
            bound_logger = logger.bind(**context) if context else logger
            bound_logger.log(level, f"{message} (occurrence #{self._counts[key]})")


class ProgressTracker:
    """Enhanced progress tracking with structured logging."""
    
    def __init__(
        self,
        total: int,
        desc: str = "Processing",
        log_frequency: int = 10,
        **context
    ):
        self.total = total
        self.desc = desc
        self.log_frequency = log_frequency
        self.current = 0
        self.start_time = time.time()
        self._logger = logger.bind(**context) if context else logger
        self._last_logged_pct = -1
    
    def update(self, n: int = 1, **metrics):
        """Update progress and optionally log metrics."""
        self.current += n
        pct = int((self.current / self.total) * 100)
        
        # Log at frequency intervals
        if pct >= self._last_logged_pct + self.log_frequency:
            elapsed = time.time() - self.start_time
            rate = self.current / elapsed if elapsed > 0 else 0
            
            message_parts = [
                f"{self.desc}: {pct}% ({self.current}/{self.total})",
                f"rate: {rate:.1f} items/s"
            ]
            
            # Add any metrics
            if metrics:
                metric_str = " | ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                                      for k, v in metrics.items())
                message_parts.append(metric_str)
            
            self._logger.info(" | ".join(message_parts))
            self._last_logged_pct = pct
    
    def __enter__(self):
        self._logger.info(f"{self.desc}: Starting ({self.total} items)")
        return self
    
    def __exit__(self, *args):
        elapsed = time.time() - self.start_time
        avg_time = elapsed / self.current if self.current > 0 else 0
        
        if self.current < self.total:
            self._logger.warning(
                f"{self.desc}: Incomplete ({self.current}/{self.total}) "
                f"| elapsed: {elapsed:.1f}s | avg: {avg_time:.3f}s/item"
            )
        else:
            self._logger.info(
                f"{self.desc}: Complete "
                f"| elapsed: {elapsed:.1f}s | avg: {avg_time:.3f}s/item"
            )


def log_mlx_info(array, name: str = "array", level: str = "DEBUG"):
    """Log MLX array information for debugging."""
    import mlx.core as mx
    
    if logger._core.min_level <= 10:  # DEBUG level
        info = {
            "name": name,
            "shape": array.shape,
            "dtype": str(array.dtype),
            "device": "gpu",  # MLX uses unified memory
            "size_mb": array.nbytes / (1024 * 1024),
        }
        
        # Check for common issues
        if any(d == 0 for d in array.shape):
            logger.warning(f"MLX array '{name}' has zero dimension: {array.shape}")
        
        logger.log(level, f"MLX array info: {info}")


# Export commonly used functions
__all__ = [
    # Basic configuration
    "configure_logging",
    "get_module_logger",
    "add_file_logger",
    "configure_module_levels",
    # Utility functions
    "log_once",
    "is_debug_enabled",
    "is_verbose_enabled",
    # Basic logging helpers
    "log_config",
    "log_metrics",
    "log_training_start",
    "log_epoch_metrics",
    # Performance and progress
    "log_performance",
    "progress_logger",
    "ProgressTracker",
    "log_timing",
    # Advanced features
    "bind_context",
    "bind_training_context",
    "lazy_debug",
    "MetricsLogger",
    "catch_and_log",
    "FrequencyLogger",
    "log_mlx_info",
]