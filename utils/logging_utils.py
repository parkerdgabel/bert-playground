"""Logging utilities for standardized logging across the application."""

import os
import sys
from typing import Optional

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


# Export commonly used functions
__all__ = [
    "configure_logging",
    "get_module_logger",
    "log_once",
    "progress_logger",
    "is_debug_enabled",
    "is_verbose_enabled",
    "log_performance",
    "log_config",
    "log_metrics",
    "configure_module_levels",
    "log_training_start",
    "log_epoch_metrics",
]