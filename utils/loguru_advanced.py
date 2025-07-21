"""Advanced Loguru features and utilities for enhanced logging."""

import functools
import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from loguru import logger


# Structured logging helpers
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


# Performance timing utilities
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


# Lazy evaluation for expensive computations
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


# Structured metrics logging
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


# Error handling decorators
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


# Conditional logging based on frequency
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


# Progress tracking with structured logging
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


# MLX-specific logging utilities
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


# Export utilities
__all__ = [
    "bind_context",
    "bind_training_context",
    "log_timing",
    "lazy_debug",
    "MetricsLogger",
    "catch_and_log",
    "FrequencyLogger",
    "ProgressTracker",
    "log_mlx_info",
]