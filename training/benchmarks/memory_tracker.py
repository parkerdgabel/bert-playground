"""
Memory tracking utilities for MLX training.
"""

import gc
from collections.abc import Callable
from typing import Any

import mlx.core as mx
from loguru import logger


class MemoryTracker:
    """
    Track memory usage during MLX training.

    Provides unified memory tracking for Apple Silicon.
    """

    def __init__(self):
        """Initialize memory tracker."""
        self.baseline_memory: float | None = None
        self.peak_memory: float = 0.0
        self.current_memory: float = 0.0

    def reset(self):
        """Reset memory tracking."""
        gc.collect()
        mx.eval(mx.array(0))  # Force any pending computations

        if hasattr(mx, "metal"):
            self.baseline_memory = mx.metal.get_active_memory() / (1024**3)
        else:
            self.baseline_memory = 0.0

        self.peak_memory = self.baseline_memory
        self.current_memory = self.baseline_memory

    def update(self):
        """Update current memory usage."""
        if hasattr(mx, "metal"):
            self.current_memory = mx.metal.get_active_memory() / (1024**3)
            self.peak_memory = max(self.peak_memory, self.current_memory)

    def get_stats(self) -> dict[str, float]:
        """Get memory statistics."""
        self.update()

        stats = {
            "current_gb": self.current_memory,
            "peak_gb": self.peak_memory,
        }

        if self.baseline_memory is not None:
            stats["baseline_gb"] = self.baseline_memory
            stats["used_gb"] = self.current_memory - self.baseline_memory
            stats["peak_used_gb"] = self.peak_memory - self.baseline_memory

        return stats

    def log_stats(self):
        """Log current memory statistics."""
        stats = self.get_stats()
        logger.info(
            f"Memory - Current: {stats['current_gb']:.2f}GB, Peak: {stats['peak_gb']:.2f}GB"
        )
        if "used_gb" in stats:
            logger.info(
                f"Memory - Used: {stats['used_gb']:.2f}GB, Peak Used: {stats['peak_used_gb']:.2f}GB"
            )

    def track_operation(self, operation: Callable, operation_name: str) -> Any:
        """
        Track memory usage for a specific operation.

        Args:
            operation: Function to execute
            operation_name: Name for logging

        Returns:
            Result from operation
        """
        # Memory before
        before_memory = (
            mx.metal.get_active_memory() / (1024**3) if hasattr(mx, "metal") else 0
        )

        # Execute operation
        result = operation()

        # Force evaluation to get accurate memory
        if hasattr(result, "item"):
            mx.eval(result)

        # Memory after
        after_memory = (
            mx.metal.get_active_memory() / (1024**3) if hasattr(mx, "metal") else 0
        )

        memory_delta = after_memory - before_memory

        if memory_delta > 0.01:  # Only log if significant (>10MB)
            logger.debug(f"{operation_name} memory delta: {memory_delta:.3f}GB")

        self.update()

        return result

    def check_memory_pressure(self, threshold_gb: float = 0.8) -> bool:
        """
        Check if memory usage is approaching system limits.

        Args:
            threshold_gb: Memory threshold as fraction of total

        Returns:
            True if memory pressure is high
        """
        if not hasattr(mx, "metal"):
            return False

        try:
            # Set memory limit if not already set
            mx.metal.set_memory_limit(threshold_gb)

            # Check current usage
            self.update()

            # Estimate total available (this is approximate)
            # MLX doesn't expose total memory, so we use a heuristic
            if self.current_memory > 50:  # Over 50GB suggests high usage
                logger.warning(
                    f"High memory usage detected: {self.current_memory:.1f}GB"
                )
                return True

        except Exception as e:
            logger.debug(f"Memory pressure check failed: {e}")

        return False

    def optimize_batch_size(self, base_batch_size: int, model_size_gb: float) -> int:
        """
        Suggest optimal batch size based on available memory.

        Args:
            base_batch_size: Desired batch size
            model_size_gb: Estimated model size in GB

        Returns:
            Recommended batch size
        """
        if not hasattr(mx, "metal"):
            return base_batch_size

        # Get current memory usage
        self.update()

        # Rough heuristics for Apple Silicon
        # M1: 8-16GB, M1 Pro/Max: 16-64GB, M2: similar, M3: up to 128GB
        if self.current_memory < 8:
            # Likely base M1/M2 with shared memory
            if model_size_gb > 2:
                return min(base_batch_size, 16)
            return min(base_batch_size, 32)
        elif self.current_memory < 32:
            # Likely M1/M2 Pro
            if model_size_gb > 4:
                return min(base_batch_size, 32)
            return min(base_batch_size, 64)
        else:
            # M1/M2 Max or Ultra
            if model_size_gb > 8:
                return min(base_batch_size, 64)
            return base_batch_size

    @staticmethod
    def get_memory_info() -> dict[str, Any]:
        """Get system memory information."""
        info = {
            "mlx_version": mx.__version__,
            "metal_available": hasattr(mx, "metal"),
        }

        if hasattr(mx, "metal"):
            info["current_memory_gb"] = mx.metal.get_active_memory() / (1024**3)
            # Note: MLX doesn't expose total memory directly

        return info
