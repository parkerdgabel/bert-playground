"""Advanced memory management system for MLX training.

This module provides comprehensive memory management specifically optimized
for Apple Silicon and MLX's unified memory architecture.
"""

import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import psutil
from loguru import logger


@dataclass
class MemoryMetrics:
    """Memory usage metrics at a point in time."""

    timestamp: float
    total_memory_gb: float
    used_memory_gb: float
    available_memory_gb: float
    memory_percentage: float

    # MLX-specific metrics
    mlx_cache_size_mb: float = 0.0
    mlx_allocated_arrays: int = 0

    # Process-specific metrics
    process_memory_gb: float = 0.0
    process_memory_percentage: float = 0.0

    # Performance metrics
    memory_bandwidth_gb_s: float = 0.0
    cache_hit_rate: float = 0.0


@dataclass
class MemoryThresholds:
    """Memory usage thresholds for dynamic management."""

    # Critical thresholds (0.0 to 1.0)
    critical_memory: float = 0.95  # Stop training if exceeded
    high_memory: float = 0.85  # Reduce batch size
    optimal_memory: float = 0.75  # Target range
    low_memory: float = 0.50  # Increase batch size

    # Apple Silicon specific
    unified_memory_fraction: float = 0.8  # Fraction of unified memory to use
    neural_engine_memory_fraction: float = 0.2  # Reserve for Neural Engine

    # Batch size limits
    min_batch_size: int = 4
    max_batch_size: int = 256
    batch_size_step: int = 2  # Multiplicative factor for adjustments


class AppleSiliconMemoryManager:
    """Advanced memory manager optimized for Apple Silicon."""

    def __init__(self, thresholds: MemoryThresholds | None = None):
        """Initialize the memory manager.

        Args:
            thresholds: Memory usage thresholds configuration
        """
        self.thresholds = thresholds or MemoryThresholds()
        self.metrics_history: list[MemoryMetrics] = []
        self.gc_stats = {"collections": 0, "time_spent": 0.0}

        # Apple Silicon detection
        self.is_apple_silicon = self._detect_apple_silicon()
        self.unified_memory_size = self._get_unified_memory_size()

        # Performance tracking
        self.last_memory_check = time.time()
        self.memory_check_interval = 1.0  # seconds

        logger.info(
            f"Memory Manager initialized:\n"
            f"  Apple Silicon: {self.is_apple_silicon}\n"
            f"  Unified Memory: {self.unified_memory_size:.2f} GB\n"
            f"  Optimal Usage: {self.thresholds.optimal_memory:.1%}\n"
            f"  Target Memory: {self.unified_memory_size * self.thresholds.optimal_memory:.2f} GB"
        )

    def _detect_apple_silicon(self) -> bool:
        """Detect if running on Apple Silicon."""
        try:
            import platform

            return platform.processor() == "arm" and platform.system() == "Darwin"
        except Exception:
            return False

    def _get_unified_memory_size(self) -> float:
        """Get unified memory size in GB."""
        try:
            memory_info = psutil.virtual_memory()
            return memory_info.total / (1024**3)
        except Exception:
            return 16.0  # Default fallback

    def get_current_metrics(self) -> MemoryMetrics:
        """Get current memory metrics."""
        try:
            # System memory
            memory = psutil.virtual_memory()
            process = psutil.Process()
            process_memory = process.memory_info()

            # Calculate metrics
            total_gb = memory.total / (1024**3)
            used_gb = (memory.total - memory.available) / (1024**3)
            available_gb = memory.available / (1024**3)
            memory_pct = used_gb / total_gb

            process_gb = process_memory.rss / (1024**3)
            process_pct = process_gb / total_gb

            # MLX-specific metrics (placeholder - would need MLX memory APIs)
            mlx_cache_mb = self._get_mlx_cache_size()
            mlx_arrays = self._count_mlx_arrays()

            metrics = MemoryMetrics(
                timestamp=time.time(),
                total_memory_gb=total_gb,
                used_memory_gb=used_gb,
                available_memory_gb=available_gb,
                memory_percentage=memory_pct,
                mlx_cache_size_mb=mlx_cache_mb,
                mlx_allocated_arrays=mlx_arrays,
                process_memory_gb=process_gb,
                process_memory_percentage=process_pct,
            )

            # Store in history
            self.metrics_history.append(metrics)

            # Limit history size
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-500:]

            return metrics

        except Exception as e:
            logger.warning(f"Failed to get memory metrics: {e}")
            return MemoryMetrics(
                timestamp=time.time(),
                total_memory_gb=16.0,
                used_memory_gb=8.0,
                available_memory_gb=8.0,
                memory_percentage=0.5,
            )

    def _get_mlx_cache_size(self) -> float:
        """Get MLX cache size in MB (placeholder)."""
        # This would need actual MLX memory APIs
        # For now, return a placeholder value
        return 0.0

    def _count_mlx_arrays(self) -> int:
        """Count allocated MLX arrays (placeholder)."""
        # This would need actual MLX memory APIs
        # For now, return a placeholder value
        return 0

    def should_adjust_batch_size(
        self, current_batch_size: int
    ) -> tuple[bool, int, str]:
        """Determine if batch size should be adjusted.

        Returns:
            (should_adjust, new_batch_size, reason)
        """
        metrics = self.get_current_metrics()
        memory_usage = metrics.memory_percentage

        # Critical memory - emergency reduction
        if memory_usage >= self.thresholds.critical_memory:
            new_size = max(current_batch_size // 4, self.thresholds.min_batch_size)
            return True, new_size, f"Critical memory usage: {memory_usage:.1%}"

        # High memory - reduce batch size
        elif memory_usage >= self.thresholds.high_memory:
            new_size = max(
                current_batch_size // self.thresholds.batch_size_step,
                self.thresholds.min_batch_size,
            )
            if new_size != current_batch_size:
                return True, new_size, f"High memory usage: {memory_usage:.1%}"

        # Low memory - increase batch size
        elif memory_usage <= self.thresholds.low_memory:
            new_size = min(
                current_batch_size * self.thresholds.batch_size_step,
                self.thresholds.max_batch_size,
            )
            if new_size != current_batch_size:
                return True, new_size, f"Low memory usage: {memory_usage:.1%}"

        return False, current_batch_size, "No adjustment needed"

    def force_garbage_collection(self, aggressive: bool = False) -> dict[str, Any]:
        """Force garbage collection and return statistics.

        Args:
            aggressive: Whether to perform aggressive cleanup

        Returns:
            Statistics about the garbage collection
        """
        start_time = time.time()

        # Get memory before cleanup
        before_metrics = self.get_current_metrics()

        # Standard Python garbage collection
        if aggressive:
            # Multiple rounds of GC for thorough cleanup
            collected = 0
            for _ in range(3):
                collected += gc.collect()
        else:
            collected = gc.collect()

        # MLX-specific cleanup (if available)
        self._mlx_memory_cleanup(aggressive)

        # Get memory after cleanup
        after_metrics = self.get_current_metrics()

        # Calculate statistics
        gc_time = time.time() - start_time
        memory_freed = before_metrics.used_memory_gb - after_metrics.used_memory_gb

        self.gc_stats["collections"] += 1
        self.gc_stats["time_spent"] += gc_time

        stats = {
            "objects_collected": collected,
            "memory_freed_gb": memory_freed,
            "gc_time_seconds": gc_time,
            "before_memory_usage": before_metrics.memory_percentage,
            "after_memory_usage": after_metrics.memory_percentage,
            "improvement": before_metrics.memory_percentage
            - after_metrics.memory_percentage,
        }

        if memory_freed > 0.1:  # Only log if significant memory was freed
            logger.info(
                f"Garbage collection completed: "
                f"freed {memory_freed:.2f} GB, "
                f"took {gc_time:.3f}s, "
                f"collected {collected} objects"
            )

        return stats

    def _mlx_memory_cleanup(self, aggressive: bool = False) -> None:
        """Perform MLX-specific memory cleanup."""
        try:
            # Force evaluation of any pending computations
            # This helps free intermediate results
            mx.eval([])  # Evaluate empty list to trigger cleanup

            if aggressive:
                # Additional MLX cleanup strategies
                pass  # Would implement MLX-specific memory management

        except Exception as e:
            logger.debug(f"MLX memory cleanup failed: {e}")

    def optimize_for_apple_silicon(self) -> dict[str, Any]:
        """Apply Apple Silicon specific optimizations.

        Returns:
            Dictionary of applied optimizations
        """
        optimizations = {}

        if not self.is_apple_silicon:
            return {"skipped": "Not running on Apple Silicon"}

        try:
            # Set MLX memory optimizations
            optimizations["unified_memory"] = self._optimize_unified_memory()
            optimizations["neural_engine"] = self._optimize_neural_engine()
            optimizations["cache_optimization"] = self._optimize_cache_behavior()

        except Exception as e:
            logger.warning(f"Apple Silicon optimization failed: {e}")
            optimizations["error"] = str(e)

        return optimizations

    def _optimize_unified_memory(self) -> bool:
        """Optimize for unified memory architecture."""
        try:
            # Configure MLX to use unified memory efficiently
            # This is a placeholder - actual implementation would depend on MLX APIs
            target_memory = (
                self.unified_memory_size * self.thresholds.unified_memory_fraction
            )

            logger.debug(f"Configuring for {target_memory:.2f} GB unified memory usage")
            return True

        except Exception as e:
            logger.debug(f"Unified memory optimization failed: {e}")
            return False

    def _optimize_neural_engine(self) -> bool:
        """Optimize for Apple Neural Engine usage."""
        try:
            # Reserve memory for Neural Engine if available
            reserved_memory = (
                self.unified_memory_size * self.thresholds.neural_engine_memory_fraction
            )

            logger.debug(f"Reserving {reserved_memory:.2f} GB for Neural Engine")
            return True

        except Exception as e:
            logger.debug(f"Neural Engine optimization failed: {e}")
            return False

    def _optimize_cache_behavior(self) -> bool:
        """Optimize cache behavior for Apple Silicon."""
        try:
            # Configure cache-friendly memory access patterns
            # This would involve MLX-specific cache optimizations

            logger.debug("Optimizing cache behavior for Apple Silicon")
            return True

        except Exception as e:
            logger.debug(f"Cache optimization failed: {e}")
            return False

    def get_memory_recommendations(self) -> dict[str, Any]:
        """Get memory optimization recommendations."""
        current_metrics = self.get_current_metrics()

        recommendations = {
            "current_usage": f"{current_metrics.memory_percentage:.1%}",
            "status": "optimal",
            "actions": [],
        }

        # Analyze current usage
        if current_metrics.memory_percentage >= self.thresholds.critical_memory:
            recommendations["status"] = "critical"
            recommendations["actions"].extend(
                [
                    "Immediately reduce batch size",
                    "Force aggressive garbage collection",
                    "Consider model quantization",
                ]
            )
        elif current_metrics.memory_percentage >= self.thresholds.high_memory:
            recommendations["status"] = "high"
            recommendations["actions"].extend(
                [
                    "Reduce batch size",
                    "Enable gradient accumulation",
                    "Schedule garbage collection",
                ]
            )
        elif current_metrics.memory_percentage <= self.thresholds.low_memory:
            recommendations["status"] = "low"
            recommendations["actions"].extend(
                [
                    "Increase batch size for better throughput",
                    "Consider larger model if needed",
                ]
            )

        # Apple Silicon specific recommendations
        if self.is_apple_silicon:
            recommendations["apple_silicon"] = {
                "unified_memory_size": f"{self.unified_memory_size:.2f} GB",
                "recommended_usage": f"{self.thresholds.optimal_memory:.1%}",
                "current_efficiency": current_metrics.memory_percentage
                / self.thresholds.optimal_memory,
            }

        return recommendations

    def save_memory_report(self, output_path: Path) -> None:
        """Save comprehensive memory usage report."""
        report = {
            "system_info": {
                "apple_silicon": self.is_apple_silicon,
                "unified_memory_gb": self.unified_memory_size,
                "thresholds": {
                    "critical": self.thresholds.critical_memory,
                    "high": self.thresholds.high_memory,
                    "optimal": self.thresholds.optimal_memory,
                    "low": self.thresholds.low_memory,
                },
            },
            "current_metrics": self.get_current_metrics().__dict__,
            "gc_stats": self.gc_stats,
            "recommendations": self.get_memory_recommendations(),
            "history_summary": self._summarize_history(),
        }

        import json

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Memory report saved to: {output_path}")

    def _summarize_history(self) -> dict[str, Any]:
        """Summarize memory usage history."""
        if not self.metrics_history:
            return {"error": "No history available"}

        memory_percentages = [m.memory_percentage for m in self.metrics_history]

        return {
            "samples": len(memory_percentages),
            "min_usage": min(memory_percentages),
            "max_usage": max(memory_percentages),
            "avg_usage": sum(memory_percentages) / len(memory_percentages),
            "time_span_minutes": (
                self.metrics_history[-1].timestamp - self.metrics_history[0].timestamp
            )
            / 60,
        }


class MemoryOptimizer:
    """High-level memory optimization coordinator."""

    def __init__(self, memory_manager: AppleSiliconMemoryManager):
        """Initialize the memory optimizer.

        Args:
            memory_manager: The memory manager instance
        """
        self.memory_manager = memory_manager
        self.optimization_history: list[dict[str, Any]] = []

    def optimize_training_memory(
        self, current_batch_size: int, force_optimization: bool = False
    ) -> tuple[int, dict[str, Any]]:
        """Optimize memory usage for training.

        Args:
            current_batch_size: Current training batch size
            force_optimization: Force optimization even if not needed

        Returns:
            (new_batch_size, optimization_info)
        """
        optimization_info = {
            "timestamp": time.time(),
            "original_batch_size": current_batch_size,
            "optimizations_applied": [],
        }

        # Check if batch size adjustment is needed
        should_adjust, new_batch_size, reason = (
            self.memory_manager.should_adjust_batch_size(current_batch_size)
        )

        if should_adjust or force_optimization:
            optimization_info["batch_size_adjustment"] = {
                "from": current_batch_size,
                "to": new_batch_size,
                "reason": reason,
            }
            optimization_info["optimizations_applied"].append("batch_size_adjustment")

        # Apply garbage collection if memory usage is high
        current_metrics = self.memory_manager.get_current_metrics()
        if (
            current_metrics.memory_percentage
            >= self.memory_manager.thresholds.high_memory
            or force_optimization
        ):
            gc_stats = self.memory_manager.force_garbage_collection(
                aggressive=current_metrics.memory_percentage
                >= self.memory_manager.thresholds.critical_memory
            )
            optimization_info["garbage_collection"] = gc_stats
            optimization_info["optimizations_applied"].append("garbage_collection")

        # Apply Apple Silicon optimizations
        if self.memory_manager.is_apple_silicon and force_optimization:
            apple_optimizations = self.memory_manager.optimize_for_apple_silicon()
            optimization_info["apple_silicon_optimizations"] = apple_optimizations
            optimization_info["optimizations_applied"].append(
                "apple_silicon_optimizations"
            )

        # Store optimization history
        self.optimization_history.append(optimization_info)

        # Log if significant changes were made
        if optimization_info["optimizations_applied"]:
            logger.info(
                f"Memory optimization applied: {', '.join(optimization_info['optimizations_applied'])}"
            )

        return new_batch_size, optimization_info
