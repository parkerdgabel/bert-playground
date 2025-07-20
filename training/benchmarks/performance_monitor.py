"""
Performance monitoring utilities for MLX training optimization.
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from statistics import mean, stdev
import mlx.core as mx
from loguru import logger


@dataclass
class BenchmarkResults:
    """Results from performance benchmarking."""
    
    step_times: List[float] = field(default_factory=list)
    memory_usage: List[float] = field(default_factory=list)
    throughput: List[float] = field(default_factory=list)
    
    # Gradient statistics
    grad_compute_times: List[float] = field(default_factory=list)
    grad_norm_times: List[float] = field(default_factory=list)
    
    # Optimizer times
    optimizer_step_times: List[float] = field(default_factory=list)
    
    # Evaluation times
    eval_times: List[float] = field(default_factory=list)
    
    # Metal backend stats
    metal_memory_gb: List[float] = field(default_factory=list)
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics."""
        def safe_stats(data: List[float]) -> Dict[str, float]:
            if not data:
                return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
            return {
                "mean": mean(data),
                "std": stdev(data) if len(data) > 1 else 0.0,
                "min": min(data),
                "max": max(data),
            }
        
        return {
            "step_time": safe_stats(self.step_times),
            "memory_usage_gb": safe_stats(self.memory_usage),
            "throughput_samples_per_sec": safe_stats(self.throughput),
            "grad_compute_time": safe_stats(self.grad_compute_times),
            "grad_norm_time": safe_stats(self.grad_norm_times),
            "optimizer_step_time": safe_stats(self.optimizer_step_times),
            "eval_time": safe_stats(self.eval_times),
            "metal_memory_gb": safe_stats(self.metal_memory_gb),
        }
    
    def log_summary(self):
        """Log performance summary."""
        summary = self.get_summary()
        logger.info("=== Performance Benchmark Results ===")
        
        for metric, stats in summary.items():
            if isinstance(stats, dict) and stats["mean"] > 0:
                logger.info(
                    f"{metric}: mean={stats['mean']:.4f}, "
                    f"std={stats['std']:.4f}, "
                    f"min={stats['min']:.4f}, "
                    f"max={stats['max']:.4f}"
                )


class PerformanceMonitor:
    """
    Monitor and benchmark MLX training performance.
    
    Tracks:
    - Step execution time
    - Memory usage
    - Throughput (samples/sec)
    - Gradient computation time
    - Optimizer step time
    """
    
    def __init__(self, warmup_steps: int = 10):
        """
        Initialize performance monitor.
        
        Args:
            warmup_steps: Number of steps to skip for warmup
        """
        self.warmup_steps = warmup_steps
        self.current_step = 0
        self.results = BenchmarkResults()
        self._timers: Dict[str, float] = {}
        
    def start_timer(self, name: str):
        """Start a named timer."""
        self._timers[name] = time.perf_counter()
        
    def end_timer(self, name: str) -> float:
        """End a named timer and return elapsed time."""
        if name not in self._timers:
            return 0.0
        
        elapsed = time.perf_counter() - self._timers[name]
        del self._timers[name]
        return elapsed
    
    def track_step(self, step_fn: Callable, batch_size: int) -> Any:
        """
        Track a training step execution.
        
        Args:
            step_fn: Step function to execute
            batch_size: Batch size for throughput calculation
            
        Returns:
            Result from step function
        """
        self.current_step += 1
        
        # Execute step with timing
        start_time = time.perf_counter()
        result = step_fn()
        mx.eval(result)  # Ensure computation completes
        end_time = time.perf_counter()
        
        # Skip warmup steps
        if self.current_step <= self.warmup_steps:
            return result
        
        # Record metrics
        step_time = end_time - start_time
        self.results.step_times.append(step_time)
        
        # Calculate throughput
        throughput = batch_size / step_time
        self.results.throughput.append(throughput)
        
        # Track memory if available
        try:
            if hasattr(mx, 'metal'):
                memory_gb = mx.metal.get_active_memory() / (1024**3)
                self.results.metal_memory_gb.append(memory_gb)
        except:
            pass
        
        return result
    
    def track_gradient_computation(self, grad_fn: Callable) -> Any:
        """Track gradient computation time."""
        start_time = time.perf_counter()
        result = grad_fn()
        mx.eval(result)  # Ensure computation completes
        elapsed = time.perf_counter() - start_time
        
        if self.current_step > self.warmup_steps:
            self.results.grad_compute_times.append(elapsed)
        
        return result
    
    def track_gradient_norm(self, norm_fn: Callable) -> Any:
        """Track gradient norm computation time."""
        start_time = time.perf_counter()
        result = norm_fn()
        if hasattr(result, 'item'):
            result.item()  # Force evaluation
        elapsed = time.perf_counter() - start_time
        
        if self.current_step > self.warmup_steps:
            self.results.grad_norm_times.append(elapsed)
        
        return result
    
    def track_optimizer_step(self, opt_fn: Callable) -> Any:
        """Track optimizer step time."""
        start_time = time.perf_counter()
        result = opt_fn()
        mx.eval(result) if result is not None else mx.eval(mx.array(0))
        elapsed = time.perf_counter() - start_time
        
        if self.current_step > self.warmup_steps:
            self.results.optimizer_step_times.append(elapsed)
        
        return result
    
    def track_evaluation(self, eval_fn: Callable) -> Any:
        """Track evaluation time."""
        start_time = time.perf_counter()
        result = eval_fn()
        elapsed = time.perf_counter() - start_time
        
        self.results.eval_times.append(elapsed)
        
        return result
    
    def get_current_stats(self) -> Dict[str, float]:
        """Get current performance statistics."""
        if not self.results.step_times:
            return {}
        
        latest_idx = -min(10, len(self.results.step_times))
        recent_steps = self.results.step_times[latest_idx:]
        recent_throughput = self.results.throughput[latest_idx:]
        
        stats = {
            "avg_step_time": mean(recent_steps),
            "avg_throughput": mean(recent_throughput),
        }
        
        if self.results.metal_memory_gb:
            stats["current_memory_gb"] = self.results.metal_memory_gb[-1]
        
        return stats
    
    def compare_implementations(
        self,
        old_impl: Callable,
        new_impl: Callable,
        num_iterations: int = 100,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """
        Compare performance of two implementations.
        
        Args:
            old_impl: Original implementation
            new_impl: Optimized implementation
            num_iterations: Number of iterations to run
            batch_size: Batch size for testing
            
        Returns:
            Comparison results
        """
        logger.info(f"Comparing implementations over {num_iterations} iterations...")
        
        # Benchmark old implementation
        old_times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            old_impl()
            mx.eval(mx.array(0))  # Ensure eval
            old_times.append(time.perf_counter() - start)
        
        # Benchmark new implementation
        new_times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            new_impl()
            mx.eval(mx.array(0))  # Ensure eval
            new_times.append(time.perf_counter() - start)
        
        # Calculate statistics
        old_mean = mean(old_times[10:])  # Skip warmup
        new_mean = mean(new_times[10:])
        speedup = old_mean / new_mean
        
        comparison = {
            "old_mean_time": old_mean,
            "new_mean_time": new_mean,
            "speedup": speedup,
            "speedup_percentage": (speedup - 1) * 100,
            "old_throughput": batch_size / old_mean,
            "new_throughput": batch_size / new_mean,
        }
        
        logger.info(f"Speedup: {speedup:.2f}x ({comparison['speedup_percentage']:.1f}% faster)")
        logger.info(f"Old throughput: {comparison['old_throughput']:.1f} samples/sec")
        logger.info(f"New throughput: {comparison['new_throughput']:.1f} samples/sec")
        
        return comparison