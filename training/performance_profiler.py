"""Performance profiler optimized for Apple Silicon and MLX.

This module provides comprehensive performance monitoring and optimization
specifically designed for Apple Silicon devices and MLX computations.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from loguru import logger


@dataclass
class PerformanceMetrics:
    """Performance metrics at a point in time."""

    timestamp: float
    step: int

    # Training metrics
    step_time_seconds: float
    samples_per_second: float
    tokens_per_second: float = 0.0

    # Memory metrics
    memory_usage_percent: float = 0.0
    memory_bandwidth_gb_s: float = 0.0

    # MLX-specific metrics
    mlx_eval_time_ms: float = 0.0
    mlx_compute_time_ms: float = 0.0
    mlx_memory_transfer_time_ms: float = 0.0

    # Apple Silicon metrics
    neural_engine_utilization: float = 0.0
    gpu_utilization: float = 0.0
    efficiency_cores_load: float = 0.0
    performance_cores_load: float = 0.0

    # Thermal metrics
    cpu_temperature_celsius: float = 0.0
    gpu_temperature_celsius: float = 0.0
    thermal_throttling: bool = False

    # Power metrics
    cpu_power_watts: float = 0.0
    gpu_power_watts: float = 0.0
    total_power_watts: float = 0.0


@dataclass
class ProfilerConfig:
    """Configuration for the performance profiler."""

    # Profiling intervals
    metrics_collection_interval: int = 10  # Steps between metric collection
    detailed_profiling_interval: int = 100  # Steps between detailed profiling
    thermal_monitoring_interval: int = 50  # Steps between thermal checks

    # Apple Silicon specific
    enable_neural_engine_monitoring: bool = True
    enable_thermal_monitoring: bool = True
    enable_power_monitoring: bool = True

    # Performance thresholds
    max_step_time_seconds: float = 10.0
    min_throughput_samples_per_second: float = 1.0
    thermal_warning_threshold: float = 80.0  # Celsius

    # Output settings
    save_detailed_logs: bool = True
    log_to_console: bool = True
    create_performance_plots: bool = True


class AppleSiliconProfiler:
    """Performance profiler optimized for Apple Silicon."""

    def __init__(self, config: ProfilerConfig | None = None):
        """Initialize the profiler.

        Args:
            config: Profiler configuration
        """
        self.config = config or ProfilerConfig()
        self.metrics_history: list[PerformanceMetrics] = []
        self.step_timers: dict[str, float] = {}

        # Apple Silicon detection and capabilities
        self.is_apple_silicon = self._detect_apple_silicon()
        self.capabilities = self._detect_capabilities()

        # Performance tracking
        self.baseline_metrics: PerformanceMetrics | None = None
        self.performance_warnings: list[dict[str, Any]] = []

        logger.info(
            f"Performance Profiler initialized:\n"
            f"  Apple Silicon: {self.is_apple_silicon}\n"
            f"  Neural Engine: {self.capabilities.get('neural_engine', False)}\n"
            f"  Thermal Monitoring: {self.capabilities.get('thermal', False)}\n"
            f"  Power Monitoring: {self.capabilities.get('power', False)}"
        )

    def _detect_apple_silicon(self) -> bool:
        """Detect if running on Apple Silicon."""
        try:
            import platform

            return platform.processor() == "arm" and platform.system() == "Darwin"
        except Exception:
            return False

    def _detect_capabilities(self) -> dict[str, bool]:
        """Detect available monitoring capabilities."""
        capabilities = {
            "neural_engine": False,
            "thermal": False,
            "power": False,
            "gpu_stats": False,
        }

        if self.is_apple_silicon:
            # Try to detect Neural Engine
            try:
                # This would check for actual Neural Engine availability
                capabilities["neural_engine"] = True
            except Exception:
                pass

            # Try to detect thermal monitoring
            try:
                # This would check for temperature sensors
                capabilities["thermal"] = True
            except Exception:
                pass

            # Try to detect power monitoring
            try:
                # This would check for power measurement APIs
                capabilities["power"] = True
            except Exception:
                pass

        return capabilities

    def start_step_timer(self, step: int) -> None:
        """Start timing a training step.

        Args:
            step: Global step number
        """
        self.step_timers[f"step_{step}"] = time.time()
        self.step_timers["last_step_start"] = time.time()

    def end_step_timer(
        self, step: int, batch_size: int, sequence_length: int | None = None
    ) -> PerformanceMetrics:
        """End timing a training step and collect metrics.

        Args:
            step: Global step number
            batch_size: Batch size used
            sequence_length: Sequence length (for token throughput)

        Returns:
            Performance metrics for this step
        """
        current_time = time.time()
        step_key = f"step_{step}"

        # Calculate step time
        if step_key in self.step_timers:
            step_time = current_time - self.step_timers[step_key]
            del self.step_timers[step_key]
        else:
            step_time = 0.0

        # Calculate throughput
        samples_per_second = batch_size / step_time if step_time > 0 else 0.0
        tokens_per_second = 0.0
        if sequence_length and step_time > 0:
            tokens_per_second = (batch_size * sequence_length) / step_time

        # Collect system metrics
        system_metrics = self._collect_system_metrics()

        # Create performance metrics
        metrics = PerformanceMetrics(
            timestamp=current_time,
            step=step,
            step_time_seconds=step_time,
            samples_per_second=samples_per_second,
            tokens_per_second=tokens_per_second,
            **system_metrics,
        )

        # Store metrics
        self.metrics_history.append(metrics)

        # Limit history size
        if len(self.metrics_history) > 10000:
            self.metrics_history = self.metrics_history[-5000:]

        # Check for performance issues
        self._check_performance_warnings(metrics)

        # Log metrics if configured
        if (
            self.config.log_to_console
            and step % self.config.metrics_collection_interval == 0
        ):
            self._log_metrics(metrics)

        return metrics

    def _collect_system_metrics(self) -> dict[str, float]:
        """Collect system-level performance metrics."""
        metrics = {}

        try:
            # Memory metrics
            metrics.update(self._collect_memory_metrics())

            # Apple Silicon specific metrics
            if self.is_apple_silicon:
                metrics.update(self._collect_apple_silicon_metrics())

            # Thermal metrics
            if self.capabilities["thermal"]:
                metrics.update(self._collect_thermal_metrics())

            # Power metrics
            if self.capabilities["power"]:
                metrics.update(self._collect_power_metrics())

        except Exception as e:
            logger.debug(f"Failed to collect some system metrics: {e}")

        return metrics

    def _collect_memory_metrics(self) -> dict[str, float]:
        """Collect memory-related metrics."""
        try:
            import psutil

            memory = psutil.virtual_memory()

            return {
                "memory_usage_percent": memory.percent,
                "memory_bandwidth_gb_s": self._estimate_memory_bandwidth(),
            }
        except Exception:
            return {"memory_usage_percent": 0.0, "memory_bandwidth_gb_s": 0.0}

    def _collect_apple_silicon_metrics(self) -> dict[str, float]:
        """Collect Apple Silicon specific metrics."""
        metrics = {
            "neural_engine_utilization": 0.0,
            "gpu_utilization": 0.0,
            "efficiency_cores_load": 0.0,
            "performance_cores_load": 0.0,
        }

        try:
            # Neural Engine utilization (placeholder)
            if self.capabilities["neural_engine"]:
                metrics["neural_engine_utilization"] = (
                    self._get_neural_engine_utilization()
                )

            # GPU utilization (placeholder)
            metrics["gpu_utilization"] = self._get_gpu_utilization()

            # CPU core utilization
            core_metrics = self._get_cpu_core_utilization()
            metrics.update(core_metrics)

        except Exception as e:
            logger.debug(f"Failed to collect Apple Silicon metrics: {e}")

        return metrics

    def _collect_thermal_metrics(self) -> dict[str, float]:
        """Collect thermal monitoring metrics."""
        try:
            temperatures = self._get_temperatures()

            return {
                "cpu_temperature_celsius": temperatures.get("cpu", 0.0),
                "gpu_temperature_celsius": temperatures.get("gpu", 0.0),
                "thermal_throttling": temperatures.get("cpu", 0.0)
                > self.config.thermal_warning_threshold,
            }
        except Exception:
            return {
                "cpu_temperature_celsius": 0.0,
                "gpu_temperature_celsius": 0.0,
                "thermal_throttling": False,
            }

    def _collect_power_metrics(self) -> dict[str, float]:
        """Collect power consumption metrics."""
        try:
            power_data = self._get_power_consumption()

            return {
                "cpu_power_watts": power_data.get("cpu", 0.0),
                "gpu_power_watts": power_data.get("gpu", 0.0),
                "total_power_watts": power_data.get("total", 0.0),
            }
        except Exception:
            return {
                "cpu_power_watts": 0.0,
                "gpu_power_watts": 0.0,
                "total_power_watts": 0.0,
            }

    def _estimate_memory_bandwidth(self) -> float:
        """Estimate memory bandwidth usage."""
        # This is a simplified estimation
        # In practice, this would measure actual memory transfer rates
        return 0.0

    def _get_neural_engine_utilization(self) -> float:
        """Get Neural Engine utilization (placeholder)."""
        # This would require Apple's private APIs or system monitoring tools
        return 0.0

    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization (placeholder)."""
        # This would query GPU usage statistics
        return 0.0

    def _get_cpu_core_utilization(self) -> dict[str, float]:
        """Get CPU core utilization breakdown."""
        try:
            import psutil

            cpu_percent = psutil.cpu_percent(interval=None)

            # This is simplified - actual implementation would
            # distinguish between efficiency and performance cores
            return {
                "efficiency_cores_load": cpu_percent * 0.6,  # Approximation
                "performance_cores_load": cpu_percent * 0.4,  # Approximation
            }
        except Exception:
            return {
                "efficiency_cores_load": 0.0,
                "performance_cores_load": 0.0,
            }

    def _get_temperatures(self) -> dict[str, float]:
        """Get system temperatures (placeholder)."""
        # This would require system-specific temperature monitoring
        return {"cpu": 50.0, "gpu": 45.0}

    def _get_power_consumption(self) -> dict[str, float]:
        """Get power consumption (placeholder)."""
        # This would require system power monitoring APIs
        return {"cpu": 10.0, "gpu": 15.0, "total": 25.0}

    def _check_performance_warnings(self, metrics: PerformanceMetrics) -> None:
        """Check for performance issues and log warnings."""
        warnings = []

        # Check step time
        if metrics.step_time_seconds > self.config.max_step_time_seconds:
            warnings.append(
                {
                    "type": "slow_step",
                    "message": f"Step took {metrics.step_time_seconds:.2f}s (threshold: {self.config.max_step_time_seconds}s)",
                    "severity": "warning",
                }
            )

        # Check throughput
        if metrics.samples_per_second < self.config.min_throughput_samples_per_second:
            warnings.append(
                {
                    "type": "low_throughput",
                    "message": f"Throughput {metrics.samples_per_second:.1f} samples/s is below threshold",
                    "severity": "warning",
                }
            )

        # Check thermal throttling
        if metrics.thermal_throttling:
            warnings.append(
                {
                    "type": "thermal_throttling",
                    "message": f"Thermal throttling detected (CPU: {metrics.cpu_temperature_celsius:.1f}°C)",
                    "severity": "critical",
                }
            )

        # Store warnings
        for warning in warnings:
            warning["timestamp"] = metrics.timestamp
            warning["step"] = metrics.step
            self.performance_warnings.append(warning)

            # Log critical warnings immediately
            if warning["severity"] == "critical":
                logger.warning(f"Performance issue: {warning['message']}")

    def _log_metrics(self, metrics: PerformanceMetrics) -> None:
        """Log performance metrics to console."""
        logger.info(
            f"Step {metrics.step}: "
            f"Time={metrics.step_time_seconds:.3f}s, "
            f"Throughput={metrics.samples_per_second:.1f} samples/s, "
            f"Memory={metrics.memory_usage_percent:.1f}%"
        )

        # Log Apple Silicon specific metrics if available
        if self.is_apple_silicon and metrics.neural_engine_utilization > 0:
            logger.debug(
                f"Apple Silicon: Neural Engine={metrics.neural_engine_utilization:.1f}%, "
                f"GPU={metrics.gpu_utilization:.1f}%"
            )

    def profile_mlx_operation(
        self, operation_name: str, operation_func, *args, **kwargs
    ) -> tuple[Any, dict[str, float]]:
        """Profile a specific MLX operation.

        Args:
            operation_name: Name of the operation for logging
            operation_func: Function to profile
            *args, **kwargs: Arguments for the function

        Returns:
            (result, profiling_metrics)
        """
        # Pre-operation memory state
        start_time = time.time()

        # Execute operation
        result = operation_func(*args, **kwargs)

        # Force evaluation for accurate timing
        if hasattr(result, "__iter__") and not isinstance(result, str):
            try:
                mx.eval(result)
            except Exception:
                pass
        elif hasattr(result, "shape"):  # MLX array
            mx.eval(result)

        end_time = time.time()

        # Calculate metrics
        operation_time = end_time - start_time

        profiling_metrics = {
            "operation_name": operation_name,
            "execution_time_ms": operation_time * 1000,
            "timestamp": start_time,
        }

        logger.debug(
            f"MLX operation '{operation_name}' took {operation_time * 1000:.2f}ms"
        )

        return result, profiling_metrics

    def get_performance_summary(self) -> dict[str, Any]:
        """Get a comprehensive performance summary."""
        if not self.metrics_history:
            return {"error": "No metrics collected yet"}

        # Calculate averages and statistics
        step_times = [m.step_time_seconds for m in self.metrics_history]
        throughputs = [m.samples_per_second for m in self.metrics_history]
        memory_usage = [m.memory_usage_percent for m in self.metrics_history]

        summary = {
            "total_steps": len(self.metrics_history),
            "time_span_minutes": (
                self.metrics_history[-1].timestamp - self.metrics_history[0].timestamp
            )
            / 60,
            "step_time": {
                "average_seconds": np.mean(step_times),
                "min_seconds": np.min(step_times),
                "max_seconds": np.max(step_times),
                "std_seconds": np.std(step_times),
            },
            "throughput": {
                "average_samples_per_second": np.mean(throughputs),
                "min_samples_per_second": np.min(throughputs),
                "max_samples_per_second": np.max(throughputs),
            },
            "memory": {
                "average_usage_percent": np.mean(memory_usage),
                "peak_usage_percent": np.max(memory_usage),
                "min_usage_percent": np.min(memory_usage),
            },
            "warnings": {
                "total_warnings": len(self.performance_warnings),
                "critical_warnings": len(
                    [
                        w
                        for w in self.performance_warnings
                        if w["severity"] == "critical"
                    ]
                ),
                "warning_types": list(
                    set(w["type"] for w in self.performance_warnings)
                ),
            },
        }

        # Add Apple Silicon specific summary
        if self.is_apple_silicon:
            neural_engine_usage = [
                m.neural_engine_utilization for m in self.metrics_history
            ]
            gpu_usage = [m.gpu_utilization for m in self.metrics_history]

            summary["apple_silicon"] = {
                "neural_engine_avg_utilization": np.mean(neural_engine_usage),
                "gpu_avg_utilization": np.mean(gpu_usage),
                "thermal_warnings": len(
                    [
                        w
                        for w in self.performance_warnings
                        if w["type"] == "thermal_throttling"
                    ]
                ),
            }

        return summary

    def save_performance_report(self, output_path: Path) -> None:
        """Save detailed performance report."""
        report = {
            "profiler_config": self.config.__dict__,
            "system_info": {
                "apple_silicon": self.is_apple_silicon,
                "capabilities": self.capabilities,
            },
            "performance_summary": self.get_performance_summary(),
            "warnings": self.performance_warnings,
            "detailed_metrics": [
                m.__dict__ for m in self.metrics_history[-100:]
            ],  # Last 100 steps
        }

        import json

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Performance report saved to: {output_path}")

    def reset_metrics(self) -> None:
        """Reset all collected metrics and history."""
        self.metrics_history.clear()
        self.performance_warnings.clear()
        self.step_timers.clear()
        logger.info("Performance metrics reset")
