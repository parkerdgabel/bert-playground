"""Memory profiling utilities for MLX training.

This module provides memory profiling and monitoring:
- Real-time memory usage tracking
- Memory leak detection
- Peak memory analysis
- Memory optimization suggestions
"""

import gc
import os
import platform
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import psutil
from loguru import logger


@dataclass
class MemorySnapshot:
    """Memory usage snapshot."""

    timestamp: float
    step: int
    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    available_mb: float  # Available system memory
    percent: float  # Memory usage percentage
    mlx_allocated_mb: float  # MLX allocated memory (placeholder)
    description: str = ""


class MemoryProfiler:
    """Memory profiler for MLX training."""

    def __init__(
        self,
        window_size: int = 100,
        alert_threshold: float = 0.9,
        log_interval: int = 10,
        save_plots: bool = True,
        output_dir: str = "output/memory_profiles",
    ):
        """Initialize memory profiler.

        Args:
            window_size: Size of sliding window for trend analysis
            alert_threshold: Memory usage threshold for alerts (0-1)
            log_interval: Steps between memory logs
            save_plots: Whether to save memory plots
            output_dir: Directory to save plots and logs
        """
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        self.log_interval = log_interval
        self.save_plots = save_plots
        self.output_dir = Path(output_dir)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Memory tracking
        self.snapshots: list[MemorySnapshot] = []
        self.recent_snapshots = deque(maxlen=window_size)
        self.peak_memory = 0.0
        self.baseline_memory = None

        # Process info
        self.process = psutil.Process(os.getpid())
        self.system_info = self._get_system_info()

        # Start baseline
        self._record_baseline()

        logger.info(
            f"Memory Profiler initialized - "
            f"System: {self.system_info['system']}, "
            f"Total RAM: {self.system_info['total_ram_gb']:.1f} GB"
        )

    def _get_system_info(self) -> dict[str, any]:
        """Get system information."""
        vm = psutil.virtual_memory()

        info = {
            "system": platform.system(),
            "processor": platform.processor(),
            "total_ram_gb": vm.total / (1024**3),
            "available_ram_gb": vm.available / (1024**3),
            "cpu_count": psutil.cpu_count(),
        }

        # Add Apple Silicon specific info
        if platform.system() == "Darwin":
            try:
                import subprocess

                # Get chip info
                chip_info = subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    text=True,
                ).strip()
                info["chip"] = chip_info

                # Check if it's Apple Silicon
                if "Apple" in chip_info:
                    info["apple_silicon"] = True
                    # Get unified memory info
                    vm_stat = subprocess.check_output(["vm_stat"], text=True)
                    info["vm_stat"] = vm_stat
            except Exception:
                pass

        return info

    def _record_baseline(self):
        """Record baseline memory usage."""
        gc.collect()  # Force garbage collection
        time.sleep(0.1)  # Let system settle

        snapshot = self._take_snapshot(0, "baseline")
        self.baseline_memory = snapshot.rss_mb
        logger.info(f"Baseline memory: {self.baseline_memory:.1f} MB")

    def _take_snapshot(self, step: int, description: str = "") -> MemorySnapshot:
        """Take a memory snapshot."""
        memory_info = self.process.memory_info()
        vm = psutil.virtual_memory()

        # Get MLX memory usage (placeholder - MLX doesn't expose this yet)
        mlx_allocated = self._estimate_mlx_memory()

        snapshot = MemorySnapshot(
            timestamp=time.time(),
            step=step,
            rss_mb=memory_info.rss / (1024**2),
            vms_mb=memory_info.vms / (1024**2),
            available_mb=vm.available / (1024**2),
            percent=self.process.memory_percent(),
            mlx_allocated_mb=mlx_allocated,
            description=description,
        )

        return snapshot

    def _estimate_mlx_memory(self) -> float:
        """Estimate MLX memory usage."""
        # MLX doesn't provide direct memory stats yet
        # This is a placeholder that estimates based on process memory
        if self.baseline_memory:
            current = self.process.memory_info().rss / (1024**2)
            return max(0, current - self.baseline_memory)
        return 0.0

    def record(self, step: int, description: str = ""):
        """Record memory usage at current step."""
        snapshot = self._take_snapshot(step, description)

        self.snapshots.append(snapshot)
        self.recent_snapshots.append(snapshot)

        # Update peak memory
        if snapshot.rss_mb > self.peak_memory:
            self.peak_memory = snapshot.rss_mb

        # Log if needed
        if step % self.log_interval == 0:
            self._log_memory_stats(snapshot)

        # Check for alerts
        self._check_memory_alerts(snapshot)

    def _log_memory_stats(self, snapshot: MemorySnapshot):
        """Log memory statistics."""
        if self.baseline_memory:
            delta = snapshot.rss_mb - self.baseline_memory
            logger.info(
                f"Memory - Step {snapshot.step}: "
                f"RSS: {snapshot.rss_mb:.1f} MB "
                f"(+{delta:.1f} MB from baseline), "
                f"Available: {snapshot.available_mb:.1f} MB, "
                f"Usage: {snapshot.percent:.1f}%"
            )

    def _check_memory_alerts(self, snapshot: MemorySnapshot):
        """Check for memory alerts."""
        # High memory usage alert
        if snapshot.percent / 100 > self.alert_threshold:
            logger.warning(
                f"HIGH MEMORY USAGE: {snapshot.percent:.1f}% "
                f"(threshold: {self.alert_threshold * 100:.0f}%)"
            )

        # Memory leak detection
        if len(self.recent_snapshots) >= self.window_size:
            if self._detect_memory_leak():
                logger.warning(
                    "POTENTIAL MEMORY LEAK DETECTED: "
                    "Memory usage is consistently increasing"
                )

    def _detect_memory_leak(self) -> bool:
        """Detect potential memory leaks using trend analysis."""
        if len(self.recent_snapshots) < 10:
            return False

        # Get recent memory values
        memory_values = [s.rss_mb for s in self.recent_snapshots]

        # Simple linear regression to detect trend
        x = np.arange(len(memory_values))
        y = np.array(memory_values)

        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]

        # Leak if consistent positive slope > 1 MB per step
        return slope > 1.0

    def get_memory_summary(self) -> dict[str, float]:
        """Get memory usage summary."""
        if not self.snapshots:
            return {}

        current = self.snapshots[-1]

        summary = {
            "current_rss_mb": current.rss_mb,
            "peak_rss_mb": self.peak_memory,
            "baseline_mb": self.baseline_memory,
            "delta_from_baseline_mb": current.rss_mb - self.baseline_memory,
            "available_mb": current.available_mb,
            "usage_percent": current.percent,
            "mlx_estimated_mb": current.mlx_allocated_mb,
        }

        # Add trend info if available
        if len(self.recent_snapshots) >= 10:
            memory_values = [s.rss_mb for s in list(self.recent_snapshots)[-10:]]
            summary["recent_trend_mb_per_step"] = np.polyfit(
                np.arange(len(memory_values)), memory_values, 1
            )[0]

        return summary

    def save_profile(self, filename: str | None = None):
        """Save memory profile to file."""
        if not filename:
            filename = f"memory_profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = self.output_dir / filename

        import json

        profile_data = {
            "system_info": self.system_info,
            "summary": self.get_memory_summary(),
            "snapshots": [
                {
                    "step": s.step,
                    "timestamp": s.timestamp,
                    "rss_mb": s.rss_mb,
                    "vms_mb": s.vms_mb,
                    "available_mb": s.available_mb,
                    "percent": s.percent,
                    "description": s.description,
                }
                for s in self.snapshots
            ],
        }

        with open(filepath, "w") as f:
            json.dump(profile_data, f, indent=2)

        logger.info(f"Memory profile saved to {filepath}")

    def plot_memory_usage(self, save_path: str | None = None):
        """Plot memory usage over time."""
        if not self.snapshots:
            return

        if not save_path:
            save_path = (
                self.output_dir
                / f"memory_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )

        # Extract data
        steps = [s.step for s in self.snapshots]
        rss = [s.rss_mb for s in self.snapshots]
        available = [s.available_mb for s in self.snapshots]

        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # RSS memory plot
        ax1.plot(steps, rss, "b-", label="RSS Memory", linewidth=2)
        ax1.axhline(y=self.baseline_memory, color="g", linestyle="--", label="Baseline")
        ax1.axhline(y=self.peak_memory, color="r", linestyle="--", label="Peak")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Memory (MB)")
        ax1.set_title("Process Memory Usage")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Available memory plot
        ax2.plot(steps, available, "g-", label="Available System Memory", linewidth=2)
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Memory (MB)")
        ax2.set_title("Available System Memory")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Memory plot saved to {save_path}")

    def get_optimization_suggestions(self) -> list[str]:
        """Get memory optimization suggestions based on profile."""
        suggestions = []
        summary = self.get_memory_summary()

        if not summary:
            return suggestions

        # High memory usage
        if summary.get("usage_percent", 0) > 80:
            suggestions.append(
                "Consider reducing batch size or using gradient accumulation"
            )

        # Large delta from baseline
        delta = summary.get("delta_from_baseline_mb", 0)
        if delta > 1000:  # More than 1GB increase
            suggestions.append(
                f"Memory increased by {delta:.0f} MB from baseline. "
                "Check for memory leaks or unnecessary caching"
            )

        # Memory trend
        trend = summary.get("recent_trend_mb_per_step", 0)
        if trend > 0.5:
            suggestions.append(
                f"Memory increasing at {trend:.2f} MB/step. "
                "Consider periodic garbage collection or model checkpointing"
            )

        # Peak vs current
        if self.peak_memory > summary.get("current_rss_mb", 0) * 1.5:
            suggestions.append(
                "Peak memory was significantly higher than current. "
                "Consider memory profiling during peak usage"
            )

        return suggestions


# Context manager for memory profiling
class profile_memory:
    """Context manager for memory profiling."""

    def __init__(self, profiler: MemoryProfiler, description: str = ""):
        self.profiler = profiler
        self.description = description
        self.start_memory = None

    def __enter__(self):
        gc.collect()
        self.start_memory = self.profiler.process.memory_info().rss / (1024**2)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        gc.collect()
        end_memory = self.profiler.process.memory_info().rss / (1024**2)
        delta = end_memory - self.start_memory

        if abs(delta) > 10:  # Log if change > 10 MB
            logger.info(
                f"Memory delta for '{self.description}': "
                f"{delta:+.1f} MB (start: {self.start_memory:.1f} MB, "
                f"end: {end_memory:.1f} MB)"
            )
