"""Performance monitoring middleware for CLI commands."""

import asyncio
import gc
import psutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from loguru import logger
from rich.console import Console
from rich.table import Table

from cli.middleware.base import CommandContext, Middleware, MiddlewareResult


@dataclass
class PerformanceMetrics:
    """Performance metrics for command execution."""
    
    command: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: float = 0.0
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    memory_delta_mb: float = 0.0
    gc_stats: Dict[str, int] = field(default_factory=dict)
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "command": self.command,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "cpu_percent": self.cpu_percent,
            "memory_mb": self.memory_mb,
            "memory_delta_mb": self.memory_delta_mb,
            "gc_stats": self.gc_stats,
            "custom_metrics": self.custom_metrics,
        }


class PerformanceMiddleware(Middleware):
    """Middleware for performance monitoring."""
    
    def __init__(
        self,
        name: str = "PerformanceMiddleware",
        track_memory: bool = True,
        track_cpu: bool = True,
        track_gc: bool = True,
        console: Optional[Console] = None,
        threshold_warn_seconds: float = 5.0,
        threshold_error_seconds: float = 30.0
    ):
        """Initialize performance middleware.
        
        Args:
            name: Middleware name
            track_memory: Whether to track memory usage
            track_cpu: Whether to track CPU usage
            track_gc: Whether to track garbage collection
            console: Rich console for output
            threshold_warn_seconds: Warning threshold for execution time
            threshold_error_seconds: Error threshold for execution time
        """
        super().__init__(name)
        self.track_memory = track_memory
        self.track_cpu = track_cpu
        self.track_gc = track_gc
        self.console = console
        self.threshold_warn = threshold_warn_seconds
        self.threshold_error = threshold_error_seconds
        self.metrics_history: List[PerformanceMetrics] = []
        self._process = psutil.Process()
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self._process.memory_info().rss / 1024 / 1024
    
    def _get_gc_stats(self) -> Dict[str, int]:
        """Get garbage collection statistics."""
        return {
            f"gen{i}_collections": gc.get_count()[i]
            for i in range(gc.get_count().__len__())
        }
    
    async def process(
        self,
        context: CommandContext,
        next_handler: Callable[[CommandContext], Any]
    ) -> MiddlewareResult:
        """Process performance monitoring."""
        # Initialize metrics
        metrics = PerformanceMetrics(
            command=context.command_name,
            start_time=datetime.now()
        )
        
        # Capture initial state
        if self.track_memory:
            initial_memory = self._get_memory_usage()
        
        if self.track_gc:
            initial_gc = self._get_gc_stats()
        
        if self.track_cpu:
            self._process.cpu_percent()  # Initialize CPU tracking
        
        start_time = time.perf_counter()
        
        try:
            # Execute command
            if asyncio.iscoroutinefunction(next_handler):
                result = await next_handler(context)
            else:
                result = next_handler(context)
            
            # Capture final state
            end_time = time.perf_counter()
            metrics.end_time = datetime.now()
            metrics.duration = end_time - start_time
            
            if self.track_memory:
                final_memory = self._get_memory_usage()
                metrics.memory_mb = final_memory
                metrics.memory_delta_mb = final_memory - initial_memory
            
            if self.track_cpu:
                metrics.cpu_percent = self._process.cpu_percent()
            
            if self.track_gc:
                final_gc = self._get_gc_stats()
                metrics.gc_stats = {
                    k: final_gc.get(k, 0) - initial_gc.get(k, 0)
                    for k in final_gc
                }
            
            # Add custom metrics from context
            if "metrics" in context.metadata:
                metrics.custom_metrics.update(context.metadata["metrics"])
            
            # Store metrics
            self.metrics_history.append(metrics)
            result.metadata["performance_metrics"] = metrics.to_dict()
            
            # Check thresholds
            if metrics.duration > self.threshold_error:
                logger.error(
                    f"Command {context.command_name} took {metrics.duration:.2f}s "
                    f"(exceeds error threshold of {self.threshold_error}s)"
                )
            elif metrics.duration > self.threshold_warn:
                logger.warning(
                    f"Command {context.command_name} took {metrics.duration:.2f}s "
                    f"(exceeds warning threshold of {self.threshold_warn}s)"
                )
            else:
                logger.debug(f"Command {context.command_name} completed in {metrics.duration:.2f}s")
            
            return result
            
        except Exception as e:
            # Still record metrics on error
            metrics.end_time = datetime.now()
            metrics.duration = time.perf_counter() - start_time
            self.metrics_history.append(metrics)
            raise
    
    def get_summary(self, last_n: Optional[int] = None) -> Dict[str, Any]:
        """Get performance summary."""
        metrics = self.metrics_history[-last_n:] if last_n else self.metrics_history
        
        if not metrics:
            return {"message": "No performance data available"}
        
        total_duration = sum(m.duration for m in metrics)
        avg_duration = total_duration / len(metrics)
        max_duration = max(m.duration for m in metrics)
        min_duration = min(m.duration for m in metrics)
        
        summary = {
            "total_commands": len(metrics),
            "total_duration": total_duration,
            "average_duration": avg_duration,
            "max_duration": max_duration,
            "min_duration": min_duration,
        }
        
        if self.track_memory:
            summary["average_memory_mb"] = sum(m.memory_mb for m in metrics) / len(metrics)
            summary["total_memory_delta_mb"] = sum(m.memory_delta_mb for m in metrics)
        
        if self.track_cpu:
            cpu_values = [m.cpu_percent for m in metrics if m.cpu_percent > 0]
            if cpu_values:
                summary["average_cpu_percent"] = sum(cpu_values) / len(cpu_values)
        
        return summary
    
    def print_report(self, last_n: Optional[int] = None) -> None:
        """Print performance report."""
        if not self.console:
            self.console = Console()
        
        metrics = self.metrics_history[-last_n:] if last_n else self.metrics_history
        
        if not metrics:
            self.console.print("[yellow]No performance data to report[/yellow]")
            return
        
        # Create table
        table = Table(title="Performance Report")
        table.add_column("Command", style="cyan")
        table.add_column("Duration (s)", justify="right")
        
        if self.track_memory:
            table.add_column("Memory (MB)", justify="right")
            table.add_column("Δ Memory (MB)", justify="right")
        
        if self.track_cpu:
            table.add_column("CPU %", justify="right")
        
        table.add_column("Time", style="dim")
        
        # Add rows
        for metric in metrics:
            row = [
                metric.command,
                f"{metric.duration:.3f}",
            ]
            
            if self.track_memory:
                row.extend([
                    f"{metric.memory_mb:.1f}",
                    f"{metric.memory_delta_mb:+.1f}",
                ])
            
            if self.track_cpu:
                row.append(f"{metric.cpu_percent:.1f}")
            
            row.append(metric.start_time.strftime("%H:%M:%S"))
            
            # Color based on duration
            style = None
            if metric.duration > self.threshold_error:
                style = "red"
            elif metric.duration > self.threshold_warn:
                style = "yellow"
            
            table.add_row(*row, style=style)
        
        # Add summary
        summary = self.get_summary(last_n)
        table.add_section()
        table.add_row(
            "TOTAL",
            f"{summary['total_duration']:.3f}",
            *[""] * (len(table.columns) - 3),
            f"{summary['total_commands']} commands",
            style="bold"
        )
        
        self.console.print(table)


class ResourceLimitMiddleware(PerformanceMiddleware):
    """Middleware for enforcing resource limits."""
    
    def __init__(
        self,
        name: str = "ResourceLimitMiddleware",
        max_memory_mb: Optional[float] = None,
        max_duration_seconds: Optional[float] = None,
        **kwargs
    ):
        """Initialize resource limit middleware."""
        super().__init__(name=name, **kwargs)
        self.max_memory_mb = max_memory_mb
        self.max_duration_seconds = max_duration_seconds
    
    async def process(
        self,
        context: CommandContext,
        next_handler: Callable[[CommandContext], Any]
    ) -> MiddlewareResult:
        """Process with resource limits."""
        # Set up monitoring
        initial_memory = self._get_memory_usage() if self.max_memory_mb else 0
        start_time = time.perf_counter()
        
        # Create monitoring task
        stop_monitoring = asyncio.Event()
        
        async def monitor():
            """Monitor resource usage."""
            while not stop_monitoring.is_set():
                # Check memory
                if self.max_memory_mb:
                    current_memory = self._get_memory_usage()
                    if current_memory - initial_memory > self.max_memory_mb:
                        logger.error(f"Memory limit exceeded: {current_memory:.1f}MB")
                        return MiddlewareResult.fail(
                            MemoryError(f"Memory limit {self.max_memory_mb}MB exceeded")
                        )
                
                # Check duration
                if self.max_duration_seconds:
                    duration = time.perf_counter() - start_time
                    if duration > self.max_duration_seconds:
                        logger.error(f"Time limit exceeded: {duration:.1f}s")
                        return MiddlewareResult.fail(
                            TimeoutError(f"Time limit {self.max_duration_seconds}s exceeded")
                        )
                
                await asyncio.sleep(0.1)
        
        # Start monitoring
        monitor_task = asyncio.create_task(monitor())
        
        try:
            # Execute command
            result = await super().process(context, next_handler)
            return result
        finally:
            # Stop monitoring
            stop_monitoring.set()
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass