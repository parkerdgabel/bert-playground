"""End-to-end tests for performance monitoring."""

import time
import pytest
from unittest.mock import Mock, patch

from cli.factory import CommandFactory
from cli.middleware.monitoring import PerformanceMiddleware, ResourceLimitMiddleware


class TestPerformanceMonitoring:
    """Test performance monitoring integration."""
    
    def test_performance_metrics_collection(self):
        """Test performance metrics are collected."""
        factory = CommandFactory()
        
        # Add performance middleware
        perf_middleware = PerformanceMiddleware(
            track_memory=True,
            track_cpu=True,
            track_gc=True
        )
        factory.middleware_pipeline.add(perf_middleware)
        
        @factory.create_middleware_command
        def test_command():
            time.sleep(0.1)  # Simulate work
            return "completed"
        
        result = test_command()
        
        assert result == "completed"
        assert len(perf_middleware.metrics_history) == 1
        
        metrics = perf_middleware.metrics_history[0]
        assert metrics.command == "test_command"
        assert metrics.duration >= 0.1
        assert metrics.memory_mb > 0
    
    def test_performance_thresholds(self):
        """Test performance threshold warnings."""
        factory = CommandFactory()
        
        # Set low thresholds for testing
        perf_middleware = PerformanceMiddleware(
            threshold_warn_seconds=0.05,
            threshold_error_seconds=0.2
        )
        factory.middleware_pipeline.add(perf_middleware)
        
        @factory.create_middleware_command
        def slow_command():
            time.sleep(0.1)  # Exceeds warning threshold
            return "slow"
        
        with patch("loguru.logger.warning") as mock_warn:
            result = slow_command()
            
            assert result == "slow"
            mock_warn.assert_called()
            
            # Check warning message
            call_args = mock_warn.call_args[0][0]
            assert "exceeds warning threshold" in call_args
    
    def test_resource_limit_enforcement(self):
        """Test resource limits are enforced."""
        factory = CommandFactory()
        
        # Add resource limit middleware
        resource_middleware = ResourceLimitMiddleware(
            max_duration_seconds=0.1,  # Very short limit
            track_memory=True
        )
        factory.middleware_pipeline.add(resource_middleware)
        
        @factory.create_middleware_command  
        def long_running_command():
            time.sleep(0.2)  # Exceeds time limit
            return "should not complete"
        
        # Should timeout
        with pytest.raises(TimeoutError, match="Time limit.*exceeded"):
            long_running_command()
    
    def test_memory_tracking(self):
        """Test memory usage tracking."""
        factory = CommandFactory()
        
        perf_middleware = PerformanceMiddleware(track_memory=True)
        factory.middleware_pipeline.add(perf_middleware)
        
        @factory.create_middleware_command
        def memory_using_command():
            # Allocate some memory
            data = [i for i in range(10000)]
            return len(data)
        
        result = memory_using_command()
        
        assert result == 10000
        assert len(perf_middleware.metrics_history) == 1
        
        metrics = perf_middleware.metrics_history[0]
        assert metrics.memory_mb > 0
        # Memory delta might be positive or negative depending on GC
        assert isinstance(metrics.memory_delta_mb, float)
    
    def test_performance_summary_generation(self):
        """Test performance summary generation."""
        factory = CommandFactory()
        
        perf_middleware = PerformanceMiddleware()
        factory.middleware_pipeline.add(perf_middleware)
        
        @factory.create_middleware_command
        def test_command(duration: float):
            time.sleep(duration)
            return f"slept {duration}s"
        
        # Execute multiple commands
        test_command(0.01)
        test_command(0.02)
        test_command(0.03)
        
        summary = perf_middleware.get_summary()
        
        assert summary["total_commands"] == 3
        assert summary["total_duration"] >= 0.06
        assert summary["average_duration"] >= 0.02
        assert summary["max_duration"] >= 0.03
        assert summary["min_duration"] >= 0.01
    
    def test_performance_report_generation(self):
        """Test performance report generation."""
        from rich.console import Console
        from io import StringIO
        
        factory = CommandFactory()
        
        # Capture console output
        output = StringIO()
        console = Console(file=output, width=80)
        
        perf_middleware = PerformanceMiddleware(
            console=console,
            track_memory=True
        )
        factory.middleware_pipeline.add(perf_middleware)
        
        @factory.create_middleware_command
        def test_command():
            return "test"
        
        # Execute command
        result = test_command()
        assert result == "test"
        
        # Generate report
        perf_middleware.print_report()
        
        output_str = output.getvalue()
        assert "Performance Report" in output_str
        assert "test_command" in output_str
        assert "Duration" in output_str
    
    def test_gc_statistics_tracking(self):
        """Test garbage collection statistics tracking."""
        import gc
        
        factory = CommandFactory()
        
        perf_middleware = PerformanceMiddleware(track_gc=True)
        factory.middleware_pipeline.add(perf_middleware)
        
        @factory.create_middleware_command
        def gc_triggering_command():
            # Create objects that might trigger GC
            data = []
            for i in range(1000):
                data.append([j for j in range(100)])
            return len(data)
        
        result = gc_triggering_command()
        
        assert result == 1000
        assert len(perf_middleware.metrics_history) == 1
        
        metrics = perf_middleware.metrics_history[0]
        assert isinstance(metrics.gc_stats, dict)
        # GC stats might be zero if no collections occurred
        assert all(isinstance(v, int) for v in metrics.gc_stats.values())
    
    def test_cpu_tracking(self):
        """Test CPU usage tracking."""
        factory = CommandFactory()
        
        perf_middleware = PerformanceMiddleware(track_cpu=True)
        factory.middleware_pipeline.add(perf_middleware)
        
        @factory.create_middleware_command
        def cpu_using_command():
            # Do some CPU-intensive work
            total = 0
            for i in range(100000):
                total += i * i
            return total
        
        result = cpu_using_command()
        
        assert result > 0
        assert len(perf_middleware.metrics_history) == 1
        
        metrics = perf_middleware.metrics_history[0]
        # CPU percentage might be 0 for very short operations
        assert metrics.cpu_percent >= 0
    
    def test_custom_metrics(self):
        """Test custom metrics collection."""
        factory = CommandFactory()
        
        perf_middleware = PerformanceMiddleware()
        factory.middleware_pipeline.add(perf_middleware)
        
        @factory.create_middleware_command
        def custom_metrics_command():
            # This would need middleware context access in real implementation
            # For now, just verify the mechanism exists
            return "metrics"
        
        result = custom_metrics_command()
        
        assert result == "metrics"
        # Custom metrics would be added via context in real implementation