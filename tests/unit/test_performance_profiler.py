"""Tests for the performance profiler."""

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from training.performance_profiler import (
    AppleSiliconProfiler,
    PerformanceMetrics,
    ProfilerConfig,
)


class TestProfilerConfig:
    """Test profiler configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ProfilerConfig()
        
        assert config.metrics_collection_interval == 10
        assert config.detailed_profiling_interval == 100
        assert config.thermal_monitoring_interval == 50
        assert config.enable_neural_engine_monitoring is True
        assert config.enable_thermal_monitoring is True
        assert config.enable_power_monitoring is True
        assert config.save_detailed_logs is True
        assert config.log_to_console is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = ProfilerConfig(
            metrics_collection_interval=5,
            enable_neural_engine_monitoring=False,
            save_detailed_logs=False,
            max_step_time_seconds=5.0,
        )
        
        assert config.metrics_collection_interval == 5
        assert config.enable_neural_engine_monitoring is False
        assert config.save_detailed_logs is False
        assert config.max_step_time_seconds == 5.0


class TestPerformanceMetrics:
    """Test performance metrics data structure."""
    
    def test_performance_metrics_creation(self):
        """Test creating performance metrics."""
        timestamp = time.time()
        metrics = PerformanceMetrics(
            timestamp=timestamp,
            step=100,
            step_time_seconds=0.5,
            samples_per_second=64.0,
            tokens_per_second=16384.0,
        )
        
        assert metrics.timestamp == timestamp
        assert metrics.step == 100
        assert metrics.step_time_seconds == 0.5
        assert metrics.samples_per_second == 64.0
        assert metrics.tokens_per_second == 16384.0
        assert metrics.memory_usage_percent == 0.0  # Default value


class TestAppleSiliconProfiler:
    """Test the Apple Silicon profiler."""
    
    @pytest.fixture
    def profiler(self):
        """Create a profiler for testing."""
        config = ProfilerConfig(
            metrics_collection_interval=1,  # Frequent collection for tests
            log_to_console=False,  # Disable console logging in tests
        )
        return AppleSiliconProfiler(config)
    
    def test_initialization(self, profiler):
        """Test profiler initialization."""
        assert profiler.config is not None
        assert isinstance(profiler.metrics_history, list)
        assert len(profiler.metrics_history) == 0
        assert isinstance(profiler.step_timers, dict)
        assert len(profiler.step_timers) == 0
        assert isinstance(profiler.performance_warnings, list)
    
    def test_step_timing(self, profiler):
        """Test step timing functionality."""
        step = 42
        batch_size = 32
        sequence_length = 128
        
        # Start timer
        profiler.start_step_timer(step)
        assert f"step_{step}" in profiler.step_timers
        assert "last_step_start" in profiler.step_timers
        
        # Simulate some work
        time.sleep(0.01)
        
        # End timer
        with patch.object(profiler, '_collect_system_metrics') as mock_system:
            mock_system.return_value = {
                "memory_usage_percent": 50.0,
                "neural_engine_utilization": 25.0,
            }
            
            metrics = profiler.end_step_timer(step, batch_size, sequence_length)
            
            assert metrics.step == step
            assert metrics.step_time_seconds > 0
            assert metrics.samples_per_second > 0
            assert metrics.tokens_per_second > 0
            assert metrics.memory_usage_percent == 50.0
            assert metrics.neural_engine_utilization == 25.0
    
    def test_step_timing_without_sequence_length(self, profiler):
        """Test step timing without sequence length."""
        step = 1
        batch_size = 16
        
        profiler.start_step_timer(step)
        time.sleep(0.01)
        
        with patch.object(profiler, '_collect_system_metrics') as mock_system:
            mock_system.return_value = {"memory_usage_percent": 30.0}
            
            metrics = profiler.end_step_timer(step, batch_size)
            
            assert metrics.tokens_per_second == 0.0  # No sequence length provided
            assert metrics.samples_per_second > 0
    
    @patch("psutil.virtual_memory")
    def test_collect_memory_metrics(self, mock_memory, profiler):
        """Test memory metrics collection."""
        mock_memory.return_value = MagicMock(percent=75.0)
        
        with patch.object(profiler, '_estimate_memory_bandwidth', return_value=50.0):
            metrics = profiler._collect_memory_metrics()
            
            assert metrics["memory_usage_percent"] == 75.0
            assert metrics["memory_bandwidth_gb_s"] == 50.0
    
    def test_collect_apple_silicon_metrics(self, profiler):
        """Test Apple Silicon specific metrics collection."""
        with patch.object(profiler, '_get_neural_engine_utilization', return_value=30.0):
            with patch.object(profiler, '_get_gpu_utilization', return_value=45.0):
                with patch.object(profiler, '_get_cpu_core_utilization') as mock_cpu:
                    mock_cpu.return_value = {
                        "efficiency_cores_load": 20.0,
                        "performance_cores_load": 60.0,
                    }
                    
                    metrics = profiler._collect_apple_silicon_metrics()
                    
                    assert metrics["neural_engine_utilization"] == 30.0
                    assert metrics["gpu_utilization"] == 45.0
                    assert metrics["efficiency_cores_load"] == 20.0
                    assert metrics["performance_cores_load"] == 60.0
    
    def test_collect_thermal_metrics(self, profiler):
        """Test thermal metrics collection."""
        with patch.object(profiler, '_get_temperatures') as mock_temps:
            mock_temps.return_value = {"cpu": 65.0, "gpu": 60.0}
            
            metrics = profiler._collect_thermal_metrics()
            
            assert metrics["cpu_temperature_celsius"] == 65.0
            assert metrics["gpu_temperature_celsius"] == 60.0
            assert metrics["thermal_throttling"] is False  # Below threshold
    
    def test_collect_thermal_metrics_with_throttling(self, profiler):
        """Test thermal metrics with throttling detected."""
        with patch.object(profiler, '_get_temperatures') as mock_temps:
            mock_temps.return_value = {"cpu": 85.0, "gpu": 75.0}
            
            metrics = profiler._collect_thermal_metrics()
            
            assert metrics["cpu_temperature_celsius"] == 85.0
            assert metrics["thermal_throttling"] is True  # Above threshold
    
    def test_collect_power_metrics(self, profiler):
        """Test power metrics collection."""
        with patch.object(profiler, '_get_power_consumption') as mock_power:
            mock_power.return_value = {"cpu": 15.0, "gpu": 25.0, "total": 40.0}
            
            metrics = profiler._collect_power_metrics()
            
            assert metrics["cpu_power_watts"] == 15.0
            assert metrics["gpu_power_watts"] == 25.0
            assert metrics["total_power_watts"] == 40.0
    
    def test_performance_warnings_slow_step(self, profiler):
        """Test performance warning for slow steps."""
        profiler.config.max_step_time_seconds = 1.0
        
        # Create metrics with slow step
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            step=1,
            step_time_seconds=2.0,  # Exceeds threshold
            samples_per_second=10.0,
        )
        
        profiler._check_performance_warnings(metrics)
        
        assert len(profiler.performance_warnings) == 1
        warning = profiler.performance_warnings[0]
        assert warning["type"] == "slow_step"
        assert warning["severity"] == "warning"
    
    def test_performance_warnings_low_throughput(self, profiler):
        """Test performance warning for low throughput."""
        profiler.config.min_throughput_samples_per_second = 5.0
        
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            step=1,
            step_time_seconds=0.5,
            samples_per_second=2.0,  # Below threshold
        )
        
        profiler._check_performance_warnings(metrics)
        
        assert len(profiler.performance_warnings) == 1
        warning = profiler.performance_warnings[0]
        assert warning["type"] == "low_throughput"
        assert warning["severity"] == "warning"
    
    def test_performance_warnings_thermal_throttling(self, profiler):
        """Test performance warning for thermal throttling."""
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            step=1,
            step_time_seconds=0.5,
            samples_per_second=10.0,
            thermal_throttling=True,
            cpu_temperature_celsius=85.0,
        )
        
        profiler._check_performance_warnings(metrics)
        
        assert len(profiler.performance_warnings) == 1
        warning = profiler.performance_warnings[0]
        assert warning["type"] == "thermal_throttling"
        assert warning["severity"] == "critical"
    
    def test_profile_mlx_operation(self, profiler):
        """Test profiling MLX operations."""
        def dummy_operation(x, y):
            time.sleep(0.01)  # Simulate work
            return x + y
        
        result, profiling_metrics = profiler.profile_mlx_operation(
            "test_operation", dummy_operation, 5, 10
        )
        
        assert result == 15
        assert profiling_metrics["operation_name"] == "test_operation"
        assert profiling_metrics["execution_time_ms"] > 0
        assert "timestamp" in profiling_metrics
    
    def test_get_performance_summary_empty(self, profiler):
        """Test performance summary with no metrics."""
        summary = profiler.get_performance_summary()
        
        assert "error" in summary
        assert summary["error"] == "No metrics collected yet"
    
    def test_get_performance_summary_with_data(self, profiler):
        """Test performance summary with collected metrics."""
        # Add some fake metrics
        for i in range(5):
            metrics = PerformanceMetrics(
                timestamp=time.time() + i,
                step=i,
                step_time_seconds=0.5 + i * 0.1,
                samples_per_second=20.0 - i,
                memory_usage_percent=50.0 + i * 5,
            )
            profiler.metrics_history.append(metrics)
        
        summary = profiler.get_performance_summary()
        
        assert summary["total_steps"] == 5
        assert "step_time" in summary
        assert "throughput" in summary
        assert "memory" in summary
        assert "warnings" in summary
        
        # Check step time statistics
        assert "average_seconds" in summary["step_time"]
        assert "min_seconds" in summary["step_time"]
        assert "max_seconds" in summary["step_time"]
        assert "std_seconds" in summary["step_time"]
    
    def test_save_performance_report(self, profiler):
        """Test saving performance report."""
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "performance_report.json"
            
            # Add some fake metrics
            metrics = PerformanceMetrics(
                timestamp=time.time(),
                step=1,
                step_time_seconds=0.5,
                samples_per_second=32.0,
            )
            profiler.metrics_history.append(metrics)
            
            profiler.save_performance_report(report_path)
            
            assert report_path.exists()
            
            # Load and verify report structure
            import json
            with open(report_path) as f:
                report = json.load(f)
            
            assert "profiler_config" in report
            assert "system_info" in report
            assert "performance_summary" in report
            assert "warnings" in report
            assert "detailed_metrics" in report
    
    def test_reset_metrics(self, profiler):
        """Test resetting profiler metrics."""
        # Add some data
        profiler.metrics_history.append(MagicMock())
        profiler.performance_warnings.append({"type": "test"})
        profiler.step_timers["test"] = time.time()
        
        # Reset
        profiler.reset_metrics()
        
        assert len(profiler.metrics_history) == 0
        assert len(profiler.performance_warnings) == 0
        assert len(profiler.step_timers) == 0