"""Tests for the comprehensive monitoring system."""

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from training.config import TrainingConfig
from training.memory_manager import AppleSiliconMemoryManager
from training.monitoring import (
    ComprehensiveMonitor,
    MetricTracker,
    MLflowTracker,
    RichConsoleMonitor,
    TrainingMetrics,
)
from training.performance_profiler import AppleSiliconProfiler


class TestMetricTracker:
    """Test the metric tracker."""
    
    def test_metric_tracker_initialization(self):
        """Test metric tracker initialization."""
        tracker = MetricTracker("test_metric")
        
        assert tracker.name == "test_metric"
        assert tracker.values == []
        assert tracker.steps == []
        assert tracker.timestamps == []
        assert tracker.best_value is None
        assert tracker.best_step is None
        assert tracker.higher_is_better is True
    
    def test_metric_tracker_update_higher_better(self):
        """Test metric updates for higher-is-better metrics."""
        tracker = MetricTracker("accuracy", higher_is_better=True)
        
        # First update
        is_best = tracker.update(0.8, 1)
        assert is_best is True
        assert tracker.best_value == 0.8
        assert tracker.best_step == 1
        
        # Better value
        is_best = tracker.update(0.9, 2)
        assert is_best is True
        assert tracker.best_value == 0.9
        assert tracker.best_step == 2
        
        # Worse value
        is_best = tracker.update(0.7, 3)
        assert is_best is False
        assert tracker.best_value == 0.9
        assert tracker.best_step == 2
    
    def test_metric_tracker_update_lower_better(self):
        """Test metric updates for lower-is-better metrics."""
        tracker = MetricTracker("loss", higher_is_better=False)
        
        # First update
        is_best = tracker.update(0.5, 1)
        assert is_best is True
        assert tracker.best_value == 0.5
        assert tracker.best_step == 1
        
        # Better value (lower)
        is_best = tracker.update(0.3, 2)
        assert is_best is True
        assert tracker.best_value == 0.3
        assert tracker.best_step == 2
        
        # Worse value (higher)
        is_best = tracker.update(0.7, 3)
        assert is_best is False
        assert tracker.best_value == 0.3
        assert tracker.best_step == 2
    
    def test_metric_tracker_recent_average(self):
        """Test recent average calculation."""
        tracker = MetricTracker("test_metric")
        
        # Add some values
        for i, value in enumerate([0.1, 0.2, 0.3, 0.4, 0.5]):
            tracker.update(value, i)
        
        # Test recent average
        avg = tracker.get_recent_average(n=3)
        expected = (0.3 + 0.4 + 0.5) / 3
        assert abs(avg - expected) < 1e-6
    
    def test_metric_tracker_trend(self):
        """Test trend calculation."""
        tracker = MetricTracker("accuracy", higher_is_better=True)
        
        # No values
        assert tracker.get_trend() == "→"
        
        # Single value
        tracker.update(0.5, 1)
        assert tracker.get_trend() == "→"
        
        # Improving trend
        tracker.update(0.7, 2)
        assert tracker.get_trend() == "↗"
        
        # Declining trend
        tracker_loss = MetricTracker("loss", higher_is_better=False)
        tracker_loss.update(0.5, 1)
        tracker_loss.update(0.7, 2)
        assert tracker_loss.get_trend() == "↘"  # Worse for loss


class TestTrainingMetrics:
    """Test the training metrics container."""
    
    def test_training_metrics_initialization(self):
        """Test training metrics initialization."""
        metrics = TrainingMetrics()
        
        trackers = metrics.get_all_trackers()
        assert "train_loss" in trackers
        assert "train_accuracy" in trackers
        assert "val_loss" in trackers
        assert "val_accuracy" in trackers
        assert "learning_rate" in trackers
        assert "step_time" in trackers
        assert "throughput" in trackers
        assert "memory_usage" in trackers
        
        # Check that loss metrics are lower-is-better
        assert trackers["train_loss"].higher_is_better is False
        assert trackers["val_loss"].higher_is_better is False
        
        # Check that accuracy metrics are higher-is-better
        assert trackers["train_accuracy"].higher_is_better is True
        assert trackers["val_accuracy"].higher_is_better is True


class TestMLflowTracker:
    """Test MLflow integration."""
    
    @pytest.fixture
    def training_config(self):
        """Create training config for testing."""
        return TrainingConfig(
            experiment_name="test_experiment",
            batch_size=4,
            learning_rate=1e-4,
            epochs=1,
            output_dir="/tmp/test"
        )
    
    def test_mlflow_tracker_disabled(self, training_config):
        """Test MLflow tracker when disabled."""
        training_config.monitoring.enable_mlflow = False
        tracker = MLflowTracker(training_config)
        
        assert tracker.run_id is None
        
        # These should not raise errors when MLflow is disabled
        tracker.log_metrics({"test_metric": 1.0}, 1)
        tracker.log_artifacts({"test": "/tmp/test.txt"})
        tracker.end_run()
    
    @patch("training.monitoring.mlflow")
    @patch("training.monitoring.MLflowCentral")
    def test_mlflow_tracker_enabled(self, mock_central, mock_mlflow, training_config):
        """Test MLflow tracker when enabled."""
        training_config.monitoring.enable_mlflow = True
        
        # Mock MLflow run
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.start_run.return_value = mock_run
        
        tracker = MLflowTracker(training_config)
        
        # Verify initialization
        mock_central.return_value.initialize.assert_called_once()
        mock_mlflow.start_run.assert_called_once()
        
        # Test metric logging
        tracker.log_metrics({"test_metric": 1.0}, 1)
        mock_mlflow.log_metric.assert_called_with("test_metric", 1.0, step=1)
        
        # Test run ending
        tracker.end_run()
        mock_mlflow.end_run.assert_called_with(status="FINISHED")


class TestRichConsoleMonitor:
    """Test rich console monitoring."""
    
    @pytest.fixture
    def training_config(self):
        """Create training config for testing."""
        return TrainingConfig(
            experiment_name="test_experiment",
            batch_size=4,
            learning_rate=1e-4,
            epochs=1,
            output_dir="/tmp/test"
        )
    
    def test_console_monitor_disabled(self, training_config):
        """Test console monitor when disabled."""
        training_config.monitoring.enable_rich_console = False
        monitor = RichConsoleMonitor(training_config)
        
        assert monitor.enabled is False
        
        # These should not raise errors when disabled
        monitor.start_training(1, 10)
        monitor.update_training_progress(0, 5, None, None, None)
        monitor.stop_training()
    
    def test_console_monitor_enabled(self, training_config):
        """Test console monitor when enabled."""
        training_config.monitoring.enable_rich_console = True
        monitor = RichConsoleMonitor(training_config)
        
        assert monitor.enabled is True
        
        # Start training (should not raise errors)
        monitor.start_training(1, 10)
        assert monitor.progress is not None
        assert monitor.training_task is not None
        assert monitor.epoch_task is not None
        
        # Stop training
        monitor.stop_training()


class TestComprehensiveMonitor:
    """Test the comprehensive monitoring system."""
    
    @pytest.fixture
    def training_config(self):
        """Create training config for testing."""
        return TrainingConfig(
            experiment_name="test_experiment",
            batch_size=4,
            learning_rate=1e-4,
            epochs=1,
            output_dir="/tmp/test"
        )
    
    @pytest.fixture
    def memory_manager(self):
        """Create memory manager for testing."""
        return AppleSiliconMemoryManager()
    
    @pytest.fixture
    def profiler(self):
        """Create profiler for testing."""
        return AppleSiliconProfiler()
    
    def test_comprehensive_monitor_initialization(self, training_config, memory_manager, profiler):
        """Test comprehensive monitor initialization."""
        monitor = ComprehensiveMonitor(training_config, memory_manager, profiler)
        
        assert monitor.config == training_config
        assert monitor.memory_manager == memory_manager
        assert monitor.profiler == profiler
        assert monitor.metrics is not None
        assert monitor.mlflow_tracker is not None
        assert monitor.console_monitor is not None
    
    def test_log_step(self, training_config, memory_manager, profiler):
        """Test step logging."""
        training_config.monitoring.enable_mlflow = False  # Disable for testing
        training_config.monitoring.enable_rich_console = False
        
        monitor = ComprehensiveMonitor(training_config, memory_manager, profiler)
        
        # Log a step
        monitor.log_step(
            step=1,
            epoch=0,
            train_loss=0.5,
            train_accuracy=0.8,
            learning_rate=1e-4,
            batch_size=4
        )
        
        # Check that metrics were updated
        assert len(monitor.metrics.train_loss.values) == 1
        assert monitor.metrics.train_loss.values[0] == 0.5
        assert len(monitor.metrics.train_accuracy.values) == 1
        assert monitor.metrics.train_accuracy.values[0] == 0.8
    
    def test_log_validation(self, training_config, memory_manager, profiler):
        """Test validation logging."""
        training_config.monitoring.enable_mlflow = False  # Disable for testing
        training_config.monitoring.enable_rich_console = False
        
        monitor = ComprehensiveMonitor(training_config, memory_manager, profiler)
        
        # Log validation metrics
        improved = monitor.log_validation(
            step=1,
            val_loss=0.3,
            val_accuracy=0.9,
            additional_metrics={"val_f1": 0.85}
        )
        
        # Should be improved since it's the first validation
        assert improved is True
        assert len(monitor.metrics.val_loss.values) == 1
        assert monitor.metrics.val_loss.values[0] == 0.3
        assert len(monitor.metrics.val_accuracy.values) == 1
        assert monitor.metrics.val_accuracy.values[0] == 0.9
    
    def test_end_training(self, training_config, memory_manager, profiler):
        """Test training end."""
        training_config.monitoring.enable_mlflow = False  # Disable for testing
        training_config.monitoring.enable_rich_console = False
        
        monitor = ComprehensiveMonitor(training_config, memory_manager, profiler)
        
        # Add some metrics first
        monitor.log_step(1, 0, 0.5, 0.8, 1e-4, 4)
        monitor.log_validation(1, 0.3, 0.9)
        
        # End training
        summary = monitor.end_training("FINISHED")
        
        assert summary["status"] == "FINISHED"
        assert "elapsed_time_seconds" in summary
        assert "total_steps" in summary
        assert "best_metrics" in summary
        assert summary["best_metrics"]["train_loss"] == 0.5
        assert summary["best_metrics"]["val_accuracy"] == 0.9
    
    @patch("training.monitoring.Path")
    def test_save_checkpoint_artifacts(self, mock_path, training_config, memory_manager, profiler):
        """Test saving checkpoint artifacts."""
        training_config.monitoring.enable_mlflow = False  # Disable for testing
        
        monitor = ComprehensiveMonitor(training_config, memory_manager, profiler)
        
        # Mock file operations
        mock_path.return_value.exists.return_value = True
        
        with patch.object(monitor.memory_manager, 'save_memory_report') as mock_memory_save, \
             patch.object(monitor.profiler, 'save_performance_report') as mock_perf_save:
            
            monitor.save_checkpoint_artifacts("/tmp/checkpoint.safetensors")
            
            # Verify reports were saved
            mock_memory_save.assert_called_once()
            mock_perf_save.assert_called_once()