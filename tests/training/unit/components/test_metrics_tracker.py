"""Unit tests for MetricsTracker component."""

import json
from pathlib import Path
import pytest

from training.components.metrics_tracker import MetricsTracker


class TestMetricsTracker:
    """Test cases for MetricsTracker component."""
    
    @pytest.fixture
    def metrics_tracker(self, tmp_output_dir):
        """Create MetricsTracker instance."""
        return MetricsTracker(
            window_size=5,
            output_dir=tmp_output_dir,
        )
    
    def test_initialization(self, tmp_output_dir):
        """Test MetricsTracker initialization."""
        tracker = MetricsTracker(window_size=10, output_dir=tmp_output_dir)
        
        assert tracker.window_size == 10
        assert tracker.output_dir == tmp_output_dir
        assert len(tracker._metrics) == 0
        assert len(tracker._best_metrics) == 0
        assert len(tracker._metric_modes) == 0
    
    def test_add_metric(self, metrics_tracker):
        """Test adding a single metric."""
        metrics_tracker.add_metric("loss", 0.5, step=1)
        
        # Check metric was stored
        history = metrics_tracker.get_metric("loss")
        assert len(history) == 1
        assert history[0] == (1, 0.5)
        
        # Check moving average
        avg = metrics_tracker.get_moving_average("loss")
        assert avg == 0.5
    
    def test_add_metric_without_step(self, metrics_tracker):
        """Test adding metric without explicit step."""
        metrics_tracker.add_metric("accuracy", 0.8)
        
        history = metrics_tracker.get_metric("accuracy")
        assert len(history) == 1
        assert history[0][0] == 0  # Step should be 0 (length of empty history)
        assert history[0][1] == 0.8
    
    def test_add_metrics_batch(self, metrics_tracker):
        """Test adding multiple metrics at once."""
        metrics = {"loss": 0.4, "accuracy": 0.85, "f1": 0.82}
        metrics_tracker.add_metrics(metrics, step=5)
        
        # Check all metrics were added
        for name, value in metrics.items():
            history = metrics_tracker.get_metric(name)
            assert len(history) == 1
            assert history[0] == (5, value)
    
    def test_get_latest_metrics(self, metrics_tracker):
        """Test getting latest metric values."""
        # Add some metrics
        metrics_tracker.add_metric("loss", 0.8, step=1)
        metrics_tracker.add_metric("loss", 0.6, step=2)
        metrics_tracker.add_metric("accuracy", 0.7, step=1)
        
        latest = metrics_tracker.get_latest_metrics()
        
        assert latest["loss"] == 0.6  # Latest loss value
        assert latest["accuracy"] == 0.7
    
    def test_moving_average_window(self, tmp_output_dir):
        """Test moving average with window size."""
        tracker = MetricsTracker(window_size=3, output_dir=tmp_output_dir)
        
        # Add values that exceed window size
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for i, value in enumerate(values):
            tracker.add_metric("test", value, step=i)
        
        # Moving average should only use last 3 values
        avg = tracker.get_moving_average("test")
        expected = (3.0 + 4.0 + 5.0) / 3
        assert abs(avg - expected) < 1e-6
    
    def test_configure_metric_tracking(self, metrics_tracker):
        """Test configuring metric tracking modes."""
        # Configure loss as min
        metrics_tracker.configure_metric("loss", "min")
        # Configure accuracy as max
        metrics_tracker.configure_metric("accuracy", "max")
        
        assert metrics_tracker._metric_modes["loss"] == "min"
        assert metrics_tracker._metric_modes["accuracy"] == "max"
    
    def test_configure_metric_invalid_mode(self, metrics_tracker):
        """Test invalid metric mode raises error."""
        with pytest.raises(ValueError, match="Invalid mode: invalid"):
            metrics_tracker.configure_metric("loss", "invalid")
    
    def test_best_metric_tracking_min(self, metrics_tracker):
        """Test best metric tracking with min mode."""
        metrics_tracker.configure_metric("loss", "min")
        
        # Add decreasing loss values
        metrics_tracker.add_metric("loss", 1.0, step=1)
        assert metrics_tracker.is_best_metric("loss", 1.0)
        
        metrics_tracker.add_metric("loss", 0.8, step=2)
        assert metrics_tracker.is_best_metric("loss", 0.8)
        
        metrics_tracker.add_metric("loss", 0.9, step=3)
        assert not metrics_tracker.is_best_metric("loss", 0.9)
        
        # Check best metric
        best = metrics_tracker.get_best_metric("loss")
        assert best == (0.8, 2)
    
    def test_best_metric_tracking_max(self, metrics_tracker):
        """Test best metric tracking with max mode."""
        metrics_tracker.configure_metric("accuracy", "max")
        
        # Add increasing accuracy values
        metrics_tracker.add_metric("accuracy", 0.7, step=1)
        metrics_tracker.add_metric("accuracy", 0.8, step=2)
        metrics_tracker.add_metric("accuracy", 0.75, step=3)
        
        assert metrics_tracker.is_best_metric("accuracy", 0.9)
        assert not metrics_tracker.is_best_metric("accuracy", 0.6)
        
        # Check best metric
        best = metrics_tracker.get_best_metric("accuracy")
        assert best == (0.8, 2)
    
    def test_get_summary(self, metrics_tracker):
        """Test getting metrics summary."""
        # Configure metrics
        metrics_tracker.configure_metric("loss", "min")
        metrics_tracker.configure_metric("accuracy", "max")
        
        # Add some data
        metrics_tracker.add_metric("loss", 0.8, step=1)
        metrics_tracker.add_metric("loss", 0.6, step=2)
        metrics_tracker.add_metric("accuracy", 0.7, step=1)
        metrics_tracker.add_metric("accuracy", 0.8, step=2)
        
        summary = metrics_tracker.get_summary()
        
        # Check structure
        assert "latest" in summary
        assert "moving_averages" in summary
        assert "best" in summary
        
        # Check latest
        assert summary["latest"]["loss"] == 0.6
        assert summary["latest"]["accuracy"] == 0.8
        
        # Check moving averages
        assert "loss" in summary["moving_averages"]
        assert "accuracy" in summary["moving_averages"]
        
        # Check best
        assert summary["best"]["loss"]["value"] == 0.6
        assert summary["best"]["loss"]["step"] == 2
        assert summary["best"]["accuracy"]["value"] == 0.8
        assert summary["best"]["accuracy"]["step"] == 2
    
    def test_clear_metrics(self, metrics_tracker):
        """Test clearing all metrics."""
        # Add some data
        metrics_tracker.add_metric("loss", 0.5)
        metrics_tracker.configure_metric("loss", "min")
        
        # Verify data exists
        assert len(metrics_tracker.get_metric("loss")) > 0
        assert "loss" in metrics_tracker._metric_modes
        
        # Clear
        metrics_tracker.clear()
        
        # Verify data is gone
        assert len(metrics_tracker.get_metric("loss")) == 0
        assert "loss" not in metrics_tracker._metric_modes
        assert len(metrics_tracker._best_metrics) == 0
    
    def test_save_and_load(self, metrics_tracker, tmp_output_dir):
        """Test saving and loading metrics."""
        # Configure and add data
        metrics_tracker.configure_metric("loss", "min")
        metrics_tracker.add_metric("loss", 0.8, step=1)
        metrics_tracker.add_metric("loss", 0.6, step=2)
        metrics_tracker.add_metric("accuracy", 0.75, step=1)
        
        # Save to file
        save_path = tmp_output_dir / "test_metrics.json"
        metrics_tracker.save(save_path)
        
        assert save_path.exists()
        
        # Create new tracker and load
        new_tracker = MetricsTracker(window_size=5)
        new_tracker.load(save_path)
        
        # Verify data was loaded
        assert len(new_tracker.get_metric("loss")) == 2
        assert len(new_tracker.get_metric("accuracy")) == 1
        assert new_tracker._metric_modes["loss"] == "min"
        
        # Check moving averages were rebuilt
        avg = new_tracker.get_moving_average("loss")
        assert avg == (0.8 + 0.6) / 2
    
    def test_log_to_console(self, metrics_tracker, caplog):
        """Test logging metrics to console."""
        metrics = {"loss": 0.5234, "accuracy": 0.8567}
        
        metrics_tracker.log_to_console(metrics, prefix="Epoch 5")
        
        # Check that log message contains formatted metrics
        log_messages = [record.message for record in caplog.records]
        assert any("Epoch 5" in msg for msg in log_messages)
        assert any("loss: 0.5234" in msg for msg in log_messages)
        assert any("accuracy: 0.8567" in msg for msg in log_messages)
    
    def test_log_to_console_without_prefix(self, metrics_tracker, caplog):
        """Test logging metrics without prefix."""
        metrics = {"f1": 0.7890}
        
        metrics_tracker.log_to_console(metrics)
        
        log_messages = [record.message for record in caplog.records]
        assert any("f1: 0.7890" in msg for msg in log_messages)
    
    def test_save_epoch_metrics(self, metrics_tracker):
        """Test saving epoch metrics to JSONL."""
        if not metrics_tracker.output_dir:
            pytest.skip("No output directory configured")
        
        train_metrics = {"loss": 0.5, "accuracy": 0.8}
        val_metrics = {"eval_loss": 0.6, "eval_accuracy": 0.75}
        
        # Save epoch metrics
        metrics_tracker.save_epoch_metrics(
            epoch=1,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
        )
        
        # Check JSONL file was created
        metrics_file = metrics_tracker.output_dir / "metrics.jsonl"
        assert metrics_file.exists()
        
        # Check content
        with open(metrics_file, "r") as f:
            line = f.readline()
            record = json.loads(line)
        
        assert record["epoch"] == 1
        assert record["train"] == train_metrics
        assert record["validation"] == val_metrics
    
    def test_save_epoch_metrics_train_only(self, metrics_tracker):
        """Test saving epoch metrics without validation."""
        if not metrics_tracker.output_dir:
            pytest.skip("No output directory configured")
        
        train_metrics = {"loss": 0.4}
        
        metrics_tracker.save_epoch_metrics(
            epoch=2,
            train_metrics=train_metrics,
        )
        
        metrics_file = metrics_tracker.output_dir / "metrics.jsonl"
        assert metrics_file.exists()
        
        with open(metrics_file, "r") as f:
            line = f.readline()
            record = json.loads(line)
        
        assert record["epoch"] == 2
        assert record["train"] == train_metrics
        assert "validation" not in record
    
    def test_get_metric_nonexistent(self, metrics_tracker):
        """Test getting non-existent metric returns empty list."""
        history = metrics_tracker.get_metric("nonexistent")
        assert history == []
    
    def test_get_moving_average_nonexistent(self, metrics_tracker):
        """Test getting moving average for non-existent metric returns None."""
        avg = metrics_tracker.get_moving_average("nonexistent")
        assert avg is None
    
    def test_get_best_metric_nonexistent(self, metrics_tracker):
        """Test getting best metric for non-existent metric returns None."""
        best = metrics_tracker.get_best_metric("nonexistent")
        assert best is None
    
    def test_is_best_metric_unconfigured(self, metrics_tracker):
        """Test is_best_metric returns False for unconfigured metric."""
        assert not metrics_tracker.is_best_metric("unconfigured", 0.5)