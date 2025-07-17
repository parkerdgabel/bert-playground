"""Unit tests for MLflow integration components."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
import mlflow
import mlx.core as mx

from utils.mlflow_central import MLflowCentral, MLflowConfigurationError
from utils.mlflow_health import MLflowHealthChecker
from training.monitoring import MLflowTracker, TrainingMetrics
from training.config import TrainingConfig, MonitoringConfig


class TestMLflowCentral(unittest.TestCase):
    """Test cases for MLflowCentral class."""
    
    def setUp(self):
        """Set up test environment."""
        # Reset the singleton instance
        MLflowCentral._instance = None
        
        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = Path(self.temp_dir) / "test.db"
        self.test_artifacts_path = Path(self.temp_dir) / "artifacts"
        
        # Test configuration
        self.test_tracking_uri = f"sqlite:///{self.test_db_path}"
        self.test_artifact_root = str(self.test_artifacts_path)
    
    def tearDown(self):
        """Clean up test environment."""
        # Clean up temporary files
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
        
        # Reset singleton
        MLflowCentral._instance = None
    
    def test_singleton_pattern(self):
        """Test that MLflowCentral follows singleton pattern."""
        instance1 = MLflowCentral()
        instance2 = MLflowCentral()
        
        self.assertIs(instance1, instance2)
    
    def test_initialization_with_valid_config(self):
        """Test successful initialization with valid configuration."""
        central = MLflowCentral()
        
        # Should not raise exception
        central.initialize(
            tracking_uri=self.test_tracking_uri,
            artifact_root=self.test_artifact_root,
            experiment_name="test_experiment"
        )
        
        self.assertEqual(central.tracking_uri, self.test_tracking_uri)
        self.assertEqual(central.artifact_root, self.test_artifact_root)
        self.assertTrue(central._initialized)
    
    def test_initialization_creates_directories(self):
        """Test that initialization creates necessary directories."""
        central = MLflowCentral()
        
        central.initialize(
            tracking_uri=self.test_tracking_uri,
            artifact_root=self.test_artifact_root
        )
        
        # Directories should be created
        self.assertTrue(self.test_db_path.parent.exists())
        self.assertTrue(self.test_artifacts_path.exists())
    
    def test_invalid_tracking_uri(self):
        """Test initialization with invalid tracking URI."""
        central = MLflowCentral()
        
        with self.assertRaises(MLflowConfigurationError):
            central.initialize(tracking_uri="")
    
    def test_invalid_artifact_root(self):
        """Test initialization with invalid artifact root."""
        central = MLflowCentral()
        
        with self.assertRaises(MLflowConfigurationError):
            central.initialize(
                tracking_uri=self.test_tracking_uri,
                artifact_root=""
            )
    
    @patch('os.access')
    def test_permission_validation(self, mock_access):
        """Test permission validation for database directory."""
        mock_access.return_value = False  # Simulate no write permission
        
        central = MLflowCentral()
        
        with self.assertRaises(MLflowConfigurationError):
            central.initialize(
                tracking_uri=self.test_tracking_uri,
                artifact_root=self.test_artifact_root
            )
    
    @patch('mlflow.search_experiments')
    def test_validate_connection_success(self, mock_search):
        """Test successful connection validation."""
        mock_search.return_value = []
        
        central = MLflowCentral()
        central.initialize(
            tracking_uri=self.test_tracking_uri,
            artifact_root=self.test_artifact_root
        )
        
        # Create empty database file
        self.test_db_path.touch()
        
        result = central.validate_connection()
        
        self.assertEqual(result["status"], "CONNECTED")
        self.assertIn("experiments found", result["message"])
    
    def test_validate_connection_not_initialized(self):
        """Test connection validation when not initialized."""
        central = MLflowCentral()
        
        result = central.validate_connection()
        
        self.assertEqual(result["status"], "NOT_INITIALIZED")
    
    def test_validate_connection_missing_database(self):
        """Test connection validation with missing database."""
        central = MLflowCentral()
        central.initialize(
            tracking_uri=self.test_tracking_uri,
            artifact_root=self.test_artifact_root
        )
        
        # Don't create database file
        result = central.validate_connection()
        
        self.assertEqual(result["status"], "DATABASE_MISSING")
    
    def test_environment_variable_override(self):
        """Test that environment variables override defaults."""
        test_uri = "sqlite:///env_test.db"
        test_artifacts = "./env_artifacts"
        
        with patch.dict(os.environ, {
            'MLFLOW_TRACKING_URI': test_uri,
            'MLFLOW_ARTIFACT_ROOT': test_artifacts
        }):
            central = MLflowCentral()
            central.initialize()
            
            self.assertEqual(central.tracking_uri, test_uri)
            self.assertEqual(central.artifact_root, test_artifacts)


class TestMLflowHealthChecker(unittest.TestCase):
    """Test cases for MLflowHealthChecker class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.health_checker = MLflowHealthChecker()
        
        # Mock MLflow central
        self.mock_central = Mock()
        self.health_checker.mlflow_central = self.mock_central
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_database_connectivity_check_success(self):
        """Test successful database connectivity check."""
        # Mock successful initialization
        self.mock_central.initialize.return_value = None
        self.mock_central.tracking_uri = f"sqlite:///{self.temp_dir}/test.db"
        
        # Create test database file
        db_path = Path(self.temp_dir) / "test.db"
        db_path.touch()
        
        with patch('sqlite3.connect') as mock_connect:
            mock_cursor = Mock()
            mock_cursor.fetchall.return_value = [('table1',), ('table2',)]
            mock_connect.return_value.__enter__.return_value.cursor.return_value = mock_cursor
            
            with patch('mlflow.search_experiments') as mock_search:
                mock_search.return_value = []
                
                result = self.health_checker._check_database_connectivity()
                
                self.assertEqual(result["status"], "PASS")
                self.assertIn("Database connected successfully", result["message"])
    
    def test_database_connectivity_check_missing_file(self):
        """Test database connectivity check with missing database file."""
        self.mock_central.initialize.return_value = None
        self.mock_central.tracking_uri = f"sqlite:///{self.temp_dir}/missing.db"
        
        result = self.health_checker._check_database_connectivity()
        
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("Database file does not exist", result["message"])
    
    def test_directory_permissions_check_success(self):
        """Test successful directory permissions check."""
        self.mock_central.artifact_root = str(Path(self.temp_dir) / "artifacts")
        
        result = self.health_checker._check_directory_permissions()
        
        self.assertEqual(result["status"], "PASS")
        self.assertIn("All directory permissions are correct", result["message"])
    
    @patch('pathlib.Path.mkdir')
    def test_directory_permissions_check_failure(self, mock_mkdir):
        """Test directory permissions check with permission error."""
        mock_mkdir.side_effect = PermissionError("Permission denied")
        
        result = self.health_checker._check_directory_permissions()
        
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("permission", result["message"].lower())
    
    def test_configuration_validity_check_success(self):
        """Test successful configuration validity check."""
        self.mock_central.tracking_uri = f"sqlite:///{self.temp_dir}/test.db"
        self.mock_central.artifact_root = str(Path(self.temp_dir) / "artifacts")
        
        # Create database directory
        Path(self.temp_dir).mkdir(exist_ok=True)
        
        result = self.health_checker._check_configuration_validity()
        
        self.assertEqual(result["status"], "PASS")
        self.assertIn("Configuration is valid", result["message"])
    
    def test_configuration_validity_check_empty_uri(self):
        """Test configuration validity check with empty tracking URI."""
        self.mock_central.tracking_uri = ""
        self.mock_central.artifact_root = str(Path(self.temp_dir) / "artifacts")
        
        result = self.health_checker._check_configuration_validity()
        
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("Tracking URI is not configured", result["message"])
    
    @patch('mlflow.create_experiment')
    @patch('mlflow.get_experiment')
    @patch('mlflow.delete_experiment')
    def test_experiment_creation_check_success(self, mock_delete, mock_get, mock_create):
        """Test successful experiment creation check."""
        mock_create.return_value = "test_experiment_id"
        mock_get.return_value = Mock()
        mock_delete.return_value = None
        
        self.mock_central.initialize.return_value = None
        self.mock_central.artifact_root = str(Path(self.temp_dir) / "artifacts")
        
        result = self.health_checker._check_experiment_creation()
        
        self.assertEqual(result["status"], "PASS")
        self.assertIn("Experiment creation successful", result["message"])
    
    @patch('mlflow.create_experiment')
    def test_experiment_creation_check_failure(self, mock_create):
        """Test experiment creation check with failure."""
        mock_create.side_effect = Exception("Database error")
        
        self.mock_central.initialize.return_value = None
        
        result = self.health_checker._check_experiment_creation()
        
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("Experiment creation failed", result["message"])
    
    def test_run_full_check(self):
        """Test running all health checks."""
        # Mock all check methods to return success
        for check_name in self.health_checker.checks:
            setattr(self.health_checker, f'_{check_name}', Mock(return_value={
                "status": "PASS",
                "message": f"{check_name} passed"
            }))
        
        results = self.health_checker.run_full_check()
        
        self.assertEqual(len(results), len(self.health_checker.checks))
        for result in results.values():
            self.assertEqual(result["status"], "PASS")


class TestMLflowTracker(unittest.TestCase):
    """Test cases for MLflowTracker class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create minimal training config
        self.config = TrainingConfig(
            learning_rate=0.001,
            epochs=1,
            batch_size=32,
            train_path="dummy_train.csv",
            monitoring=MonitoringConfig(enable_mlflow=True)
        )
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_initialization_with_mlflow_disabled(self):
        """Test tracker initialization with MLflow disabled."""
        self.config.monitoring.enable_mlflow = False
        
        tracker = MLflowTracker(self.config)
        
        self.assertIsNone(tracker.run_id)
        self.assertIsNone(tracker.experiment_id)
    
    @patch('mlflow.start_run')
    @patch('mlflow.log_param')
    def test_initialization_with_mlflow_enabled(self, mock_log_param, mock_start_run):
        """Test tracker initialization with MLflow enabled."""
        mock_run = Mock()
        mock_run.info.run_id = "test_run_id"
        mock_start_run.return_value = mock_run
        
        # Mock MLflow central
        with patch.object(MLflowTracker, '_initialize_mlflow') as mock_init:
            mock_init.return_value = None
            
            tracker = MLflowTracker(self.config)
            
            mock_init.assert_called_once()
    
    @patch('mlflow.log_metric')
    def test_log_metrics_success(self, mock_log_metric):
        """Test successful metric logging."""
        tracker = MLflowTracker(self.config)
        
        metrics = {
            "loss": 0.5,
            "accuracy": 0.95,
            "learning_rate": 0.001
        }
        
        tracker.log_metrics(metrics, step=1)
        
        self.assertEqual(mock_log_metric.call_count, 3)
    
    @patch('mlflow.log_metric')
    def test_log_metrics_with_invalid_values(self, mock_log_metric):
        """Test metric logging with invalid values."""
        tracker = MLflowTracker(self.config)
        
        metrics = {
            "loss": 0.5,
            "accuracy": True,  # Boolean value should be skipped
            "learning_rate": "invalid"  # String value should be skipped
        }
        
        tracker.log_metrics(metrics, step=1)
        
        # Only valid numeric metric should be logged
        self.assertEqual(mock_log_metric.call_count, 1)
        mock_log_metric.assert_called_with("loss", 0.5, step=1)
    
    @patch('mlflow.log_metric')
    def test_log_metrics_failure_disables_mlflow(self, mock_log_metric):
        """Test that MLflow is disabled after connection failures."""
        mock_log_metric.side_effect = Exception("Connection failed")
        
        tracker = MLflowTracker(self.config)
        
        # First call should fail and disable MLflow
        tracker.log_metrics({"loss": 0.5}, step=1)
        
        # MLflow should be disabled
        self.assertFalse(tracker.config.monitoring.enable_mlflow)
    
    @patch('mlflow.log_artifact')
    def test_log_artifacts_success(self, mock_log_artifact):
        """Test successful artifact logging."""
        tracker = MLflowTracker(self.config)
        
        # Create test artifact file
        test_file = Path(self.temp_dir) / "test_artifact.txt"
        test_file.write_text("test content")
        
        artifacts = {
            "model": str(test_file),
            "missing_file": "/nonexistent/path"
        }
        
        tracker.log_artifacts(artifacts)
        
        # Only existing file should be logged
        self.assertEqual(mock_log_artifact.call_count, 1)
        mock_log_artifact.assert_called_with(str(test_file), artifact_path="model")
    
    @patch('mlflow.end_run')
    def test_end_run_success(self, mock_end_run):
        """Test successful run ending."""
        tracker = MLflowTracker(self.config)
        tracker.run_id = "test_run_id"
        
        tracker.end_run("FINISHED")
        
        mock_end_run.assert_called_once_with(status="FINISHED")
    
    @patch('mlflow.end_run')
    def test_end_run_failure(self, mock_end_run):
        """Test run ending with failure."""
        mock_end_run.side_effect = Exception("End run failed")
        
        tracker = MLflowTracker(self.config)
        tracker.run_id = "test_run_id"
        
        # Should not raise exception
        tracker.end_run("FAILED")


class TestTrainingMetrics(unittest.TestCase):
    """Test cases for TrainingMetrics class."""
    
    def setUp(self):
        """Set up test environment."""
        self.metrics = TrainingMetrics()
    
    def test_metric_tracker_initialization(self):
        """Test that metric trackers are properly initialized."""
        trackers = self.metrics.get_all_trackers()
        
        # Check that all expected trackers exist
        expected_trackers = [
            "train_loss", "train_accuracy", "val_loss", "val_accuracy",
            "learning_rate", "step_time", "throughput", "memory_usage"
        ]
        
        for tracker_name in expected_trackers:
            self.assertIn(tracker_name, trackers)
            self.assertIsNotNone(trackers[tracker_name])
    
    def test_metric_update_and_best_tracking(self):
        """Test metric updating and best value tracking."""
        tracker = self.metrics.train_accuracy
        
        # Update with increasing values
        self.assertTrue(tracker.update(0.8, step=1))  # Should be new best
        self.assertTrue(tracker.update(0.9, step=2))  # Should be new best
        self.assertFalse(tracker.update(0.85, step=3))  # Should not be new best
        
        self.assertEqual(tracker.best_value, 0.9)
        self.assertEqual(tracker.best_step, 2)
    
    def test_metric_trend_calculation(self):
        """Test trend calculation for metrics."""
        tracker = self.metrics.train_loss
        
        # Add decreasing values (good trend for loss)
        tracker.update(1.0, step=1)
        tracker.update(0.8, step=2)
        tracker.update(0.6, step=3)
        
        trend = tracker.get_trend()
        self.assertEqual(trend, "↗")  # Should show improvement for loss
    
    def test_metric_recent_average(self):
        """Test recent average calculation."""
        tracker = self.metrics.train_accuracy
        
        # Add values
        for i, value in enumerate([0.1, 0.2, 0.3, 0.4, 0.5]):
            tracker.update(value, step=i)
        
        # Recent average of last 3 values should be (0.3 + 0.4 + 0.5) / 3 = 0.4
        recent_avg = tracker.get_recent_average(n=3)
        self.assertAlmostEqual(recent_avg, 0.4, places=5)


if __name__ == '__main__':
    unittest.main()