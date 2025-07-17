"""Unit tests for MLflow central configuration."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import mlflow
import pytest

from utils.mlflow_central import MLflowCentral, setup_central_mlflow


class TestMLflowCentral:
    """Test MLflow central configuration manager."""
    
    def test_singleton_pattern(self):
        """Test that MLflowCentral follows singleton pattern."""
        instance1 = MLflowCentral()
        instance2 = MLflowCentral()
        assert instance1 is instance2
    
    def test_initialize(self, temp_dir, monkeypatch):
        """Test MLflow initialization."""
        # Reset singleton
        MLflowCentral._instance = None
        
        tracking_uri = f"sqlite:///{temp_dir}/test.db"
        monkeypatch.setattr(MLflowCentral, "TRACKING_URI", tracking_uri)
        monkeypatch.setattr(MLflowCentral, "ARTIFACT_ROOT", str(temp_dir / "artifacts"))
        
        central = MLflowCentral()
        
        with patch("mlflow.set_tracking_uri") as mock_set_uri:
            central.initialize()
            mock_set_uri.assert_called_once_with(tracking_uri)
            assert central.initialized
    
    def test_create_experiment(self, mock_mlflow_central):
        """Test experiment creation."""
        with patch("mlflow.get_experiment_by_name", return_value=None), \
             patch("mlflow.create_experiment", return_value="exp_id_123") as mock_create:
            
            exp_id = mock_mlflow_central.create_experiment("test_experiment")
            
            assert exp_id == "exp_id_123"
            mock_create.assert_called_once()
    
    def test_get_existing_experiment(self, mock_mlflow_central):
        """Test getting existing experiment."""
        mock_exp = MagicMock()
        mock_exp.experiment_id = "existing_id"
        
        with patch("mlflow.get_experiment_by_name", return_value=mock_exp):
            exp_id = mock_mlflow_central.create_experiment("test_experiment")
            assert exp_id == "existing_id"
    
    def test_list_experiments(self, mock_mlflow_central):
        """Test listing experiments."""
        mock_exps = [MagicMock(name="exp1"), MagicMock(name="exp2")]
        
        with patch("mlflow.search_experiments", return_value=mock_exps):
            experiments = mock_mlflow_central.list_experiments()
            assert len(experiments) == 2
    
    def test_migrate_database(self, temp_dir, monkeypatch):
        """Test database migration."""
        # Create source and target directories
        source_dir = temp_dir / "source"
        target_dir = temp_dir / "target"
        source_dir.mkdir()
        
        # Create a dummy database file
        source_db = source_dir / "mlflow.db"
        source_db.write_text("dummy")
        
        # Reset singleton
        MLflowCentral._instance = None
        
        monkeypatch.setattr(MLflowCentral, "TRACKING_URI", f"sqlite:///{target_dir}/mlflow.db")
        
        central = MLflowCentral()
        
        with patch("shutil.copy2") as mock_copy:
            central.migrate_database(str(source_db))
            mock_copy.assert_called_once()
    
    def test_get_tracking_info(self, mock_mlflow_central):
        """Test getting tracking information."""
        info = mock_mlflow_central.get_tracking_info()
        
        assert "tracking_uri" in info
        assert "artifact_root" in info
        assert "experiments" in info
        assert isinstance(info["experiments"], list)


class TestSetupCentralMLflow:
    """Test the setup_central_mlflow helper function."""
    
    def test_setup_with_defaults(self, temp_dir, monkeypatch):
        """Test setup with default parameters."""
        # Reset singleton
        MLflowCentral._instance = None
        
        monkeypatch.setattr(MLflowCentral, "TRACKING_URI", f"sqlite:///{temp_dir}/test.db")
        
        with patch("mlflow.set_tracking_uri"), \
             patch("mlflow.set_experiment") as mock_set_exp:
            
            central = setup_central_mlflow()
            
            assert isinstance(central, MLflowCentral)
            mock_set_exp.assert_called_once_with("mlx_training")
    
    def test_setup_with_custom_experiment(self, temp_dir, monkeypatch):
        """Test setup with custom experiment name."""
        # Reset singleton
        MLflowCentral._instance = None
        
        monkeypatch.setattr(MLflowCentral, "TRACKING_URI", f"sqlite:///{temp_dir}/test.db")
        
        with patch("mlflow.set_tracking_uri"), \
             patch("mlflow.set_experiment") as mock_set_exp:
            
            central = setup_central_mlflow(experiment_name="custom_exp")
            
            mock_set_exp.assert_called_once_with("custom_exp")
    
    def test_setup_with_custom_uri(self, temp_dir):
        """Test setup with custom tracking URI."""
        # Reset singleton
        MLflowCentral._instance = None
        
        custom_uri = f"sqlite:///{temp_dir}/custom.db"
        
        with patch("mlflow.set_tracking_uri") as mock_set_uri, \
             patch("mlflow.set_experiment"):
            
            central = setup_central_mlflow(tracking_uri=custom_uri)
            
            mock_set_uri.assert_called_with(custom_uri)


@pytest.mark.integration
class TestMLflowCentralIntegration:
    """Integration tests for MLflow central configuration."""
    
    def test_full_workflow(self, temp_dir):
        """Test complete MLflow workflow."""
        # Reset singleton
        MLflowCentral._instance = None
        
        tracking_uri = f"sqlite:///{temp_dir}/integration.db"
        
        # Create central configuration
        central = MLflowCentral()
        central.TRACKING_URI = tracking_uri
        central.ARTIFACT_ROOT = str(temp_dir / "artifacts")
        central.initialize()
        
        # Create experiment
        exp_id = central.create_experiment("integration_test")
        assert exp_id is not None
        
        # Set experiment and start run
        mlflow.set_experiment("integration_test")
        
        with mlflow.start_run() as run:
            # Log some metrics
            mlflow.log_metric("test_metric", 0.95)
            mlflow.log_param("test_param", "value")
            
            # Create and log artifact
            artifact_path = temp_dir / "test_artifact.txt"
            artifact_path.write_text("test content")
            mlflow.log_artifact(str(artifact_path))
        
        # Verify experiment exists
        experiments = central.list_experiments()
        exp_names = [exp.name for exp in experiments]
        assert "integration_test" in exp_names
        
        # Get tracking info
        info = central.get_tracking_info()
        assert info["tracking_uri"] == tracking_uri
        assert len(info["experiments"]) > 0