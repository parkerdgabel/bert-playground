"""MLflow health checking and diagnostic tools.

This module provides comprehensive health checking for MLflow integration,
including database connectivity, configuration validation, and performance testing.
"""

import os
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import mlx.core as mx
from loguru import logger

from utils.mlflow_central import MLflowCentral


class MLflowHealthChecker:
    """Comprehensive MLflow health checker."""
    
    def __init__(self):
        """Initialize health checker."""
        self.mlflow_central = MLflowCentral()
        self.checks = {
            "database_connectivity": self._check_database_connectivity,
            "directory_permissions": self._check_directory_permissions,
            "configuration_validity": self._check_configuration_validity,
            "experiment_creation": self._check_experiment_creation,
            "metric_logging": self._check_metric_logging,
            "artifact_logging": self._check_artifact_logging,
            "run_management": self._check_run_management,
            "performance": self._check_performance,
            "cleanup": self._check_cleanup_capabilities,
        }
    
    def run_full_check(self) -> Dict[str, Dict[str, Any]]:
        """Run all health checks and return results."""
        results = {}
        
        for check_name, check_func in self.checks.items():
            try:
                result = check_func()
                results[check_name] = result
            except Exception as e:
                results[check_name] = {
                    "status": "FAIL",
                    "message": f"Check failed with error: {str(e)}",
                    "suggestions": ["Check logs for detailed error information"]
                }
        
        return results
    
    def _check_database_connectivity(self) -> Dict[str, Any]:
        """Check database connectivity and basic operations."""
        try:
            # Initialize MLflow central
            self.mlflow_central.initialize()
            
            # Test database connection
            tracking_uri = self.mlflow_central.tracking_uri
            
            if tracking_uri.startswith("sqlite:"):
                db_path = tracking_uri.replace("sqlite:///", "")
                
                # Check if database exists
                if not Path(db_path).exists():
                    return {
                        "status": "FAIL",
                        "message": f"Database file does not exist: {db_path}",
                        "suggestions": [
                            "Run a training command to create the database",
                            "Check if the directory has write permissions"
                        ]
                    }
                
                # Test SQLite connection
                try:
                    with sqlite3.connect(db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                        tables = cursor.fetchall()
                        
                        if not tables:
                            return {
                                "status": "FAIL",
                                "message": "Database exists but has no tables",
                                "suggestions": ["Run MLflow server or create an experiment"]
                            }
                        
                except sqlite3.Error as e:
                    return {
                        "status": "FAIL",
                        "message": f"SQLite error: {str(e)}",
                        "suggestions": [
                            "Check database file permissions",
                            "Verify database is not corrupted"
                        ]
                    }
            
            # Test MLflow connection
            mlflow.set_tracking_uri(tracking_uri)
            experiments = mlflow.search_experiments()
            
            return {
                "status": "PASS",
                "message": f"Database connected successfully. Found {len(experiments)} experiments."
            }
            
        except Exception as e:
            return {
                "status": "FAIL",
                "message": f"Database connectivity failed: {str(e)}",
                "suggestions": [
                    "Check tracking URI configuration",
                    "Verify database permissions",
                    "Check network connectivity if using remote MLflow"
                ]
            }
    
    def _check_directory_permissions(self) -> Dict[str, Any]:
        """Check directory permissions for MLflow operations."""
        try:
            # Check mlruns directory
            mlruns_dir = Path("mlruns")
            
            # Test directory creation
            if not mlruns_dir.exists():
                try:
                    mlruns_dir.mkdir(parents=True, exist_ok=True)
                except PermissionError:
                    return {
                        "status": "FAIL",
                        "message": "Cannot create mlruns directory due to permission error",
                        "suggestions": [
                            "Run with appropriate permissions",
                            "Check parent directory permissions"
                        ]
                    }
            
            # Test write permissions
            test_file = mlruns_dir / "test_write.tmp"
            try:
                test_file.write_text("test")
                test_file.unlink()
            except PermissionError:
                return {
                    "status": "FAIL",
                    "message": "Cannot write to mlruns directory",
                    "suggestions": [
                        "Check directory permissions",
                        "Run with appropriate user privileges"
                    ]
                }
            
            # Check artifact directory
            artifact_dir = Path(self.mlflow_central.artifact_root)
            if not artifact_dir.exists():
                try:
                    artifact_dir.mkdir(parents=True, exist_ok=True)
                except PermissionError:
                    return {
                        "status": "FAIL",
                        "message": "Cannot create artifact directory",
                        "suggestions": [
                            "Check artifact root permissions",
                            "Configure artifact root to writable location"
                        ]
                    }
            
            return {
                "status": "PASS",
                "message": "All directory permissions are correct"
            }
            
        except Exception as e:
            return {
                "status": "FAIL",
                "message": f"Directory permission check failed: {str(e)}",
                "suggestions": [
                    "Check file system permissions",
                    "Verify disk space availability"
                ]
            }
    
    def _check_configuration_validity(self) -> Dict[str, Any]:
        """Check MLflow configuration validity."""
        try:
            # Check tracking URI format
            tracking_uri = self.mlflow_central.tracking_uri
            
            if not tracking_uri:
                return {
                    "status": "FAIL",
                    "message": "Tracking URI is not configured",
                    "suggestions": [
                        "Set MLFLOW_TRACKING_URI environment variable",
                        "Initialize MLflowCentral with tracking URI"
                    ]
                }
            
            # Check if URI is valid
            if tracking_uri.startswith("sqlite:"):
                db_path = tracking_uri.replace("sqlite:///", "")
                if not Path(db_path).parent.exists():
                    return {
                        "status": "FAIL",
                        "message": f"Database directory does not exist: {Path(db_path).parent}",
                        "suggestions": [
                            "Create database directory",
                            "Use valid database path"
                        ]
                    }
            
            # Check artifact root
            artifact_root = self.mlflow_central.artifact_root
            if not artifact_root:
                return {
                    "status": "FAIL",
                    "message": "Artifact root is not configured",
                    "suggestions": [
                        "Set MLFLOW_ARTIFACT_ROOT environment variable",
                        "Configure artifact root in MLflowCentral"
                    ]
                }
            
            # Check environment variables
            env_vars = {
                "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI"),
                "MLFLOW_ARTIFACT_ROOT": os.getenv("MLFLOW_ARTIFACT_ROOT"),
            }
            
            configured_vars = {k: v for k, v in env_vars.items() if v is not None}
            
            return {
                "status": "PASS",
                "message": f"Configuration is valid. Environment variables: {configured_vars}"
            }
            
        except Exception as e:
            return {
                "status": "FAIL",
                "message": f"Configuration validation failed: {str(e)}",
                "suggestions": [
                    "Check MLflow configuration",
                    "Verify environment variables"
                ]
            }
    
    def _check_experiment_creation(self) -> Dict[str, Any]:
        """Check experiment creation capabilities."""
        try:
            test_experiment_name = f"health_check_test_{int(time.time())}"
            
            # Initialize MLflow
            self.mlflow_central.initialize()
            
            # Create test experiment
            experiment_id = mlflow.create_experiment(
                test_experiment_name,
                artifact_location=self.mlflow_central.artifact_root
            )
            
            # Verify experiment was created
            experiment = mlflow.get_experiment(experiment_id)
            if not experiment:
                return {
                    "status": "FAIL",
                    "message": "Experiment creation failed - experiment not found",
                    "suggestions": [
                        "Check database write permissions",
                        "Verify MLflow server is running"
                    ]
                }
            
            # Clean up test experiment
            mlflow.delete_experiment(experiment_id)
            
            return {
                "status": "PASS",
                "message": f"Experiment creation successful (ID: {experiment_id})"
            }
            
        except Exception as e:
            return {
                "status": "FAIL",
                "message": f"Experiment creation failed: {str(e)}",
                "suggestions": [
                    "Check database permissions",
                    "Verify MLflow configuration",
                    "Check for existing experiment with same name"
                ]
            }
    
    def _check_metric_logging(self) -> Dict[str, Any]:
        """Check metric logging capabilities."""
        try:
            test_experiment_name = f"health_check_metrics_{int(time.time())}"
            
            # Initialize MLflow
            self.mlflow_central.initialize()
            
            # Create test experiment
            experiment_id = mlflow.create_experiment(test_experiment_name)
            
            # Start test run
            with mlflow.start_run(experiment_id=experiment_id):
                # Test parameter logging
                mlflow.log_param("test_param", "test_value")
                
                # Test metric logging
                mlflow.log_metric("test_metric", 0.5)
                mlflow.log_metric("test_metric", 0.7, step=1)
                
                # Test batch metric logging
                metrics = {"accuracy": 0.95, "loss": 0.1}
                for name, value in metrics.items():
                    mlflow.log_metric(name, value)
                
                run_info = mlflow.active_run().info
                run_id = run_info.run_id
            
            # Verify logged data
            run_data = mlflow.get_run(run_id)
            if not run_data.data.params.get("test_param"):
                return {
                    "status": "FAIL",
                    "message": "Parameter logging failed",
                    "suggestions": [
                        "Check database write permissions",
                        "Verify MLflow run is active"
                    ]
                }
            
            if not run_data.data.metrics.get("test_metric"):
                return {
                    "status": "FAIL",
                    "message": "Metric logging failed",
                    "suggestions": [
                        "Check database write permissions",
                        "Verify metric values are numeric"
                    ]
                }
            
            # Clean up
            mlflow.delete_experiment(experiment_id)
            
            return {
                "status": "PASS",
                "message": f"Metric logging successful (Run ID: {run_id})"
            }
            
        except Exception as e:
            return {
                "status": "FAIL",
                "message": f"Metric logging failed: {str(e)}",
                "suggestions": [
                    "Check MLflow run context",
                    "Verify database permissions",
                    "Check for network connectivity issues"
                ]
            }
    
    def _check_artifact_logging(self) -> Dict[str, Any]:
        """Check artifact logging capabilities."""
        try:
            test_experiment_name = f"health_check_artifacts_{int(time.time())}"
            
            # Initialize MLflow
            self.mlflow_central.initialize()
            
            # Create test experiment
            experiment_id = mlflow.create_experiment(test_experiment_name)
            
            # Start test run
            with mlflow.start_run(experiment_id=experiment_id):
                # Create test artifact
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                    f.write("Test artifact content")
                    test_file_path = f.name
                
                # Log artifact
                mlflow.log_artifact(test_file_path, "test_artifacts")
                
                # Test text logging
                mlflow.log_text("Test text content", "test_text.txt")
                
                run_id = mlflow.active_run().info.run_id
            
            # Verify artifacts using the client API
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            artifacts = client.list_artifacts(run_id)
            if not artifacts:
                return {
                    "status": "FAIL",
                    "message": "No artifacts found after logging",
                    "suggestions": [
                        "Check artifact storage permissions",
                        "Verify artifact root configuration"
                    ]
                }
            
            # Clean up
            Path(test_file_path).unlink(missing_ok=True)
            mlflow.delete_experiment(experiment_id)
            
            return {
                "status": "PASS",
                "message": f"Artifact logging successful ({len(artifacts)} artifacts)"
            }
            
        except Exception as e:
            return {
                "status": "FAIL",
                "message": f"Artifact logging failed: {str(e)}",
                "suggestions": [
                    "Check artifact storage permissions",
                    "Verify artifact root configuration",
                    "Check disk space availability"
                ]
            }
    
    def _check_run_management(self) -> Dict[str, Any]:
        """Check run management capabilities."""
        try:
            test_experiment_name = f"health_check_runs_{int(time.time())}"
            
            # Initialize MLflow
            self.mlflow_central.initialize()
            
            # Create test experiment
            experiment_id = mlflow.create_experiment(test_experiment_name)
            
            # Test run creation and management
            run_ids = []
            for i in range(3):
                with mlflow.start_run(experiment_id=experiment_id):
                    mlflow.log_param("run_number", i)
                    mlflow.log_metric("test_metric", i * 0.1)
                    run_ids.append(mlflow.active_run().info.run_id)
            
            # Test run search
            runs = mlflow.search_runs(experiment_ids=[experiment_id])
            if len(runs) != 3:
                return {
                    "status": "FAIL",
                    "message": f"Expected 3 runs, found {len(runs)}",
                    "suggestions": [
                        "Check database write operations",
                        "Verify run completion"
                    ]
                }
            
            # Test run retrieval
            for run_id in run_ids:
                run = mlflow.get_run(run_id)
                if not run:
                    return {
                        "status": "FAIL",
                        "message": f"Cannot retrieve run {run_id}",
                        "suggestions": [
                            "Check database consistency",
                            "Verify run storage"
                        ]
                    }
            
            # Clean up
            mlflow.delete_experiment(experiment_id)
            
            return {
                "status": "PASS",
                "message": f"Run management successful ({len(run_ids)} runs created and managed)"
            }
            
        except Exception as e:
            return {
                "status": "FAIL",
                "message": f"Run management failed: {str(e)}",
                "suggestions": [
                    "Check database operations",
                    "Verify MLflow run lifecycle",
                    "Check for concurrent access issues"
                ]
            }
    
    def _check_performance(self) -> Dict[str, Any]:
        """Check MLflow performance with typical workload."""
        try:
            test_experiment_name = f"health_check_performance_{int(time.time())}"
            
            # Initialize MLflow
            self.mlflow_central.initialize()
            
            # Create test experiment
            experiment_id = mlflow.create_experiment(test_experiment_name)
            
            # Performance test
            start_time = time.time()
            
            with mlflow.start_run(experiment_id=experiment_id):
                # Log multiple parameters
                for i in range(20):
                    mlflow.log_param(f"param_{i}", f"value_{i}")
                
                # Log multiple metrics with steps
                for step in range(100):
                    mlflow.log_metric("training_loss", 1.0 - step * 0.01, step=step)
                    mlflow.log_metric("validation_accuracy", step * 0.01, step=step)
                
                run_id = mlflow.active_run().info.run_id
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Clean up
            mlflow.delete_experiment(experiment_id)
            
            # Evaluate performance
            if duration > 30:  # More than 30 seconds is concerning
                return {
                    "status": "FAIL",
                    "message": f"Performance test took {duration:.2f}s (too slow)",
                    "suggestions": [
                        "Check database performance",
                        "Consider using faster storage",
                        "Check for network latency if using remote MLflow"
                    ]
                }
            
            return {
                "status": "PASS",
                "message": f"Performance test completed in {duration:.2f}s"
            }
            
        except Exception as e:
            return {
                "status": "FAIL",
                "message": f"Performance test failed: {str(e)}",
                "suggestions": [
                    "Check MLflow configuration",
                    "Verify database performance",
                    "Check system resources"
                ]
            }
    
    def _check_cleanup_capabilities(self) -> Dict[str, Any]:
        """Check cleanup and maintenance capabilities."""
        try:
            # Check database size
            tracking_uri = self.mlflow_central.tracking_uri
            
            if tracking_uri.startswith("sqlite:"):
                db_path = tracking_uri.replace("sqlite:///", "")
                if Path(db_path).exists():
                    db_size_mb = Path(db_path).stat().st_size / (1024 * 1024)
                    
                    if db_size_mb > 100:  # More than 100MB
                        return {
                            "status": "FAIL",
                            "message": f"Database size is {db_size_mb:.1f}MB (consider cleanup)",
                            "suggestions": [
                                "Run experiment cleanup",
                                "Archive old experiments",
                                "Consider database maintenance"
                            ]
                        }
            
            # Check artifact directory size
            artifact_dir = Path(self.mlflow_central.artifact_root)
            if artifact_dir.exists():
                total_size = sum(f.stat().st_size for f in artifact_dir.rglob('*') if f.is_file())
                total_size_mb = total_size / (1024 * 1024)
                
                if total_size_mb > 500:  # More than 500MB
                    return {
                        "status": "FAIL",
                        "message": f"Artifact directory size is {total_size_mb:.1f}MB (consider cleanup)",
                        "suggestions": [
                            "Clean up old artifacts",
                            "Archive completed experiments",
                            "Use external artifact storage"
                        ]
                    }
            
            # Test cleanup function
            test_experiment_name = f"health_check_cleanup_{int(time.time())}"
            
            # Initialize MLflow
            self.mlflow_central.initialize()
            
            # Create and delete test experiment
            experiment_id = mlflow.create_experiment(test_experiment_name)
            mlflow.delete_experiment(experiment_id)
            
            # Verify deletion
            try:
                deleted_exp = mlflow.get_experiment(experiment_id)
                if deleted_exp.lifecycle_stage != "deleted":
                    return {
                        "status": "FAIL",
                        "message": "Experiment deletion failed",
                        "suggestions": [
                            "Check database permissions",
                            "Verify MLflow version supports deletion"
                        ]
                    }
            except Exception:
                # This is expected if experiment is truly deleted
                pass
            
            return {
                "status": "PASS",
                "message": "Cleanup capabilities verified"
            }
            
        except Exception as e:
            return {
                "status": "FAIL",
                "message": f"Cleanup check failed: {str(e)}",
                "suggestions": [
                    "Check file system permissions",
                    "Verify database maintenance capabilities"
                ]
            }


def run_health_check() -> None:
    """Convenience function to run health check from command line."""
    checker = MLflowHealthChecker()
    results = checker.run_full_check()
    
    print("\n" + "="*60)
    print("MLflow Health Check Results")
    print("="*60)
    
    for check_name, result in results.items():
        status = "✓ PASS" if result["status"] == "PASS" else "✗ FAIL"
        print(f"{status} {check_name}: {result['message']}")
        
        if result["status"] == "FAIL" and result.get("suggestions"):
            print("  Suggestions:")
            for suggestion in result["suggestions"]:
                print(f"    • {suggestion}")
    
    # Summary
    passed = sum(1 for r in results.values() if r["status"] == "PASS")
    total = len(results)
    
    print(f"\nSummary: {passed}/{total} checks passed")
    
    if passed != total:
        print("\nPlease address the failed checks above before running training.")


if __name__ == "__main__":
    run_health_check()