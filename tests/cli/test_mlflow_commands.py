"""Unit tests for MLflow CLI commands."""

import pytest
from unittest.mock import patch, MagicMock, call
from pathlib import Path
from typer.testing import CliRunner
from datetime import datetime

import sys
from pathlib import Path as SysPath

# Add project root to path
sys.path.insert(0, str(SysPath(__file__).parent.parent.parent))

from cli.app import app


class TestMLflowHealthCommand:
    """Test suite for the mlflow-health command."""
    
    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()
    
    @pytest.fixture
    def mock_health_results(self):
        """Create mock health check results."""
        return {
            'database_connectivity': {
                'status': 'PASS',
                'message': 'Successfully connected to MLflow backend store'
            },
            'artifact_store_access': {
                'status': 'PASS',
                'message': 'Artifact store is accessible at ./mlruns'
            },
            'tracking_server': {
                'status': 'PASS',
                'message': 'MLflow tracking URI is properly configured'
            },
            'dependencies': {
                'status': 'PASS',
                'message': 'All required dependencies are installed'
            },
            'permissions': {
                'status': 'PASS',
                'message': 'Read/write permissions verified'
            }
        }
    
    @patch('cli.commands.mlflow.health.MLflowHealthChecker')
    def test_health_check_all_pass(self, mock_health_class, runner, mock_health_results):
        """Test health check with all checks passing."""
        # Setup mock
        mock_checker = MagicMock()
        mock_checker.run_full_check.return_value = mock_health_results
        mock_health_class.return_value = mock_checker
        
        # Run command
        result = runner.invoke(app, ["mlflow-health"])
        
        # Check success
        assert result.exit_code == 0
        assert "MLflow Health Check" in result.stdout
        assert "PASS" in result.stdout
        assert "All checks passed" in result.stdout.lower() or "healthy" in result.stdout.lower()
        
        # Critical: Verify run_full_check takes NO parameters
        mock_checker.run_full_check.assert_called_once_with()
    
    @patch('cli.commands.mlflow.health.MLflowHealthChecker')
    def test_health_check_with_failures(self, mock_health_class, runner):
        """Test health check with some failures."""
        # Setup mock with failures
        mock_results = {
            'database_connectivity': {
                'status': 'FAIL',
                'message': 'Cannot connect to backend store',
                'suggestions': [
                    'Check if MLflow server is running',
                    'Verify database connection string'
                ]
            },
            'artifact_store_access': {
                'status': 'PASS',
                'message': 'Artifact store is accessible'
            },
            'permissions': {
                'status': 'FAIL',
                'message': 'Cannot write to artifact store',
                'suggestions': [
                    'Check directory permissions',
                    'Run with appropriate user privileges'
                ]
            }
        }
        
        mock_checker = MagicMock()
        mock_checker.run_full_check.return_value = mock_results
        mock_health_class.return_value = mock_checker
        
        # Run command
        result = runner.invoke(app, ["mlflow-health"])
        
        # Check that failures are reported
        assert result.exit_code != 0 or "FAIL" in result.stdout
        assert "FAIL" in result.stdout
        assert "suggestions" in result.stdout.lower() or "Check" in result.stdout
        
        # Verify no parameters passed
        mock_checker.run_full_check.assert_called_once_with()
    
    @patch('cli.commands.mlflow.health.MLflowHealthChecker')
    def test_health_check_output_formatting(self, mock_health_class, runner, mock_health_results):
        """Test health check output formatting."""
        mock_checker = MagicMock()
        mock_checker.run_full_check.return_value = mock_health_results
        mock_health_class.return_value = mock_checker
        
        result = runner.invoke(app, ["mlflow-health"])
        
        assert result.exit_code == 0
        # Check formatting elements
        assert "Status" in result.stdout or "PASS" in result.stdout
        assert "database" in result.stdout.lower() or "Database" in result.stdout
        assert "artifact" in result.stdout.lower() or "Artifact" in result.stdout
    
    @patch('cli.commands.mlflow.health.MLflowHealthChecker')
    def test_health_check_error_handling(self, mock_health_class, runner):
        """Test health check error handling."""
        mock_health_class.side_effect = Exception("Failed to initialize health checker")
        
        result = runner.invoke(app, ["mlflow-health"])
        
        assert result.exit_code != 0
        assert "Error" in result.stdout or "Failed" in result.stdout


class TestMLflowExperimentsCommand:
    """Test suite for the mlflow-experiments command."""
    
    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()
    
    @pytest.fixture
    def mock_experiments(self):
        """Create mock experiments data."""
        from collections import namedtuple
        Experiment = namedtuple('Experiment', ['experiment_id', 'name', 'artifact_location', 
                                               'lifecycle_stage', 'creation_time'])
        
        return [
            Experiment('1', 'Default', './mlruns/1', 'active', 1700000000000),
            Experiment('2', 'titanic_modernbert', './mlruns/2', 'active', 1700100000000),
            Experiment('3', 'archived_exp', './mlruns/3', 'deleted', 1700200000000)
        ]
    
    @patch('cli.commands.mlflow.experiments.mlflow')
    def test_experiments_list_active(self, mock_mlflow, runner, mock_experiments):
        """Test listing active experiments."""
        # Setup mock
        mock_mlflow.search_experiments.return_value = mock_experiments[:2]  # Only active
        
        # Run command
        result = runner.invoke(app, ["mlflow-experiments"])
        
        # Check success
        assert result.exit_code == 0
        assert "MLflow Experiments" in result.stdout
        assert "titanic_modernbert" in result.stdout
        assert "Default" in result.stdout
        assert "archived_exp" not in result.stdout
        
        # Verify search was called correctly
        mock_mlflow.search_experiments.assert_called_once()
        call_args = mock_mlflow.search_experiments.call_args
        # Should filter for active experiments by default
        assert call_args[1].get('filter_string') is None or 'active' in str(call_args)
    
    @patch('cli.commands.mlflow.experiments.mlflow')
    def test_experiments_list_all(self, mock_mlflow, runner, mock_experiments):
        """Test listing all experiments including archived."""
        mock_mlflow.search_experiments.return_value = mock_experiments
        
        result = runner.invoke(app, [
            "mlflow-experiments",
            "--all"
        ])
        
        assert result.exit_code == 0
        assert "archived_exp" in result.stdout
        assert "deleted" in result.stdout.lower() or "Deleted" in result.stdout
    
    @patch('cli.commands.mlflow.experiments.mlflow')
    def test_experiments_with_runs(self, mock_mlflow, runner, mock_experiments):
        """Test showing experiments with run counts."""
        mock_mlflow.search_experiments.return_value = mock_experiments[:2]
        
        # Mock run counts
        mock_mlflow.search_runs.side_effect = [
            MagicMock(shape=(5, 1)),  # 5 runs for experiment 1
            MagicMock(shape=(12, 1))  # 12 runs for experiment 2
        ]
        
        result = runner.invoke(app, [
            "mlflow-experiments",
            "--show-runs"
        ])
        
        assert result.exit_code == 0
        assert "5" in result.stdout or "runs" in result.stdout.lower()
        assert "12" in result.stdout
    
    @patch('cli.commands.mlflow.experiments.mlflow')
    def test_experiments_create(self, mock_mlflow, runner):
        """Test creating a new experiment."""
        mock_mlflow.create_experiment.return_value = "4"
        
        result = runner.invoke(app, [
            "mlflow-experiments",
            "--create", "new_experiment"
        ])
        
        assert result.exit_code == 0
        assert "Created" in result.stdout or "new_experiment" in result.stdout
        
        mock_mlflow.create_experiment.assert_called_once_with("new_experiment")
    
    @patch('cli.commands.mlflow.experiments.mlflow')
    def test_experiments_empty_list(self, mock_mlflow, runner):
        """Test handling empty experiment list."""
        mock_mlflow.search_experiments.return_value = []
        
        result = runner.invoke(app, ["mlflow-experiments"])
        
        assert result.exit_code == 0
        assert "No experiments found" in result.stdout


class TestMLflowRunsCommand:
    """Test suite for the mlflow-runs command."""
    
    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()
    
    @pytest.fixture
    def mock_runs_df(self):
        """Create mock runs DataFrame."""
        import pandas as pd
        return pd.DataFrame({
            'run_id': ['run1', 'run2', 'run3'],
            'experiment_id': ['2', '2', '2'],
            'status': ['FINISHED', 'RUNNING', 'FAILED'],
            'start_time': [
                datetime(2024, 1, 1, 10, 0, 0),
                datetime(2024, 1, 1, 11, 0, 0),
                datetime(2024, 1, 1, 12, 0, 0)
            ],
            'end_time': [
                datetime(2024, 1, 1, 10, 30, 0),
                None,
                datetime(2024, 1, 1, 12, 15, 0)
            ],
            'metrics.accuracy': [0.85, None, 0.75],
            'metrics.loss': [0.35, 0.45, 0.55],
            'params.learning_rate': ['2e-5', '3e-5', '2e-5'],
            'params.batch_size': ['32', '64', '32'],
            'tags.mlflow.runName': ['baseline', 'improved_lr', 'debug_run']
        })
    
    @patch('cli.commands.mlflow.runs.mlflow')
    def test_runs_list_by_experiment(self, mock_mlflow, runner, mock_runs_df):
        """Test listing runs for a specific experiment."""
        mock_mlflow.search_runs.return_value = mock_runs_df
        
        result = runner.invoke(app, [
            "mlflow-runs",
            "--experiment", "titanic_modernbert"
        ])
        
        assert result.exit_code == 0
        assert "MLflow Runs" in result.stdout
        assert "baseline" in result.stdout
        assert "improved_lr" in result.stdout
        assert "FINISHED" in result.stdout
        assert "RUNNING" in result.stdout
    
    @patch('cli.commands.mlflow.runs.mlflow')
    def test_runs_limit(self, mock_mlflow, runner, mock_runs_df):
        """Test limiting number of runs displayed."""
        mock_mlflow.search_runs.return_value = mock_runs_df.head(2)
        
        result = runner.invoke(app, [
            "mlflow-runs",
            "--limit", "2"
        ])
        
        assert result.exit_code == 0
        # Should only show 2 runs
        assert "baseline" in result.stdout
        assert "improved_lr" in result.stdout
        assert "debug_run" not in result.stdout
    
    @patch('cli.commands.mlflow.runs.mlflow')
    def test_runs_filter_by_status(self, mock_mlflow, runner, mock_runs_df):
        """Test filtering runs by status."""
        # Filter for only finished runs
        finished_runs = mock_runs_df[mock_runs_df['status'] == 'FINISHED']
        mock_mlflow.search_runs.return_value = finished_runs
        
        result = runner.invoke(app, [
            "mlflow-runs",
            "--status", "FINISHED"
        ])
        
        assert result.exit_code == 0
        assert "baseline" in result.stdout
        assert "RUNNING" not in result.stdout
        assert "FAILED" not in result.stdout
    
    @patch('cli.commands.mlflow.runs.mlflow')
    def test_runs_show_artifacts(self, mock_mlflow, runner, mock_runs_df):
        """Test showing run artifacts."""
        mock_mlflow.search_runs.return_value = mock_runs_df.head(1)
        
        # Mock artifact listing
        mock_client = MagicMock()
        mock_client.list_artifacts.return_value = [
            MagicMock(path='model', is_dir=True),
            MagicMock(path='metrics.json', is_dir=False),
            MagicMock(path='params.json', is_dir=False)
        ]
        mock_mlflow.tracking.MlflowClient.return_value = mock_client
        
        result = runner.invoke(app, [
            "mlflow-runs",
            "--show-artifacts",
            "--limit", "1"
        ])
        
        assert result.exit_code == 0
        assert "Artifacts" in result.stdout or "model" in result.stdout
    
    @patch('cli.commands.mlflow.runs.mlflow')
    def test_runs_delete(self, mock_mlflow, runner):
        """Test deleting a run."""
        result = runner.invoke(app, [
            "mlflow-runs",
            "--delete", "run1"
        ])
        
        assert result.exit_code == 0
        assert "Deleted" in result.stdout or "run1" in result.stdout
        
        mock_mlflow.delete_run.assert_called_once_with("run1")
    
    @patch('cli.commands.mlflow.runs.mlflow')
    def test_runs_empty_results(self, mock_mlflow, runner):
        """Test handling empty run results."""
        import pandas as pd
        mock_mlflow.search_runs.return_value = pd.DataFrame()
        
        result = runner.invoke(app, ["mlflow-runs"])
        
        assert result.exit_code == 0
        assert "No runs found" in result.stdout
    
    @patch('cli.commands.mlflow.runs.mlflow')
    def test_runs_error_handling(self, mock_mlflow, runner):
        """Test error handling in runs command."""
        mock_mlflow.search_runs.side_effect = Exception("Connection error")
        
        result = runner.invoke(app, ["mlflow-runs"])
        
        assert result.exit_code != 0
        assert "Error" in result.stdout or "Failed" in result.stdout


class TestMLflowUICommand:
    """Test suite for the mlflow-ui command."""
    
    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()
    
    @patch('cli.commands.mlflow.ui.subprocess.Popen')
    @patch('cli.commands.mlflow.ui.requests.get')
    @patch('cli.commands.mlflow.ui.time.sleep')
    def test_ui_launch_success(self, mock_sleep, mock_requests, mock_popen, runner):
        """Test successful MLflow UI launch."""
        # Mock process
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Process is running
        mock_popen.return_value = mock_process
        
        # Mock server response (becomes available after 2 attempts)
        mock_requests.side_effect = [
            Exception("Connection refused"),
            MagicMock(status_code=200)
        ]
        
        result = runner.invoke(app, ["mlflow-ui"])
        
        assert result.exit_code == 0
        assert "MLflow UI" in result.stdout
        assert "http://localhost:5000" in result.stdout
        
        # Verify launch command
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args[0][0]
        assert "mlflow" in call_args
        assert "ui" in call_args
    
    @patch('cli.commands.mlflow.ui.subprocess.Popen')
    def test_ui_custom_port(self, mock_popen, runner):
        """Test launching UI on custom port."""
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process
        
        result = runner.invoke(app, [
            "mlflow-ui",
            "--port", "8080"
        ])
        
        assert result.exit_code == 0
        assert "8080" in result.stdout
        
        # Verify port in command
        call_args = mock_popen.call_args[0][0]
        assert "--port" in call_args
        assert "8080" in call_args
    
    @patch('cli.commands.mlflow.ui.subprocess.Popen')
    def test_ui_custom_backend_uri(self, mock_popen, runner):
        """Test launching UI with custom backend URI."""
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process
        
        result = runner.invoke(app, [
            "mlflow-ui",
            "--backend-uri", "./custom_mlruns"
        ])
        
        assert result.exit_code == 0
        
        # Verify backend URI in command
        call_args = mock_popen.call_args[0][0]
        assert "--backend-store-uri" in call_args
        assert "./custom_mlruns" in call_args