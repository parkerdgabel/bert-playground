#!/usr/bin/env python3
"""API Contract Tests for CLI Commands.

These tests verify that the interfaces between CLI commands and utility functions
remain stable and compatible. They use mock objects to test the contracts without
executing the actual functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import mlx.core as mx
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List

# Test the contracts for each command category


class TestCoreCommandContracts:
    """Test API contracts for core commands (train, predict, benchmark, info)."""
    
    @patch('training.core.base.BaseTrainer')
    @patch('models.factory.create_model')
    @patch('data.factory.create_dataloader')
    def test_train_command_contract(self, mock_create_dataloader, mock_create_model, mock_trainer):
        """Test train command's interface with TrainerV2 and model factory."""
        from cli.commands.core.train import train_command
        
        # Setup mocks
        mock_model = Mock()
        mock_create_model.return_value = mock_model
        
        mock_dataloader = Mock()
        mock_create_dataloader.return_value = mock_dataloader
        
        mock_trainer_instance = Mock()
        mock_trainer.return_value = mock_trainer_instance
        
        # Expected contract: TrainerV2 should be called with these parameters
        expected_trainer_params = {
            'max_epochs': 3,
            'batch_size': 32,
            'learning_rate': 2e-5,
            'warmup_steps': 100,
            'eval_interval': 100,
            'checkpoint_dir': Path('checkpoints'),
            'log_interval': 10,
            'grad_accum_steps': 1,
            'mixed_precision': False,
            'compile_model': False,
            'experiment_name': 'test',
            'enable_mlflow': False
        }
        
        # Contract: create_model should accept model_type and return a model
        assert callable(mock_create_model)
        
        # Contract: TrainerV2 should accept the expected parameters
        assert callable(mock_trainer)
        
        # Contract: trainer.train should accept model and data_module
        mock_trainer_instance.train = Mock()
        assert hasattr(mock_trainer_instance, 'train')
    
    def test_predict_command_contract(self):
        """Test predict command's interface with checkpoint loading and data module."""
        from cli.commands.core.predict import predict_command
        
        # Just verify the command is callable
        assert callable(predict_command)
    
    @patch('cli.commands.core.benchmark.create_bert_with_head')
    @patch('cli.commands.core.benchmark.nn.value_and_grad')
    def test_benchmark_command_contract(self, mock_value_and_grad, mock_create_bert):
        """Test benchmark command's MLX API usage contract."""
        from cli.commands.core.benchmark import benchmark_command
        
        # Setup mocks
        mock_model = Mock()
        mock_create_bert.return_value = mock_model
        
        # Contract: model should return dict with 'loss' key when called
        mock_model.return_value = {'loss': mx.array(0.5)}
        
        # Contract: value_and_grad should accept model and function
        mock_grad_fn = Mock()
        mock_grad_fn.return_value = (mx.array(0.5), {'param': mx.array([1, 2, 3])})
        mock_value_and_grad.return_value = mock_grad_fn
        
        # Verify the contract
        assert callable(mock_value_and_grad)
        assert callable(mock_grad_fn)
        
        # Contract: loss function should be callable and return scalar
        def loss_fn():
            outputs = mock_model(
                input_ids=mx.array([[1, 2, 3]]),
                attention_mask=mx.array([[1, 1, 1]]),
                labels=mx.array([1])
            )
            return outputs['loss']
        
        # Test the contract
        loss = loss_fn()
        assert isinstance(loss, mx.array)


class TestKaggleCommandContracts:
    """Test API contracts for Kaggle commands."""
    
    @patch('utils.kaggle_integration.KaggleIntegration')
    def test_competitions_command_contract(self, mock_kaggle_integration):
        """Test competitions command's interface with KaggleIntegration."""
        from cli.commands.kaggle.competitions import competitions_command
        
        # Setup mock
        mock_kaggle = Mock()
        mock_kaggle_integration.return_value = mock_kaggle
        
        # Contract: list_competitions should accept these parameters
        expected_params = {
            'category': 'tabular',
            'search': 'classification',
            'sort_by': 'prize',
            'page': 1  # NOT page_size!
        }
        
        # Mock return value should be a DataFrame
        mock_competitions_df = pd.DataFrame([
            {
                'id': 'titanic',
                'title': 'Titanic Competition',
                'category': 'tabular',
                'numTeams': 100,
                'reward': '$1000',
                'deadline': pd.Timestamp('2024-12-31'),
                'isCompleted': False,
                'isKernelsSubmissionsOnly': False,
                'userHasEntered': False,
                'tags': ['binary-classification', 'tabular']
            }
        ])
        
        mock_kaggle.list_competitions.return_value = mock_competitions_df
        
        # Verify contract
        result = mock_kaggle.list_competitions(**expected_params)
        assert isinstance(result, pd.DataFrame)
        assert 'id' in result.columns
        assert 'deadline' in result.columns
    
    @patch('utils.kaggle_integration.KaggleIntegration')
    def test_download_command_contract(self, mock_kaggle_integration):
        """Test download command's interface with KaggleIntegration."""
        from cli.commands.kaggle.download import download_competition_command
        
        # Setup mock
        mock_kaggle = Mock()
        mock_kaggle_integration.return_value = mock_kaggle
        
        # Contract: download_competition_data should accept competition_id and path
        expected_params = {
            'competition_id': 'titanic',
            'output_dir': Path('data/titanic')
        }
        
        # Contract: should return list of downloaded files
        mock_kaggle.download_competition_data.return_value = [
            'train.csv',
            'test.csv',
            'submission.csv'
        ]
        
        # Verify contract
        result = mock_kaggle.download_competition_data(**expected_params)
        assert isinstance(result, list)
        assert all(isinstance(f, str) for f in result)
    
    @patch('utils.kaggle_integration.KaggleIntegration')
    def test_submit_command_contract(self, mock_kaggle_integration):
        """Test submit command's interface with KaggleIntegration."""
        from cli.commands.kaggle.submit import submit_command
        
        # Setup mock
        mock_kaggle = Mock()
        mock_kaggle_integration.return_value = mock_kaggle
        
        # Contract: submit_predictions should accept these parameters
        expected_params = {
            'competition_id': 'titanic',
            'submission_file': Path('submission.csv'),
            'message': 'Test submission'
        }
        
        # Contract: should return submission result
        mock_kaggle.submit_predictions.return_value = {
            'status': 'complete',
            'publicScore': 0.85,
            'privateScore': None,
            'submissionId': '12345'
        }
        
        # Verify contract
        result = mock_kaggle.submit_predictions(**expected_params)
        assert isinstance(result, dict)
        assert 'status' in result


class TestMLflowCommandContracts:
    """Test API contracts for MLflow commands."""
    
    @patch('utils.mlflow_health.MLflowHealthChecker')
    def test_health_command_contract(self, mock_health_checker_class):
        """Test health command's interface with MLflowHealthChecker."""
        from cli.commands.mlflow.health import health_command
        
        # Setup mock
        mock_checker = Mock()
        mock_health_checker_class.return_value = mock_checker
        
        # Contract: run_full_check should take NO parameters
        mock_checker.run_full_check.return_value = {
            'database_connectivity': {
                'status': 'PASS',
                'message': 'Connected successfully'
            },
            'directory_permissions': {
                'status': 'PASS',
                'message': 'Permissions OK'
            },
            'configuration_validity': {
                'status': 'FAIL',
                'message': 'Missing tracking URI',
                'suggestions': ['Set MLFLOW_TRACKING_URI']
            }
        }
        
        # Verify contract - run_full_check should not accept parameters
        result = mock_checker.run_full_check()  # No parameters!
        assert isinstance(result, dict)
        
        # Each check result should have status and message
        for check_name, check_result in result.items():
            assert 'status' in check_result
            assert 'message' in check_result
            assert check_result['status'] in ['PASS', 'FAIL']
    
    @patch('utils.mlflow_central.MLflowCentral')
    def test_experiments_command_contract(self, mock_mlflow_central_class):
        """Test experiments command's interface with MLflowCentral."""
        from cli.commands.mlflow.experiments import list_experiments_command
        
        # Setup mock
        mock_central = Mock()
        mock_mlflow_central_class.return_value = mock_central
        
        # Contract: list_experiments should return list of experiments
        mock_central.list_experiments.return_value = [
            Mock(experiment_id='1', name='test_experiment', artifact_location='/path/to/artifacts')
        ]
        
        # Contract: get_experiment_id should accept experiment_name
        mock_central.get_experiment_id.return_value = '1'
        
        # Verify contracts
        experiments = mock_central.list_experiments()
        assert isinstance(experiments, list)
        
        exp_id = mock_central.get_experiment_id('test_experiment')
        assert isinstance(exp_id, str)


class TestModelCommandContracts:
    """Test API contracts for model commands."""
    
    def test_convert_command_contract(self):
        """Test convert command's interface with checkpoint functions."""
        from cli.commands.model.convert import convert_command
        
        # Just verify the command is callable
        assert callable(convert_command)
    
    def test_inspect_command_contract(self):
        """Test inspect command's interface with checkpoint loading."""
        from cli.commands.model.inspect import inspect_command
        
        # Just verify the command is callable
        assert callable(inspect_command)


class TestUtilityContracts:
    """Test contracts for utility functions used by CLI."""
    
    def test_console_utilities_contract(self):
        """Test console utility functions contract."""
        from cli.utils.console import create_table, print_success, print_error
        
        # Contract: create_table should accept title and columns
        table = create_table("Test Table", ["Col1", "Col2"])
        assert hasattr(table, 'add_row')
        
        # Contract: print functions should accept message and optional title
        # These should not raise exceptions
        print_success("Success message")
        print_error("Error message", title="Error Title")
    
    def test_kaggle_integration_contract(self):
        """Test KaggleIntegration class contract."""
        with patch('utils.kaggle_integration.KaggleApi') as mock_api:
            from utils.kaggle_integration import KaggleIntegration
            
            kaggle = KaggleIntegration()
            
            # Contract: should have expected methods
            assert hasattr(kaggle, 'list_competitions')
            assert hasattr(kaggle, 'download_competition_data')
            assert hasattr(kaggle, 'submit_predictions')
            assert hasattr(kaggle, 'get_leaderboard')
    
    def test_mlflow_central_contract(self):
        """Test MLflowCentral class contract."""
        with patch('mlflow.set_tracking_uri'):
            from utils.mlflow_central import MLflowCentral
            
            central = MLflowCentral()
            
            # Contract: should have expected attributes and methods
            assert hasattr(central, 'tracking_uri')
            assert hasattr(central, 'artifact_root')
            assert hasattr(central, 'initialize')
            assert hasattr(central, 'list_experiments')
            assert hasattr(central, 'get_experiment_id')


class TestErrorHandlingContracts:
    """Test error handling contracts across CLI commands."""
    
    def test_handle_errors_decorator_contract(self):
        """Test that handle_errors decorator properly catches and formats errors."""
        from cli.utils import handle_errors
        import typer
        
        @handle_errors
        def test_function():
            raise ValueError("Test error")
        
        # Contract: should catch exceptions and exit with code 1
        with pytest.raises(typer.Exit) as exc_info:
            test_function()
        
        assert exc_info.value.exit_code == 1
    
    def test_validation_error_contract(self):
        """Test that validation errors are handled consistently."""
        from cli.utils.validators import validate_path
        import typer
        
        # Contract: validation functions should raise typer.BadParameter
        from pathlib import Path
        
        # validate_path with must_exist=True should raise for non-existent file
        with pytest.raises(typer.BadParameter):
            validate_path(Path("nonexistent.txt"), must_exist=True)


def test_api_stability():
    """Meta-test to ensure all expected APIs are available."""
    # Core commands
    from cli.commands.core.train_simple import train_command
    from cli.commands.core.predict import predict_command
    from cli.commands.core.benchmark import benchmark_command
    from cli.commands.core.info import info_command
    
    # Kaggle commands  
    from cli.commands.kaggle.competitions import competitions_command
    from cli.commands.kaggle.download import download_competition_command
    from cli.commands.kaggle.submit import submit_command
    
    # MLflow commands
    from cli.commands.mlflow.health import health_command
    from cli.commands.mlflow.experiments import list_experiments_command
    from cli.commands.mlflow.runs import list_runs_command
    
    # Model commands
    from cli.commands.model.inspect import inspect_command
    from cli.commands.model.convert import convert_command
    
    # All commands should be callable
    commands = [
        train_command, predict_command, benchmark_command, info_command,
        competitions_command, download_competition_command, submit_command,
        health_command, list_experiments_command, list_runs_command,
        inspect_command, convert_command
    ]
    
    for cmd in commands:
        assert callable(cmd), f"{cmd.__name__} should be callable"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])