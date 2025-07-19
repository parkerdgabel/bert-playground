"""Unit tests for the benchmark command."""

import pytest
from unittest.mock import patch, MagicMock, call
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from typer.testing import CliRunner

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cli.commands.core.benchmark import benchmark_command
from cli.app import app


class TestBenchmarkCommand:
    """Test suite for the benchmark command."""
    
    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model that behaves like an MLX model."""
        model = MagicMock()
        model.parameters = MagicMock(return_value={
            "embedding.weight": mx.random.normal((30522, 768)),
            "encoder.layer.0.attention.query.weight": mx.random.normal((768, 768)),
        })
        # Make model callable
        model.__call__ = MagicMock(return_value={"loss": mx.array(2.5)})
        return model
    
    @patch('cli.commands.core.benchmark.ModernBertForSequenceClassificationOptimized')
    @patch('cli.commands.core.benchmark.mx.eval')
    def test_benchmark_basic(self, mock_eval, mock_model_class, runner, mock_model):
        """Test basic benchmark functionality."""
        # Setup mocks
        mock_model_class.return_value = mock_model
        mock_eval.return_value = None
        
        # Run command
        result = runner.invoke(app, [
            "benchmark",
            "--batch-size", "16",
            "--seq-length", "128",
            "--steps", "5",
            "--warm-up", "2"
        ])
        
        # Check success
        assert result.exit_code == 0
        assert "Benchmark Results" in result.stdout
        assert "Steps: 5" in result.stdout
        
        # Verify model was created with correct config
        mock_model_class.assert_called_once()
        config = mock_model_class.call_args[0][0]
        assert config.num_labels == 2
        assert config.batch_size == 16
        assert config.max_length == 128
    
    @patch('cli.commands.core.benchmark.ModernBertForSequenceClassificationOptimized')
    def test_benchmark_memory_tracking(self, mock_model_class, runner, mock_model):
        """Test benchmark with memory tracking enabled."""
        mock_model_class.return_value = mock_model
        
        result = runner.invoke(app, [
            "benchmark",
            "--batch-size", "32",
            "--track-memory"
        ])
        
        assert result.exit_code == 0
        assert "Memory Usage" in result.stdout
    
    @patch('cli.commands.core.benchmark.ModernBertForSequenceClassificationOptimized')
    def test_benchmark_gradient_computation(self, mock_model_class, runner, mock_model):
        """Test that gradients are computed correctly."""
        mock_model_class.return_value = mock_model
        
        # Track value_and_grad calls
        with patch('cli.commands.core.benchmark.nn.value_and_grad') as mock_value_grad:
            # Setup the value_and_grad mock
            mock_grad_fn = MagicMock()
            mock_grad_fn.return_value = (mx.array(2.5), {"embedding.weight": mx.random.normal((30522, 768))})
            mock_value_grad.return_value = mock_grad_fn
            
            result = runner.invoke(app, [
                "benchmark",
                "--steps", "3"
            ])
            
            assert result.exit_code == 0
            
            # Check value_and_grad was called correctly
            mock_value_grad.assert_called_once()
            assert mock_value_grad.call_args[0][0] == mock_model
            
            # Check that the loss function was created properly
            loss_fn = mock_value_grad.call_args[0][1]
            assert callable(loss_fn)
    
    def test_benchmark_invalid_batch_size(self, runner):
        """Test benchmark with invalid batch size."""
        result = runner.invoke(app, [
            "benchmark",
            "--batch-size", "0"
        ])
        
        assert result.exit_code != 0
        assert "Invalid value" in result.stdout or "Error" in result.stdout
    
    def test_benchmark_invalid_seq_length(self, runner):
        """Test benchmark with invalid sequence length."""
        result = runner.invoke(app, [
            "benchmark",
            "--seq-length", "-1"
        ])
        
        assert result.exit_code != 0
        assert "Invalid value" in result.stdout or "Error" in result.stdout
    
    @patch('cli.commands.core.benchmark.ModernBertForSequenceClassificationOptimized')
    def test_benchmark_model_types(self, mock_model_class, runner, mock_model):
        """Test different model types."""
        mock_model_class.return_value = mock_model
        
        # Test multiclass
        result = runner.invoke(app, [
            "benchmark",
            "--model-type", "multiclass",
            "--num-labels", "5"
        ])
        
        assert result.exit_code == 0
        config = mock_model_class.call_args[0][0]
        assert config.num_labels == 5
        assert config.problem_type == "multiclass"
        
        # Test regression
        result = runner.invoke(app, [
            "benchmark",
            "--model-type", "regression"
        ])
        
        assert result.exit_code == 0
        config = mock_model_class.call_args[0][0]
        assert config.num_labels == 1
        assert config.problem_type == "regression"
    
    @patch('cli.commands.core.benchmark.ModernBertForSequenceClassificationOptimized')
    def test_benchmark_output_formatting(self, mock_model_class, runner, mock_model):
        """Test output formatting."""
        mock_model_class.return_value = mock_model
        
        result = runner.invoke(app, [
            "benchmark",
            "--batch-size", "64",
            "--seq-length", "256",
            "--steps", "10"
        ])
        
        assert result.exit_code == 0
        
        # Check key outputs are present
        assert "Configuration" in result.stdout
        assert "Batch Size: 64" in result.stdout
        assert "Sequence Length: 256" in result.stdout
        assert "Model Parameters:" in result.stdout
        assert "Performance Metrics" in result.stdout
        assert "Average time per step" in result.stdout
        assert "Throughput" in result.stdout
    
    @patch('cli.commands.core.benchmark.ModernBertForSequenceClassificationOptimized')
    @patch('cli.commands.core.benchmark.time.time')
    def test_benchmark_timing_calculation(self, mock_time, mock_model_class, runner, mock_model):
        """Test timing calculations."""
        mock_model_class.return_value = mock_model
        
        # Simulate timing
        times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]  # 1 second per step
        mock_time.side_effect = times
        
        result = runner.invoke(app, [
            "benchmark",
            "--steps", "5",
            "--warm-up", "0"
        ])
        
        assert result.exit_code == 0
        assert "1.00 seconds" in result.stdout  # Average time per step
    
    @patch('cli.commands.core.benchmark.ModernBertForSequenceClassificationOptimized')
    def test_benchmark_error_handling(self, mock_model_class, runner):
        """Test error handling in benchmark."""
        # Simulate model creation failure
        mock_model_class.side_effect = Exception("Model creation failed")
        
        result = runner.invoke(app, [
            "benchmark"
        ])
        
        assert result.exit_code != 0
        assert "Error" in result.stdout or "failed" in result.stdout.lower()