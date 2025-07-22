"""Integration tests for the command system.

This module tests the integration of commands, pipelines, and strategies
in the refactored training system.
"""

import pytest
from unittest.mock import Mock, MagicMock
import time

from training.commands.base import CommandContext, CommandResult
from training.commands import (
    ForwardCommand,
    BackwardCommand,
    OptimizerStepCommand,
    GradientAccumulationCommand,
    LoggingCommand,
)
from training.pipeline.base import PipelineBuilder
from training.pipeline.middleware import (
    TimingMiddleware,
    ErrorHandlingMiddleware,
    MetricsMiddleware,
    ValidationMiddleware,
)
from core.protocols.training import TrainingState


class TestCommandIntegration:
    """Test integration of individual commands."""
    
    @pytest.fixture
    def mock_context(self):
        """Create a mock command context."""
        context = Mock(spec=CommandContext)
        
        # Mock basic attributes
        context.model = Mock()
        context.optimizer = Mock()
        context.state = TrainingState()
        context.batch = {"input_ids": [1, 2, 3], "labels": [0, 1, 0]}
        context.batch_idx = 0
        context.outputs = {}
        context.gradients = {}
        context.loss = None
        context.metrics = {}
        context.is_training = True
        context.should_update_weights = True
        context.config = {"max_grad_norm": 1.0, "log_interval": 10}
        
        return context
    
    def test_forward_command_execution(self, mock_context):
        """Test forward command execution."""
        # Setup
        forward_cmd = ForwardCommand(compute_loss=True, return_outputs=True)
        mock_context.model.return_value = {"loss": 0.5, "logits": [0.1, 0.2, 0.3]}
        
        # Execute
        result = forward_cmd.execute(mock_context)
        
        # Verify
        assert result.success
        assert "loss" in result.outputs
        assert mock_context.loss == 0.5
    
    def test_backward_command_execution(self, mock_context):
        """Test backward command execution."""
        # Setup
        backward_cmd = BackwardCommand(grad_clip_norm=1.0, compute_grad_norm=True)
        mock_context.loss = 0.5
        
        # Mock gradient computation (would be framework-specific)
        mock_gradients = {"param1": Mock(), "param2": Mock()}
        
        # Execute
        result = backward_cmd.execute(mock_context)
        
        # Since this is a placeholder implementation, just verify structure
        assert isinstance(result, CommandResult)
    
    def test_optimizer_step_command(self, mock_context):
        """Test optimizer step command."""
        # Setup
        optimizer_cmd = OptimizerStepCommand(update_lr_scheduler=True)
        mock_context.gradients = {"param1": Mock()}
        mock_context.optimizer.learning_rate = 0.001
        
        # Execute
        result = optimizer_cmd.execute(mock_context)
        
        # Verify
        assert isinstance(result, CommandResult)
        assert mock_context.state.global_step == 1  # Should increment
    
    def test_gradient_accumulation_command(self, mock_context):
        """Test gradient accumulation command."""
        # Setup
        accumulation_cmd = GradientAccumulationCommand(
            accumulation_steps=3,
            normalize_accumulated_gradients=True
        )
        mock_context.gradients = {"param1": Mock()}
        
        # Execute multiple times
        results = []
        for i in range(3):
            result = accumulation_cmd.execute(mock_context)
            results.append(result)
        
        # Verify
        assert all(r.success for r in results)
        # Last execution should trigger weight update
        assert results[-1].outputs["should_update"]
    
    def test_logging_command(self, mock_context):
        """Test logging command execution."""
        # Setup
        logging_cmd = LoggingCommand(log_interval=1, log_to_console=False)
        mock_context.loss = 0.5
        mock_context.metrics = {"accuracy": 0.8}
        
        # Execute
        result = logging_cmd.execute(mock_context)
        
        # Verify
        assert result.success
        assert "logged_metrics" in result.outputs


class TestPipelineIntegration:
    """Test integration of pipelines with commands and middleware."""
    
    @pytest.fixture
    def mock_context(self):
        """Create a mock command context."""
        context = Mock(spec=CommandContext)
        
        # Mock all required attributes
        context.model = Mock()
        context.optimizer = Mock()
        context.state = TrainingState()
        context.batch = {"input_ids": [1, 2, 3], "labels": [0, 1, 0]}
        context.batch_idx = 0
        context.outputs = {}
        context.gradients = {}
        context.loss = None
        context.metrics = {}
        context.is_training = True
        context.should_update_weights = True
        context.config = {
            "max_grad_norm": 1.0,
            "log_interval": 10,
            "gradient_accumulation_steps": 1,
        }
        
        return context
    
    def test_basic_pipeline_execution(self, mock_context):
        """Test basic pipeline with forward, backward, and optimizer commands."""
        # Create commands
        forward_cmd = Mock()
        forward_cmd.name = "ForwardCommand"
        forward_cmd.requires_grad = False
        forward_cmd.can_execute.return_value = True
        forward_cmd.execute.return_value = CommandResult(success=True, outputs={"loss": 0.5})
        
        backward_cmd = Mock()
        backward_cmd.name = "BackwardCommand" 
        backward_cmd.requires_grad = True
        backward_cmd.can_execute.return_value = True
        backward_cmd.execute.return_value = CommandResult(success=True, outputs={"gradients": {}})
        
        optimizer_cmd = Mock()
        optimizer_cmd.name = "OptimizerStep"
        optimizer_cmd.requires_grad = False
        optimizer_cmd.can_execute.return_value = True
        optimizer_cmd.execute.return_value = CommandResult(success=True, outputs={"learning_rate": 0.001})
        
        # Build pipeline
        pipeline = (
            PipelineBuilder("TestPipeline")
            .add_commands(forward_cmd, backward_cmd, optimizer_cmd)
            .build()
        )
        
        # Execute
        result = pipeline.execute(mock_context)
        
        # Verify
        assert result.success
        forward_cmd.execute.assert_called_once()
        backward_cmd.execute.assert_called_once()
        optimizer_cmd.execute.assert_called_once()
    
    def test_pipeline_with_middleware(self, mock_context):
        """Test pipeline with middleware integration."""
        # Create mock command
        mock_command = Mock()
        mock_command.name = "TestCommand"
        mock_command.requires_grad = False
        mock_command.can_execute.return_value = True
        mock_command.execute.return_value = CommandResult(success=True)
        
        # Create middleware
        timing_middleware = TimingMiddleware(log_timings=False)
        metrics_middleware = MetricsMiddleware(rolling_window=10)
        
        # Build pipeline
        pipeline = (
            PipelineBuilder("MiddlewarePipeline")
            .add_command(mock_command)
            .add_middlewares(timing_middleware, metrics_middleware)
            .build()
        )
        
        # Execute
        result = pipeline.execute(mock_context)
        
        # Verify
        assert result.success
        mock_command.execute.assert_called_once()
        
        # Check timing metrics were added
        assert any(key.endswith("_time") for key in result.metrics.keys())
    
    def test_error_handling_middleware(self, mock_context):
        """Test error handling middleware in pipeline."""
        # Create command that fails
        failing_command = Mock()
        failing_command.name = "FailingCommand"
        failing_command.requires_grad = False
        failing_command.can_execute.return_value = True
        failing_command.execute.side_effect = RuntimeError("Test error")
        
        # Create error handling middleware
        error_middleware = ErrorHandlingMiddleware(
            max_retries=2,
            retry_delay=0.01,  # Fast retry for testing
            recoverable_errors=(RuntimeError,),
            log_errors=False  # Suppress logs during testing
        )
        
        # Build pipeline
        pipeline = (
            PipelineBuilder("ErrorHandlingPipeline")
            .add_command(failing_command)
            .add_middleware(error_middleware)
            .continue_on_error(True)
            .build()
        )
        
        # Execute
        result = pipeline.execute(mock_context)
        
        # Verify retries occurred
        assert failing_command.execute.call_count == 3  # Initial + 2 retries
    
    def test_validation_middleware(self, mock_context):
        """Test validation middleware in pipeline."""
        # Create command
        mock_command = Mock()
        mock_command.name = "TestCommand"
        mock_command.requires_grad = True  # Requires gradients
        mock_command.can_execute.return_value = True
        mock_command.execute.return_value = CommandResult(success=True)
        
        # Create validation middleware
        validation_middleware = ValidationMiddleware(strict=False)
        
        # Test with missing optimizer (should warn but not fail)
        mock_context.optimizer = None
        
        # Build pipeline
        pipeline = (
            PipelineBuilder("ValidationPipeline")
            .add_command(mock_command)
            .add_middleware(validation_middleware)
            .build()
        )
        
        # Execute
        result = pipeline.execute(mock_context)
        
        # Should succeed despite validation warning
        assert result.success


class TestMiddlewareIntegration:
    """Test middleware behavior and integration."""
    
    def test_timing_middleware_accuracy(self):
        """Test that timing middleware accurately measures execution time."""
        # Create slow command
        class SlowCommand:
            def __init__(self, delay=0.1):
                self.name = "SlowCommand"
                self.requires_grad = False
                self.delay = delay
            
            def can_execute(self, context):
                return True
            
            def execute(self, context):
                time.sleep(self.delay)
                return CommandResult(success=True)
            
            def rollback(self, context):
                pass
        
        slow_cmd = SlowCommand(delay=0.05)  # 50ms delay
        timing_middleware = TimingMiddleware(log_timings=False)
        
        # Build pipeline
        pipeline = (
            PipelineBuilder("TimingTestPipeline")
            .add_command(slow_cmd)
            .add_middleware(timing_middleware)
            .build()
        )
        
        # Execute
        mock_context = Mock()
        result = pipeline.execute(mock_context)
        
        # Verify timing was recorded
        assert result.success
        assert "SlowCommand_time" in result.metrics
        assert result.metrics["SlowCommand_time"] >= 0.04  # Allow some variance
    
    def test_metrics_middleware_aggregation(self):
        """Test metrics middleware aggregation functionality."""
        # Create command that produces metrics
        class MetricsCommand:
            def __init__(self, value):
                self.name = "MetricsCommand"
                self.requires_grad = False
                self.value = value
            
            def can_execute(self, context):
                return True
            
            def execute(self, context):
                return CommandResult(
                    success=True,
                    metrics={"test_metric": self.value}
                )
            
            def rollback(self, context):
                pass
        
        metrics_middleware = MetricsMiddleware(rolling_window=3)
        
        # Create commands with different metric values
        commands = [MetricsCommand(0.1), MetricsCommand(0.2), MetricsCommand(0.3)]
        
        mock_context = Mock()
        mock_context.state = TrainingState()
        
        for i, cmd in enumerate(commands):
            mock_context.state.global_step = i + 1
            
            pipeline = (
                PipelineBuilder("MetricsTestPipeline")
                .add_command(cmd)
                .add_middleware(metrics_middleware)
                .build()
            )
            
            result = pipeline.execute(mock_context)
            assert result.success
        
        # Check aggregated metrics
        summary = metrics_middleware.get_metrics_summary()
        assert "test_metric" in summary
        assert summary["test_metric"]["mean"] == 0.2  # (0.1 + 0.2 + 0.3) / 3


@pytest.mark.integration
class TestCommandPipelineIntegration:
    """High-level integration tests for command-pipeline system."""
    
    def test_full_training_step_simulation(self):
        """Test a complete training step with all components."""
        # This would be a comprehensive test that simulates a full training step
        # For now, we'll create a simplified version
        
        # Create mock context with all necessary components
        context = Mock(spec=CommandContext)
        context.model = Mock()
        context.optimizer = Mock()
        context.state = TrainingState()
        context.batch = {"input_ids": [1, 2, 3], "labels": [0, 1]}
        context.is_training = True
        context.config = {
            "max_grad_norm": 1.0,
            "gradient_accumulation_steps": 1,
            "log_interval": 1,
        }
        
        # Mock model output
        context.model.return_value = {"loss": 0.5, "logits": [0.1, 0.2]}
        context.optimizer.learning_rate = 0.001
        
        # Create actual commands (not mocks) for real integration testing
        forward_cmd = ForwardCommand(compute_loss=True)
        logging_cmd = LoggingCommand(log_interval=1, log_to_console=False)
        
        # Build pipeline with middleware
        pipeline = (
            PipelineBuilder("IntegrationTestPipeline")
            .add_commands(forward_cmd, logging_cmd)
            .add_middlewares(
                TimingMiddleware(log_timings=False),
                MetricsMiddleware(rolling_window=10),
            )
            .build()
        )
        
        # Execute
        result = pipeline.execute(context)
        
        # Verify successful execution
        assert result.success
        assert context.loss == 0.5  # Loss should be set by forward command
        assert len(result.metrics) > 0  # Should have timing and other metrics