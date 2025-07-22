"""Integration tests for the training strategy system.

This module tests the integration of training strategies, pipelines, and adapters.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

from training.strategies import (
    StandardTraining,
    GradientAccumulationTraining,
    MixedPrecisionTraining,
    MLXOptimizedTraining,
    get_strategy,
    register_strategy,
    list_strategies,
)
from training.commands.base import CommandContext
from training.adapters import get_framework_adapter
from core.protocols.training import TrainingState


class TestStrategyRegistration:
    """Test strategy registration and retrieval system."""
    
    def test_strategy_registration(self):
        """Test registering and retrieving strategies."""
        # Create custom strategy
        class CustomStrategy:
            def __init__(self):
                self._name = "CustomTest"
                self._description = "Test strategy"
                self.config = {}
            
            @property
            def name(self):
                return self._name
            
            @property
            def description(self):
                return self._description
            
            def create_pipeline(self, context):
                from training.pipeline.base import PipelineBuilder
                return PipelineBuilder("TestPipeline").build()
            
            def configure_context(self, context):
                return context
            
            def validate_requirements(self, context):
                return []
            
            def get_default_config(self):
                return {}
        
        custom_strategy = CustomStrategy()
        
        # Register strategy
        register_strategy(custom_strategy)
        
        # Retrieve strategy
        retrieved = get_strategy("CustomTest")
        assert retrieved.name == "CustomTest"
        assert retrieved.description == "Test strategy"
        
        # Check it appears in list
        strategies = list_strategies()
        assert "CustomTest" in strategies
    
    def test_default_strategies_available(self):
        """Test that default strategies are registered and available."""
        strategies = list_strategies()
        
        # Check that standard strategies are available
        assert "StandardTraining" in strategies
        assert "GradientAccumulationTraining" in strategies
        assert "MixedPrecisionTraining" in strategies
        assert "MLXOptimizedTraining" in strategies
    
    def test_strategy_retrieval_error(self):
        """Test error handling for missing strategies."""
        with pytest.raises(KeyError, match="Strategy 'NonExistent' not found"):
            get_strategy("NonExistent")


class TestStandardTraining:
    """Test the standard training strategy."""
    
    @pytest.fixture
    def mock_context(self):
        """Create mock command context."""
        context = Mock(spec=CommandContext)
        context.model = Mock()
        context.optimizer = Mock()
        context.state = TrainingState()
        context.train_dataloader = Mock()
        context.config = {}
        context.is_training = True
        return context
    
    def test_standard_training_pipeline_creation(self, mock_context):
        """Test that standard training creates a valid pipeline."""
        strategy = StandardTraining()
        pipeline = strategy.create_pipeline(mock_context)
        
        # Verify pipeline structure
        assert pipeline.name == "StandardTrainingPipeline"
        assert len(pipeline.commands) == 4  # Forward, Backward, Optimizer, Logging
        assert len(pipeline.middleware) == 4  # Timing, Error, Metrics, Validation
    
    def test_standard_training_validation(self, mock_context):
        """Test standard training validation."""
        strategy = StandardTraining()
        
        # Should pass with valid context
        errors = strategy.validate_requirements(mock_context)
        assert len(errors) == 0
        
        # Should fail with missing model
        mock_context.model = None
        errors = strategy.validate_requirements(mock_context)
        assert len(errors) > 0
        assert "Model is required" in errors
    
    def test_standard_training_config(self):
        """Test standard training configuration."""
        # Test with default config
        strategy = StandardTraining()
        config = strategy.get_default_config()
        
        assert "mixed_precision" in config
        assert "max_grad_norm" in config
        assert config["mixed_precision"] == False
        
        # Test with custom config
        custom_config = {"mixed_precision": True, "max_grad_norm": 2.0}
        strategy = StandardTraining(config=custom_config)
        
        assert strategy.config["mixed_precision"] == True
        assert strategy.config["max_grad_norm"] == 2.0


class TestGradientAccumulationTraining:
    """Test gradient accumulation training strategy."""
    
    @pytest.fixture
    def mock_context(self):
        """Create mock command context."""
        context = Mock(spec=CommandContext)
        context.model = Mock()
        context.optimizer = Mock()
        context.state = TrainingState()
        context.train_dataloader = Mock()
        context.config = {}
        context.is_training = True
        return context
    
    def test_gradient_accumulation_pipeline(self, mock_context):
        """Test gradient accumulation pipeline creation."""
        config = {"gradient_accumulation_steps": 4}
        strategy = GradientAccumulationTraining(config=config)
        pipeline = strategy.create_pipeline(mock_context)
        
        # Should have gradient accumulation command
        assert len(pipeline.commands) == 5  # Forward, Backward, Accumulation, Optimizer, Logging
        assert pipeline.name == "GradientAccumulationPipeline"
        
        # Check config is applied
        assert strategy.config["gradient_accumulation_steps"] == 4
    
    def test_gradient_accumulation_config_merging(self):
        """Test that custom config is properly merged with defaults."""
        custom_config = {
            "gradient_accumulation_steps": 8,
            "mixed_precision": True,
            "custom_param": "test"
        }
        
        strategy = GradientAccumulationTraining(config=custom_config)
        
        # Should have custom values
        assert strategy.config["gradient_accumulation_steps"] == 8
        assert strategy.config["mixed_precision"] == True
        assert strategy.config["custom_param"] == "test"
        
        # Should have default values for non-specified params
        assert "max_grad_norm" in strategy.config


class TestMixedPrecisionTraining:
    """Test mixed precision training strategy."""
    
    @pytest.fixture
    def mock_context(self):
        """Create mock command context."""
        context = Mock(spec=CommandContext)
        context.model = Mock()
        context.optimizer = Mock()
        context.state = TrainingState()
        context.train_dataloader = Mock()
        context.config = {}
        context.is_training = True
        return context
    
    def test_mixed_precision_pipeline(self, mock_context):
        """Test mixed precision pipeline creation."""
        strategy = MixedPrecisionTraining()
        pipeline = strategy.create_pipeline(mock_context)
        
        # Verify mixed precision is enabled in config
        assert strategy.config["mixed_precision"] == True
        assert strategy.config["loss_scale"] == 1.0
        
        # Check pipeline structure
        assert len(pipeline.commands) == 4
        assert pipeline.name == "MixedPrecisionPipeline"
    
    def test_mixed_precision_config(self):
        """Test mixed precision configuration."""
        config = {"loss_scale": 2.0, "max_grad_norm": 0.5}
        strategy = MixedPrecisionTraining(config=config)
        
        assert strategy.config["mixed_precision"] == True  # Always enabled
        assert strategy.config["loss_scale"] == 2.0
        assert strategy.config["max_grad_norm"] == 0.5


class TestMLXOptimizedTraining:
    """Test MLX-optimized training strategy."""
    
    @pytest.fixture
    def mock_context(self):
        """Create mock command context."""
        context = Mock(spec=CommandContext)
        context.model = Mock()
        context.optimizer = Mock()
        context.state = TrainingState()
        context.train_dataloader = Mock()
        context.config = {}
        context.is_training = True
        return context
    
    def test_mlx_optimized_pipeline(self, mock_context):
        """Test MLX-optimized pipeline creation."""
        strategy = MLXOptimizedTraining()
        pipeline = strategy.create_pipeline(mock_context)
        
        # Should use MLX-specific optimizations
        assert strategy.config["mixed_precision"] == True
        assert strategy.config["use_compilation"] == True
        
        # Check pipeline structure
        assert len(pipeline.commands) >= 4
        assert pipeline.name == "MLXOptimizedPipeline"
    
    def test_mlx_optimized_with_gradient_accumulation(self, mock_context):
        """Test MLX optimization with gradient accumulation."""
        config = {"gradient_accumulation_steps": 2}
        strategy = MLXOptimizedTraining(config=config)
        pipeline = strategy.create_pipeline(mock_context)
        
        # Should have additional gradient accumulation command
        assert len(pipeline.commands) == 5  # Includes accumulation command
    
    @patch('training.strategies.standard.mx', None)  # Mock MLX not available
    def test_mlx_validation_without_mlx(self, mock_context):
        """Test validation when MLX is not available."""
        strategy = MLXOptimizedTraining()
        
        # Should fail validation if MLX not available
        with patch('importlib.import_module', side_effect=ImportError):
            errors = strategy.validate_requirements(mock_context)
            # Note: This test might need adjustment based on actual implementation


class TestStrategyContextConfiguration:
    """Test strategy context configuration."""
    
    @pytest.fixture
    def mock_context(self):
        """Create mock command context."""
        context = Mock(spec=CommandContext)
        context.model = Mock()
        context.optimizer = Mock()
        context.state = TrainingState()
        context.config = {}
        return context
    
    def test_context_configuration_merging(self, mock_context):
        """Test that strategy config is merged into context config."""
        custom_config = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "custom_param": "value"
        }
        
        strategy = StandardTraining(config=custom_config)
        configured_context = strategy.configure_context(mock_context)
        
        # Strategy config should be merged into context config
        assert configured_context.config["learning_rate"] == 0.001
        assert configured_context.config["batch_size"] == 32
        assert configured_context.config["custom_param"] == "value"


@pytest.mark.integration
class TestStrategyPipelineIntegration:
    """Integration tests for strategy-pipeline interaction."""
    
    def test_strategy_pipeline_execution_flow(self):
        """Test the complete flow from strategy to pipeline execution."""
        # Create mock context
        context = Mock(spec=CommandContext)
        context.model = Mock()
        context.optimizer = Mock()
        context.state = TrainingState()
        context.config = {}
        context.is_training = True
        
        # Mock model execution
        context.model.return_value = {"loss": 0.5}
        context.optimizer.learning_rate = 0.001
        
        # Create strategy and get pipeline
        strategy = StandardTraining()
        configured_context = strategy.configure_context(context)
        pipeline = strategy.create_pipeline(configured_context)
        
        # Verify pipeline can be created and has expected structure
        assert pipeline is not None
        assert len(pipeline.commands) > 0
        assert len(pipeline.middleware) > 0
        
        # Note: Full execution would require more complex mocking
        # of framework-specific operations
    
    def test_multiple_strategies_different_pipelines(self):
        """Test that different strategies create different pipelines."""
        context = Mock(spec=CommandContext)
        context.model = Mock()
        context.optimizer = Mock()
        context.state = TrainingState()
        context.config = {}
        context.is_training = True
        
        # Create different strategies
        standard = StandardTraining()
        gradient_acc = GradientAccumulationTraining(config={"gradient_accumulation_steps": 4})
        mixed_precision = MixedPrecisionTraining()
        
        # Get pipelines
        standard_pipeline = standard.create_pipeline(context)
        gradient_acc_pipeline = gradient_acc.create_pipeline(context)
        mixed_precision_pipeline = mixed_precision.create_pipeline(context)
        
        # Verify different characteristics
        assert standard_pipeline.name != gradient_acc_pipeline.name
        assert gradient_acc_pipeline.name != mixed_precision_pipeline.name
        
        # Gradient accumulation should have more commands
        assert len(gradient_acc_pipeline.commands) > len(standard_pipeline.commands)
        
        # All should have middleware
        assert len(standard_pipeline.middleware) > 0
        assert len(gradient_acc_pipeline.middleware) > 0
        assert len(mixed_precision_pipeline.middleware) > 0