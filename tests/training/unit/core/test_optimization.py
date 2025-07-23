"""Unit tests for optimization components."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from tests.training.fixtures.models import SimpleBinaryClassifier
from training.adapters.framework_adapter import FrameworkAdapter
from training.core.config import (
    OptimizerConfig,
    OptimizerType,
    SchedulerConfig,
    SchedulerType,
)
from training.core.optimization import (
    GradientAccumulator,
    clip_gradients,
    compute_gradient_stats,
    create_lr_scheduler,
    create_optimizer,
)


class TestOptimizers:
    """Test optimizer creation and functionality."""

    @pytest.fixture
    def framework(self):
        """Create framework adapter for tests."""
        return FrameworkAdapter(backend="mlx")

    def test_create_adam_optimizer(self, framework):
        """Test Adam optimizer creation."""
        model = SimpleBinaryClassifier()
        config = OptimizerConfig(
            type=OptimizerType.ADAM,
            learning_rate=1e-3,
            weight_decay=0.01,
            beta1=0.9,
            beta2=0.999,
        )

        optimizer = create_optimizer(model, config, framework)

        assert optimizer is not None
        assert hasattr(optimizer, "learning_rate")
        assert optimizer.learning_rate == 1e-3

    def test_create_adamw_optimizer(self, framework):
        """Test AdamW optimizer creation."""
        model = SimpleBinaryClassifier()
        config = OptimizerConfig(
            type=OptimizerType.ADAMW,
            learning_rate=2e-5,
            weight_decay=0.1,
        )

        optimizer = create_optimizer(model, config, framework)

        assert optimizer is not None
        assert optimizer.learning_rate == 2e-5

    def test_create_sgd_optimizer(self, framework):
        """Test SGD optimizer creation."""
        model = SimpleBinaryClassifier()
        config = OptimizerConfig(
            type=OptimizerType.SGD,
            learning_rate=0.1,
            momentum=0.9,
            weight_decay=1e-4,
            nesterov=True,
        )

        optimizer = create_optimizer(model, config, framework)

        assert optimizer is not None
        assert optimizer.learning_rate == 0.1

    def test_create_lion_optimizer(self, framework):
        """Test Lion optimizer creation."""
        model = SimpleBinaryClassifier()
        config = OptimizerConfig(
            type=OptimizerType.LION,
            learning_rate=1e-4,
            lion_beta1=0.9,
            lion_beta2=0.99,
            weight_decay=0.0,
        )

        optimizer = create_optimizer(model, config, framework)

        assert optimizer is not None
        assert optimizer.learning_rate == 1e-4

    def test_invalid_optimizer_type(self, framework):
        """Test error handling for invalid optimizer type."""
        model = SimpleBinaryClassifier()
        config = OptimizerConfig(
            type="invalid_optimizer",  # Invalid type
            learning_rate=1e-3,
        )

        with pytest.raises(ValueError):
            create_optimizer(model, config, framework)


class TestSchedulers:
    """Test scheduler creation and functionality."""

    @pytest.fixture
    def framework(self):
        """Create framework adapter for tests."""
        return FrameworkAdapter(backend="mlx")

    @pytest.fixture
    def optimizer(self, framework):
        """Create a test optimizer."""
        model = SimpleBinaryClassifier()
        config = OptimizerConfig(type=OptimizerType.ADAM, learning_rate=1e-3)
        return create_optimizer(model, config, framework)

    def test_create_constant_scheduler(self, optimizer, framework):
        """Test constant scheduler creation."""
        config = SchedulerConfig(
            type=SchedulerType.CONSTANT,
            num_training_steps=1000,
        )

        scheduler = create_lr_scheduler(optimizer, config, framework)

        assert scheduler is not None
        assert abs(scheduler.get_last_lr() - 1e-3) < 1e-6

        # Step should not change LR
        scheduler.step()
        assert abs(scheduler.get_last_lr() - 1e-3) < 1e-6

    def test_create_linear_scheduler(self, optimizer, framework):
        """Test linear scheduler creation."""
        config = SchedulerConfig(
            type=SchedulerType.LINEAR,
            num_training_steps=1000,
            warmup_steps=100,
        )

        scheduler = create_lr_scheduler(optimizer, config, framework)

        assert scheduler is not None
        assert abs(scheduler.get_last_lr() - 1e-3) < 1e-6

        # Test warmup
        for _ in range(50):
            scheduler.step()
        assert scheduler.get_last_lr() < 1e-3  # Should be warming up

        # Test decay after warmup
        for _ in range(100):
            scheduler.step()
        current_lr = scheduler.get_last_lr()
        scheduler.step()
        assert scheduler.get_last_lr() < current_lr  # Should be decaying

    def test_create_cosine_scheduler(self, optimizer, framework):
        """Test cosine scheduler creation."""
        config = SchedulerConfig(
            type=SchedulerType.COSINE,
            num_training_steps=1000,
            warmup_steps=100,
            num_cycles=0.5,
        )

        scheduler = create_lr_scheduler(optimizer, config, framework)

        assert scheduler is not None
        assert abs(scheduler.get_last_lr() - 1e-3) < 1e-6

    def test_scheduler_state_dict(self, optimizer, framework):
        """Test scheduler state saving and loading."""
        config = SchedulerConfig(
            type=SchedulerType.LINEAR,
            num_training_steps=1000,
        )

        scheduler = create_lr_scheduler(optimizer, config, framework)

        # Step a few times
        for _ in range(10):
            scheduler.step()

        # Save state
        state = scheduler.state_dict

        # Create new scheduler and load state
        new_scheduler = create_lr_scheduler(optimizer, config, framework)
        new_scheduler.load_state_dict(state)

        assert new_scheduler.current_step == scheduler.current_step
        assert new_scheduler.current_lr == scheduler.current_lr


class TestGradientOperations:
    """Test gradient manipulation operations."""

    @pytest.fixture
    def framework(self):
        """Create framework adapter for tests."""
        return FrameworkAdapter(backend="mlx")

    @pytest.fixture
    def gradients(self):
        """Create test gradients."""
        return {
            "layer1": {
                "weight": mx.random.normal((10, 10)),
                "bias": mx.random.normal((10,)),
            },
            "layer2": {
                "weight": mx.random.normal((5, 10)),
                "bias": mx.random.normal((5,)),
            },
        }

    def test_gradient_clipping(self, gradients, framework):
        """Test gradient clipping by norm."""
        # Make gradients large
        large_grads = {}
        for k, v in gradients.items():
            large_grads[k] = {}
            for k2, v2 in v.items():
                large_grads[k][k2] = v2 * 100.0

        # Clip gradients
        clipped, norm = clip_gradients(large_grads, max_norm=1.0, framework=framework)

        # Check that norm is reduced
        assert norm <= 1.0 + 1e-6  # Allow small numerical error

        # Check that gradients are scaled proportionally
        orig_norm = framework.compute_gradient_norm(large_grads)
        scale_factor = 1.0 / orig_norm
        for k in large_grads:
            for k2 in large_grads[k]:
                expected = large_grads[k][k2] * scale_factor
                actual = clipped[k][k2]
                mx.eval(expected, actual)
                assert mx.allclose(actual, expected, atol=1e-6)

    def test_gradient_accumulation(self, gradients, framework):
        """Test gradient accumulation."""
        accumulator = GradientAccumulator(accumulation_steps=4, framework=framework)

        # Accumulate gradients
        should_update_flags = []
        for i in range(8):
            should_update = accumulator.accumulate(gradients)
            should_update_flags.append(should_update)
            
            # Check accumulated gradients after each accumulation step
            if should_update:
                accumulated = accumulator.get_gradients(average=True)
                
                # Check that gradients are averaged
                for k in gradients:
                    for k2 in gradients[k]:
                        # Should be close to original since we accumulated same gradients
                        mx.eval(accumulated[k][k2], gradients[k][k2])
                        assert mx.allclose(accumulated[k][k2], gradients[k][k2], atol=1e-5)

        # Check update pattern
        expected = [False, False, False, True, False, False, False, True]
        assert should_update_flags == expected

    def test_gradient_stats(self, gradients, framework):
        """Test gradient statistics computation."""
        stats = compute_gradient_stats(gradients, framework, detailed=False)

        assert "grad_norm" in stats
        assert stats["grad_norm"] > 0

        # Detailed stats should include more metrics
        detailed_stats = compute_gradient_stats(gradients, framework, detailed=True)
        assert "grad_norm" in detailed_stats
        assert "grad_max" in detailed_stats
        assert "grad_min" in detailed_stats
        assert "grad_mean" in detailed_stats
        assert "grad_std" in detailed_stats

    def test_gradient_accumulator_reset(self, gradients, framework):
        """Test gradient accumulator reset."""
        accumulator = GradientAccumulator(accumulation_steps=2, framework=framework)

        # Accumulate once
        accumulator.accumulate(gradients)
        assert accumulator.step_count == 1

        # Reset
        accumulator.reset()
        assert accumulator.step_count == 0
        assert accumulator.accumulated_grads is None


class TestReduceOnPlateauScheduler:
    """Test ReduceOnPlateau scheduler functionality."""

    @pytest.fixture
    def framework(self):
        """Create framework adapter for tests."""
        return FrameworkAdapter(backend="mlx")

    @pytest.fixture
    def optimizer(self, framework):
        """Create a test optimizer."""
        model = SimpleBinaryClassifier()
        config = OptimizerConfig(type=OptimizerType.ADAM, learning_rate=1e-3)
        return create_optimizer(model, config, framework)

    def test_reduce_on_plateau(self, optimizer, framework):
        """Test learning rate reduction on plateau."""
        config = SchedulerConfig(
            type=SchedulerType.REDUCE_ON_PLATEAU,
            patience=3,
            factor=0.5,
            min_lr=1e-6,
        )

        scheduler = create_lr_scheduler(optimizer, config, framework)
        initial_lr = scheduler.get_last_lr()

        # Simulate no improvement for patience steps
        metrics = {"eval_loss": 1.0}
        for _ in range(config.patience + 1):
            scheduler.step(metrics)

        # LR should be reduced
        assert abs(scheduler.get_last_lr() - initial_lr * config.factor) < 1e-6

        # Simulate improvement
        metrics = {"eval_loss": 0.5}
        scheduler.step(metrics)

        # Further no improvement
        metrics = {"eval_loss": 0.5}
        for _ in range(config.patience + 1):
            scheduler.step(metrics)

        # LR should be reduced again
        assert abs(scheduler.get_last_lr() - initial_lr * config.factor * config.factor) < 1e-6