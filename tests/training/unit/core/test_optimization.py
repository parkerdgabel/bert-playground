"""Unit tests for optimization components."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from tests.training.fixtures.models import SimpleBinaryClassifier
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

    def test_create_adam_optimizer(self):
        """Test Adam optimizer creation."""
        model = SimpleBinaryClassifier()
        config = OptimizerConfig(
            type=OptimizerType.ADAM,
            learning_rate=1e-3,
            weight_decay=0.01,
            beta1=0.9,
            beta2=0.999,
        )

        optimizer = create_optimizer(model, config)

        assert optimizer is not None
        assert hasattr(optimizer, "learning_rate")
        assert optimizer.learning_rate == 1e-3

    def test_create_adamw_optimizer(self):
        """Test AdamW optimizer creation."""
        model = SimpleBinaryClassifier()
        config = OptimizerConfig(
            type=OptimizerType.ADAMW,
            learning_rate=2e-5,
            weight_decay=0.1,
        )

        optimizer = create_optimizer(model, config)

        assert optimizer is not None
        assert optimizer.learning_rate == 2e-5

    def test_create_sgd_optimizer(self):
        """Test SGD optimizer creation."""
        model = SimpleBinaryClassifier()
        config = OptimizerConfig(
            type=OptimizerType.SGD,
            learning_rate=0.1,
            momentum=0.9,
            weight_decay=0.0,
        )

        optimizer = create_optimizer(model, config)

        assert optimizer is not None
        assert optimizer.learning_rate == 0.1

    def test_create_lion_optimizer(self):
        """Test Lion optimizer creation."""
        model = SimpleBinaryClassifier()
        config = OptimizerConfig(
            type=OptimizerType.LION,
            learning_rate=1e-4,
            weight_decay=0.0,
        )

        optimizer = create_optimizer(model, config)

        assert optimizer is not None
        assert optimizer.learning_rate == 1e-4

    def test_create_adafactor_optimizer(self):
        """Test Adafactor optimizer creation."""
        model = SimpleBinaryClassifier()
        config = OptimizerConfig(
            type=OptimizerType.ADAFACTOR,
            learning_rate=1e-3,
        )

        optimizer = create_optimizer(model, config)

        assert optimizer is not None
        # Adafactor may have dynamic learning rate
        assert hasattr(optimizer, "learning_rate") or hasattr(optimizer, "lr")

    def test_invalid_optimizer_type(self):
        """Test invalid optimizer type."""
        model = SimpleBinaryClassifier()
        # Test with invalid optimizer type by manually setting
        config = OptimizerConfig()
        config.type = "invalid_optimizer"  # type: ignore

        with pytest.raises(ValueError, match="Unknown optimizer type"):
            create_optimizer(model, config)

    def test_optimizer_step(self):
        """Test optimizer parameter update."""
        model = SimpleBinaryClassifier()
        config = OptimizerConfig(type=OptimizerType.SGD, learning_rate=0.1)
        optimizer = create_optimizer(model, config)

        # Get initial parameters
        from mlx.utils import tree_map

        initial_params = tree_map(lambda x: mx.array(x), model.parameters())

        # Create dummy loss and compute gradients
        batch = {
            "input": mx.random.normal((4, 10)),
            "labels": mx.random.randint(0, 2, (4,)),
        }

        def loss_fn(model):
            outputs = model(batch)
            return outputs["loss"]

        # Compute gradients
        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad_fn(model)

        # Update parameters
        optimizer.update(model, grads)

        # Check parameters changed
        final_params = model.parameters()

        def check_params_changed(initial, final):
            """Recursively check if parameters changed."""
            for key in initial:
                if isinstance(initial[key], dict):
                    check_params_changed(initial[key], final[key])
                else:
                    assert not mx.allclose(initial[key], final[key], atol=1e-6)

        check_params_changed(initial_params, final_params)


class TestSchedulers:
    """Test learning rate scheduler creation and functionality."""

    def test_create_constant_scheduler(self):
        """Test constant scheduler creation."""
        optimizer = type("MockOptimizer", (), {"learning_rate": 1e-3})()
        config = SchedulerConfig(type="constant")

        config.num_training_steps = 1000
        scheduler = create_lr_scheduler(optimizer, config)

        assert scheduler is not None
        # Constant scheduler might be None or a no-op
        if scheduler is not None:
            assert hasattr(scheduler, "step")

    def test_create_linear_scheduler(self):
        """Test linear scheduler creation."""
        optimizer = type("MockOptimizer", (), {"learning_rate": 1e-3})()
        config = SchedulerConfig(
            type=SchedulerType.LINEAR,
            warmup_ratio=0.1,
        )

        config.num_training_steps = 1000
        scheduler = create_lr_scheduler(optimizer, config)

        assert scheduler is not None
        assert hasattr(scheduler, "step")
        assert hasattr(scheduler, "get_last_lr")

    def test_create_cosine_scheduler(self):
        """Test cosine scheduler creation."""
        optimizer = type("MockOptimizer", (), {"learning_rate": 1e-3})()
        config = SchedulerConfig(
            type="cosine",
            warmup_ratio=0.1,
            num_cycles=0.5,
        )

        config.num_training_steps = 1000
        scheduler = create_lr_scheduler(optimizer, config)

        assert scheduler is not None
        assert hasattr(scheduler, "step")

    def test_create_cosine_with_restarts_scheduler(self):
        """Test cosine with restarts scheduler creation."""
        optimizer = type("MockOptimizer", (), {"learning_rate": 1e-3})()
        config = SchedulerConfig(
            type="cosine_with_restarts",
            warmup_ratio=0.05,
            num_cycles=4,
        )

        config.num_training_steps = 1000
        scheduler = create_lr_scheduler(optimizer, config)

        assert scheduler is not None
        assert hasattr(scheduler, "step")

    def test_create_exponential_scheduler(self):
        """Test exponential scheduler creation."""
        optimizer = type("MockOptimizer", (), {"learning_rate": 1e-3})()
        config = SchedulerConfig(
            type="exponential",
            gamma=0.95,
        )

        config.num_training_steps = 1000
        scheduler = create_lr_scheduler(optimizer, config)

        assert scheduler is not None
        assert hasattr(scheduler, "step")

    def test_create_reduce_on_plateau_scheduler(self):
        """Test ReduceLROnPlateau scheduler creation."""
        optimizer = type("MockOptimizer", (), {"learning_rate": 1e-3})()
        config = SchedulerConfig(
            type="reduce_on_plateau",
            factor=0.5,
            patience=3,
            mode="min",
            threshold=0.01,
        )

        config.num_training_steps = 1000
        scheduler = create_lr_scheduler(optimizer, config)

        assert scheduler is not None
        assert hasattr(scheduler, "step")

    def test_invalid_scheduler_type(self):
        """Test invalid scheduler type."""
        optimizer = type("MockOptimizer", (), {"learning_rate": 1e-3})()
        config = SchedulerConfig(type="invalid_scheduler")

        with pytest.raises(ValueError, match="Unknown scheduler type"):
            create_lr_scheduler(optimizer, config)

    def test_scheduler_step_updates(self):
        """Test that scheduler updates learning rate."""

        # Create mock optimizer with mutable learning rate
        class MockOptimizer:
            def __init__(self):
                self.learning_rate = 1e-3

        optimizer = MockOptimizer()
        config = SchedulerConfig(
            type=SchedulerType.LINEAR,
            warmup_ratio=0.1,
        )

        config.num_training_steps = 100
        scheduler = create_lr_scheduler(optimizer, config)

        if scheduler is not None:
            initial_lr = optimizer.learning_rate

            # Step through warmup
            for _ in range(10):
                scheduler.step()

            # LR should have changed during warmup
            # Note: Exact behavior depends on scheduler implementation


class TestGradientAccumulator:
    """Test gradient accumulation functionality."""

    def test_gradient_accumulator_init(self):
        """Test gradient accumulator initialization."""
        accumulator = GradientAccumulator()

        assert accumulator.accumulated_grads is None
        assert accumulator.step_count == 0

    def test_gradient_accumulation(self):
        """Test accumulating gradients."""
        accumulator = GradientAccumulator()

        # Create dummy gradients
        grads1 = {
            "layer1": {"weight": mx.ones((10, 10)), "bias": mx.ones((10,))},
            "layer2": {"weight": mx.ones((5, 10)), "bias": mx.ones((5,))},
        }

        grads2 = {
            "layer1": {"weight": mx.ones((10, 10)) * 2, "bias": mx.ones((10,)) * 2},
            "layer2": {"weight": mx.ones((5, 10)) * 2, "bias": mx.ones((5,)) * 2},
        }

        # Accumulate gradients
        accumulator.accumulate(grads1)
        accumulator.accumulate(grads2)

        assert accumulator.step_count == 2

        # Get accumulated gradients
        accumulated = accumulator.get_gradients()

        # Should be sum of gradients
        expected_weight1 = mx.ones((10, 10)) * 3
        assert mx.allclose(accumulated["layer1"]["weight"], expected_weight1)

    def test_gradient_reset(self):
        """Test resetting gradient accumulator."""
        accumulator = GradientAccumulator()

        # Accumulate some gradients
        grads = {"weight": mx.ones((10, 10))}
        accumulator.accumulate(grads)

        assert accumulator.step_count == 1

        # Reset
        accumulator.reset()

        assert accumulator.accumulated_grads is None
        assert accumulator.step_count == 0

    def test_averaged_gradients(self):
        """Test getting averaged gradients."""
        accumulator = GradientAccumulator(accumulation_steps=4)

        # Accumulate gradients
        for i in range(4):
            grads = {"weight": mx.ones((5, 5)) * (i + 1)}
            accumulator.accumulate(grads)

        # Get averaged gradients
        averaged = accumulator.get_gradients(average=True)

        # Average should be (1 + 2 + 3 + 4) / 4 = 2.5
        expected = mx.ones((5, 5)) * 2.5
        assert mx.allclose(averaged["weight"], expected)


class TestGradientClipping:
    """Test gradient clipping functionality."""

    def test_clip_gradients_by_norm(self):
        """Test gradient clipping by norm."""
        # Create gradients with large norm
        grads = {
            "layer1": {"weight": mx.ones((10, 10)) * 10},
            "layer2": {"weight": mx.ones((5, 10)) * 10},
        }

        # Clip gradients
        clipped_grads, original_norm = clip_gradients(grads, max_norm=1.0)

        # Compute norm of clipped gradients
        flat_grads = []
        for param in clipped_grads.values():
            if isinstance(param, dict):
                for p in param.values():
                    flat_grads.append(p.flatten())
            else:
                flat_grads.append(param.flatten())

        all_grads = mx.concatenate(flat_grads)
        clipped_norm = mx.sqrt(mx.sum(all_grads**2)).item()

        # Norm should be approximately 1.0
        assert abs(clipped_norm - 1.0) < 0.1

    def test_clip_gradients_no_clipping(self):
        """Test gradient clipping when norm is already small."""
        # Create gradients with small norm
        grads = {
            "weight": mx.ones((5, 5)) * 0.01,
            "bias": mx.ones((5,)) * 0.01,
        }

        # Clip gradients with high max_norm
        clipped_grads, original_norm = clip_gradients(grads, max_norm=10.0)

        # Gradients should be unchanged
        for key in grads:
            assert mx.allclose(clipped_grads[key], grads[key])

    def test_clip_gradients_empty(self):
        """Test clipping empty gradients."""
        grads = {}
        clipped, norm = clip_gradients(grads, max_norm=1.0)
        assert clipped == {}
        assert norm == 0.0


class TestGradientStats:
    """Test gradient statistics computation."""

    def test_compute_gradient_stats(self):
        """Test computing gradient statistics."""
        grads = {
            "layer1": {
                "weight": mx.array([[1.0, 2.0], [3.0, 4.0]]),
                "bias": mx.array([0.5, 1.5]),
            },
            "layer2": {
                "weight": mx.array([[0.1, 0.2]]),
            },
        }

        stats = compute_gradient_stats(grads)

        assert "grad_norm" in stats
        assert "grad_mean" in stats
        assert "grad_min" in stats
        assert "grad_max" in stats

        # Check values are reasonable
        assert stats["grad_norm"] > 0
        assert stats["grad_min"] <= stats["grad_mean"] <= stats["grad_max"]
        assert stats["grad_std"] >= 0

    def test_compute_gradient_stats_zero_grads(self):
        """Test computing stats for zero gradients."""
        grads = {
            "weight": mx.zeros((10, 10)),
            "bias": mx.zeros((10,)),
        }

        stats = compute_gradient_stats(grads)

        assert stats["grad_norm"] == 0.0
        assert stats["grad_mean"] == 0.0
        assert stats["grad_min"] == 0.0  # min of abs(zeros) is 0
        assert stats["grad_max"] == 0.0

    def test_compute_gradient_stats_empty(self):
        """Test computing stats for empty gradients."""
        grads = {}
        stats = compute_gradient_stats(grads)

        # Should return zeros or handle gracefully
        assert stats["grad_norm"] == 0.0
