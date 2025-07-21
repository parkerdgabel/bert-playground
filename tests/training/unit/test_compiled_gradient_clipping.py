"""Test gradient clipping in compiled mode."""

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from training.core.compiled import _clip_gradients_compiled


class TestCompiledGradientClipping:
    """Test gradient clipping functions for compiled training."""

    def test_clip_gradients_with_tree_flatten(self):
        """Test that gradient clipping works with tree_flatten returning tuples."""

        # Create a simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(10, 20)
                self.linear2 = nn.Linear(20, 2)

            def __call__(self, x):
                x = self.linear1(x)
                x = nn.relu(x)
                x = self.linear2(x)
                return x

        model = SimpleModel()

        # Create sample input and compute gradients
        x = mx.random.normal((4, 10))
        y = mx.array([0, 1, 0, 1])

        def loss_fn(model, x, y):
            logits = model(x)
            return mx.mean(nn.losses.cross_entropy(logits, y))

        loss_and_grad_fn = mx.value_and_grad(loss_fn)
        loss, grads = loss_and_grad_fn(model, x, y)

        # Test gradient clipping
        clipped_grads = _clip_gradients_compiled(grads, max_norm=1.0)

        # Verify structure is preserved
        assert isinstance(clipped_grads, dict)
        assert set(clipped_grads.keys()) == set(grads.keys())

        for key in grads:
            assert key in clipped_grads
            assert isinstance(clipped_grads[key], dict)
            assert set(clipped_grads[key].keys()) == set(grads[key].keys())

            for param_key in grads[key]:
                assert (
                    clipped_grads[key][param_key].shape == grads[key][param_key].shape
                )

    def test_gradient_norm_clipping(self):
        """Test that gradients are actually clipped to the specified norm."""
        # Create gradients with known large norm
        grads = {
            "layer1": {
                "weight": mx.ones((100, 100)) * 10.0,  # Large gradients
                "bias": mx.ones((100,)) * 10.0,
            },
            "layer2": {
                "weight": mx.ones((50, 100)) * 10.0,
                "bias": mx.ones((50,)) * 10.0,
            },
        }

        # Compute original norm
        total_norm_sq = 0.0
        for path, g in tree_flatten(grads):
            if g is not None:
                total_norm_sq = total_norm_sq + mx.sum(g * g)
        original_norm = mx.sqrt(total_norm_sq)

        # Clip gradients
        max_norm = 5.0
        clipped_grads = _clip_gradients_compiled(grads, max_norm=max_norm)

        # Compute clipped norm
        clipped_norm_sq = 0.0
        for path, g in tree_flatten(clipped_grads):
            if g is not None:
                clipped_norm_sq = clipped_norm_sq + mx.sum(g * g)
        clipped_norm = mx.sqrt(clipped_norm_sq)

        # Verify clipping worked
        assert float(original_norm) > max_norm  # Original was larger
        assert (
            abs(float(clipped_norm) - max_norm) < 1e-4
        )  # Clipped to max_norm (allow for float32 precision)

    def test_none_gradients_handling(self):
        """Test that None gradients are handled correctly."""
        grads = {
            "layer1": {
                "weight": mx.ones((10, 10)),
                "bias": None,  # Some gradients might be None
            },
            "layer2": {"weight": mx.ones((5, 10)), "bias": mx.ones((5,))},
        }

        # Should not raise an error
        clipped_grads = _clip_gradients_compiled(grads, max_norm=1.0)

        # Verify None is preserved
        assert clipped_grads["layer1"]["bias"] is None
        assert clipped_grads["layer1"]["weight"] is not None
        assert clipped_grads["layer2"]["weight"] is not None
        assert clipped_grads["layer2"]["bias"] is not None
