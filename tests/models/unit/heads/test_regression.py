"""Tests for regression heads."""

import mlx.core as mx
import pytest

from models.heads import RegressionHead, create_head
from tests.models.fixtures.configs import create_regression_config
from tests.models.fixtures.data import create_regression_targets


class TestRegressionHead:
    """Test RegressionHead implementation."""

    def test_initialization(self):
        """Test RegressionHead initialization."""
        config = create_regression_config(hidden_size=768, dropout_prob=0.1)
        head = RegressionHead(config)

        assert head.config.input_size == 768
        assert head.config.dropout_prob == 0.1
        assert hasattr(head.config, "output_size")

    def test_forward_pass(self, create_embeddings):
        """Test forward pass through regression head."""
        config = create_regression_config(hidden_size=128, output_dim=1)
        head = RegressionHead(config)

        batch_size, seq_length = 4, 32
        hidden_states = create_embeddings(batch_size, seq_length, 128)
        attention_mask = mx.ones((batch_size, seq_length))
        targets = create_regression_targets(batch_size, output_dim=1)

        outputs = head(
            hidden_states=hidden_states, attention_mask=attention_mask, labels=targets
        )

        assert "logits" in outputs
        assert outputs["logits"].shape == (batch_size, 1)

    def test_multi_output_regression(self, create_embeddings):
        """Test regression with multiple outputs."""
        output_dim = 5
        config = create_regression_config(hidden_size=128, output_dim=output_dim)
        head = RegressionHead(config)

        batch_size = 4
        hidden_states = create_embeddings(batch_size, 32, 128)
        targets = create_regression_targets(batch_size, output_dim=output_dim)

        outputs = head(hidden_states=hidden_states, labels=targets)

        assert outputs["logits"].shape == (batch_size, output_dim)

    def test_pooling_strategies(self, create_embeddings):
        """Test different pooling strategies for regression."""
        batch_size, seq_length = 4, 32
        hidden_states = create_embeddings(batch_size, seq_length, 128)
        attention_mask = mx.ones((batch_size, seq_length))

        for pooling_type in ["cls", "mean", "max"]:
            config = create_regression_config(
                hidden_size=128, pooling_type=pooling_type
            )
            head = RegressionHead(config)

            outputs = head(hidden_states=hidden_states, attention_mask=attention_mask)
            assert outputs["logits"].shape == (batch_size, 1)

    def test_loss_computation(self, create_embeddings):
        """Test MSE loss computation."""
        config = create_regression_config(hidden_size=128)
        head = RegressionHead(config)

        batch_size = 4
        hidden_states = create_embeddings(batch_size, 32, 128)
        targets = create_regression_targets(
            batch_size, output_dim=1, target_type="linear", range_min=0.0, range_max=1.0
        )

        outputs = head(hidden_states=hidden_states, labels=targets)

        # Compute MSE loss manually
        predictions = outputs["logits"]
        loss = mx.mean((predictions - targets) ** 2)
        assert mx.isfinite(loss)

    def test_gradient_flow(self, create_embeddings, check_gradients):
        """Test gradient flow through regression head."""
        config = create_regression_config(hidden_size=128)
        head = RegressionHead(config)

        batch_size = 2
        data = {
            "hidden_states": create_embeddings(batch_size, 16, 128),
            "targets": create_regression_targets(batch_size, output_dim=1),
        }

        def loss_fn(model, batch):
            outputs = model(
                hidden_states=batch["hidden_states"], labels=batch["targets"]
            )
            predictions = outputs["logits"]
            loss = mx.mean((predictions - batch["targets"]) ** 2)
            return loss

        result = check_gradients(head, loss_fn, data)
        assert result is True

    @pytest.mark.parametrize("output_dim", [1, 3, 10])
    def test_different_output_dimensions(self, output_dim, create_embeddings):
        """Test regression with different output dimensions."""
        config = create_regression_config(hidden_size=128, output_dim=output_dim)
        head = RegressionHead(config)

        batch_size = 4
        hidden_states = create_embeddings(batch_size, 32, 128)

        outputs = head(hidden_states=hidden_states)
        assert outputs["logits"].shape == (batch_size, output_dim)

    def test_with_masked_input(self, create_embeddings, create_attention_mask):
        """Test regression with masked input sequences."""
        config = create_regression_config(hidden_size=128, pooling_type="mean")
        head = RegressionHead(config)

        batch_size, seq_length = 4, 32
        hidden_states = create_embeddings(batch_size, seq_length, 128)
        attention_mask = create_attention_mask(
            batch_size, seq_length, mask_type="padding", mask_ratio=0.3
        )

        outputs = head(hidden_states=hidden_states, attention_mask=attention_mask)
        assert mx.all(mx.isfinite(outputs["logits"]))

    def test_target_types(self, create_embeddings):
        """Test regression with different target distributions."""
        config = create_regression_config(hidden_size=128, output_dim=1)
        head = RegressionHead(config)

        batch_size = 8
        hidden_states = create_embeddings(batch_size, 32, 128)

        for target_type in ["random", "linear", "sine"]:
            targets = create_regression_targets(
                batch_size, output_dim=1, target_type=target_type
            )

            outputs = head(hidden_states=hidden_states, labels=targets)
            loss = mx.mean((outputs["logits"] - targets) ** 2)
            assert mx.isfinite(loss)


class TestRegressionHeadFactory:
    """Test regression head factory functions."""

    def test_create_regression_head(self):
        """Test creating regression head via factory."""
        head = create_head(head_type="regression", input_size=768, output_size=1)
        assert isinstance(head, RegressionHead)

        # Multi-output regression
        head = create_head(head_type="regression", input_size=768, output_size=5)
        assert isinstance(head, RegressionHead)

    def test_factory_with_config(self):
        """Test factory with regression config."""
        config = create_regression_config(
            hidden_size=256, output_dim=3, dropout_prob=0.2, pooling_type="mean"
        )

        head = create_head(
            head_type="regression",
            input_size=config.input_size,
            output_size=config.output_size,
            dropout_prob=config.dropout_prob,
            pooling_type=config.pooling_type,
            activation=config.activation,
        )

        assert isinstance(head, RegressionHead)
        assert head.config.input_size == 256
        assert head.config.output_size == 3


@pytest.mark.integration
class TestRegressionHeadIntegration:
    """Integration tests for regression heads."""

    def test_with_bert_outputs(self, mock_bert_model, create_test_batch):
        """Test regression head with BERT model outputs."""
        bert = mock_bert_model
        head = create_head(
            head_type="regression", input_size=bert.config.hidden_size, output_size=1
        )

        # Get BERT outputs
        batch = create_test_batch(
            batch_size=4,
            seq_length=32,
            vocab_size=bert.config.vocab_size,
            include_labels=False,  # No classification labels
        )
        bert_outputs = bert(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )

        # Create regression targets
        targets = create_regression_targets(4, output_dim=1)

        # Pass through head
        head_outputs = head(
            hidden_states=bert_outputs["last_hidden_state"],
            attention_mask=batch["attention_mask"],
            labels=targets,
        )

        assert "logits" in head_outputs
        assert head_outputs["logits"].shape == (4, 1)

    def test_multi_task_regression(self, mock_bert_model):
        """Test multi-task regression setup."""
        bert = mock_bert_model

        # Create multiple regression heads for different tasks
        heads = {
            "task1": create_head(
                head_type="regression",
                input_size=bert.config.hidden_size,
                output_size=1,
            ),
            "task2": create_head(
                head_type="regression",
                input_size=bert.config.hidden_size,
                output_size=3,
            ),
            "task3": create_head(
                head_type="regression",
                input_size=bert.config.hidden_size,
                output_size=5,
            ),
        }

        # Test data
        batch_size = 4
        input_ids = mx.random.randint(0, bert.config.vocab_size, (batch_size, 32))

        # Get shared BERT representations
        bert_outputs = bert(input_ids=input_ids)
        hidden_states = bert_outputs["last_hidden_state"]

        # Apply each head
        outputs = {}
        for task_name, head in heads.items():
            task_outputs = head(hidden_states=hidden_states)
            outputs[task_name] = task_outputs["logits"]

        # Check outputs
        assert outputs["task1"].shape == (batch_size, 1)
        assert outputs["task2"].shape == (batch_size, 3)
        assert outputs["task3"].shape == (batch_size, 5)


@pytest.mark.slow
class TestRegressionHeadPerformance:
    """Performance tests for regression heads."""

    def test_large_batch_handling(self, create_embeddings):
        """Test regression head with large batches."""
        config = create_regression_config(hidden_size=128)
        head = RegressionHead(config)

        # Large batch
        batch_size = 256
        hidden_states = create_embeddings(batch_size, 128, 128)

        outputs = head(hidden_states=hidden_states)
        assert outputs["logits"].shape == (batch_size, 1)

    def test_memory_efficiency(self, memory_profiler, create_embeddings):
        """Test memory usage of regression head."""
        config = create_regression_config(hidden_size=256, output_dim=10)
        head = RegressionHead(config)

        batch_size = 64
        hidden_states = create_embeddings(batch_size, 128, 256)

        with memory_profiler() as prof:
            outputs = head(hidden_states=hidden_states)
            mx.eval(outputs["logits"])

        memory_used = prof.get_memory_used()
        # Regression head should use minimal memory
        assert memory_used < 100_000_000  # Less than 100MB
