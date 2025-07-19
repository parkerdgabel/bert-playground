"""Tests for BERT feedforward layers."""

import pytest
import mlx.core as mx
import mlx.nn as nn

from models.bert.layers import BertIntermediate, BertOutput
from models.bert.config import BertConfig
from tests.models.fixtures.configs import (
    create_bert_config,
    create_small_bert_config,
)
from tests.models.fixtures.data import create_embeddings
from tests.models.fixtures.utils import check_gradient_flow


class TestBertIntermediate:
    """Test BertIntermediate layer implementation."""
    
    def test_initialization(self):
        """Test BertIntermediate initialization."""
        config = create_bert_config()
        intermediate = BertIntermediate(config)
        
        # Check layer dimensions
        assert intermediate.dense.weight.shape == (config.intermediate_size, config.hidden_size)
        assert intermediate.dense.bias.shape == (config.intermediate_size,)
    
    def test_forward_pass(self, create_embeddings):
        """Test forward pass through intermediate layer."""
        config = create_small_bert_config()
        intermediate = BertIntermediate(config)
        
        batch_size, seq_length = 4, 32
        hidden_states = create_embeddings(batch_size, seq_length, config.hidden_size)
        
        output = intermediate(hidden_states)
        assert output.shape == (batch_size, seq_length, config.intermediate_size)
        assert mx.all(mx.isfinite(output))
    
    def test_activation_function(self, create_embeddings):
        """Test activation function in intermediate layer."""
        # Test with GELU activation (default)
        config = create_bert_config(hidden_act="gelu")
        intermediate = BertIntermediate(config)
        
        batch_size, seq_length = 2, 16
        hidden_states = create_embeddings(batch_size, seq_length, config.hidden_size)
        
        output = intermediate(hidden_states)
        
        # Manually compute expected output
        dense_output = intermediate.dense(hidden_states)
        expected = nn.gelu(dense_output)
        
        assert mx.allclose(output, expected, rtol=1e-5)
    
    @pytest.mark.parametrize("hidden_act", ["gelu", "relu", "silu"])
    def test_different_activations(self, hidden_act, create_embeddings):
        """Test different activation functions."""
        config = create_bert_config(hidden_act=hidden_act)
        intermediate = BertIntermediate(config)
        
        batch_size, seq_length = 2, 16
        hidden_states = create_embeddings(batch_size, seq_length, config.hidden_size)
        
        output = intermediate(hidden_states)
        assert output.shape == (batch_size, seq_length, config.intermediate_size)
        
        # Check that activation is applied (output should differ from linear)
        linear_output = intermediate.dense(hidden_states)
        assert not mx.allclose(output, linear_output)
    
    def test_gradient_flow(self, create_embeddings, check_gradients):
        """Test gradient flow through intermediate layer."""
        config = create_small_bert_config()
        intermediate = BertIntermediate(config)
        
        batch_size, seq_length = 2, 16
        hidden_states = create_embeddings(batch_size, seq_length, config.hidden_size)
        
        def loss_fn(model, inputs):
            output = model(inputs["hidden_states"])
            return mx.mean(output)
        
        result = check_gradients(intermediate, loss_fn, {"hidden_states": hidden_states})
        assert result is True
    
    @pytest.mark.parametrize("intermediate_size", [1024, 2048, 3072, 4096])
    def test_different_intermediate_sizes(self, intermediate_size, create_embeddings):
        """Test with different intermediate layer sizes."""
        config = create_bert_config(
            hidden_size=768,
            intermediate_size=intermediate_size
        )
        intermediate = BertIntermediate(config)
        
        batch_size, seq_length = 2, 32
        hidden_states = create_embeddings(batch_size, seq_length, config.hidden_size)
        
        output = intermediate(hidden_states)
        assert output.shape == (batch_size, seq_length, intermediate_size)


class TestBertOutput:
    """Test BertOutput layer implementation."""
    
    def test_initialization(self):
        """Test BertOutput initialization."""
        config = create_bert_config()
        bert_output = BertOutput(config)
        
        # Check layer dimensions
        assert bert_output.dense.weight.shape == (config.hidden_size, config.intermediate_size)
        assert bert_output.dense.bias.shape == (config.hidden_size,)
        assert hasattr(bert_output, 'LayerNorm')
        assert hasattr(bert_output, 'dropout')
    
    def test_forward_pass(self, create_embeddings):
        """Test forward pass through output layer."""
        config = create_small_bert_config()
        bert_output = BertOutput(config)
        
        batch_size, seq_length = 4, 32
        # Hidden states from intermediate layer
        hidden_states = create_embeddings(batch_size, seq_length, config.intermediate_size)
        # Input tensor for residual connection
        input_tensor = create_embeddings(batch_size, seq_length, config.hidden_size)
        
        output = bert_output(hidden_states, input_tensor)
        assert output.shape == (batch_size, seq_length, config.hidden_size)
        assert mx.all(mx.isfinite(output))
    
    def test_residual_connection(self, create_embeddings):
        """Test residual connection in output layer."""
        config = create_small_bert_config()
        bert_output = BertOutput(config)
        
        batch_size, seq_length = 2, 16
        hidden_states = create_embeddings(batch_size, seq_length, config.intermediate_size)
        input_tensor = create_embeddings(batch_size, seq_length, config.hidden_size)
        
        # Set to eval mode to avoid dropout
        bert_output.eval()
        
        output = bert_output(hidden_states, input_tensor)
        
        # Manually compute expected output
        dense_output = bert_output.dense(hidden_states)
        # Residual connection is added before layer norm
        expected = bert_output.LayerNorm(dense_output + input_tensor)
        
        assert mx.allclose(output, expected, rtol=1e-5)
    
    def test_dropout_behavior(self, create_embeddings):
        """Test dropout in output layer."""
        config = create_bert_config(hidden_dropout_prob=0.5)
        bert_output = BertOutput(config)
        
        batch_size, seq_length = 2, 16
        hidden_states = create_embeddings(batch_size, seq_length, config.intermediate_size)
        input_tensor = create_embeddings(batch_size, seq_length, config.hidden_size)
        
        # Training mode
        bert_output.train()
        output1 = bert_output(hidden_states, input_tensor)
        output2 = bert_output(hidden_states, input_tensor)
        assert not mx.allclose(output1, output2)
        
        # Eval mode
        bert_output.eval()
        output3 = bert_output(hidden_states, input_tensor)
        output4 = bert_output(hidden_states, input_tensor)
        assert mx.allclose(output3, output4)
    
    def test_layer_normalization(self, create_embeddings):
        """Test layer normalization in output layer."""
        config = create_bert_config()
        bert_output = BertOutput(config)
        
        batch_size, seq_length = 4, 32
        hidden_states = create_embeddings(batch_size, seq_length, config.intermediate_size)
        input_tensor = create_embeddings(batch_size, seq_length, config.hidden_size)
        
        output = bert_output(hidden_states, input_tensor)
        
        # Check normalization
        mean = mx.mean(output, axis=-1)
        std = mx.std(output, axis=-1)
        
        assert mx.all(mx.abs(mean) < 0.1)  # Mean close to 0
        assert mx.all(mx.abs(std - 1.0) < 0.2)  # Std close to 1


@pytest.mark.integration
class TestFeedForwardIntegration:
    """Integration tests for feedforward components."""
    
    def test_full_feedforward_block(self, create_embeddings):
        """Test full feedforward block (intermediate + output)."""
        config = create_small_bert_config()
        intermediate = BertIntermediate(config)
        output = BertOutput(config)
        
        batch_size, seq_length = 4, 32
        hidden_states = create_embeddings(batch_size, seq_length, config.hidden_size)
        
        # Full feedforward block
        intermediate_output = intermediate(hidden_states)
        final_output = output(intermediate_output, hidden_states)
        
        assert final_output.shape == hidden_states.shape
        assert mx.all(mx.isfinite(final_output))
    
    def test_feedforward_with_attention_output(self, create_embeddings):
        """Test feedforward block with attention layer output."""
        config = create_bert_config()
        intermediate = BertIntermediate(config)
        output = BertOutput(config)
        
        # Simulate attention output
        batch_size, seq_length = 2, 64
        attention_output = create_embeddings(batch_size, seq_length, config.hidden_size)
        
        # Apply feedforward
        intermediate_output = intermediate(attention_output)
        final_output = output(intermediate_output, attention_output)
        
        # Should maintain shape and add residual
        assert final_output.shape == attention_output.shape
    
    def test_gradient_flow_through_block(self, create_embeddings, check_gradients):
        """Test gradient flow through complete feedforward block."""
        config = create_small_bert_config()
        
        class FeedForwardBlock(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.intermediate = BertIntermediate(config)
                self.output = BertOutput(config)
            
            def __call__(self, hidden_states):
                intermediate_output = self.intermediate(hidden_states)
                return self.output(intermediate_output, hidden_states)
        
        block = FeedForwardBlock(config)
        
        batch_size, seq_length = 2, 16
        hidden_states = create_embeddings(batch_size, seq_length, config.hidden_size)
        
        def loss_fn(model, inputs):
            output = model(inputs["hidden_states"])
            return mx.mean(output)
        
        result = check_gradients(block, loss_fn, {"hidden_states": hidden_states})
        assert result is True


@pytest.mark.slow
class TestFeedForwardPerformance:
    """Performance tests for feedforward layers."""
    
    def test_large_intermediate_size(self):
        """Test performance with large intermediate size."""
        config = create_bert_config(
            hidden_size=768,
            intermediate_size=8192  # Very large
        )
        intermediate = BertIntermediate(config)
        output = BertOutput(config)
        
        batch_size, seq_length = 4, 128
        hidden_states = mx.random.normal((batch_size, seq_length, config.hidden_size))
        
        # Should handle large intermediate size
        intermediate_output = intermediate(hidden_states)
        assert intermediate_output.shape == (batch_size, seq_length, 8192)
        
        final_output = output(intermediate_output, hidden_states)
        assert final_output.shape == hidden_states.shape
    
    def test_memory_efficiency(self, memory_profiler, create_embeddings):
        """Test memory usage of feedforward layers."""
        config = create_bert_config()
        intermediate = BertIntermediate(config)
        output = BertOutput(config)
        
        batch_size, seq_length = 16, 256
        hidden_states = create_embeddings(batch_size, seq_length, config.hidden_size)
        
        with memory_profiler() as prof:
            intermediate_output = intermediate(hidden_states)
            final_output = output(intermediate_output, hidden_states)
            mx.eval(final_output)
        
        memory_used = prof.get_memory_used()
        # Feedforward block should use reasonable memory
        assert memory_used < 1_000_000_000  # Less than 1GB