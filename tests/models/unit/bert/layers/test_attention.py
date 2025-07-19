"""Tests for BERT attention layers."""

import pytest
import mlx.core as mx
import mlx.nn as nn

from models.bert.layers import BertSelfAttention, BertSelfOutput
from models.bert.config import BertConfig
from tests.models.fixtures.configs import (
    create_bert_config,
    create_small_bert_config,
)
from tests.models.fixtures.data import (
    create_embeddings,
    create_attention_mask,
)
from tests.models.fixtures.utils import check_gradient_flow


class TestBertSelfAttention:
    """Test BertSelfAttention layer implementation."""
    
    def test_initialization(self):
        """Test BertSelfAttention initialization."""
        config = create_bert_config()
        attention = BertSelfAttention(config)
        
        # Check dimensions
        assert attention.num_attention_heads == config.num_attention_heads
        assert attention.attention_head_size == config.hidden_size // config.num_attention_heads
        assert attention.all_head_size == config.hidden_size
        
        # Check linear layers
        assert hasattr(attention, 'query')
        assert hasattr(attention, 'key')
        assert hasattr(attention, 'value')
    
    def test_forward_pass(self, create_embeddings):
        """Test forward pass through self-attention."""
        config = create_small_bert_config()
        attention = BertSelfAttention(config)
        
        batch_size, seq_length = 4, 32
        hidden_states = create_embeddings(batch_size, seq_length, config.hidden_size)
        attention_mask = mx.ones((batch_size, seq_length))
        
        output = attention(hidden_states, attention_mask)
        
        # Output should be tuple (attention_output, attention_probs) or just attention_output
        if isinstance(output, tuple):
            attention_output, attention_probs = output
            assert attention_probs.shape[1] == config.num_attention_heads  # [batch, heads, seq, seq]
        else:
            attention_output = output
        
        assert attention_output.shape == (batch_size, seq_length, config.hidden_size)
    
    def test_attention_mask_handling(self, create_embeddings, create_attention_mask):
        """Test handling of different attention masks."""
        config = create_small_bert_config()
        attention = BertSelfAttention(config)
        
        batch_size, seq_length = 4, 32
        hidden_states = create_embeddings(batch_size, seq_length, config.hidden_size)
        
        # Test different mask types
        for mask_type in ["padding", "random", "causal"]:
            if mask_type == "causal":
                # Causal mask has different shape
                attention_mask = create_attention_mask(
                    batch_size, seq_length, mask_type=mask_type
                )
            else:
                attention_mask = create_attention_mask(
                    batch_size, seq_length, mask_type=mask_type
                )
            
            output = attention(hidden_states, attention_mask)
            
            if isinstance(output, tuple):
                attention_output = output[0]
            else:
                attention_output = output
            
            assert attention_output.shape == (batch_size, seq_length, config.hidden_size)
            assert mx.all(mx.isfinite(attention_output))
    
    def test_multi_head_attention(self):
        """Test multi-head attention computation."""
        config = create_bert_config(
            hidden_size=768,
            num_attention_heads=12
        )
        attention = BertSelfAttention(config)
        
        batch_size, seq_length = 2, 16
        hidden_states = mx.random.normal((batch_size, seq_length, config.hidden_size))
        
        output = attention(hidden_states)
        
        # Each head should process hidden_size/num_heads dimensions
        assert attention.attention_head_size == 64  # 768 / 12
        
        if isinstance(output, tuple):
            attention_output = output[0]
        else:
            attention_output = output
        
        assert attention_output.shape == (batch_size, seq_length, config.hidden_size)
    
    def test_gradient_flow(self, create_embeddings, check_gradients):
        """Test gradient flow through attention."""
        config = create_small_bert_config()
        attention = BertSelfAttention(config)
        
        batch_size, seq_length = 2, 16
        hidden_states = create_embeddings(batch_size, seq_length, config.hidden_size)
        
        def loss_fn(model, inputs):
            output = model(inputs["hidden_states"])
            if isinstance(output, tuple):
                output = output[0]
            return mx.mean(output)
        
        result = check_gradients(attention, loss_fn, {"hidden_states": hidden_states})
        assert result is True
    
    @pytest.mark.parametrize("num_heads", [1, 4, 8, 16])
    def test_different_head_counts(self, num_heads):
        """Test attention with different numbers of heads."""
        hidden_size = 256
        config = create_bert_config(
            hidden_size=hidden_size,
            num_attention_heads=num_heads
        )
        
        # Ensure hidden_size is divisible by num_heads
        assert hidden_size % num_heads == 0
        
        attention = BertSelfAttention(config)
        
        batch_size, seq_length = 2, 32
        hidden_states = mx.random.normal((batch_size, seq_length, hidden_size))
        
        output = attention(hidden_states)
        if isinstance(output, tuple):
            output = output[0]
        
        assert output.shape == (batch_size, seq_length, hidden_size)
    
    def test_attention_dropout(self):
        """Test attention dropout behavior."""
        config = create_bert_config(attention_probs_dropout_prob=0.5)
        attention = BertSelfAttention(config)
        
        batch_size, seq_length = 2, 16
        hidden_states = mx.random.normal((batch_size, seq_length, config.hidden_size))
        
        # Training mode
        attention.train()
        output1 = attention(hidden_states)
        output2 = attention(hidden_states)
        
        if isinstance(output1, tuple):
            output1 = output1[0]
            output2 = output2[0]
        
        # Outputs should differ due to dropout
        assert not mx.allclose(output1, output2)
        
        # Eval mode
        attention.eval()
        output3 = attention(hidden_states)
        output4 = attention(hidden_states)
        
        if isinstance(output3, tuple):
            output3 = output3[0]
            output4 = output4[0]
        
        # Outputs should be identical
        assert mx.allclose(output3, output4)


class TestBertSelfOutput:
    """Test BertSelfOutput layer implementation."""
    
    def test_initialization(self):
        """Test BertSelfOutput initialization."""
        config = create_bert_config()
        self_output = BertSelfOutput(config)
        
        assert hasattr(self_output, 'dense')
        assert hasattr(self_output, 'LayerNorm')
        assert hasattr(self_output, 'dropout')
    
    def test_forward_pass(self, create_embeddings):
        """Test forward pass through self-output."""
        config = create_small_bert_config()
        self_output = BertSelfOutput(config)
        
        batch_size, seq_length = 4, 32
        hidden_states = create_embeddings(batch_size, seq_length, config.hidden_size)
        input_tensor = create_embeddings(batch_size, seq_length, config.hidden_size)
        
        output = self_output(hidden_states, input_tensor)
        assert output.shape == (batch_size, seq_length, config.hidden_size)
    
    def test_residual_connection(self, create_embeddings):
        """Test residual connection in self-output."""
        config = create_small_bert_config()
        self_output = BertSelfOutput(config)
        
        batch_size, seq_length = 2, 16
        hidden_states = create_embeddings(batch_size, seq_length, config.hidden_size)
        input_tensor = create_embeddings(batch_size, seq_length, config.hidden_size)
        
        # Set to eval mode to avoid dropout
        self_output.eval()
        
        # Forward pass
        output = self_output(hidden_states, input_tensor)
        
        # Manually compute expected output
        dense_output = self_output.dense(hidden_states)
        # Residual connection is added before layer norm
        expected = self_output.LayerNorm(dense_output + input_tensor)
        
        assert mx.allclose(output, expected, rtol=1e-5)
    
    def test_layer_normalization(self, create_embeddings):
        """Test layer normalization in self-output."""
        config = create_bert_config()
        self_output = BertSelfOutput(config)
        
        batch_size, seq_length = 4, 32
        hidden_states = create_embeddings(batch_size, seq_length, config.hidden_size)
        input_tensor = create_embeddings(batch_size, seq_length, config.hidden_size)
        
        output = self_output(hidden_states, input_tensor)
        
        # Check normalization along hidden dimension
        mean = mx.mean(output, axis=-1)
        std = mx.std(output, axis=-1)
        
        assert mx.all(mx.abs(mean) < 0.1)  # Mean close to 0
        assert mx.all(mx.abs(std - 1.0) < 0.2)  # Std close to 1


@pytest.mark.integration
class TestAttentionIntegration:
    """Integration tests for attention components."""
    
    def test_full_attention_block(self, create_embeddings):
        """Test full attention block (self-attention + output)."""
        config = create_small_bert_config()
        self_attention = BertSelfAttention(config)
        self_output = BertSelfOutput(config)
        
        batch_size, seq_length = 4, 32
        hidden_states = create_embeddings(batch_size, seq_length, config.hidden_size)
        attention_mask = mx.ones((batch_size, seq_length))
        
        # Full attention block
        attention_output = self_attention(hidden_states, attention_mask)
        if isinstance(attention_output, tuple):
            attention_output = attention_output[0]
        
        output = self_output(attention_output, hidden_states)
        
        assert output.shape == hidden_states.shape
        assert mx.all(mx.isfinite(output))
    
    def test_attention_with_long_sequences(self):
        """Test attention with longer sequences."""
        config = create_bert_config(
            hidden_size=256,
            num_attention_heads=8,
            num_hidden_layers=1  # Just testing attention
        )
        attention = BertSelfAttention(config)
        
        batch_size, seq_length = 2, 512  # Long sequence
        hidden_states = mx.random.normal((batch_size, seq_length, config.hidden_size))
        
        output = attention(hidden_states)
        if isinstance(output, tuple):
            output = output[0]
        
        assert output.shape == (batch_size, seq_length, config.hidden_size)


@pytest.mark.slow
class TestAttentionPerformance:
    """Performance tests for attention layers."""
    
    def test_attention_scaling(self):
        """Test attention computation scales correctly."""
        # Test that attention computation scales as O(n^2) with sequence length
        config = create_small_bert_config()
        attention = BertSelfAttention(config)
        
        import time
        times = []
        seq_lengths = [32, 64, 128, 256]
        
        for seq_length in seq_lengths:
            hidden_states = mx.random.normal((1, seq_length, config.hidden_size))
            
            start = time.time()
            output = attention(hidden_states)
            mx.eval(output)
            end = time.time()
            
            times.append(end - start)
        
        # Check that time roughly quadruples when sequence length doubles
        # (allowing for some variance)
        for i in range(1, len(times)):
            ratio = times[i] / times[i-1]
            # Should be roughly 4x (2^2) but allow 2x to 8x range
            assert 2.0 < ratio < 8.0
    
    def test_memory_efficiency(self, memory_profiler):
        """Test memory usage of attention."""
        config = create_bert_config()
        attention = BertSelfAttention(config)
        
        batch_size, seq_length = 8, 256
        hidden_states = mx.random.normal((batch_size, seq_length, config.hidden_size))
        
        with memory_profiler() as prof:
            output = attention(hidden_states)
            if isinstance(output, tuple):
                output = output[0]
            mx.eval(output)
        
        memory_used = prof.get_memory_used()
        # Attention should use reasonable memory even for larger sequences
        assert memory_used < 2_000_000_000  # Less than 2GB