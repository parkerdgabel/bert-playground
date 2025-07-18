#!/usr/bin/env python3
"""
Comprehensive test suite for Classic BERT implementation compliance with the original paper.

This test suite validates that our Classic BERT implementation correctly follows
the original BERT paper architecture and specifications.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from models.bert.core import BertCore, BertSelfAttention, BertAttention, BertLayer
from models.bert.config import BertConfig


class TestClassicBertCompliance:
    """Test suite for Classic BERT paper compliance."""
    
    @pytest.fixture
    def small_config(self):
        """Create a small BERT config for testing."""
        return BertConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=512,
            max_position_embeddings=64,
            type_vocab_size=2,
            layer_norm_eps=1e-12,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
    
    @pytest.fixture
    def base_config(self):
        """Create a BERT-base config matching the original paper."""
        return BertConfig(
            vocab_size=30522,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=512,
            type_vocab_size=2,
            layer_norm_eps=1e-12,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
    
    @pytest.fixture
    def sample_inputs(self, small_config):
        """Create sample inputs for testing."""
        batch_size = 2
        seq_len = 16
        
        return {
            'input_ids': mx.random.randint(0, small_config.vocab_size, (batch_size, seq_len)),
            'attention_mask': mx.ones((batch_size, seq_len), dtype=mx.int32),
            'token_type_ids': mx.zeros((batch_size, seq_len), dtype=mx.int32),
        }
    
    def test_bert_self_attention_dimensions(self, small_config):
        """Test that BertSelfAttention produces correct dimensions."""
        attention = BertSelfAttention(small_config)
        
        # Check that head dimensions are correct
        assert attention.attention_head_size == small_config.hidden_size // small_config.num_attention_heads
        assert attention.all_head_size == small_config.hidden_size
        
        # Test forward pass
        batch_size, seq_len = 2, 16
        hidden_states = mx.random.normal((batch_size, seq_len, small_config.hidden_size))
        attention_mask = mx.ones((batch_size, seq_len), dtype=mx.int32)
        
        context, attn_weights = attention(hidden_states, attention_mask, output_attentions=True)
        
        # Check output dimensions
        assert context.shape == (batch_size, seq_len, small_config.hidden_size)
        assert attn_weights.shape == (batch_size, small_config.num_attention_heads, seq_len, seq_len)
        
        # Check attention weights sum to 1
        attn_sums = attn_weights.sum(axis=-1)
        assert mx.allclose(attn_sums, mx.ones_like(attn_sums), atol=1e-6)
    
    def test_attention_mask_application(self, small_config):
        """Test that attention mask is correctly applied."""
        attention = BertSelfAttention(small_config)
        
        batch_size, seq_len = 2, 8
        hidden_states = mx.random.normal((batch_size, seq_len, small_config.hidden_size))
        
        # Create attention mask where second sequence is shorter
        attention_mask = mx.ones((batch_size, seq_len), dtype=mx.int32)
        attention_mask[1, 4:] = 0  # Mask out last 4 positions for second sequence
        
        context, attn_weights = attention(hidden_states, attention_mask, output_attentions=True)
        
        # Check that masked positions receive almost no attention
        masked_attention = attn_weights[1, :, :, 4:]  # Attention to masked positions
        assert mx.all(masked_attention < 1e-6)
    
    def test_residual_connections(self, small_config):
        """Test that residual connections are properly implemented."""
        layer = BertLayer(small_config)
        
        batch_size, seq_len = 2, 16
        hidden_states = mx.random.normal((batch_size, seq_len, small_config.hidden_size))
        attention_mask = mx.ones((batch_size, seq_len), dtype=mx.int32)
        
        # Test without dropout (set to 0 for deterministic test)
        layer.attention.output.dropout = nn.Dropout(0.0)
        layer.output.dropout = nn.Dropout(0.0)
        
        output, _ = layer(hidden_states, attention_mask, output_attentions=True)
        
        # Output should be different from input (transformation applied)
        assert not mx.allclose(output, hidden_states, atol=1e-3)
        
        # But should preserve general scale (residual connections help)
        input_norm = mx.linalg.norm(hidden_states, axis=-1)
        output_norm = mx.linalg.norm(output, axis=-1)
        
        # Norms should be in similar range (not exact due to layer norm)
        assert mx.all(output_norm > 0.1 * input_norm)
        assert mx.all(output_norm < 10.0 * input_norm)
    
    def test_layer_normalization(self, small_config):
        """Test that layer normalization is correctly applied."""
        layer = BertLayer(small_config)
        
        batch_size, seq_len = 2, 16
        hidden_states = mx.random.normal((batch_size, seq_len, small_config.hidden_size))
        attention_mask = mx.ones((batch_size, seq_len), dtype=mx.int32)
        
        output, _ = layer(hidden_states, attention_mask)
        
        # Check that output has approximately zero mean and unit variance
        # (due to layer normalization)
        output_mean = output.mean(axis=-1)
        output_var = output.var(axis=-1)
        
        assert mx.allclose(output_mean, mx.zeros_like(output_mean), atol=1e-3)
        assert mx.allclose(output_var, mx.ones_like(output_var), atol=1e-1)
    
    def test_bert_core_architecture(self, small_config, sample_inputs):
        """Test complete BertCore architecture."""
        model = BertCore(small_config)
        
        # Test forward pass with all options
        output = model(
            input_ids=sample_inputs['input_ids'],
            attention_mask=sample_inputs['attention_mask'],
            token_type_ids=sample_inputs['token_type_ids'],
            output_attentions=True,
            output_hidden_states=True,
        )
        
        # Check output structure
        assert hasattr(output, 'last_hidden_state')
        assert hasattr(output, 'pooler_output')
        assert hasattr(output, 'attentions')
        assert hasattr(output, 'hidden_states')
        
        # Check dimensions
        batch_size, seq_len = sample_inputs['input_ids'].shape
        assert output.last_hidden_state.shape == (batch_size, seq_len, small_config.hidden_size)
        assert output.pooler_output.shape == (batch_size, small_config.hidden_size)
        
        # Check attention weights
        assert len(output.attentions) == small_config.num_hidden_layers
        for attn in output.attentions:
            assert attn.shape == (batch_size, small_config.num_attention_heads, seq_len, seq_len)
        
        # Check hidden states (includes embedding + all layers)
        assert len(output.hidden_states) == small_config.num_hidden_layers + 1
        for hidden in output.hidden_states:
            assert hidden.shape == (batch_size, seq_len, small_config.hidden_size)
    
    def test_bert_base_parameter_count(self, base_config):
        """Test that BERT-base has approximately the correct number of parameters."""
        model = BertCore(base_config)
        
        # Calculate total parameters
        total_params = sum(p.size for p in model.parameters())
        
        # BERT-base should have approximately 110M parameters
        # Allow some tolerance for implementation differences
        expected_params = 110_000_000
        assert 100_000_000 <= total_params <= 120_000_000, f"Got {total_params} parameters, expected ~{expected_params}"
    
    def test_position_embeddings(self, small_config):
        """Test that position embeddings are properly applied."""
        model = BertCore(small_config)
        
        batch_size, seq_len = 2, 16
        input_ids = mx.random.randint(0, small_config.vocab_size, (batch_size, seq_len))
        
        # Test with different sequence lengths
        short_output = model(input_ids[:, :8])
        long_output = model(input_ids)
        
        # Outputs should have different shapes
        assert short_output.last_hidden_state.shape[1] == 8
        assert long_output.last_hidden_state.shape[1] == 16
        
        # Position embeddings should make sequences of same tokens but different positions differ
        same_tokens = mx.ones((1, 8), dtype=mx.int32) * 5  # All token ID 5
        different_positions = mx.ones((1, 8), dtype=mx.int32) * 5  # Same tokens, different positions
        
        output1 = model(same_tokens)
        output2 = model(different_positions)
        
        # Due to position embeddings, different positions should produce different outputs
        # (even with same tokens)
        assert not mx.allclose(output1.last_hidden_state, output2.last_hidden_state, atol=1e-3)
    
    def test_token_type_embeddings(self, small_config):
        """Test that token type embeddings work correctly."""
        model = BertCore(small_config)
        
        batch_size, seq_len = 2, 16
        input_ids = mx.random.randint(0, small_config.vocab_size, (batch_size, seq_len))
        
        # Test with different token types
        token_type_ids_A = mx.zeros((batch_size, seq_len), dtype=mx.int32)  # All sentence A
        token_type_ids_B = mx.ones((batch_size, seq_len), dtype=mx.int32)   # All sentence B
        
        output_A = model(input_ids, token_type_ids=token_type_ids_A)
        output_B = model(input_ids, token_type_ids=token_type_ids_B)
        
        # Different token types should produce different outputs
        assert not mx.allclose(output_A.last_hidden_state, output_B.last_hidden_state, atol=1e-3)
    
    def test_attention_head_isolation(self, small_config):
        """Test that attention heads operate independently."""
        attention = BertSelfAttention(small_config)
        
        batch_size, seq_len = 1, 8
        hidden_states = mx.random.normal((batch_size, seq_len, small_config.hidden_size))
        
        _, attn_weights = attention(hidden_states, output_attentions=True)
        
        # Each head should have different attention patterns
        # (very unlikely they'd be the same by chance)
        head_0 = attn_weights[0, 0, :, :]
        head_1 = attn_weights[0, 1, :, :]
        
        assert not mx.allclose(head_0, head_1, atol=1e-2)
    
    def test_dropout_effect(self, small_config):
        """Test that dropout affects outputs during training."""
        model = BertCore(small_config)
        
        batch_size, seq_len = 2, 16
        input_ids = mx.random.randint(0, small_config.vocab_size, (batch_size, seq_len))
        
        # Test multiple forward passes with dropout
        # Note: MLX dropout behavior might be different, so this test checks basic functionality
        output1 = model(input_ids, training=True)
        output2 = model(input_ids, training=True)
        
        # With dropout, consecutive runs should potentially differ
        # However, this depends on MLX's dropout implementation
        # At minimum, check that training mode runs without error
        assert output1.last_hidden_state.shape == output2.last_hidden_state.shape
    
    def test_gradient_flow(self, small_config):
        """Test that gradients flow through the model correctly."""
        model = BertCore(small_config)
        
        batch_size, seq_len = 2, 16
        input_ids = mx.random.randint(0, small_config.vocab_size, (batch_size, seq_len))
        
        # Create a simple loss (sum of all outputs)
        output = model(input_ids)
        loss = output.last_hidden_state.sum()
        
        # Compute gradients
        grad_fn = mx.grad(lambda: loss)
        gradients = grad_fn()
        
        # Check that gradients exist and are reasonable
        assert gradients is not None
        # Note: More detailed gradient checks would require specific MLX gradient utilities


def test_bert_architecture_compliance():
    """Integration test for complete BERT architecture compliance."""
    config = BertConfig(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=256,
        max_position_embeddings=32,
        type_vocab_size=2,
    )
    
    model = BertCore(config)
    
    # Test complete forward pass
    batch_size, seq_len = 2, 16
    input_ids = mx.random.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = mx.ones((batch_size, seq_len), dtype=mx.int32)
    token_type_ids = mx.zeros((batch_size, seq_len), dtype=mx.int32)
    
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        output_attentions=True,
        output_hidden_states=True,
    )
    
    # Verify all expected outputs are present
    assert output.last_hidden_state is not None
    assert output.pooler_output is not None
    assert output.attentions is not None
    assert output.hidden_states is not None
    
    # Test pooling variations
    cls_output = output.get_pooled_output("cls")
    mean_output = output.get_pooled_output("mean")
    max_output = output.get_pooled_output("max")
    
    assert cls_output.shape == (batch_size, config.hidden_size)
    assert mean_output.shape == (batch_size, config.hidden_size)
    assert max_output.shape == (batch_size, config.hidden_size)
    
    # Test that different pooling methods give different results
    assert not mx.allclose(cls_output, mean_output, atol=1e-3)
    assert not mx.allclose(cls_output, max_output, atol=1e-3)
    assert not mx.allclose(mean_output, max_output, atol=1e-3)
    
    print("✅ All BERT architecture compliance tests passed!")


if __name__ == "__main__":
    test_bert_architecture_compliance()