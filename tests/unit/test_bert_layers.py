"""Tests for BERT layer components that are actually available."""

import pytest
import mlx.core as mx
import mlx.nn as nn

from models.bert.layers import (
    BertEmbeddings,
    BertSelfAttention,
    BertSelfOutput,
    BertIntermediate,
    BertOutput,
)
from models.bert import BertConfig


class TestBertLayerComponents:
    """Test individual BERT layer components."""
    
    @pytest.fixture
    def config(self):
        return BertConfig(
            hidden_size=768,
            num_attention_heads=12,
            intermediate_size=3072,
            vocab_size=50265,
        )
    
    def test_bert_embeddings(self, config):
        """Test BertEmbeddings layer."""
        embeddings = BertEmbeddings(config)
        
        # Test forward pass
        batch_size, seq_length = 2, 10
        input_ids = mx.random.randint(0, config.vocab_size, (batch_size, seq_length))
        
        output = embeddings(input_ids)
        assert output.shape == (batch_size, seq_length, config.hidden_size)
        
    def test_bert_self_attention(self, config):
        """Test BertSelfAttention layer."""
        attention = BertSelfAttention(config)
        
        # Test forward pass
        batch_size, seq_length = 2, 10
        hidden_states = mx.random.normal((batch_size, seq_length, config.hidden_size))
        attention_mask = mx.ones((batch_size, seq_length))
        
        output = attention(hidden_states, attention_mask)
        assert output[0].shape == (batch_size, seq_length, config.hidden_size)
        
    def test_bert_self_output(self, config):
        """Test BertSelfOutput layer."""
        self_output = BertSelfOutput(config)
        
        # Test forward pass
        batch_size, seq_length = 2, 10
        hidden_states = mx.random.normal((batch_size, seq_length, config.hidden_size))
        input_tensor = mx.random.normal((batch_size, seq_length, config.hidden_size))
        
        output = self_output(hidden_states, input_tensor)
        assert output.shape == (batch_size, seq_length, config.hidden_size)
        
    def test_bert_intermediate(self, config):
        """Test BertIntermediate layer."""
        intermediate = BertIntermediate(config)
        
        # Test forward pass
        batch_size, seq_length = 2, 10
        hidden_states = mx.random.normal((batch_size, seq_length, config.hidden_size))
        
        output = intermediate(hidden_states)
        assert output.shape == (batch_size, seq_length, config.intermediate_size)
        
    def test_bert_output(self, config):
        """Test BertOutput layer."""
        bert_output = BertOutput(config)
        
        # Test forward pass
        batch_size, seq_length = 2, 10
        hidden_states = mx.random.normal((batch_size, seq_length, config.intermediate_size))
        input_tensor = mx.random.normal((batch_size, seq_length, config.hidden_size))
        
        output = bert_output(hidden_states, input_tensor)
        assert output.shape == (batch_size, seq_length, config.hidden_size)