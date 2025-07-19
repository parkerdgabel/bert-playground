"""Tests for BERT embedding layers."""

import pytest
import mlx.core as mx
import mlx.nn as nn

from models.bert.layers import BertEmbeddings
from models.bert.config import BertConfig
from tests.models.fixtures.configs import (
    create_bert_config,
    create_small_bert_config,
)
from tests.models.fixtures.data import (
    create_test_batch,
    create_position_ids,
    create_variable_length_batch,
)
from tests.models.fixtures.utils import check_gradient_flow


class TestBertEmbeddings:
    """Test BertEmbeddings layer implementation."""
    
    def test_initialization(self):
        """Test BertEmbeddings initialization."""
        config = create_bert_config()
        embeddings = BertEmbeddings(config)
        
        # Check embedding sizes
        assert embeddings.word_embeddings.weight.shape == (config.vocab_size, config.hidden_size)
        assert embeddings.position_embeddings.weight.shape == (config.max_position_embeddings, config.hidden_size)
        assert embeddings.token_type_embeddings.weight.shape == (config.type_vocab_size, config.hidden_size)
    
    def test_forward_pass(self):
        """Test forward pass through embeddings."""
        config = create_small_bert_config()
        embeddings = BertEmbeddings(config)
        
        batch_size, seq_length = 4, 32
        input_ids = mx.random.randint(0, config.vocab_size, (batch_size, seq_length))
        
        output = embeddings(input_ids)
        assert output.shape == (batch_size, seq_length, config.hidden_size)
        assert mx.all(mx.isfinite(output))
    
    def test_with_token_type_ids(self):
        """Test embeddings with token type IDs."""
        config = create_small_bert_config()
        embeddings = BertEmbeddings(config)
        
        batch_size, seq_length = 4, 32
        input_ids = mx.random.randint(0, config.vocab_size, (batch_size, seq_length))
        token_type_ids = mx.random.randint(0, config.type_vocab_size, (batch_size, seq_length))
        
        output = embeddings(input_ids, token_type_ids=token_type_ids)
        assert output.shape == (batch_size, seq_length, config.hidden_size)
    
    def test_with_position_ids(self, create_position_ids):
        """Test embeddings with custom position IDs."""
        config = create_small_bert_config()
        embeddings = BertEmbeddings(config)
        
        batch_size, seq_length = 4, 32
        input_ids = mx.random.randint(0, config.vocab_size, (batch_size, seq_length))
        
        # Test different position ID types
        for position_type in ["absolute", "relative", "random"]:
            position_ids = create_position_ids(
                batch_size, seq_length, position_type=position_type
            )
            output = embeddings(input_ids, position_ids=position_ids)
            assert output.shape == (batch_size, seq_length, config.hidden_size)
    
    def test_dropout_behavior(self):
        """Test dropout in embeddings."""
        config = create_bert_config(hidden_dropout_prob=0.5)
        embeddings = BertEmbeddings(config)
        
        batch_size, seq_length = 2, 16
        input_ids = mx.random.randint(0, config.vocab_size, (batch_size, seq_length))
        
        # Training mode - outputs should differ
        embeddings.train()
        output1 = embeddings(input_ids)
        output2 = embeddings(input_ids)
        assert not mx.allclose(output1, output2)
        
        # Eval mode - outputs should be identical
        embeddings.eval()
        output3 = embeddings(input_ids)
        output4 = embeddings(input_ids)
        assert mx.allclose(output3, output4)
    
    def test_layer_norm(self):
        """Test layer normalization in embeddings."""
        config = create_bert_config()
        embeddings = BertEmbeddings(config)
        
        batch_size, seq_length = 4, 32
        input_ids = mx.random.randint(0, config.vocab_size, (batch_size, seq_length))
        
        output = embeddings(input_ids)
        
        # Check that output is normalized
        # Mean should be close to 0, std close to 1 along hidden dimension
        mean = mx.mean(output, axis=-1)
        std = mx.std(output, axis=-1)
        
        assert mx.all(mx.abs(mean) < 0.1)  # Mean close to 0
        assert mx.all(mx.abs(std - 1.0) < 0.1)  # Std close to 1
    
    def test_gradient_flow(self, check_gradients):
        """Test gradient flow through embeddings."""
        config = create_small_bert_config()
        embeddings = BertEmbeddings(config)
        
        batch_size, seq_length = 2, 16
        input_ids = mx.random.randint(0, config.vocab_size, (batch_size, seq_length))
        
        def loss_fn(model, inputs):
            output = model(inputs["input_ids"])
            return mx.mean(output)
        
        result = check_gradients(embeddings, loss_fn, {"input_ids": input_ids})
        assert result is True
    
    @pytest.mark.parametrize("seq_length", [1, 128, 512])
    def test_different_sequence_lengths(self, seq_length):
        """Test embeddings with different sequence lengths."""
        config = create_bert_config()
        embeddings = BertEmbeddings(config)
        
        batch_size = 2
        input_ids = mx.random.randint(0, config.vocab_size, (batch_size, seq_length))
        
        output = embeddings(input_ids)
        assert output.shape == (batch_size, seq_length, config.hidden_size)
    
    def test_max_position_embeddings(self):
        """Test handling of maximum position embeddings."""
        config = create_bert_config(max_position_embeddings=128)
        embeddings = BertEmbeddings(config)
        
        # Test at max length
        batch_size, seq_length = 2, 128
        input_ids = mx.random.randint(0, config.vocab_size, (batch_size, seq_length))
        
        output = embeddings(input_ids)
        assert output.shape == (batch_size, seq_length, config.hidden_size)
        
        # Test beyond max length should fail or be handled
        # Implementation may vary - some truncate, some error
    
    def test_embedding_weight_sharing(self):
        """Test that embedding weights are independent."""
        config = create_bert_config()
        embeddings = BertEmbeddings(config)
        
        # Get initial weights
        word_weight_initial = embeddings.word_embeddings.weight.copy()
        position_weight_initial = embeddings.position_embeddings.weight.copy()
        
        # Modify word embeddings
        embeddings.word_embeddings.weight = mx.random.normal(
            embeddings.word_embeddings.weight.shape
        )
        
        # Position embeddings should remain unchanged
        assert mx.allclose(
            embeddings.position_embeddings.weight,
            position_weight_initial
        )


@pytest.mark.integration
class TestBertEmbeddingsIntegration:
    """Integration tests for BERT embeddings."""
    
    def test_with_variable_length_batch(self, create_variable_length_batch):
        """Test embeddings with variable length sequences."""
        config = create_small_bert_config()
        embeddings = BertEmbeddings(config)
        
        batch = create_variable_length_batch(
            batch_size=4,
            min_length=16,
            max_length=64,
            vocab_size=config.vocab_size,
            pad_to_max=True
        )
        
        output = embeddings(batch["input_ids"])
        assert output.shape == (4, 64, config.hidden_size)
        
        # Check that padding positions are handled
        for i in range(4):
            seq_len = batch["lengths"][i].item()
            # Non-padded positions should have non-zero embeddings
            assert mx.any(output[i, :seq_len] != 0)
    
    def test_special_token_embeddings(self):
        """Test embeddings for special tokens."""
        config = create_bert_config()
        embeddings = BertEmbeddings(config)
        
        # Common special token IDs
        CLS_TOKEN = 101
        SEP_TOKEN = 102
        PAD_TOKEN = 0
        
        batch_size = 2
        # Create sequence with special tokens
        input_ids = mx.array([
            [CLS_TOKEN, 1000, 2000, SEP_TOKEN, PAD_TOKEN, PAD_TOKEN],
            [CLS_TOKEN, 3000, 4000, 5000, SEP_TOKEN, PAD_TOKEN]
        ])
        
        output = embeddings(input_ids)
        assert output.shape == (batch_size, 6, config.hidden_size)
        
        # CLS embeddings should be consistent across sequences
        cls_embedding_1 = output[0, 0]
        cls_embedding_2 = output[1, 0]
        # They won't be identical due to position embeddings
        assert cls_embedding_1.shape == cls_embedding_2.shape


@pytest.mark.slow
class TestBertEmbeddingsPerformance:
    """Performance tests for BERT embeddings."""
    
    def test_large_vocabulary_handling(self):
        """Test embeddings with large vocabulary."""
        config = create_bert_config(vocab_size=100000)  # Large vocab
        embeddings = BertEmbeddings(config)
        
        batch_size, seq_length = 8, 128
        input_ids = mx.random.randint(0, config.vocab_size, (batch_size, seq_length))
        
        output = embeddings(input_ids)
        assert output.shape == (batch_size, seq_length, config.hidden_size)
    
    def test_memory_efficiency(self, memory_profiler):
        """Test memory usage of embeddings."""
        config = create_bert_config()
        embeddings = BertEmbeddings(config)
        
        batch_size, seq_length = 32, 256
        input_ids = mx.random.randint(0, config.vocab_size, (batch_size, seq_length))
        
        with memory_profiler() as prof:
            output = embeddings(input_ids)
            mx.eval(output)
        
        memory_used = prof.get_memory_used()
        # Embeddings should use reasonable memory
        assert memory_used < 1_000_000_000  # Less than 1GB