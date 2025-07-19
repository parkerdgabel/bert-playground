"""Tests for BERT core model implementations."""

import pytest
import mlx.core as mx
import tempfile
from pathlib import Path

from models.bert import BertConfig, BertCore, ModernBertCore
from models.bert.modernbert_config import ModernBertConfig
from tests.models.fixtures.configs import (
    create_bert_config,
    create_small_bert_config,
    create_modernbert_config,
    create_small_modernbert_config,
)
from tests.models.fixtures.data import create_test_batch
from tests.models.fixtures.utils import (
    compare_model_parameters,
    check_gradient_flow,
    save_and_load_model,
)


class TestBertCore:
    """Test BertCore model implementation."""
    
    def test_initialization(self):
        """Test BertCore initialization with various configurations."""
        # Standard config
        config = create_bert_config()
        bert = BertCore(config)
        
        assert bert.config.hidden_size == 768
        assert bert.config.num_hidden_layers == 12
        assert bert.config.num_attention_heads == 12
        assert bert.config.vocab_size == 30522
        
        # Small config
        small_config = create_small_bert_config()
        small_bert = BertCore(small_config)
        
        assert small_bert.config.hidden_size == 128
        assert small_bert.config.num_hidden_layers == 2
    
    def test_forward_pass(self, create_test_batch):
        """Test BertCore forward pass."""
        config = create_small_bert_config()
        bert = BertCore(config)
        
        # Create test batch
        batch = create_test_batch(
            batch_size=4,
            seq_length=32,
            vocab_size=config.vocab_size
        )
        
        # Forward pass
        outputs = bert(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"]
        )
        
        # Check outputs
        assert hasattr(outputs, 'last_hidden_state')
        assert outputs.last_hidden_state.shape == (4, 32, config.hidden_size)
        
        # Check pooler output if available
        if hasattr(outputs, 'pooler_output'):
            assert outputs.pooler_output.shape == (4, config.hidden_size)
    
    def test_forward_without_attention_mask(self, create_test_batch):
        """Test forward pass without attention mask."""
        config = create_small_bert_config()
        bert = BertCore(config)
        
        batch = create_test_batch(
            batch_size=2,
            seq_length=16,
            vocab_size=config.vocab_size
        )
        
        # Forward pass without attention mask
        outputs = bert(input_ids=batch["input_ids"])
        
        assert outputs.last_hidden_state.shape == (2, 16, config.hidden_size)
    
    def test_gradient_flow(self, create_test_batch, check_gradients):
        """Test gradient flow through BertCore."""
        config = create_small_bert_config()
        bert = BertCore(config)
        
        batch = create_test_batch(
            batch_size=2,
            seq_length=16,
            vocab_size=config.vocab_size
        )
        
        def loss_fn(model, batch):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            # Simple loss for gradient checking
            return mx.mean(outputs.last_hidden_state)
        
        # Check gradients
        result = check_gradients(bert, loss_fn, batch)
        assert result is True
    
    def test_save_load(self, tmp_model_dir):
        """Test saving and loading BertCore."""
        config = create_small_bert_config()
        bert = BertCore(config)
        
        # Save model
        save_path = tmp_model_dir / "bert_core"
        bert.save_pretrained(str(save_path))
        
        # Check saved files exist
        assert (save_path / "config.json").exists()
        assert (save_path / "model.safetensors").exists() or (save_path / "pytorch_model.bin").exists()
        
        # Load model
        loaded_bert = BertCore.from_pretrained(str(save_path))
        
        # Compare configs
        assert loaded_bert.config.hidden_size == bert.config.hidden_size
        assert loaded_bert.config.num_hidden_layers == bert.config.num_hidden_layers
        
        # Test loaded model works
        batch = create_test_batch(
            batch_size=2,
            seq_length=16,
            vocab_size=config.vocab_size
        )
        
        outputs = loaded_bert(input_ids=batch["input_ids"])
        assert outputs.last_hidden_state.shape == (2, 16, config.hidden_size)
    
    @pytest.mark.parametrize("num_layers", [1, 6, 12])
    def test_different_layer_counts(self, num_layers):
        """Test BertCore with different numbers of layers."""
        config = create_bert_config(num_hidden_layers=num_layers)
        bert = BertCore(config)
        
        batch_size, seq_length = 2, 32
        input_ids = mx.random.randint(0, config.vocab_size, (batch_size, seq_length))
        
        outputs = bert(input_ids=input_ids)
        assert outputs.last_hidden_state.shape == (batch_size, seq_length, config.hidden_size)
    
    @pytest.mark.parametrize("hidden_size,num_heads", [(128, 4), (256, 8), (768, 12)])
    def test_different_hidden_sizes(self, hidden_size, num_heads):
        """Test BertCore with different hidden sizes and attention heads."""
        config = create_bert_config(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_hidden_layers=2  # Small for faster tests
        )
        bert = BertCore(config)
        
        batch_size, seq_length = 2, 16
        input_ids = mx.random.randint(0, config.vocab_size, (batch_size, seq_length))
        
        outputs = bert(input_ids=input_ids)
        assert outputs.last_hidden_state.shape == (batch_size, seq_length, hidden_size)


class TestModernBertCore:
    """Test ModernBertCore model implementation."""
    
    def test_initialization(self):
        """Test ModernBertCore initialization."""
        config = create_modernbert_config()
        bert = ModernBertCore(config)
        
        assert bert.config.hidden_size == 768
        assert bert.config.num_hidden_layers == 12
        assert bert.config.max_position_embeddings == 8192  # Longer context
        assert hasattr(bert.config, 'rope_base')  # RoPE specific
    
    def test_forward_pass(self, create_test_batch):
        """Test ModernBertCore forward pass."""
        config = create_small_modernbert_config()
        bert = ModernBertCore(config)
        
        batch = create_test_batch(
            batch_size=4,
            seq_length=64,
            vocab_size=config.vocab_size
        )
        
        outputs = bert(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        
        assert hasattr(outputs, 'last_hidden_state')
        assert outputs.last_hidden_state.shape == (4, 64, config.hidden_size)
    
    def test_long_context(self):
        """Test ModernBertCore with long context."""
        config = create_modernbert_config(
            max_position_embeddings=512,
            num_hidden_layers=2  # Small for faster test
        )
        bert = ModernBertCore(config)
        
        # Test with longer sequence
        batch_size, seq_length = 2, 256
        input_ids = mx.random.randint(0, config.vocab_size, (batch_size, seq_length))
        
        outputs = bert(input_ids=input_ids)
        assert outputs.last_hidden_state.shape == (batch_size, seq_length, config.hidden_size)
    
    def test_rope_embeddings(self):
        """Test that RoPE embeddings are properly configured."""
        config = create_modernbert_config(rope_base=10000.0)
        bert = ModernBertCore(config)
        
        # RoPE should be integrated in attention layers
        assert hasattr(bert.config, 'rope_base')
        assert bert.config.rope_base == 10000.0
    
    def test_no_bias_attention(self):
        """Test ModernBERT with no attention bias."""
        config = create_modernbert_config(attention_bias=False)
        bert = ModernBertCore(config)
        
        assert bert.config.attention_bias is False
        
        # Forward pass should still work
        batch_size, seq_length = 2, 32
        input_ids = mx.random.randint(0, config.vocab_size, (batch_size, seq_length))
        
        outputs = bert(input_ids=input_ids)
        assert outputs.last_hidden_state.shape == (batch_size, seq_length, config.hidden_size)
    
    def test_save_load(self, tmp_model_dir):
        """Test saving and loading ModernBertCore."""
        config = create_small_modernbert_config()
        bert = ModernBertCore(config)
        
        # Save model
        save_path = tmp_model_dir / "modernbert_core"
        bert.save_pretrained(str(save_path))
        
        # Load model
        loaded_bert = ModernBertCore.from_pretrained(str(save_path))
        
        # Test loaded model
        batch_size, seq_length = 2, 32
        input_ids = mx.random.randint(0, config.vocab_size, (batch_size, seq_length))
        
        outputs = loaded_bert(input_ids=input_ids)
        assert outputs.last_hidden_state.shape == (batch_size, seq_length, config.hidden_size)
    
    @pytest.mark.parametrize("sliding_window", [None, 128, 256])
    def test_sliding_window_attention(self, sliding_window):
        """Test ModernBERT with sliding window attention."""
        config = create_modernbert_config(
            sliding_window=sliding_window,
            num_hidden_layers=2
        )
        bert = ModernBertCore(config)
        
        batch_size, seq_length = 2, 128
        input_ids = mx.random.randint(0, config.vocab_size, (batch_size, seq_length))
        
        outputs = bert(input_ids=input_ids)
        assert outputs.last_hidden_state.shape == (batch_size, seq_length, config.hidden_size)


class TestBertCoreComparison:
    """Test comparisons between BERT variants."""
    
    def test_bert_vs_modernbert_outputs(self, create_test_batch):
        """Compare outputs of BERT and ModernBERT with similar configs."""
        # Create similar configs
        hidden_size = 128
        num_layers = 2
        num_heads = 4
        
        bert_config = create_bert_config(
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            vocab_size=1000
        )
        
        modernbert_config = create_modernbert_config(
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            vocab_size=1000
        )
        
        bert = BertCore(bert_config)
        modernbert = ModernBertCore(modernbert_config)
        
        # Same input
        batch = create_test_batch(
            batch_size=2,
            seq_length=32,
            vocab_size=1000
        )
        
        bert_outputs = bert(input_ids=batch["input_ids"])
        modernbert_outputs = modernbert(input_ids=batch["input_ids"])
        
        # Should have same output shape
        assert bert_outputs.last_hidden_state.shape == modernbert_outputs.last_hidden_state.shape
        
        # Values will be different due to different architectures
        # Just ensure both produce valid outputs
        assert mx.all(mx.isfinite(bert_outputs.last_hidden_state))
        assert mx.all(mx.isfinite(modernbert_outputs.last_hidden_state))


@pytest.mark.slow
class TestBertCoreMemory:
    """Memory and performance tests for BERT models."""
    
    def test_memory_usage(self, memory_profiler):
        """Test memory usage of BERT models."""
        config = create_small_bert_config()
        bert = BertCore(config)
        
        batch = create_test_batch(
            batch_size=8,
            seq_length=128,
            vocab_size=config.vocab_size
        )
        
        with memory_profiler() as prof:
            outputs = bert(input_ids=batch["input_ids"])
            mx.eval(outputs.last_hidden_state)
        
        memory_used = prof.get_memory_used()
        # Ensure reasonable memory usage (less than 500MB for small model)
        assert memory_used < 500_000_000
    
    def test_large_batch_handling(self):
        """Test handling of large batches."""
        config = create_small_bert_config()
        bert = BertCore(config)
        
        # Large batch
        batch_size, seq_length = 64, 128
        input_ids = mx.random.randint(0, config.vocab_size, (batch_size, seq_length))
        
        outputs = bert(input_ids=input_ids)
        assert outputs.last_hidden_state.shape == (batch_size, seq_length, config.hidden_size)