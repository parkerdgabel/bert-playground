"""
Tests for MLX Embeddings Integration

Tests tokenizer parity, model loading, and embedding generation.
"""

import pytest
import mlx.core as mx
import numpy as np
from pathlib import Path

from embeddings.tokenizer_wrapper import TokenizerWrapper
from embeddings.mlx_adapter import MLXEmbeddingsAdapter
from embeddings.model_wrapper import MLXEmbeddingModel
from embeddings.config import MLXEmbeddingsConfig, get_default_config
from embeddings.migration import CheckpointMigrator, check_mlx_embeddings_compatibility


class TestTokenizerWrapper:
    """Test TokenizerWrapper functionality."""
    
    @pytest.mark.parametrize("backend", ["auto", "huggingface"])
    def test_tokenizer_initialization(self, backend):
        """Test tokenizer initialization with different backends."""
        tokenizer = TokenizerWrapper(
            model_name="bert-base-uncased",
            backend=backend
        )
        assert tokenizer is not None
        assert tokenizer.vocab_size > 0
    
    def test_single_text_encoding(self):
        """Test encoding single text."""
        tokenizer = TokenizerWrapper(backend="huggingface")
        text = "Hello, world!"
        
        # Test encoding
        encoded = tokenizer.encode(text)
        assert isinstance(encoded, mx.array)
        assert encoded.shape[0] > 0
        
        # Test decoding
        decoded = tokenizer.decode(encoded)
        assert isinstance(decoded, str)
    
    def test_batch_encoding(self):
        """Test batch encoding of multiple texts."""
        tokenizer = TokenizerWrapper(backend="huggingface")
        texts = ["Hello world", "How are you?", "MLX is great!"]
        
        # Batch encode
        encoded = tokenizer.batch_encode_plus(
            texts,
            padding=True,
            truncation=True,
            max_length=32
        )
        
        assert "input_ids" in encoded
        assert "attention_mask" in encoded
        assert isinstance(encoded["input_ids"], mx.array)
        assert encoded["input_ids"].shape[0] == len(texts)
    
    def test_special_tokens(self):
        """Test special token handling."""
        tokenizer = TokenizerWrapper(backend="huggingface")
        
        assert tokenizer.pad_token_id is not None
        assert tokenizer.vocab_size > 0


@pytest.mark.skipif(
    not MLXEmbeddingsAdapter(use_mlx_embeddings=False).is_available,
    reason="MLX embeddings not available"
)
class TestMLXEmbeddingsAdapter:
    """Test MLX embeddings adapter."""
    
    def test_adapter_initialization(self):
        """Test adapter initialization."""
        adapter = MLXEmbeddingsAdapter(
            model_name="bert-base-uncased",
            use_mlx_embeddings=False  # Use fallback for testing
        )
        assert adapter is not None
        assert not adapter.is_available  # Should be False with fallback
    
    def test_model_name_conversion(self):
        """Test HuggingFace to MLX model name conversion."""
        adapter = MLXEmbeddingsAdapter()
        
        # Test known mappings
        assert adapter._convert_to_mlx_model_name("bert-base-uncased") == "mlx-community/bert-base-uncased-4bit"
        assert adapter._convert_to_mlx_model_name("answerdotai/ModernBERT-base") == "mlx-community/answerdotai-ModernBERT-base-4bit"
        
        # Test unknown model (should return original)
        assert adapter._convert_to_mlx_model_name("unknown/model") == "unknown/model"


class TestMLXEmbeddingModel:
    """Test MLX embedding model wrapper."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = MLXEmbeddingModel(
            model_name="bert-base-uncased",
            num_labels=2,
            use_mlx_embeddings=False  # Use fallback for testing
        )
        assert model is not None
        assert model.num_labels == 2
        assert model.hidden_size == 768  # Default BERT hidden size
    
    def test_model_forward_pass(self):
        """Test model forward pass with dummy input."""
        model = MLXEmbeddingModel(
            num_labels=2,
            use_mlx_embeddings=False
        )
        
        # Create dummy input
        batch_size = 2
        seq_length = 10
        input_ids = mx.ones((batch_size, seq_length), dtype=mx.int32)
        
        # Forward pass
        output = model(input_ids)
        
        # Check output shape
        assert output.shape == (batch_size, 2)  # num_labels = 2
    
    def test_embedding_extraction(self):
        """Test embedding extraction without classification head."""
        model = MLXEmbeddingModel(
            num_labels=None,  # No classification head
            use_mlx_embeddings=False
        )
        
        # Create dummy input
        batch_size = 2
        seq_length = 10
        input_ids = mx.ones((batch_size, seq_length), dtype=mx.int32)
        
        # Forward pass
        embeddings = model(input_ids)
        
        # Check embedding shape
        assert embeddings.shape == (batch_size, model.hidden_size)


class TestConfiguration:
    """Test configuration utilities."""
    
    def test_mlx_embeddings_config(self):
        """Test MLXEmbeddingsConfig creation."""
        config = MLXEmbeddingsConfig(
            model_name="bert-base-uncased",
            hidden_size=768,
            num_hidden_layers=12
        )
        
        assert config.model_name == "bert-base-uncased"
        assert config.hidden_size == 768
        assert config.use_mlx_embeddings is True
    
    def test_config_from_model_name(self):
        """Test creating config from model name."""
        config = MLXEmbeddingsConfig.from_model_name("bert-large-uncased")
        
        assert config.hidden_size == 1024  # Large model
        assert config.num_hidden_layers == 24
    
    def test_get_default_config(self):
        """Test getting default configurations."""
        config = get_default_config("modernbert-base")
        
        assert config.model_name == "answerdotai/ModernBERT-base"
        assert config.hidden_size == 768
        assert config.max_position_embeddings == 8192


class TestMigration:
    """Test checkpoint migration utilities."""
    
    def test_compatibility_check(self):
        """Test model compatibility checking."""
        # Test known model
        info = check_mlx_embeddings_compatibility("bert-base-uncased")
        assert info["supported"] is True
        assert info["has_mlx_mapping"] is True
        
        # Test MLX model
        info = check_mlx_embeddings_compatibility("mlx-community/bert-base-4bit")
        assert info["supported"] is True
        
        # Test unknown model
        info = check_mlx_embeddings_compatibility("unknown/model")
        assert len(info["notes"]) > 0
    
    @pytest.fixture
    def mock_checkpoint(self, tmp_path):
        """Create a mock checkpoint for testing."""
        checkpoint_dir = tmp_path / "test_checkpoint"
        checkpoint_dir.mkdir()
        
        # Create mock config
        config = {
            "model_name": "bert-base-uncased",
            "hidden_size": 768,
            "num_labels": 2
        }
        
        import json
        with open(checkpoint_dir / "config.json", "w") as f:
            json.dump(config, f)
        
        # Create mock weights file
        (checkpoint_dir / "model.safetensors").touch()
        
        return checkpoint_dir
    
    def test_checkpoint_migration(self, mock_checkpoint):
        """Test checkpoint migration."""
        migrator = CheckpointMigrator(mock_checkpoint)
        
        # Test config migration
        migrated_config = migrator._migrate_config()
        assert migrated_config["use_mlx_embeddings"] is True
        assert migrated_config["tokenizer_backend"] == "mlx"
        assert "original_model_name" in migrated_config


@pytest.mark.integration
class TestIntegration:
    """Integration tests for the full pipeline."""
    
    def test_tokenizer_model_integration(self):
        """Test tokenizer and model integration."""
        # Create tokenizer
        tokenizer = TokenizerWrapper(
            model_name="bert-base-uncased",
            backend="huggingface"
        )
        
        # Create model
        model = MLXEmbeddingModel(
            model_name="bert-base-uncased",
            num_labels=2,
            use_mlx_embeddings=False
        )
        
        # Tokenize text
        texts = ["Hello world", "MLX is great"]
        encoded = tokenizer.batch_encode_plus(texts, padding=True)
        
        # Forward pass
        output = model(encoded["input_ids"], encoded["attention_mask"])
        
        # Check output
        assert output.shape == (len(texts), 2)
    
    def test_data_loader_integration(self):
        """Test integration with data loaders."""
        from data.mlx_dataloader import KaggleDataLoader
        
        # This would require a test CSV file
        # Skipping for now as it requires test data setup
        pass