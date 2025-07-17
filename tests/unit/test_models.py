"""Unit tests for model components."""

from unittest.mock import MagicMock, patch

import mlx.core as mx
import mlx.nn as nn
import pytest

from models.classification import TitanicClassifier
from models.factory import create_model
from models.modernbert import ModernBertModel, ModernBertConfig
from models.modernbert_cnn_hybrid import (
    CNNEnhancedModernBERT,
    CNNHybridConfig,
    MultiScaleConv1D,
)


class TestModernBertConfig:
    """Test ModernBertConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ModernBertConfig()
        
        assert config.vocab_size == 50265
        assert config.hidden_size == 768
        assert config.num_hidden_layers == 12
        assert config.num_attention_heads == 12
        assert config.hidden_dropout_prob == 0.1
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = ModernBertConfig(
            vocab_size=30000,
            hidden_size=512,
            num_hidden_layers=6,
        )
        
        assert config.vocab_size == 30000
        assert config.hidden_size == 512
        assert config.num_hidden_layers == 6


class TestModernBertModel:
    """Test ModernBertModel."""
    
    def test_initialization(self):
        """Test model initialization."""
        config = ModernBertConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
        )
        
        model = ModernBertModel(config)
        
        assert model.config == config
        assert hasattr(model, "embeddings")
        assert hasattr(model, "encoder")
        assert hasattr(model, "dropout")
    
    def test_forward_pass(self):
        """Test forward pass through model."""
        config = ModernBertConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
        )
        
        model = ModernBertModel(config)
        
        # Create dummy input
        batch_size, seq_length = 2, 10
        input_ids = mx.random.randint(0, 1000, (batch_size, seq_length))
        attention_mask = mx.ones((batch_size, seq_length))
        
        # Forward pass
        outputs = model(input_ids, attention_mask)
        
        assert outputs.shape == (batch_size, seq_length, config.hidden_size)
    
    def test_save_and_load(self, temp_dir):
        """Test model saving and loading."""
        config = ModernBertConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
        )
        
        model = ModernBertModel(config)
        
        # Save model
        save_path = temp_dir / "model"
        model.save_pretrained(str(save_path))
        
        # Check files exist
        assert (save_path / "config.json").exists()
        assert (save_path / "model.safetensors").exists()
        
        # Load model
        loaded_model = ModernBertModel.from_pretrained(str(save_path))
        
        assert loaded_model.config.vocab_size == config.vocab_size
        assert loaded_model.config.hidden_size == config.hidden_size


class TestTitanicClassifier:
    """Test TitanicClassifier wrapper."""
    
    def test_initialization(self, base_model):
        """Test classifier initialization."""
        classifier = TitanicClassifier(base_model)
        
        assert classifier.bert == base_model
        assert hasattr(classifier, "classifier")
        assert classifier.num_labels == 2
    
    def test_forward_pass(self, titanic_classifier):
        """Test forward pass with labels."""
        batch_size, seq_length = 2, 10
        input_ids = mx.random.randint(0, 50000, (batch_size, seq_length))
        attention_mask = mx.ones((batch_size, seq_length))
        labels = mx.array([0, 1])
        
        outputs = titanic_classifier(input_ids, attention_mask, labels)
        
        assert "loss" in outputs
        assert "logits" in outputs
        assert outputs["logits"].shape == (batch_size, 2)
    
    def test_forward_pass_without_labels(self, titanic_classifier):
        """Test forward pass without labels."""
        batch_size, seq_length = 2, 10
        input_ids = mx.random.randint(0, 50000, (batch_size, seq_length))
        attention_mask = mx.ones((batch_size, seq_length))
        
        outputs = titanic_classifier(input_ids, attention_mask)
        
        assert "logits" in outputs
        assert "loss" not in outputs
    
    def test_custom_loss_function(self, base_model):
        """Test classifier with custom loss function."""
        # Create classifier with custom loss
        classifier = TitanicClassifier(
            base_model,
            num_labels=2,
            loss_type="weighted_cross_entropy",
            class_weights=[0.4, 0.6],
        )
        
        assert classifier.loss_type == "weighted_cross_entropy"
        assert classifier.class_weights is not None
    
    def test_save_pretrained(self, titanic_classifier, temp_dir):
        """Test saving classifier."""
        save_path = temp_dir / "classifier"
        titanic_classifier.save_pretrained(str(save_path))
        
        assert (save_path / "bert").exists()
        assert (save_path / "classifier_head.npz").exists()
    
    def test_load_pretrained(self, titanic_classifier, temp_dir):
        """Test loading classifier."""
        # Save first
        save_path = temp_dir / "classifier"
        titanic_classifier.save_pretrained(str(save_path))
        
        # Create new classifier and load
        new_classifier = TitanicClassifier(titanic_classifier.bert)
        new_classifier.load_pretrained(str(save_path))
        
        # Verify loaded correctly
        assert new_classifier.num_labels == titanic_classifier.num_labels


class TestMultiScaleConv1D:
    """Test MultiScaleConv1D module."""
    
    def test_initialization(self):
        """Test multi-scale conv initialization."""
        conv = MultiScaleConv1D(
            in_channels=768,
            num_filters=128,
            kernel_sizes=[2, 3, 4],
        )
        
        assert conv.num_filters == 128
        assert conv.kernel_sizes == [2, 3, 4]
        assert conv.in_channels == 768
    
    def test_forward_pass(self):
        """Test multi-scale conv forward pass."""
        conv = MultiScaleConv1D(
            in_channels=768,
            num_filters=128,
            kernel_sizes=[2, 3, 4],
        )
        
        # Create dummy input
        batch_size, seq_length, hidden_size = 2, 10, 768
        inputs = mx.random.normal((batch_size, seq_length, hidden_size))
        
        outputs = conv(inputs)
        
        # Output should have num_filters * len(kernel_sizes) channels
        assert outputs.shape[-1] == 128 * 3


class TestCNNEnhancedModernBERT:
    """Test CNN-enhanced ModernBERT model."""
    
    def test_initialization(self):
        """Test hybrid model initialization."""
        config = CNNHybridConfig(hidden_size=768)
        
        model = CNNEnhancedModernBERT(config)
        
        assert hasattr(model, "embeddings")
        assert hasattr(model, "encoder")
        assert hasattr(model, "cnn_module")
        assert hasattr(model, "fusion")
    
    def test_forward_pass(self):
        """Test hybrid model forward pass."""
        config = CNNHybridConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            cnn_kernel_sizes=[2, 3],
            cnn_num_filters=64,
        )
        
        model = CNNEnhancedModernBERT(config)
        
        batch_size, seq_length = 2, 10
        input_ids = mx.random.randint(0, 1000, (batch_size, seq_length))
        attention_mask = mx.ones((batch_size, seq_length))
        
        outputs = model(input_ids, attention_mask)
        
        assert "last_hidden_state" in outputs
        assert "pooler_output" in outputs
        assert outputs["last_hidden_state"].shape[0] == batch_size
    
    def test_attention_fusion(self):
        """Test hybrid model with attention fusion."""
        config = CNNHybridConfig(
            hidden_size=128,
            use_attention_fusion=True,
            cnn_kernel_sizes=[2, 3],
            cnn_num_filters=64,
        )
        
        model = CNNEnhancedModernBERT(config)
        
        assert hasattr(model.fusion, "attention_fusion")
        
        batch_size, seq_length = 2, 10
        input_ids = mx.random.randint(0, 1000, (batch_size, seq_length))
        attention_mask = mx.ones((batch_size, seq_length))
        
        outputs = model(input_ids, attention_mask)
        assert outputs["pooler_output"].shape[0] == batch_size
    
    def test_highway_connections(self):
        """Test hybrid model with highway connections."""
        config = CNNHybridConfig(
            hidden_size=128,
            use_highway=True,
            cnn_kernel_sizes=[2, 3],
            cnn_num_filters=64,
        )
        
        model = CNNEnhancedModernBERT(config)
        
        assert hasattr(model.fusion, "highway")


class TestCNNHybridConfig:
    """Test CNN hybrid configuration."""
    
    def test_default_config(self):
        """Test default CNN hybrid config."""
        config = CNNHybridConfig()
        
        assert config.cnn_kernel_sizes == (2, 3, 4, 5)
        assert config.cnn_num_filters == 128
        assert config.use_dilated_conv == True
        assert config.use_attention_fusion == True
        assert config.use_highway == True
    
    def test_custom_config(self):
        """Test custom CNN hybrid config."""
        config = CNNHybridConfig(
            hidden_size=512,
            cnn_kernel_sizes=[3, 4, 5],
            cnn_num_filters=256,
            use_dilated_conv=False,
        )
        
        assert config.hidden_size == 512
        assert config.cnn_kernel_sizes == [3, 4, 5]
        assert config.cnn_num_filters == 256
        assert config.use_dilated_conv == False


class TestModelFactory:
    """Test model factory."""
    
    def test_create_standard_model(self):
        """Test creating standard model."""
        model = create_model("standard")
        
        assert isinstance(model, ModernBertModel)
        assert model.config is not None
    
    def test_create_mlx_optimized_model(self):
        """Test creating MLX optimized model."""
        model = create_model("mlx_optimized")
        
        assert isinstance(model, ModernBertModel)
    
    def test_create_invalid_model(self):
        """Test creating model with invalid type."""
        with pytest.raises(ValueError):
            create_model("invalid_type")


@pytest.mark.integration
@pytest.mark.mlx
class TestModelsIntegration:
    """Integration tests for models."""
    
    def test_end_to_end_classification(self, sample_titanic_data):
        """Test end-to-end classification pipeline."""
        from data.unified_loader import UnifiedTitanicDataPipeline
        
        # Create model
        bert_model = create_model("standard")
        classifier = TitanicClassifier(bert_model, num_labels=2)
        
        # Create data loader
        pipeline = UnifiedTitanicDataPipeline(
            data_path=str(sample_titanic_data),
            tokenizer_name="answerdotai/ModernBERT-base",
            batch_size=4,
            max_length=128,
        )
        
        # Get a batch
        batch = next(iter(pipeline.get_dataloader()))
        
        # Forward pass
        outputs = classifier(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        
        assert "loss" in outputs
        assert "logits" in outputs
        assert outputs["loss"].item() > 0
        
        # Test predictions
        predictions = mx.argmax(outputs["logits"], axis=-1)
        assert predictions.shape == batch["labels"].shape
    
    def test_cnn_hybrid_classification(self, sample_titanic_data):
        """Test CNN hybrid model classification."""
        from data.unified_loader import UnifiedTitanicDataPipeline
        
        # Create CNN hybrid model
        config = CNNHybridConfig(
            vocab_size=50265,
            hidden_size=768,
            num_hidden_layers=12,
            cnn_kernel_sizes=[2, 3, 4],
            cnn_num_filters=128,
        )
        bert_model = CNNEnhancedModernBERT(config)
        
        classifier = TitanicClassifier(bert_model, num_labels=2)
        
        # Create data loader
        pipeline = UnifiedTitanicDataPipeline(
            data_path=str(sample_titanic_data),
            tokenizer_name="answerdotai/ModernBERT-base",
            batch_size=4,
            max_length=128,
        )
        
        # Get a batch
        batch = next(iter(pipeline.get_dataloader()))
        
        # Forward pass
        outputs = classifier(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        
        assert "loss" in outputs
        assert "logits" in outputs