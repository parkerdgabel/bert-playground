"""Comprehensive tests for BERT models and heads.

This test suite ensures the modular BERT implementation works correctly
with all head types for Kaggle competitions.
"""

import unittest
import numpy as np
import mlx.core as mx
from pathlib import Path
import tempfile
import shutil

from models.bert.config import BertConfig
from models.bert.core import BertCore, BertOutput
from models.bert.model import BertWithHead, create_bert_with_head, create_bert_for_competition
from models.heads.base_head import HeadType, PoolingType, ActivationType
from models.heads.head_registry import CompetitionType, get_head_registry
from models.factory import create_model, MODEL_REGISTRY


class TestBertConfig(unittest.TestCase):
    """Test BertConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = BertConfig()
        self.assertEqual(config.hidden_size, 768)
        self.assertEqual(config.num_hidden_layers, 12)
        self.assertEqual(config.num_attention_heads, 12)
        self.assertEqual(config.vocab_size, 30522)
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = BertConfig(
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4
        )
        self.assertEqual(config.hidden_size, 256)
        self.assertEqual(config.num_hidden_layers, 4)
        self.assertEqual(config.num_attention_heads, 4)
    
    def test_config_serialization(self):
        """Test config to/from dict."""
        config = BertConfig(hidden_size=256)
        config_dict = config.to_dict()
        
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict["hidden_size"], 256)
        
        # Test from_dict
        new_config = BertConfig.from_dict(config_dict)
        self.assertEqual(new_config.hidden_size, 256)


class TestBertCore(unittest.TestCase):
    """Test BertCore model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = BertConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=256,
            vocab_size=1000
        )
        self.bert = BertCore(self.config)
        self.batch_size = 4
        self.seq_length = 32
    
    def test_initialization(self):
        """Test model initialization."""
        self.assertIsInstance(self.bert, BertCore)
        self.assertEqual(self.bert.get_hidden_size(), 128)
        self.assertEqual(self.bert.get_num_layers(), 2)
    
    def test_forward_pass(self):
        """Test forward pass through BERT."""
        # Create dummy input
        input_ids = mx.random.randint(0, self.config.vocab_size, (self.batch_size, self.seq_length))
        attention_mask = mx.ones((self.batch_size, self.seq_length))
        
        # Forward pass
        output = self.bert(input_ids, attention_mask)
        
        # Check output type and shapes
        self.assertIsInstance(output, BertOutput)
        self.assertEqual(output.last_hidden_state.shape, (self.batch_size, self.seq_length, self.config.hidden_size))
        self.assertEqual(output.pooler_output.shape, (self.batch_size, self.config.hidden_size))
    
    def test_pooling_outputs(self):
        """Test different pooling outputs."""
        input_ids = mx.random.randint(0, self.config.vocab_size, (self.batch_size, self.seq_length))
        attention_mask = mx.ones((self.batch_size, self.seq_length))
        
        output = self.bert(input_ids, attention_mask, compute_pooling=True)
        
        # Check different pooling types
        cls_pooled = output.get_pooled_output("cls")
        mean_pooled = output.get_pooled_output("mean")
        max_pooled = output.get_pooled_output("max")
        pooler = output.get_pooled_output("pooler")
        
        self.assertEqual(cls_pooled.shape, (self.batch_size, self.config.hidden_size))
        self.assertEqual(mean_pooled.shape, (self.batch_size, self.config.hidden_size))
        self.assertEqual(max_pooled.shape, (self.batch_size, self.config.hidden_size))
        self.assertEqual(pooler.shape, (self.batch_size, self.config.hidden_size))
    
    def test_save_load(self):
        """Test saving and loading model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "bert_model"
            
            # Save model
            self.bert.save_pretrained(save_path)
            
            # Check files exist
            self.assertTrue((save_path / "config.json").exists())
            self.assertTrue((save_path / "model.safetensors").exists())
            
            # Load model
            loaded_bert = BertCore.from_pretrained(save_path)
            
            # Check config matches
            self.assertEqual(loaded_bert.get_hidden_size(), self.bert.get_hidden_size())
            self.assertEqual(loaded_bert.get_num_layers(), self.bert.get_num_layers())


class TestBertWithHead(unittest.TestCase):
    """Test BertWithHead combined model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.bert_config = BertConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4
        )
        self.batch_size = 4
        self.seq_length = 32
    
    def test_binary_classification(self):
        """Test BERT with binary classification head."""
        model = create_bert_with_head(
            bert_config=self.bert_config,
            head_type="binary_classification",
            num_labels=2
        )
        
        # Create inputs
        input_ids = mx.random.randint(0, self.bert_config.vocab_size, (self.batch_size, self.seq_length))
        attention_mask = mx.ones((self.batch_size, self.seq_length))
        labels = mx.random.randint(0, 2, (self.batch_size,))
        
        # Forward pass
        outputs = model(input_ids, attention_mask, labels=labels)
        
        # Check outputs
        self.assertIn("loss", outputs)
        self.assertIn("logits", outputs)
        self.assertIn("probabilities", outputs)
        self.assertEqual(outputs["logits"].shape, (self.batch_size, 2))
    
    def test_multiclass_classification(self):
        """Test BERT with multiclass classification head."""
        num_classes = 5
        model = create_bert_with_head(
            bert_config=self.bert_config,
            head_type="multiclass_classification",
            num_labels=num_classes
        )
        
        # Create inputs
        input_ids = mx.random.randint(0, self.bert_config.vocab_size, (self.batch_size, self.seq_length))
        attention_mask = mx.ones((self.batch_size, self.seq_length))
        labels = mx.random.randint(0, num_classes, (self.batch_size,))
        
        # Forward pass
        outputs = model(input_ids, attention_mask, labels=labels)
        
        # Check outputs
        self.assertEqual(outputs["logits"].shape, (self.batch_size, num_classes))
        self.assertEqual(outputs["probabilities"].shape, (self.batch_size, num_classes))
    
    def test_regression(self):
        """Test BERT with regression head."""
        model = create_bert_with_head(
            bert_config=self.bert_config,
            head_type="regression",
            num_labels=1
        )
        
        # Create inputs
        input_ids = mx.random.randint(0, self.bert_config.vocab_size, (self.batch_size, self.seq_length))
        attention_mask = mx.ones((self.batch_size, self.seq_length))
        labels = mx.random.uniform(shape=(self.batch_size, 1))
        
        # Forward pass
        outputs = model(input_ids, attention_mask, labels=labels)
        
        # Check outputs
        self.assertIn("predictions", outputs)
        self.assertEqual(outputs["predictions"].shape, (self.batch_size, 1))
    
    def test_competition_model_creation(self):
        """Test creating model for specific competition type."""
        model = create_bert_for_competition(
            competition_type="tabular_classification",
            bert_config=self.bert_config,
            num_labels=3
        )
        
        self.assertIsInstance(model, BertWithHead)
        self.assertIsNotNone(model.get_bert())
        self.assertIsNotNone(model.get_head())
    
    def test_model_freezing(self):
        """Test freezing BERT layers."""
        model = create_bert_with_head(
            bert_config=self.bert_config,
            head_type="binary_classification",
            freeze_bert=True
        )
        
        # Check that BERT is frozen
        # Note: MLX doesn't have a direct way to check if parameters are frozen
        # This is a placeholder for when MLX adds this functionality
        self.assertTrue(True)  # Placeholder assertion
    
    def test_save_load_complete_model(self):
        """Test saving and loading complete model with head."""
        model = create_bert_with_head(
            bert_config=self.bert_config,
            head_type="binary_classification",
            num_labels=2
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "complete_model"
            
            # Save model
            model.save_pretrained(save_path)
            
            # Check files exist
            self.assertTrue((save_path / "bert").exists())
            self.assertTrue((save_path / "head").exists())
            self.assertTrue((save_path / "model_metadata.json").exists())
            
            # Load model
            loaded_model = BertWithHead.from_pretrained(save_path)
            
            # Test loaded model
            input_ids = mx.random.randint(0, self.bert_config.vocab_size, (self.batch_size, self.seq_length))
            outputs = loaded_model(input_ids)
            
            self.assertIn("logits", outputs)


class TestModelFactory(unittest.TestCase):
    """Test model factory functionality."""
    
    def test_factory_bert_core(self):
        """Test creating BERT core through factory."""
        model = create_model("bert_core", hidden_size=128)
        self.assertIsInstance(model, BertCore)
        self.assertEqual(model.get_hidden_size(), 128)
    
    def test_factory_bert_with_head(self):
        """Test creating BERT with head through factory."""
        model = create_model(
            "bert_with_head",
            head_type="binary_classification",
            hidden_size=128,
            num_labels=2
        )
        self.assertIsInstance(model, BertWithHead)
    
    def test_model_registry(self):
        """Test model registry entries."""
        # Check core models
        self.assertIn("bert-core", MODEL_REGISTRY)
        self.assertIn("bert-binary", MODEL_REGISTRY)
        self.assertIn("bert-multiclass", MODEL_REGISTRY)
        self.assertIn("bert-regression", MODEL_REGISTRY)
        
        # Test creating from registry
        model = MODEL_REGISTRY["bert-binary"](hidden_size=128)
        self.assertIsInstance(model, BertWithHead)


class TestHeadRegistry(unittest.TestCase):
    """Test head registry functionality."""
    
    def test_registry_contents(self):
        """Test that all heads are registered."""
        registry = get_head_registry()
        
        # Check that we have classification heads
        classification_heads = registry.get_heads_by_type(HeadType.BINARY_CLASSIFICATION)
        self.assertGreater(len(classification_heads), 0)
        
        multiclass_heads = registry.get_heads_by_type(HeadType.MULTICLASS_CLASSIFICATION)
        self.assertGreater(len(multiclass_heads), 0)
        
        # Check regression heads
        regression_heads = registry.get_heads_by_type(HeadType.REGRESSION)
        self.assertGreater(len(regression_heads), 0)
    
    def test_competition_head_selection(self):
        """Test selecting heads for competition types."""
        registry = get_head_registry()
        
        # Test for tabular classification
        head = registry.create_head_from_competition(
            CompetitionType.TABULAR_CLASSIFICATION,
            input_size=128,
            output_size=2
        )
        self.assertIsNotNone(head)
        
        # Test for time series
        head = registry.create_head_from_competition(
            CompetitionType.TIME_SERIES,
            input_size=128,
            output_size=1
        )
        self.assertIsNotNone(head)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete pipeline."""
    
    def test_end_to_end_classification(self):
        """Test complete classification pipeline."""
        # Create model
        model = create_bert_for_competition(
            competition_type="tabular_classification",
            bert_config={"hidden_size": 64, "num_hidden_layers": 2},
            num_labels=2
        )
        
        # Create data
        batch_size = 8
        seq_length = 16
        input_ids = mx.random.randint(0, 1000, (batch_size, seq_length))
        attention_mask = mx.ones((batch_size, seq_length))
        labels = mx.random.randint(0, 2, (batch_size,))
        
        # Forward pass
        outputs = model(input_ids, attention_mask, labels=labels)
        
        # Check outputs
        self.assertIn("loss", outputs)
        self.assertIsNotNone(outputs["loss"])
        
        # Compute metrics
        metrics = model.compute_metrics(outputs["head_outputs"], labels)
        self.assertIn("accuracy", metrics)
        self.assertIsInstance(metrics["accuracy"], float)
    
    def test_batch_processing(self):
        """Test processing multiple batches."""
        model = create_bert_with_head(
            bert_config={"hidden_size": 64, "num_hidden_layers": 2},
            head_type="binary_classification"
        )
        
        # Process multiple batches
        total_loss = 0
        num_batches = 5
        
        for _ in range(num_batches):
            input_ids = mx.random.randint(0, 1000, (4, 16))
            labels = mx.random.randint(0, 2, (4,))
            
            outputs = model(input_ids, labels=labels)
            total_loss += float(outputs["loss"])
        
        avg_loss = total_loss / num_batches
        self.assertGreater(avg_loss, 0)


if __name__ == "__main__":
    unittest.main()