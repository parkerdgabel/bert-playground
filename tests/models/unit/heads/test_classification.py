"""Tests for classification heads."""

import pytest
import mlx.core as mx
import mlx.nn as nn

from models.heads import (
    BinaryClassificationHead,
    MulticlassClassificationHead,
    create_head,
)
from models.heads.config import ClassificationHeadConfig
from tests.models.fixtures.configs import create_classification_config
from tests.models.fixtures.data import create_embeddings, create_classification_targets
from tests.models.fixtures.utils import check_gradient_flow, save_and_load_model


class TestBinaryClassificationHead:
    """Test BinaryClassificationHead implementation."""
    
    def test_initialization(self):
        """Test BinaryClassificationHead initialization."""
        config = create_classification_config(
            hidden_size=768,
            num_labels=2,
            dropout_prob=0.1
        )
        head = BinaryClassificationHead(config)
        
        assert head.config.hidden_size == 768
        assert head.config.num_labels == 2
        assert head.config.dropout_prob == 0.1
    
    def test_forward_pass(self, create_embeddings):
        """Test forward pass through binary classification head."""
        config = create_classification_config(
            hidden_size=128,
            num_labels=2,
            pooling_type="cls"
        )
        head = BinaryClassificationHead(config)
        
        # Create test data
        batch_size, seq_length = 4, 32
        hidden_states = create_embeddings(batch_size, seq_length, 128)
        attention_mask = mx.ones((batch_size, seq_length))
        labels = create_classification_targets(batch_size, num_classes=2)
        
        # Forward pass
        outputs = head(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            labels=labels
        )
        
        assert "logits" in outputs
        # Binary classification can output (batch_size,) or (batch_size, 1)
        assert outputs["logits"].shape == (batch_size,) or \
               outputs["logits"].shape == (batch_size, 1)
    
    def test_pooling_strategies(self, create_embeddings):
        """Test different pooling strategies."""
        batch_size, seq_length = 4, 32
        hidden_states = create_embeddings(batch_size, seq_length, 128)
        attention_mask = mx.ones((batch_size, seq_length))
        
        for pooling_type in ["cls", "mean", "max"]:
            config = create_classification_config(
                hidden_size=128,
                num_labels=2,
                pooling_type=pooling_type
            )
            head = BinaryClassificationHead(config)
            
            outputs = head(hidden_states=hidden_states, attention_mask=attention_mask)
            assert "logits" in outputs
    
    def test_with_masked_input(self, create_embeddings, create_attention_mask):
        """Test with partially masked input."""
        config = create_classification_config(
            hidden_size=128,
            num_labels=2,
            pooling_type="mean"
        )
        head = BinaryClassificationHead(config)
        
        batch_size, seq_length = 4, 32
        hidden_states = create_embeddings(batch_size, seq_length, 128)
        attention_mask = create_attention_mask(
            batch_size, seq_length, mask_type="padding"
        )
        
        outputs = head(hidden_states=hidden_states, attention_mask=attention_mask)
        assert "logits" in outputs
        assert mx.all(mx.isfinite(outputs["logits"]))
    
    def test_gradient_flow(self, create_embeddings, check_gradients):
        """Test gradient flow through head."""
        config = create_classification_config(hidden_size=128, num_labels=2)
        head = BinaryClassificationHead(config)
        
        batch_size, seq_length = 2, 16
        data = {
            "hidden_states": create_embeddings(batch_size, seq_length, 128),
            "labels": create_classification_targets(batch_size, num_classes=2)
        }
        
        def loss_fn(model, batch):
            outputs = model(
                hidden_states=batch["hidden_states"],
                labels=batch["labels"]
            )
            logits = outputs["logits"]
            # Simple binary cross entropy loss
            labels = batch["labels"]
            if logits.shape[-1] == 1:
                logits = logits.squeeze(-1)
            probs = mx.sigmoid(logits)
            loss = -mx.mean(
                labels * mx.log(probs + 1e-8) + 
                (1 - labels) * mx.log(1 - probs + 1e-8)
            )
            return loss
        
        result = check_gradients(head, loss_fn, data)
        assert result is True
    
    @pytest.mark.parametrize("dropout_prob", [0.0, 0.1, 0.5])
    def test_dropout_rates(self, dropout_prob, create_embeddings):
        """Test different dropout rates."""
        config = create_classification_config(
            hidden_size=128,
            num_labels=2,
            dropout_prob=dropout_prob
        )
        head = BinaryClassificationHead(config)
        
        # Training mode - dropout should affect outputs
        head.train()
        hidden_states = create_embeddings(4, 32, 128)
        outputs1 = head(hidden_states=hidden_states)
        outputs2 = head(hidden_states=hidden_states)
        
        if dropout_prob > 0:
            # Outputs should differ in training mode
            assert not mx.allclose(outputs1["logits"], outputs2["logits"])
        
        # Eval mode - dropout should not affect outputs
        head.eval()
        outputs3 = head(hidden_states=hidden_states)
        outputs4 = head(hidden_states=hidden_states)
        assert mx.allclose(outputs3["logits"], outputs4["logits"])


class TestMulticlassClassificationHead:
    """Test MulticlassClassificationHead implementation."""
    
    def test_initialization(self):
        """Test MulticlassClassificationHead initialization."""
        config = create_classification_config(
            hidden_size=768,
            num_labels=10,
            dropout_prob=0.1
        )
        head = MulticlassClassificationHead(config)
        
        assert head.config.hidden_size == 768
        assert head.config.num_labels == 10
    
    def test_forward_pass(self, create_embeddings):
        """Test forward pass through multiclass classification head."""
        num_classes = 5
        config = create_classification_config(
            hidden_size=128,
            num_labels=num_classes
        )
        head = MulticlassClassificationHead(config)
        
        batch_size, seq_length = 4, 32
        hidden_states = create_embeddings(batch_size, seq_length, 128)
        attention_mask = mx.ones((batch_size, seq_length))
        labels = create_classification_targets(batch_size, num_classes=num_classes)
        
        outputs = head(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            labels=labels
        )
        
        assert "logits" in outputs
        assert outputs["logits"].shape == (batch_size, num_classes)
    
    def test_loss_computation(self, create_embeddings):
        """Test loss computation for multiclass classification."""
        num_classes = 5
        config = create_classification_config(
            hidden_size=128,
            num_labels=num_classes
        )
        head = MulticlassClassificationHead(config)
        
        batch_size = 4
        hidden_states = create_embeddings(batch_size, 32, 128)
        labels = create_classification_targets(
            batch_size,
            num_classes=num_classes,
            target_type="balanced"
        )
        
        outputs = head(hidden_states=hidden_states, labels=labels)
        
        # Compute cross entropy loss manually
        logits = outputs["logits"]
        loss = mx.mean(nn.losses.cross_entropy(logits, labels))
        assert mx.isfinite(loss)
    
    @pytest.mark.parametrize("num_classes", [2, 5, 10, 100])
    def test_different_class_counts(self, num_classes, create_embeddings):
        """Test with different numbers of classes."""
        config = create_classification_config(
            hidden_size=128,
            num_labels=num_classes
        )
        head = MulticlassClassificationHead(config)
        
        batch_size = 4
        hidden_states = create_embeddings(batch_size, 32, 128)
        
        outputs = head(hidden_states=hidden_states)
        assert outputs["logits"].shape == (batch_size, num_classes)
    
    def test_imbalanced_classes(self, create_embeddings):
        """Test with imbalanced class distribution."""
        num_classes = 5
        config = create_classification_config(
            hidden_size=128,
            num_labels=num_classes
        )
        head = MulticlassClassificationHead(config)
        
        batch_size = 100
        hidden_states = create_embeddings(batch_size, 32, 128)
        # Create imbalanced labels (mostly class 0)
        labels = create_classification_targets(
            batch_size,
            num_classes=num_classes,
            target_type="weighted",
            class_weights=[0.8, 0.05, 0.05, 0.05, 0.05]
        )
        
        outputs = head(hidden_states=hidden_states, labels=labels)
        assert outputs["logits"].shape == (batch_size, num_classes)


class TestHeadFactory:
    """Test head factory functions."""
    
    def test_create_head_factory(self):
        """Test create_head factory function."""
        # Binary classification
        head = create_head(
            head_type="binary_classification",
            input_size=768,
            output_size=2,
        )
        assert isinstance(head, BinaryClassificationHead)
        
        # Multiclass classification
        head = create_head(
            head_type="multiclass_classification",
            input_size=768,
            output_size=10,
        )
        assert isinstance(head, MulticlassClassificationHead)
        
        # Invalid type should raise error
        with pytest.raises(ValueError):
            create_head(
                head_type="invalid_type",
                input_size=768,
                output_size=2,
            )
    
    def test_factory_with_config(self):
        """Test factory with config object."""
        config = create_classification_config(
            hidden_size=256,
            num_labels=5,
            dropout_prob=0.2
        )
        
        head = create_head(
            head_type="multiclass_classification",
            config=config
        )
        
        assert isinstance(head, MulticlassClassificationHead)
        assert head.config.hidden_size == 256
        assert head.config.num_labels == 5
        assert head.config.dropout_prob == 0.2


@pytest.mark.integration
class TestClassificationHeadIntegration:
    """Integration tests for classification heads."""
    
    def test_with_bert_outputs(self, mock_bert_model, create_test_batch):
        """Test classification heads with BERT model outputs."""
        bert = mock_bert_model
        head = create_head(
            head_type="binary_classification",
            input_size=bert.config.hidden_size,
            output_size=2
        )
        
        # Get BERT outputs
        batch = create_test_batch(
            batch_size=4,
            seq_length=32,
            vocab_size=bert.config.vocab_size
        )
        bert_outputs = bert(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        
        # Pass through head
        head_outputs = head(
            hidden_states=bert_outputs["last_hidden_state"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        
        assert "logits" in head_outputs
        assert head_outputs["logits"].shape[0] == 4  # batch_size
    
    def test_end_to_end_classification(self, mock_bert_model):
        """Test end-to-end classification pipeline."""
        # Create BERT + classification head
        bert = mock_bert_model
        head = create_head(
            head_type="multiclass_classification",
            input_size=bert.config.hidden_size,
            output_size=5
        )
        
        # Combined forward pass
        def classify(input_ids, attention_mask=None, labels=None):
            bert_outputs = bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            return head(
                hidden_states=bert_outputs["last_hidden_state"],
                attention_mask=attention_mask,
                labels=labels
            )
        
        # Test
        batch_size = 4
        input_ids = mx.random.randint(0, bert.config.vocab_size, (batch_size, 32))
        labels = mx.random.randint(0, 5, (batch_size,))
        
        outputs = classify(input_ids, labels=labels)
        assert outputs["logits"].shape == (batch_size, 5)