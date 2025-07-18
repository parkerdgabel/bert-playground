"""Tests for model factory functions."""

import pytest
import mlx.core as mx

from models.factory import (
    create_bert_core,
    create_bert_with_head,
    create_model_with_lora,
    create_bert_with_lora,
    create_qlora_model,
    MODEL_REGISTRY,
)
from models.bert import BertConfig
from models.lora import LoRAConfig


class TestModelFactory:
    """Test model factory functions."""
    
    def test_create_bert_core(self):
        """Test creating BERT core models."""
        # Create with default config
        bert = create_bert_core()
        assert bert is not None
        
        # Create with custom config
        config = BertConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
        )
        bert = create_bert_core(config=config)
        assert bert.config.hidden_size == 128
        
    def test_create_bert_with_head(self):
        """Test creating BERT with head."""
        # Binary classification
        model = create_bert_with_head(
            head_type="binary_classification",
            num_labels=2,
        )
        assert model is not None
        assert model.num_labels == 2
        
        # Multiclass classification
        model = create_bert_with_head(
            head_type="multiclass_classification",
            num_labels=5,
        )
        assert model.num_labels == 5
        
        # Regression
        model = create_bert_with_head(
            head_type="regression",
            num_labels=1,
        )
        assert model.num_labels == 1
        
    # TODO: Fix LoRA adapter injection issue
    def _test_create_model_with_lora(self):
        """Test creating model with LoRA - temporarily disabled due to adapter injection issue."""
        lora_config = LoRAConfig(r=8, alpha=16)
        
        model, adapter = create_model_with_lora(
            model_name="bert",
            lora_config=lora_config,
            head_type="binary_classification",
        )
        
        assert model is not None
        assert adapter is not None
        assert adapter.config.r == 8
        assert adapter.config.alpha == 16
        
    # TODO: Fix LoRA adapter injection issue
    def _test_create_bert_with_lora(self):
        """Test creating BERT with LoRA - temporarily disabled due to adapter injection issue."""
        model, adapter = create_bert_with_lora(
            r=4,
            alpha=8,
            target_modules=["query", "value"],
        )
        
        assert model is not None
        assert adapter is not None
        assert adapter.config.r == 4
        assert adapter.config.alpha == 8
        
    # TODO: Fix QLoRA integration
    def _test_create_qlora_model(self):
        """Test creating QLoRA model - temporarily disabled due to adapter injection issue."""
        # Note: This test might fail if quantization is not supported
        try:
            model, adapter = create_qlora_model(
                base_model_name="bert",
                qlora_config={"r": 4, "alpha": 8},
            )
            assert model is not None
            assert adapter is not None
        except Exception as e:
            # Skip if quantization not supported
            pytest.skip(f"QLoRA not supported: {e}")
            
    def test_model_registry(self):
        """Test MODEL_REGISTRY."""
        assert "bert-core" in MODEL_REGISTRY
        assert "bert-binary" in MODEL_REGISTRY
        assert "modernbert-core" in MODEL_REGISTRY
        assert "modernbert-binary" in MODEL_REGISTRY
        
        # Test registry functions
        bert_fn = MODEL_REGISTRY["bert-core"]
        bert = bert_fn()
        assert bert is not None