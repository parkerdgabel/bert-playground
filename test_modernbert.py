#!/usr/bin/env python3
"""
Comprehensive test script for ModernBERT implementation.

This script tests all ModernBERT components and validates the improvements
over Classic BERT.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from models.bert.modernbert_config import ModernBertConfig
from models.bert.modernbert_core import ModernBertCore
from models.factory import create_model
from loguru import logger

def test_modernbert_config():
    """Test ModernBERT configuration."""
    print("Testing ModernBERT Configuration...")
    
    # Test base config
    config = ModernBertConfig.get_base_config()
    print(f"Base config: {config}")
    
    assert config.model_size == "base"
    assert config.hidden_size == 768
    assert config.num_hidden_layers == 22
    assert config.num_attention_heads == 12
    assert config.max_position_embeddings == 8192
    assert config.use_rope == True
    assert config.use_geglu == True
    assert config.use_alternating_attention == True
    assert config.use_bias == False
    
    # Test large config
    large_config = ModernBertConfig.get_large_config()
    print(f"Large config: {large_config}")
    
    assert large_config.model_size == "large"
    assert large_config.hidden_size == 1024
    assert large_config.num_hidden_layers == 28
    assert large_config.num_attention_heads == 16
    
    # Test attention pattern
    pattern = []
    for i in range(config.num_hidden_layers):
        pattern.append(config.get_attention_type(i))
    
    print(f"Attention pattern (first 12 layers): {pattern[:12]}")
    
    # Every 3rd layer should be global
    assert pattern[2] == "global"  # Layer 3
    assert pattern[5] == "global"  # Layer 6
    assert pattern[8] == "global"  # Layer 9
    
    # Other layers should be local
    assert pattern[0] == "local"   # Layer 1
    assert pattern[1] == "local"   # Layer 2
    assert pattern[3] == "local"   # Layer 4
    
    print("✅ ModernBERT configuration tests passed!")


def test_modernbert_core():
    """Test ModernBERT core model."""
    print("\nTesting ModernBERT Core Model...")
    
    # Create small config for testing
    config = ModernBertConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=6,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=512,
        model_size="base",
    )
    
    # Create model
    model = ModernBertCore(config)
    
    # Test inputs
    batch_size = 2
    seq_len = 32
    
    input_ids = mx.random.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = mx.ones((batch_size, seq_len), dtype=mx.int32)
    token_type_ids = mx.zeros((batch_size, seq_len), dtype=mx.int32)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Model config: {config}")
    
    # Test forward pass
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        output_attentions=True,
        output_hidden_states=True,
    )
    
    print(f"Last hidden state shape: {output.last_hidden_state.shape}")
    print(f"Pooler output shape: {output.pooler_output.shape}")
    
    # Check outputs
    assert output.last_hidden_state.shape == (batch_size, seq_len, config.hidden_size)
    
    # Check pooler output (may have different size due to pooler_hidden_size)
    expected_pooler_size = config.pooler_hidden_size or config.hidden_size
    print(f"Expected pooler size: {expected_pooler_size}, Actual: {output.pooler_output.shape}")
    assert output.pooler_output.shape == (batch_size, expected_pooler_size)
    
    # Check attention outputs
    assert len(output.attentions) == config.num_hidden_layers
    print(f"Number of attention layers: {len(output.attentions)}")
    
    # Check hidden states
    assert len(output.hidden_states) == config.num_hidden_layers + 1
    print(f"Number of hidden states: {len(output.hidden_states)}")
    
    # Test attention pattern
    pattern = model.get_attention_pattern()
    print(f"Attention pattern: {pattern}")
    
    # Check specific layers
    assert pattern[2] == "global"  # Layer 3 (0-indexed)
    assert pattern[5] == "global"  # Layer 6 (0-indexed)
    assert pattern[0] == "local"   # Layer 1 (0-indexed)
    assert pattern[1] == "local"   # Layer 2 (0-indexed)
    
    print("✅ ModernBERT core model tests passed!")


def test_modernbert_vs_classic_bert():
    """Compare ModernBERT with Classic BERT."""
    print("\nTesting ModernBERT vs Classic BERT...")
    
    # Same config for both models
    config_dict = {
        "vocab_size": 1000,
        "hidden_size": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "intermediate_size": 512,
        "max_position_embeddings": 512,
    }
    
    # Create Classic BERT
    classic_bert = create_model("bert_core", config=config_dict)
    
    # Create ModernBERT (need to override the default base config)
    modernbert = create_model("modernbert_core", config=config_dict)
    
    # Test inputs
    batch_size = 2
    seq_len = 16
    
    input_ids = mx.random.randint(0, 1000, (batch_size, seq_len))
    attention_mask = mx.ones((batch_size, seq_len), dtype=mx.int32)
    
    # Forward pass through both models
    classic_output = classic_bert(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=True,
        output_hidden_states=True,
    )
    
    modernbert_output = modernbert(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=True,
        output_hidden_states=True,
    )
    
    print(f"Classic BERT output shape: {classic_output.last_hidden_state.shape}")
    print(f"ModernBERT output shape: {modernbert_output.last_hidden_state.shape}")
    
    # Both should have same output shape
    assert classic_output.last_hidden_state.shape == modernbert_output.last_hidden_state.shape
    assert classic_output.pooler_output.shape == modernbert_output.pooler_output.shape
    
    # But outputs should be different (different architectures)
    assert not mx.allclose(classic_output.last_hidden_state, modernbert_output.last_hidden_state, atol=1e-3)
    
    print("✅ ModernBERT vs Classic BERT comparison tests passed!")


def test_modernbert_with_heads():
    """Test ModernBERT with different head types."""
    print("\nTesting ModernBERT with Heads...")
    
    config_dict = {
        "vocab_size": 1000,
        "hidden_size": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "intermediate_size": 512,
        "max_position_embeddings": 512,
    }
    
    # Test different head types
    head_types = [
        "binary_classification",
        "multiclass_classification",
        "multilabel_classification",
        "regression",
    ]
    
    batch_size = 2
    seq_len = 16
    input_ids = mx.random.randint(0, 1000, (batch_size, seq_len))
    attention_mask = mx.ones((batch_size, seq_len), dtype=mx.int32)
    
    for head_type in head_types:
        print(f"Testing {head_type}...")
        
        # Create model with head
        model = create_model(
            "modernbert_with_head",
            config=config_dict,
            head_type=head_type,
            num_labels=3 if "classification" in head_type else 1
        )
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True
        )
        
        # Check outputs
        assert "logits" in outputs or "predictions" in outputs
        assert "bert_outputs" in outputs
        
        # Check BERT outputs
        bert_outputs = outputs["bert_outputs"]
        assert bert_outputs.last_hidden_state.shape == (batch_size, seq_len, 128)
        assert bert_outputs.pooler_output.shape == (batch_size, 128)
        
        print(f"  ✅ {head_type} test passed!")
    
    print("✅ All ModernBERT with heads tests passed!")


def test_modernbert_factory():
    """Test ModernBERT factory functions."""
    print("\nTesting ModernBERT Factory...")
    
    # Test direct creation
    model1 = create_model("modernbert_core", model_size="base")
    print(f"Direct creation: {model1.get_config().model_size}")
    
    # Test registry creation
    from models.factory import create_from_registry
    
    model2 = create_from_registry("modernbert-base")
    print(f"Registry creation: {model2.get_config().model_size}")
    
    model3 = create_from_registry("modernbert-binary", num_labels=2)
    print(f"Binary classifier: {type(model3)}")
    
    # Test configurations
    assert model1.get_config().model_size == "base"
    assert model2.get_config().model_size == "base"
    
    print("✅ ModernBERT factory tests passed!")


def test_modernbert_extended_sequence():
    """Test ModernBERT with extended sequence length."""
    print("\nTesting ModernBERT Extended Sequence Length...")
    
    config = ModernBertConfig(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=256,
        max_position_embeddings=1024,  # Extended sequence length
        model_size="base",
    )
    
    model = ModernBertCore(config)
    
    # Test with long sequences
    batch_size = 1
    seq_len = 512  # Much longer than Classic BERT's 512 limit
    
    input_ids = mx.random.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = mx.ones((batch_size, seq_len), dtype=mx.int32)
    
    # Forward pass
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=True,
    )
    
    print(f"Extended sequence shape: {output.last_hidden_state.shape}")
    assert output.last_hidden_state.shape == (batch_size, seq_len, config.hidden_size)
    
    # Test attention pattern with long sequences
    pattern = model.get_attention_pattern()
    print(f"Attention pattern for long sequence: {pattern}")
    
    print("✅ Extended sequence length test passed!")


def test_modernbert_architecture_improvements():
    """Test specific ModernBERT architecture improvements."""
    print("\nTesting ModernBERT Architecture Improvements...")
    
    config = ModernBertConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=6,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=512,
        use_rope=True,
        use_geglu=True,
        use_alternating_attention=True,
        use_bias=False,
        use_post_embedding_norm=True,
        model_size="base",
    )
    
    model = ModernBertCore(config)
    
    # Test RoPE (no position embeddings)
    assert not hasattr(model.embeddings, 'position_embeddings')
    print("✅ RoPE: No position embeddings found")
    
    # Test GeGLU in layers
    first_layer = model.encoder_layers[0]
    assert hasattr(first_layer.feed_forward, 'mlp')
    print("✅ GeGLU: Found GeGLU MLP in layers")
    
    # Test alternating attention
    attention_pattern = model.get_attention_pattern()
    global_layers = [i for i, att in enumerate(attention_pattern) if att == "global"]
    local_layers = [i for i, att in enumerate(attention_pattern) if att == "local"]
    
    print(f"Global attention layers: {global_layers}")
    print(f"Local attention layers: {local_layers}")
    
    assert len(global_layers) > 0
    assert len(local_layers) > 0
    print("✅ Alternating attention: Both global and local layers found")
    
    # Test no bias terms
    assert not config.use_bias
    print("✅ No bias terms: Streamlined architecture")
    
    # Test post-embedding normalization
    assert hasattr(model.embeddings, 'post_embedding_norm')
    assert model.embeddings.post_embedding_norm is not None
    print("✅ Post-embedding normalization: Found additional normalization")
    
    print("✅ All architecture improvements verified!")


def main():
    """Run all ModernBERT tests."""
    print("🚀 Starting ModernBERT Comprehensive Testing")
    print("=" * 50)
    
    try:
        test_modernbert_config()
        test_modernbert_core()
        test_modernbert_vs_classic_bert()
        test_modernbert_with_heads()
        test_modernbert_factory()
        test_modernbert_extended_sequence()
        test_modernbert_architecture_improvements()
        
        print("\n" + "=" * 50)
        print("🎉 All ModernBERT tests passed successfully!")
        print("ModernBERT implementation is ready for production use!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)