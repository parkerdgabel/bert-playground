#!/usr/bin/env python3
"""Test neoBERT implementation"""

import sys
sys.path.append('.')

from models.bert import (
    create_neobert,
    create_neobert_mini,
    get_neobert_config,
    create_model_core,
)

def test_neobert_creation():
    """Test creating neoBERT models"""
    print("Testing neoBERT implementation...")
    
    # Test 1: Create neoBERT with factory function
    print("\n1. Creating neoBERT with factory function...")
    model = create_neobert()
    print(f"   ✓ Created neoBERT model: {type(model).__name__}")
    print(f"   ✓ Layers: {model.config.num_hidden_layers}")
    print(f"   ✓ Hidden size: {model.config.hidden_size}")
    print(f"   ✓ Max position embeddings: {model.config.max_position_embeddings}")
    print(f"   ✓ Uses SwiGLU: {model.config.use_swiglu}")
    print(f"   ✓ Uses RoPE: {model.config.use_rope}")
    print(f"   ✓ Uses pre-norm: {model.config.use_pre_norm}")
    print(f"   ✓ Norm type: {model.config.norm_type}")
    
    # Test 2: Create mini neoBERT
    print("\n2. Creating mini neoBERT...")
    mini_model = create_neobert_mini()
    print(f"   ✓ Created mini neoBERT: {mini_model.config.num_hidden_layers} layers")
    
    # Test 3: Create neoBERT with create_model_core
    print("\n3. Creating neoBERT with create_model_core...")
    model2 = create_model_core(model_type="neobert")
    print(f"   ✓ Created via create_model_core: {type(model2).__name__}")
    
    # Test 4: Get neoBERT config
    print("\n4. Testing neoBERT config...")
    config = get_neobert_config()
    print(f"   ✓ Config created with {config.num_hidden_layers} layers")
    print(f"   ✓ Activation: {config.hidden_act}")
    
    print("\n✅ All neoBERT tests passed!")
    return True

if __name__ == "__main__":
    try:
        test_neobert_creation()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)