"""Simple test to verify core modular BERT functionality."""

import mlx.core as mx
from models.bert import create_bert_with_head
from models.heads.base_head import HeadType

print("Testing modular BERT architecture...")

# Test 1: Create and use BertWithHead
print("\n1. Testing BertWithHead creation and forward pass...")
model = create_bert_with_head(
    bert_config={"hidden_size": 128, "num_hidden_layers": 2, "num_attention_heads": 8},
    head_type=HeadType.BINARY_CLASSIFICATION,
    num_labels=2
)

# Create dummy input
input_ids = mx.random.randint(0, 100, (4, 32))
labels = mx.random.randint(0, 2, (4,))

# Forward pass
outputs = model(input_ids, labels=labels)
print(f"✅ Forward pass successful!")
print(f"   Loss: {outputs['loss'].item():.4f}")
print(f"   Predictions shape: {outputs['predictions'].shape}")

# Test 2: Different head types
print("\n2. Testing different head types...")
for head_type in [HeadType.MULTICLASS_CLASSIFICATION, HeadType.REGRESSION]:
    model = create_bert_with_head(
        bert_config={"hidden_size": 64, "num_hidden_layers": 1, "num_attention_heads": 8},
        head_type=head_type,
        num_labels=5 if head_type == HeadType.MULTICLASS_CLASSIFICATION else 1
    )
    outputs = model(mx.random.randint(0, 50, (2, 16)))
    print(f"✅ {head_type.value}: {outputs['predictions'].shape}")

# Test 3: Save and load
print("\n3. Testing save/load...")
from pathlib import Path
import shutil

save_path = Path("test_output/simple_test")
model.save_pretrained(save_path)
print(f"✅ Model saved to {save_path}")

# Load it back
from models.bert import BertWithHead
loaded_model = BertWithHead.from_pretrained(save_path)
outputs = loaded_model(mx.random.randint(0, 50, (2, 16)))
print(f"✅ Model loaded and working: {outputs['predictions'].shape}")

# Clean up
if save_path.exists():
    shutil.rmtree(save_path)

print("\n✅ All core functionality tests passed!")