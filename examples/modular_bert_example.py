"""Example demonstrating the new modular BERT architecture.

This script shows how to:
1. Create a BertCore model
2. Attach different heads
3. Use the factory functions
4. Save and load models
"""

import mlx.core as mx
from pathlib import Path
import numpy as np

# Import the new modular architecture
from models.bert import (
    BertCore, BertWithHead, BertOutput,
    create_bert_core, create_bert_with_head, create_bert_for_competition
)
from models.heads.base_head import HeadType, PoolingType
from models.factory import create_model, create_bert_for_task, create_modular_bert


def demo_bert_core():
    """Demonstrate BertCore usage."""
    print("\n=== BertCore Demo ===")
    
    # Create a BERT core model
    bert = create_bert_core(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12
    )
    
    # Create dummy input
    batch_size = 2
    seq_length = 128
    input_ids = mx.random.randint(0, 1000, (batch_size, seq_length))
    attention_mask = mx.ones((batch_size, seq_length))
    
    # Forward pass
    outputs = bert(input_ids, attention_mask)
    
    print(f"BertOutput fields:")
    print(f"  - last_hidden_state: {outputs.last_hidden_state.shape}")
    print(f"  - pooler_output: {outputs.pooler_output.shape}")
    print(f"  - cls_output: {outputs.cls_output.shape}")
    print(f"  - mean_pooled: {outputs.mean_pooled.shape}")
    print(f"  - max_pooled: {outputs.max_pooled.shape}")
    
    # Access different pooling strategies
    print(f"\nPooling strategies:")
    print(f"  - CLS pooling: {outputs.get_pooled_output('cls').shape}")
    print(f"  - Mean pooling: {outputs.get_pooled_output('mean').shape}")
    print(f"  - Max pooling: {outputs.get_pooled_output('max').shape}")
    print(f"  - Pooler output: {outputs.get_pooled_output('pooler').shape}")


def demo_bert_with_head():
    """Demonstrate BertWithHead usage."""
    print("\n=== BertWithHead Demo ===")
    
    # Create BERT with binary classification head
    model = create_bert_with_head(
        head_type=HeadType.BINARY_CLASSIFICATION,
        num_labels=2,
        bert_config={
            "hidden_size": 768,
            "num_hidden_layers": 6,  # Smaller for demo
            "num_attention_heads": 12
        }
    )
    
    # Create dummy input
    batch_size = 4
    seq_length = 64
    input_ids = mx.random.randint(0, 1000, (batch_size, seq_length))
    attention_mask = mx.ones((batch_size, seq_length))
    labels = mx.random.randint(0, 2, (batch_size,))
    
    # Forward pass with labels
    outputs = model(input_ids, attention_mask, labels=labels)
    
    print(f"Model outputs:")
    print(f"  - loss: {outputs['loss'].item():.4f}")
    print(f"  - logits: {outputs['logits'].shape}")
    print(f"  - predictions: {outputs['predictions'].shape}")
    print(f"  - probabilities: {outputs['probabilities'].shape}")


def demo_factory_functions():
    """Demonstrate factory function usage."""
    print("\n=== Factory Functions Demo ===")
    
    # Method 1: Create with factory
    model1 = create_model(
        model_type="bert_with_head",
        head_type="multiclass_classification",
        num_labels=5,
        config={"hidden_size": 384, "num_hidden_layers": 4}
    )
    print(f"Created model with factory: {type(model1).__name__}")
    
    # Method 2: Create for specific task
    model2 = create_bert_for_task(
        task="regression",
        num_labels=1,
        freeze_bert_layers=2  # Freeze first 2 layers
    )
    print(f"Created model for task: {type(model2).__name__}")
    print(f"  - Head type: {model2.get_head().head_type}")
    
    # Method 3: Create for competition
    model3 = create_bert_for_competition(
        competition_type="multilabel_classification",
        num_labels=10
    )
    print(f"Created model for competition: {type(model3).__name__}")
    print(f"  - Head type: {model3.get_head().head_type}")


def demo_different_heads():
    """Demonstrate different head types."""
    print("\n=== Different Head Types Demo ===")
    
    # Create base BERT
    bert = create_bert_core(hidden_size=256, num_hidden_layers=3)
    
    head_types = [
        (HeadType.BINARY_CLASSIFICATION, 2),
        (HeadType.MULTICLASS_CLASSIFICATION, 5),
        (HeadType.MULTILABEL_CLASSIFICATION, 10),
        (HeadType.REGRESSION, 1),
        (HeadType.RANKING, 1),
    ]
    
    for head_type, num_labels in head_types:
        # Create model with specific head
        model = create_bert_with_head(
            bert_config={"hidden_size": 256, "num_hidden_layers": 3},
            head_type=head_type,
            num_labels=num_labels
        )
        
        # Get head info
        head = model.get_head()
        config = head.get_config()
        
        print(f"\n{head_type.value}:")
        print(f"  - Head class: {type(head).__name__}")
        print(f"  - Pooling type: {config.pooling_type.value}")
        print(f"  - Output size: {config.output_size}")
        print(f"  - Competition metric: {config.competition_metric}")


def demo_save_load():
    """Demonstrate saving and loading models."""
    print("\n=== Save/Load Demo ===")
    
    # Create a model
    model = create_bert_with_head(
        head_type=HeadType.MULTICLASS_CLASSIFICATION,
        num_labels=3,
        bert_config={"hidden_size": 256, "num_hidden_layers": 2}
    )
    
    # Save model
    save_path = Path("output/demo_model")
    model.save_pretrained(save_path)
    print(f"Model saved to: {save_path}")
    
    # Load model
    loaded_model = BertWithHead.from_pretrained(save_path)
    print(f"Model loaded successfully")
    print(f"  - BERT hidden size: {loaded_model.get_bert().get_hidden_size()}")
    print(f"  - Head type: {loaded_model.get_head().head_type}")
    
    # Clean up
    import shutil
    if save_path.exists():
        shutil.rmtree(save_path)


def demo_pooling_strategies():
    """Demonstrate different pooling strategies."""
    print("\n=== Pooling Strategies Demo ===")
    
    pooling_types = [
        PoolingType.CLS,
        PoolingType.MEAN,
        PoolingType.MAX,
        PoolingType.ATTENTION,
        PoolingType.WEIGHTED_MEAN,
        PoolingType.LAST,
    ]
    
    for pooling_type in pooling_types:
        # Create model with specific pooling
        model = create_bert_with_head(
            bert_config={"hidden_size": 128, "num_hidden_layers": 2},
            head_config={
                "head_type": HeadType.BINARY_CLASSIFICATION,
                "input_size": 128,
                "output_size": 2,
                "pooling_type": pooling_type,
            }
        )
        
        print(f"\n{pooling_type.value} pooling:")
        
        # Test forward pass
        input_ids = mx.random.randint(0, 100, (1, 32))
        attention_mask = mx.ones((1, 32))
        
        outputs = model(input_ids, attention_mask)
        print(f"  - Output shape: {outputs['logits'].shape}")


def main():
    """Run all demonstrations."""
    print("Modular BERT Architecture Examples")
    print("==================================")
    
    # Run demos
    demo_bert_core()
    demo_bert_with_head()
    demo_factory_functions()
    demo_different_heads()
    demo_pooling_strategies()
    demo_save_load()
    
    print("\n✅ All demonstrations completed successfully!")


if __name__ == "__main__":
    main()