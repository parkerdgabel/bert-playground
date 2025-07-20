#!/usr/bin/env python3
"""Test a minimal training loop without BaseTrainer."""

import sys
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from transformers import AutoTokenizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.factory import create_dataloader
from models.factory import create_model

def simple_train_loop():
    """Test training without BaseTrainer."""
    
    print("🔍 Testing simple training loop")
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    
    # Create model
    print("🔍 Creating model")
    model = create_model(
        model_type="modernbert_with_head",
        head_type="binary_classification", 
        num_labels=2
    )
    
    # Create optimizer
    print("🔍 Creating optimizer")
    optimizer = optim.AdamW(learning_rate=1e-5)
    
    # Create loader
    print("🔍 Creating data loader")
    loader = create_dataloader(
        data_path=Path("data/titanic/train.csv"),
        batch_size=4,
        shuffle=False,
        num_workers=0,
        prefetch_size=0,
        tokenizer=tokenizer,
        split="train"
    )
    
    print(f"🔍 Loader has {len(loader)} batches")
    
    # Define loss function
    def loss_fn(model, batch):
        print("  🔍 In loss_fn")
        model_inputs = {k: v for k, v in batch.items() 
                       if k not in ['metadata'] and v is not None}
        print(f"  🔍 Model inputs: {list(model_inputs.keys())}")
        
        outputs = model(**model_inputs)
        print(f"  🔍 Got outputs: {list(outputs.keys())}")
        
        return outputs["loss"]
    
    # Create value and grad function
    value_and_grad_fn = mx.value_and_grad(loss_fn)
    
    print("🔍 Starting training loop")
    
    # Train for a few steps
    for i, batch in enumerate(loader):
        print(f"\n🔍 Step {i}")
        print(f"  Batch keys: {list(batch.keys())}")
        
        # Forward and backward
        print("  🔍 Computing loss and gradients")
        loss, grads = value_and_grad_fn(model, batch)
        print(f"  🔍 Loss: {loss}")
        
        # Update
        print("  🔍 Updating model")
        optimizer.update(model, grads)
        
        # Eval to force computation
        print("  🔍 Evaluating computation")
        mx.eval(loss)
        print(f"  ✅ Step {i} complete, loss: {loss.item()}")
        
        if i >= 2:  # Just a few steps
            break
    
    print("\n✅ Simple training loop successful!")

if __name__ == "__main__":
    simple_train_loop()