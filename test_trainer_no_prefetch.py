#!/usr/bin/env python3
"""Test training with prefetching disabled to isolate the issue."""

import sys
from pathlib import Path
from transformers import AutoTokenizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from training.core.base import BaseTrainer
from training.core.config import get_quick_test_config
from models.factory import create_model
from data.factory import create_dataloader

def test_training_no_prefetch():
    """Test training with prefetching disabled."""
    
    print("🔍 Testing training with PREFETCHING DISABLED")
    
    # Create configuration
    config = get_quick_test_config()
    config.environment.output_dir = Path("no_prefetch_test_output")
    config.training.num_epochs = 1
    config.data.batch_size = 4
    config.training.logging_steps = 10
    config.environment.experiment_name = "no_prefetch_test"
    config.training.report_to = []  # Disable MLflow
    
    print("🔍 Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    
    print("🔍 Creating model")
    model = create_model(
        model_type="modernbert_with_head",
        head_type="binary_classification", 
        num_labels=2
    )
    
    print("🔍 Creating data loader with NO PREFETCHING")
    train_loader = create_dataloader(
        data_path=Path("data/titanic/train.csv"),
        batch_size=4,
        shuffle=False,
        num_workers=0,  # No workers
        prefetch_size=0,  # No prefetching
        tokenizer=tokenizer,
        split="train"
    )
    
    print(f"🔍 Data loader created with {len(train_loader)} batches")
    
    # Test iteration outside trainer first
    print("🔍 Testing data loader iteration outside trainer")
    for i, batch in enumerate(train_loader):
        print(f"🔍 Got batch {i}: {list(batch.keys())}")
        if i >= 1:
            break
    print("✅ Data loader iteration works outside trainer")
    
    print("🔍 Creating trainer")
    trainer = BaseTrainer(model=model, config=config)
    
    print("🔍 Starting training (no prefetch)")
    
    try:
        result = trainer.train(train_dataloader=train_loader)
        print(f"✅ Training completed successfully!")
        print(f"✅ Final loss: {result.final_train_loss}")
        print(f"✅ Total steps: {result.total_steps}")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_training_no_prefetch()