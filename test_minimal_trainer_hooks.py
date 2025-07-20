#!/usr/bin/env python3
"""Test trainer with all hooks and callbacks disabled."""

import sys
from pathlib import Path
from transformers import AutoTokenizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from training.core.base import BaseTrainer
from training.core.config import get_quick_test_config
from models.factory import create_model
from data.factory import create_dataloader

def test_no_hooks():
    """Test training with all hooks disabled."""
    
    print("🔍 Testing trainer with hooks disabled")
    
    # Create minimal config
    config = get_quick_test_config()
    config.environment.output_dir = Path("no_hooks_test_output")
    config.training.num_epochs = 1
    config.data.batch_size = 4
    config.training.logging_steps = 10
    config.environment.experiment_name = "no_hooks_test"
    config.training.report_to = []  # Disable MLflow
    
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    
    model = create_model(
        model_type="modernbert_with_head",
        head_type="binary_classification", 
        num_labels=2
    )
    
    loader = create_dataloader(
        data_path=Path("data/titanic/train.csv"),
        batch_size=4,
        shuffle=False,
        num_workers=0,
        prefetch_size=0,
        tokenizer=tokenizer,
        split="train"
    )
    
    print(f"🔍 Loader created with {len(loader)} batches")
    
    # Create trainer with empty callbacks
    trainer = BaseTrainer(model=model, config=config, callbacks=[])
    
    # Override _call_hooks to do nothing
    trainer._call_hooks = lambda *args, **kwargs: None
    
    print("🔍 Starting training with hooks disabled")
    
    try:
        # Manually test the exact flow
        print("🔍 Testing manual flow:")
        
        # 1. Calculate steps
        steps_per_epoch = len(loader)
        print(f"  Steps per epoch: {steps_per_epoch}")
        
        # 2. Start training
        import time
        trainer.state.training_start_time = time.time()
        
        # 3. Try to iterate
        print("  About to iterate dataloader in trainer context")
        epoch = 0
        trainer.state.epoch = epoch
        
        print("  Calling _train_epoch")
        metrics = trainer._train_epoch(loader, epoch)
        print(f"  ✅ _train_epoch completed! Metrics: {metrics}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_no_hooks()