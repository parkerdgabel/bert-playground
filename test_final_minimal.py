#!/usr/bin/env python3
"""Absolute minimal test to isolate the issue."""

import sys
from pathlib import Path
from transformers import AutoTokenizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.factory import create_dataloader
from training.core.base import BaseTrainer
from training.core.config import get_quick_test_config
from models.factory import create_model

def test_minimal():
    """Minimal test case."""
    
    print("Creating components...")
    
    # Config
    config = get_quick_test_config()
    config.environment.output_dir = Path("minimal_test")
    config.training.report_to = []
    config.training.num_epochs = 1
    
    # Model
    model = create_model(
        model_type="modernbert_with_head",
        head_type="binary_classification", 
        num_labels=2
    )
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    
    # Data loader
    loader = create_dataloader(
        data_path=Path("data/titanic/train.csv"),
        batch_size=4,
        shuffle=False,
        num_workers=0,
        prefetch_size=0,
        tokenizer=tokenizer,
        split="train"
    )
    
    print(f"Loader: {type(loader)}, batches: {len(loader)}")
    
    # Test iteration BEFORE trainer
    print("\nTesting iteration BEFORE trainer creation:")
    for i, batch in enumerate(loader):
        print(f"  Batch {i}: OK")
        if i >= 1:
            break
    
    # Create trainer
    print("\nCreating trainer...")
    trainer = BaseTrainer(model=model, config=config)
    
    # Test iteration AFTER trainer
    print("\nTesting iteration AFTER trainer creation:")
    for i, batch in enumerate(loader):
        print(f"  Batch {i}: OK")
        if i >= 1:
            break
    
    # Try the actual training method
    print("\nCalling trainer._train_epoch directly...")
    
    # Manually set required state
    trainer.state.epoch = 0
    trainer.state.global_step = 0
    trainer.state.samples_seen = 0
    
    # Call _train_epoch
    result = trainer._train_epoch(loader, 0)
    print(f"✅ _train_epoch completed! Loss: {result['loss']:.4f}")

if __name__ == "__main__":
    test_minimal()