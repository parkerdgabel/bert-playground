#!/usr/bin/env python3
"""Debug script to investigate training stall issue."""

import pdb
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from training.core.base import BaseTrainer
from training.core.config import get_quick_test_config
from models.factory import create_model
from data.factory import create_dataloader

def debug_training():
    """Debug the training stall issue using pdb."""
    
    print("Setting up debug training...")
    
    # Create configuration
    config = get_quick_test_config()
    config.environment.output_dir = Path("debug_output")
    config.training.num_epochs = 1
    config.data.batch_size = 4
    config.training.logging_steps = 1
    
    # Create model
    model = create_model(
        model_type="modernbert_with_head",
        head_type="binary_classification", 
        num_labels=2
    )
    
    # Create data loader
    train_loader = create_dataloader(
        data_path=Path("data/titanic/train.csv"),
        batch_size=config.data.batch_size,
        shuffle=config.data.shuffle_train,
        num_workers=config.data.num_workers,
        prefetch_size=config.data.prefetch_size,
        split="train"
    )
    
    # Create trainer
    trainer = BaseTrainer(model=model, config=config)
    
    print("Starting pdb session...")
    pdb.set_trace()
    
    # Run training - this is where it gets stuck
    result = trainer.train(train_dataloader=train_loader)
    
    return result

if __name__ == "__main__":
    debug_training()