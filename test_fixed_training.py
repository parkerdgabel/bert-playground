#!/usr/bin/env python3
"""Test script to verify the training fix works."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from training.core.base import BaseTrainer
from training.core.config import get_quick_test_config
from models.factory import create_model
from data.factory import create_dataloader
from transformers import AutoTokenizer

def test_fixed_training():
    """Test that training works with tokenizer fix."""
    
    print("Testing fixed training pipeline...")
    
    # Create configuration
    config = get_quick_test_config()
    config.environment.output_dir = Path("test_output")
    config.training.num_epochs = 1
    config.data.batch_size = 4
    config.training.logging_steps = 1
    config.environment.experiment_name = "test_fix"
    
    # Create tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    
    # Create model
    print("Creating model...")
    model = create_model(
        model_type="modernbert_with_head",
        head_type="binary_classification", 
        num_labels=2
    )
    
    # Create data loader with tokenizer
    print("Creating data loader...")
    train_loader = create_dataloader(
        data_path=Path("data/titanic/train.csv"),
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=1,
        prefetch_size=1,
        tokenizer=tokenizer,
        split="train"
    )
    
    # Create trainer
    print("Creating trainer...")
    trainer = BaseTrainer(model=model, config=config)
    
    # Run training - this should now work!
    print("Starting training (this should work now)...")
    result = trainer.train(train_dataloader=train_loader)
    
    print(f"✅ Training completed successfully!")
    print(f"Final loss: {result.final_train_loss:.4f}")
    print(f"Total epochs: {result.total_epochs}")
    print(f"Training time: {result.training_time:.2f}s")
    
    return result

if __name__ == "__main__":
    test_fixed_training()