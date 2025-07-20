#!/usr/bin/env python3
"""Test with the real MLX data loader to see if the fix works."""

import sys
from pathlib import Path
import mlx.core as mx
from transformers import AutoTokenizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from training.core.base import BaseTrainer
from training.core.config import get_quick_test_config
from models.factory import create_model
from data.factory import create_dataloader

def test_real_dataloader():
    """Test with the real MLX data loader."""
    
    print("🔍 MAIN: Testing with real MLX data loader")
    
    # Create configuration
    config = get_quick_test_config()
    config.environment.output_dir = Path("real_loader_test_output")
    config.training.num_epochs = 1
    config.data.batch_size = 4
    config.training.logging_steps = 1
    config.environment.experiment_name = "real_loader_test"
    config.training.report_to = []  # Disable MLflow
    
    print("🔍 MAIN: Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    
    print("🔍 MAIN: Creating model")
    model = create_model(
        model_type="modernbert_with_head",
        head_type="binary_classification", 
        num_labels=2
    )
    
    print("🔍 MAIN: Creating real MLX data loader")
    train_loader = create_dataloader(
        data_path=Path("data/titanic/train.csv"),
        batch_size=4,
        shuffle=False,
        num_workers=1,  # Minimal workers to reduce complexity
        prefetch_size=1,  # Minimal prefetch
        tokenizer=tokenizer,
        split="train"
    )
    
    print(f"🔍 MAIN: Data loader created with {len(train_loader)} batches")
    
    print("🔍 MAIN: Creating trainer")
    trainer = BaseTrainer(model=model, config=config)
    
    print("🔍 MAIN: Starting training with real MLX data loader")
    
    try:
        result = trainer.train(train_dataloader=train_loader)
        print(f"🔍 MAIN: Training completed successfully!")
        print(f"🔍 MAIN: Final loss: {result.final_train_loss}")
    except Exception as e:
        print(f"🔍 MAIN: Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_real_dataloader()