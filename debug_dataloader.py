#!/usr/bin/env python3
"""Debug script to test data loader separately."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.factory import create_dataloader
from training.core.config import get_quick_test_config
from transformers import AutoTokenizer

def test_dataloader():
    """Test the data loader to see where it gets stuck."""
    
    print("Creating data loader...")
    
    # Create configuration
    config = get_quick_test_config()
    
    # Create tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    
    # Create data loader with minimal settings
    train_loader = create_dataloader(
        data_path=Path("data/titanic/train.csv"),
        batch_size=2,  # Very small batch
        shuffle=False,  # No shuffling
        num_workers=1,  # Single worker
        prefetch_size=1,  # Minimal prefetch
        tokenizer=tokenizer,  # Add tokenizer
        split="train"
    )
    
    print(f"Data loader created with {len(train_loader)} batches")
    print("Attempting to iterate through data loader...")
    
    try:
        # Test iteration
        for i, batch in enumerate(train_loader):
            print(f"Batch {i}: keys={list(batch.keys())}, shapes={[(k, v.shape if hasattr(v, 'shape') else type(v)) for k, v in batch.items()]}")
            
            if i >= 2:  # Only test first few batches
                break
                
        print("✓ Data loader iteration successful!")
        
    except Exception as e:
        print(f"✗ Data loader failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataloader()