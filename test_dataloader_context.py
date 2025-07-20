#!/usr/bin/env python3
"""Debug script to identify where exactly the hang occurs."""

import sys
from pathlib import Path
from transformers import AutoTokenizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.factory import create_dataloader

def test_loader_in_class():
    """Test if the issue is related to being in a class context."""
    
    class FakeTrainer:
        def __init__(self, loader):
            self.loader = loader
            print("FakeTrainer initialized")
        
        def train(self):
            print(f"Starting fake training with {len(self.loader)} batches")
            print("About to iterate...")
            
            # This is where it might hang
            for i, batch in enumerate(self.loader):
                print(f"Got batch {i}")
                if i >= 2:
                    break
            
            print("Iteration complete!")
    
    print("🔍 Testing dataloader in class context")
    
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    
    # Create loader
    loader = create_dataloader(
        data_path=Path("data/titanic/train.csv"),
        batch_size=4,
        shuffle=False,
        num_workers=0,
        prefetch_size=0,
        tokenizer=tokenizer,
        split="train"
    )
    
    print(f"Created loader with {len(loader)} batches")
    
    # Test in class
    trainer = FakeTrainer(loader)
    trainer.train()
    
    print("✅ Class context test passed!")

if __name__ == "__main__":
    test_loader_in_class()