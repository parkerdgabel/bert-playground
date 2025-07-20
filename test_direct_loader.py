#!/usr/bin/env python3
"""Test data loader iteration directly to isolate issue."""

import sys
from pathlib import Path
from transformers import AutoTokenizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.factory import create_dataloader

def test_loader_iteration():
    """Test just the data loader iteration."""
    
    print("🔍 Testing data loader iteration only")
    
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    
    # Create loader with no prefetching
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
    
    # Direct iteration test
    print("🔍 Testing direct iteration")
    it = iter(loader)
    print("🔍 Iterator created")
    
    try:
        batch = next(it)
        print(f"✅ Got first batch: {list(batch.keys())}")
    except Exception as e:
        print(f"❌ Failed to get batch: {e}")
        import traceback
        traceback.print_exc()
    
    # For loop test
    print("\n🔍 Testing for loop iteration")
    try:
        for i, batch in enumerate(loader):
            print(f"✅ Got batch {i}")
            if i >= 2:
                break
    except Exception as e:
        print(f"❌ For loop failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_loader_iteration()