#!/usr/bin/env python3
"""Debug MLX data loader prefetching issue."""

import sys
from pathlib import Path
from transformers import AutoTokenizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.factory import create_dataloader, create_dataset

def test_no_prefetch():
    """Test with prefetching disabled."""
    
    print("🔍 Testing with prefetching DISABLED")
    
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    
    # Create loader with no prefetching
    loader = create_dataloader(
        data_path=Path("data/titanic/train.csv"),
        batch_size=4,
        shuffle=False,
        num_workers=0,  # No workers
        prefetch_size=0,  # No prefetching
        tokenizer=tokenizer,
        split="train"
    )
    
    print(f"🔍 Created loader with {len(loader)} batches")
    
    # Try to iterate
    try:
        for i, batch in enumerate(loader):
            print(f"🔍 Got batch {i}: {list(batch.keys())}")
            if i >= 2:  # Just a few batches
                break
        print("✅ No-prefetch iteration successful!")
    except Exception as e:
        print(f"❌ No-prefetch failed: {e}")
        import traceback
        traceback.print_exc()

def test_single_worker():
    """Test with single worker."""
    
    print("\n🔍 Testing with SINGLE WORKER")
    
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    
    # Create loader with single worker
    loader = create_dataloader(
        data_path=Path("data/titanic/train.csv"),
        batch_size=4,
        shuffle=False,
        num_workers=1,
        prefetch_size=1,
        tokenizer=tokenizer,
        split="train"
    )
    
    print(f"🔍 Created loader with {len(loader)} batches")
    
    # Try to iterate
    try:
        for i, batch in enumerate(loader):
            print(f"🔍 Got batch {i}: {list(batch.keys())}")
            if i >= 2:  # Just a few batches
                break
        print("✅ Single worker iteration successful!")
    except Exception as e:
        print(f"❌ Single worker failed: {e}")
        import traceback
        traceback.print_exc()

def test_manual_iteration():
    """Test manual iteration without prefetching."""
    
    print("\n🔍 Testing MANUAL iteration")
    
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    
    # Create dataset directly
    dataset = create_dataset(
        data_path=Path("data/titanic/train.csv"),
        split="train"
    )
    
    print(f"🔍 Dataset has {len(dataset)} samples")
    
    # Manual batching
    batch_size = 4
    texts = []
    labels = []
    
    for i in range(min(batch_size, len(dataset))):
        sample = dataset[i]
        texts.append(sample['text'])
        labels.append(sample['labels'])
    
    # Tokenize
    tokens = tokenizer(
        texts,
        max_length=512,
        truncation=True,
        padding=True,
        return_tensors="np"
    )
    
    print(f"✅ Manual tokenization successful: {tokens['input_ids'].shape}")

if __name__ == "__main__":
    test_no_prefetch()
    test_single_worker() 
    test_manual_iteration()