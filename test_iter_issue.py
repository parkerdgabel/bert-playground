#!/usr/bin/env python3
"""Test to isolate the iterator issue."""

import sys
from pathlib import Path
from transformers import AutoTokenizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.factory import create_dataloader

def test_iter_contexts():
    """Test dataloader iteration in different contexts."""
    
    print("🔍 Testing dataloader iteration contexts")
    
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
    
    print(f"🔍 Loader type: {type(loader)}")
    print(f"🔍 Loader has {len(loader)} batches")
    
    # Test 1: Direct next()
    print("\n1️⃣ Testing direct next()")
    it = iter(loader)
    batch = next(it)
    print(f"✅ Got batch: {list(batch.keys())}")
    
    # Test 2: For loop
    print("\n2️⃣ Testing for loop")
    count = 0
    for batch in loader:
        count += 1
        if count >= 2:
            break
    print(f"✅ Iterated {count} batches")
    
    # Test 3: Manual __iter__ and __next__
    print("\n3️⃣ Testing manual __iter__ and __next__")
    it = loader.__iter__()
    print(f"Iterator type: {type(it)}")
    batch = it.__next__()
    print(f"✅ Got batch via __next__")
    
    # Test 4: Check if loader is being used elsewhere
    print("\n4️⃣ Testing repeated iteration")
    for i in range(2):
        print(f"  Iteration {i}")
        for j, batch in enumerate(loader):
            if j >= 1:
                break
        print(f"  ✅ Completed iteration {i}")

if __name__ == "__main__":
    test_iter_contexts()