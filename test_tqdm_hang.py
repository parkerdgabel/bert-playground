#!/usr/bin/env python3
"""Test if tqdm is causing the hang with our data loader."""

import sys
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from minimal_trainer_debug import DebugDataLoader
from data.factory import create_dataset
from transformers import AutoTokenizer

def test_tqdm_hang():
    """Test if tqdm hangs when wrapping our data loader."""
    
    print("🔍 Creating dataset and tokenizer...")
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    
    # Create dataset
    dataset = create_dataset(
        data_path=Path("data/titanic/train.csv"),
        split="train"
    )
    
    # Create debug data loader
    debug_loader = DebugDataLoader(dataset, tokenizer, batch_size=2)
    
    print("🔍 Testing manual iteration (should work)...")
    for i, batch in enumerate(debug_loader):
        print(f"Manual iteration batch {i}: {list(batch.keys())}")
        if i >= 0:  # Just one batch
            break
    
    print("🔍 Manual iteration completed successfully")
    
    print("🔍 Testing tqdm wrapping (this might hang)...")
    
    # Try to wrap with tqdm - this is where the hang might occur
    try:
        pbar = tqdm(debug_loader, desc="Testing tqdm")
        print("🔍 tqdm created successfully")
        
        for i, batch in enumerate(pbar):
            print(f"tqdm iteration batch {i}: {list(batch.keys())}")
            if i >= 0:  # Just one batch
                break
                
        print("🔍 tqdm iteration completed successfully")
        
    except Exception as e:
        print(f"🔍 tqdm failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_tqdm_hang()