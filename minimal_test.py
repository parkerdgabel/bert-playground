#!/usr/bin/env python3
"""Minimal test to isolate the issue."""

import sys
from pathlib import Path
import mlx.core as mx
from transformers import AutoTokenizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.factory import create_dataset

def minimal_test():
    """Minimal test to see if we can iterate through data."""
    
    print("Creating dataset...")
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    
    # Create dataset directly (not data loader)
    dataset = create_dataset(
        data_path=Path("data/titanic/train.csv"),
        split="train"
    )
    
    print(f"Dataset created with {len(dataset)} samples")
    
    # Test getting a few samples
    for i in range(3):
        sample = dataset[i]
        print(f"Sample {i}: keys={list(sample.keys())}")
        
        # Test tokenizing manually
        if 'text' in sample:
            tokens = tokenizer(sample['text'], max_length=512, truncation=True, padding=True, return_tensors="np")
            print(f"  Tokenized shape: {tokens['input_ids'].shape}")
    
    print("✅ Dataset access works!")

if __name__ == "__main__":
    minimal_test()