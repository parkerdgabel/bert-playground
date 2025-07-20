#!/usr/bin/env python3
"""Minimal test to isolate the trainer hang issue."""

import sys
from pathlib import Path
import mlx.core as mx
from transformers import AutoTokenizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from training.core.base import BaseTrainer
from training.core.config import get_quick_test_config
from models.factory import create_model
from data.factory import create_dataset

class DebugDataLoader:
    """Debug data loader that just yields one batch."""
    
    def __init__(self, dataset, tokenizer, batch_size=2):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
    
    def __len__(self):
        return 1  # Just one batch for debugging
    
    def __iter__(self):
        print("🔍 DATALOADER: Starting iteration")
        # Get first few samples
        batch_samples = []
        for i in range(min(self.batch_size, len(self.dataset))):
            sample = self.dataset[i]
            batch_samples.append(sample)
        
        # Tokenize batch
        texts = [sample['text'] for sample in batch_samples]
        labels = [sample['labels'] for sample in batch_samples]
        
        tokens = self.tokenizer(
            texts, 
            max_length=512, 
            truncation=True, 
            padding=True, 
            return_tensors="np"
        )
        
        batch = {
            'input_ids': mx.array(tokens['input_ids']),
            'attention_mask': mx.array(tokens['attention_mask']),
            'labels': mx.array(labels)
        }
        
        print(f"🔍 DATALOADER: Yielding batch with keys: {list(batch.keys())}")
        print(f"🔍 DATALOADER: Batch shapes: {[(k, v.shape) for k, v in batch.items()]}")
        yield batch
        print("🔍 DATALOADER: Iteration completed")

def test_minimal_trainer():
    """Test the trainer with absolute minimal setup."""
    
    print("🔍 MAIN: Starting minimal trainer test")
    
    # Create configuration with minimal settings
    config = get_quick_test_config()
    config.environment.output_dir = Path("minimal_debug_output")
    config.training.num_epochs = 1
    config.data.batch_size = 2
    config.training.logging_steps = 1
    config.environment.experiment_name = "minimal_debug"
    
    # Disable callbacks that might cause issues
    config.training.report_to = []  # Disable MLflow
    
    print("🔍 MAIN: Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    
    print("🔍 MAIN: Creating model")
    model = create_model(
        model_type="modernbert_with_head",
        head_type="binary_classification", 
        num_labels=2
    )
    
    print("🔍 MAIN: Creating dataset")
    dataset = create_dataset(
        data_path=Path("data/titanic/train.csv"),
        split="train"
    )
    
    print("🔍 MAIN: Creating debug data loader")
    train_loader = DebugDataLoader(dataset, tokenizer, batch_size=2)
    
    print("🔍 MAIN: Testing data loader iteration")
    for i, batch in enumerate(train_loader):
        print(f"🔍 MAIN: Got batch {i}: {list(batch.keys())}")
        break
    
    print("🔍 MAIN: Creating trainer")
    trainer = BaseTrainer(model=model, config=config)
    
    print("🔍 MAIN: About to call trainer.train() - THIS IS WHERE THE HANG HAPPENS")
    
    # Add more debugging before the actual train call
    import time
    start_time = time.time()
    
    try:
        result = trainer.train(train_dataloader=train_loader)
        print(f"🔍 MAIN: trainer.train() completed in {time.time() - start_time:.2f}s")
        print(f"🔍 MAIN: Result: {result}")
    except Exception as e:
        print(f"🔍 MAIN: trainer.train() failed with exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_minimal_trainer()