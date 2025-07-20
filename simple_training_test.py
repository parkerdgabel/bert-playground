#!/usr/bin/env python3
"""Simple training test with single-threaded data loading."""

import sys
from pathlib import Path
import mlx.core as mx
from transformers import AutoTokenizer
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from training.core.base import BaseTrainer
from training.core.config import get_quick_test_config
from models.factory import create_model
from data.factory import create_dataset

class SimpleDataLoader:
    """Simple single-threaded data loader for testing."""
    
    def __init__(self, dataset, tokenizer, batch_size=4):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
    
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        for i in range(0, len(self.dataset), self.batch_size):
            batch_samples = []
            for j in range(i, min(i + self.batch_size, len(self.dataset))):
                sample = self.dataset[j]
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
            
            yield {
                'input_ids': mx.array(tokens['input_ids']),
                'attention_mask': mx.array(tokens['attention_mask']),
                'labels': mx.array(labels)
            }

def test_simple_training():
    """Test training with simple data loader."""
    
    print("Testing training with simple data loader...")
    
    # Create configuration
    config = get_quick_test_config()
    config.environment.output_dir = Path("simple_test_output")
    config.training.num_epochs = 1
    config.data.batch_size = 4
    config.training.logging_steps = 10
    config.environment.experiment_name = "simple_test"
    
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
    
    # Create dataset
    print("Creating dataset...")
    dataset = create_dataset(
        data_path=Path("data/titanic/train.csv"),
        split="train"
    )
    
    # Create simple data loader
    print("Creating simple data loader...")
    train_loader = SimpleDataLoader(dataset, tokenizer, batch_size=4)
    
    print(f"Data loader has {len(train_loader)} batches")
    
    # Test a few batches
    print("Testing data loader iteration...")
    for i, batch in enumerate(train_loader):
        print(f"Batch {i}: input_ids shape={batch['input_ids'].shape}, labels shape={batch['labels'].shape}")
        if i >= 2:  # Test first 3 batches
            break
    
    print("✅ Data loader works! Starting training...")
    
    # Create trainer
    trainer = BaseTrainer(model=model, config=config)
    
    # Run training
    result = trainer.train(train_dataloader=train_loader)
    
    print(f"✅ Training completed successfully!")
    print(f"Final loss: {result.final_train_loss:.4f}")
    print(f"Total epochs: {result.total_epochs}")
    print(f"Training time: {result.training_time:.2f}s")
    
    return result

if __name__ == "__main__":
    test_simple_training()