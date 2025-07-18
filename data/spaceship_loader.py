"""
Data loader for Spaceship Titanic dataset using MLX-Data.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import mlx.core as mx
import mlx.data as dx
from loguru import logger

from data.spaceship_text_templates import SpaceshipTitanicTextConverter, create_spaceship_dataset_splits
from embeddings.tokenizer_wrapper import TokenizerWrapper


class SpaceshipTitanicDataset:
    """Dataset class for Spaceship Titanic data."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: TokenizerWrapper,
        max_length: int = 256,
        augment: bool = False,
        n_augmentations: int = 2,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize Spaceship Titanic dataset.
        
        Args:
            data_path: Path to CSV file
            tokenizer: Tokenizer wrapper
            max_length: Maximum sequence length
            augment: Whether to augment data
            n_augmentations: Number of augmentations per sample
            cache_dir: Directory for caching tokenized data
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        self.n_augmentations = n_augmentations
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Load and convert data
        self._load_data()
    
    def _load_data(self):
        """Load and convert data to text."""
        # Check cache first
        if self.cache_dir:
            cache_file = self.cache_dir / f"spaceship_{self.data_path.stem}_cache.json"
            if cache_file.exists():
                logger.info(f"Loading cached data from {cache_file}")
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                self.texts = cache_data['texts']
                self.labels = cache_data['labels']
                self.passenger_ids = cache_data.get('passenger_ids', [None] * len(self.texts))
                return
        
        # Load CSV
        df = pd.read_csv(self.data_path)
        
        # Create text converter
        converter = SpaceshipTitanicTextConverter(augment=self.augment)
        
        # Convert to text
        self.texts = []
        self.labels = []
        self.passenger_ids = []
        
        for _, row in df.iterrows():
            # Original text
            text = converter.convert_row_to_text(row)
            self.texts.append(text)
            
            # Label (if available)
            if 'Transported' in row:
                self.labels.append(int(row['Transported']))
            else:
                self.labels.append(-1)  # Test data
            
            self.passenger_ids.append(row['PassengerId'])
            
            # Augmentations
            if self.augment and 'Transported' in row:
                aug_texts = converter.create_augmented_texts(row, self.n_augmentations)
                self.texts.extend(aug_texts)
                self.labels.extend([int(row['Transported'])] * self.n_augmentations)
                self.passenger_ids.extend([f"{row['PassengerId']}_aug{i}" for i in range(self.n_augmentations)])
        
        # Cache if directory provided
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_data = {
                'texts': self.texts,
                'labels': self.labels,
                'passenger_ids': self.passenger_ids
            }
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
            logger.info(f"Cached data to {cache_file}")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        """Get a single tokenized example."""
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize using batch_encode_plus for single text
        encoding = self.tokenizer.batch_encode_plus(
            [text],  # Wrap in list
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="mlx"
        )
        
        return {
            'input_ids': encoding['input_ids'][0],  # Remove batch dimension
            'attention_mask': encoding['attention_mask'][0],  # Remove batch dimension
            'label': label,
            'passenger_id': self.passenger_ids[idx]
        }


def create_spaceship_data_loader(
    data_path: str,
    tokenizer: TokenizerWrapper,
    batch_size: int = 32,
    max_length: int = 256,
    shuffle: bool = True,
    augment: bool = False,
    n_augmentations: int = 2,
    num_workers: int = 4,
    prefetch_size: int = 4,
    cache_dir: Optional[str] = None,
) -> dx.Stream:
    """
    Create MLX data loader for Spaceship Titanic dataset.
    
    Args:
        data_path: Path to CSV file
        tokenizer: Tokenizer wrapper
        batch_size: Batch size
        max_length: Maximum sequence length
        shuffle: Whether to shuffle data
        augment: Whether to augment data
        n_augmentations: Number of augmentations
        num_workers: Number of worker threads
        prefetch_size: Number of batches to prefetch
        cache_dir: Directory for caching
        
    Returns:
        MLX data stream
    """
    # Create dataset
    dataset = SpaceshipTitanicDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        augment=augment,
        n_augmentations=n_augmentations,
        cache_dir=cache_dir
    )
    
    # Pre-tokenize all data for efficiency
    logger.info(f"Pre-tokenizing {len(dataset)} samples...")
    
    # Separate arrays for each field
    input_ids_list = []
    attention_mask_list = []
    label_list = []
    
    for i in range(len(dataset)):
        item = dataset[i]
        # Convert to numpy/list if MLX array
        if hasattr(item['input_ids'], 'tolist'):
            input_ids_list.append(item['input_ids'].tolist())
            attention_mask_list.append(item['attention_mask'].tolist())
        else:
            input_ids_list.append(item['input_ids'])
            attention_mask_list.append(item['attention_mask'])
        label_list.append(int(item['label']))
    
    # Convert to MLX arrays
    input_ids_array = mx.array(input_ids_list, dtype=mx.int32)  # Ensure int32 for embedding lookup
    attention_mask_array = mx.array(attention_mask_list, dtype=mx.float32)  # Use float32 for buffer compatibility
    label_array = mx.array(label_list, dtype=mx.int32)
    
    # Create indices for iteration
    num_samples = len(dataset)
    indices = mx.arange(num_samples)
    
    # Shuffle indices if requested
    if shuffle:
        indices = mx.random.permutation(indices)
    
    # Create batches manually
    def batch_generator():
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            yield {
                'input_ids': input_ids_array[batch_indices],
                'attention_mask': attention_mask_array[batch_indices],
                'label': label_array[batch_indices]
            }
    
    # Convert generator to stream
    batches = list(batch_generator())
    buffer = dx.buffer_from_vector(batches)
    stream = buffer.to_stream()
    
    # Prefetch
    stream = stream.prefetch(prefetch_size, num_workers)
    
    return stream


def create_spaceship_dataloaders(
    train_path: str,
    val_path: Optional[str] = None,
    test_path: Optional[str] = None,
    tokenizer: TokenizerWrapper = None,
    batch_size: int = 32,
    max_length: int = 256,
    augment_train: bool = True,
    n_augmentations: int = 2,
    num_workers: int = 4,
    prefetch_size: int = 4,
    val_split: float = 0.2,
    cache_dir: Optional[str] = None,
) -> Dict[str, dx.Stream]:
    """
    Create train/val/test data loaders.
    
    Args:
        train_path: Path to training CSV
        val_path: Optional validation CSV path
        test_path: Optional test CSV path
        tokenizer: Tokenizer (will create if None)
        batch_size: Batch size
        max_length: Maximum sequence length
        augment_train: Whether to augment training data
        n_augmentations: Number of augmentations
        num_workers: Number of workers
        prefetch_size: Prefetch size
        val_split: Validation split if val_path not provided
        cache_dir: Cache directory
        
    Returns:
        Dictionary with 'train', 'val', and optionally 'test' loaders
    """
    # Create tokenizer if not provided
    if tokenizer is None:
        tokenizer = TokenizerWrapper(
            model_name="mlx-community/answerdotai-ModernBERT-base-4bit"
        )
    
    # If no val_path, split train data
    if val_path is None:
        logger.info(f"Creating train/val split with {val_split:.0%} validation")
        splits = create_spaceship_dataset_splits(
            train_path=train_path,
            val_split=val_split,
            augment_train=False,  # We'll augment in the loader
            n_augmentations=0
        )
        
        # Save temporary files
        import tempfile
        temp_dir = Path(tempfile.mkdtemp())
        train_path = temp_dir / "train_split.csv"
        val_path = temp_dir / "val_split.csv"
        
        # Need to add PassengerId for compatibility
        splits['train']['PassengerId'] = [f"train_{i}" for i in range(len(splits['train']))]
        splits['val']['PassengerId'] = [f"val_{i}" for i in range(len(splits['val']))]
        
        # Rename label to Transported
        splits['train']['Transported'] = splits['train']['label']
        splits['val']['Transported'] = splits['val']['label']
        
        splits['train'].to_csv(train_path, index=False)
        splits['val'].to_csv(val_path, index=False)
    
    loaders = {}
    
    # Create train loader
    loaders['train'] = create_spaceship_data_loader(
        data_path=train_path,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        shuffle=True,
        augment=augment_train,
        n_augmentations=n_augmentations,
        num_workers=num_workers,
        prefetch_size=prefetch_size,
        cache_dir=cache_dir
    )
    
    # Create val loader
    loaders['val'] = create_spaceship_data_loader(
        data_path=val_path,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        shuffle=False,
        augment=False,
        num_workers=num_workers,
        prefetch_size=prefetch_size,
        cache_dir=cache_dir
    )
    
    # Create test loader if provided
    if test_path:
        loaders['test'] = create_spaceship_data_loader(
            data_path=test_path,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            shuffle=False,
            augment=False,
            num_workers=num_workers,
            prefetch_size=prefetch_size,
            cache_dir=cache_dir
        )
    
    return loaders


if __name__ == "__main__":
    # Test the data loader
    from embeddings.tokenizer_wrapper import TokenizerWrapper
    
    # Create tokenizer
    tokenizer = TokenizerWrapper("mlx-community/answerdotai-ModernBERT-base-4bit")
    
    # Create loaders
    loaders = create_spaceship_dataloaders(
        train_path="data/spaceship-titanic/train.csv",
        test_path="data/spaceship-titanic/test.csv",
        tokenizer=tokenizer,
        batch_size=4,
        max_length=128,
        augment_train=True,
        n_augmentations=1
    )
    
    # Test train loader
    print("Testing train loader...")
    for i, batch in enumerate(loaders['train']):
        print(f"\nBatch {i}:")
        print(f"  Input IDs shape: {batch['input_ids'].shape}")
        print(f"  Attention mask shape: {batch['attention_mask'].shape}")
        print(f"  Labels shape: {batch['label'].shape}")
        print(f"  Labels: {batch['label']}")
        
        if i >= 2:  # Show first 3 batches
            break