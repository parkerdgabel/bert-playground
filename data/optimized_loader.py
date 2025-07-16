"""Optimized data loader for MLX with efficient batching and preprocessing."""

import mlx.core as mx
import mlx.data as dx
from typing import Dict, List, Optional, Tuple, Iterator
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from transformers import AutoTokenizer
import random

from data.text_templates import TitanicTextTemplates


class OptimizedTitanicDataPipeline:
    """Optimized data pipeline for Titanic dataset using MLX-Data."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer_name: str = "answerdotai/ModernBERT-base",
        max_length: int = 256,
        batch_size: int = 32,
        prefetch_size: int = 4,
        num_threads: int = 4,
        is_training: bool = True,
        augment: bool = True,
        cache_size: int = 1000,
    ):
        self.data_path = Path(data_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.batch_size = batch_size
        self.prefetch_size = prefetch_size
        self.num_threads = num_threads
        self.is_training = is_training
        self.augment = augment
        self.cache_size = cache_size
        
        # Initialize text templates
        self.text_converter = TitanicTextTemplates()
        
        # Load and prepare data
        self.df = pd.read_csv(self.data_path)
        self.prepare_data()
        
        # Create MLX data stream
        self.stream = self._create_stream()
        
        logger.info(
            f"Initialized OptimizedTitanicDataPipeline with {len(self.texts)} samples, "
            f"batch_size={batch_size}, prefetch={prefetch_size}, threads={num_threads}"
        )
    
    def prepare_data(self):
        """Prepare data with optimized preprocessing."""
        # Fill missing values efficiently
        self.df['Age'] = self.df['Age'].fillna(self.df['Age'].median())
        self.df['Embarked'] = self.df['Embarked'].fillna(self.df['Embarked'].mode()[0])
        self.df['Fare'] = self.df['Fare'].fillna(self.df['Fare'].median())
        self.df['Cabin'] = self.df['Cabin'].fillna('Unknown')
        
        # Convert to text with optional augmentation
        self.texts = []
        self.labels = []
        
        for _, row in self.df.iterrows():
            if self.augment and self.is_training:
                # Generate multiple text variations
                for _ in range(3):
                    text = self.text_converter.row_to_text(row)
                    self.texts.append(text)
                    if 'Survived' in row:
                        self.labels.append(int(row['Survived']))
                    else:
                        self.labels.append(-1)  # No label for test data
            else:
                text = self.text_converter.row_to_text(row)
                self.texts.append(text)
                if 'Survived' in row:
                    self.labels.append(int(row['Survived']))
                else:
                    self.labels.append(-1)
        
        # Pre-tokenize for efficiency
        logger.info("Pre-tokenizing texts for efficiency...")
        self.tokenized_cache = {}
        self._pretokenize_batch()
    
    def _pretokenize_batch(self):
        """Pre-tokenize texts in batches for efficiency."""
        batch_size = 100
        for i in range(0, len(self.texts), batch_size):
            batch_texts = self.texts[i:i + batch_size]
            batch_tokens = self.tokenizer(
                batch_texts,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='np'
            )
            
            for j, idx in enumerate(range(i, min(i + batch_size, len(self.texts)))):
                self.tokenized_cache[idx] = {
                    'input_ids': batch_tokens['input_ids'][j],
                    'attention_mask': batch_tokens['attention_mask'][j]
                }
    
    def _create_sample_generator(self) -> Iterator[Dict[str, np.ndarray]]:
        """Create generator for samples."""
        indices = list(range(len(self.texts)))
        
        if self.is_training:
            random.shuffle(indices)
        
        for idx in indices:
            # Use pre-tokenized data
            tokens = self.tokenized_cache.get(idx)
            if tokens is None:
                # Fallback to on-demand tokenization
                encoded = self.tokenizer(
                    self.texts[idx],
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='np'
                )
                tokens = {
                    'input_ids': encoded['input_ids'][0],
                    'attention_mask': encoded['attention_mask'][0]
                }
            
            sample = {
                'input_ids': tokens['input_ids'].astype(np.int32),
                'attention_mask': tokens['attention_mask'].astype(np.int32),
            }
            
            if self.labels[idx] != -1:
                sample['labels'] = np.array(self.labels[idx], dtype=np.int32)
            
            yield sample
    
    def _create_stream(self) -> dx.Stream:
        """Create optimized MLX data stream."""
        # Create buffer from generator
        samples = list(self._create_sample_generator())
        buffer = dx.buffer_from_vector(samples)
        
        # Create stream with optimizations
        stream = buffer.to_stream()
        
        # Shuffle for training
        if self.is_training:
            stream = stream.shuffle(buffer_size=self.cache_size)
        
        # Batch efficiently
        stream = stream.batch(self.batch_size)
        
        # Prefetch for performance
        stream = stream.prefetch(self.prefetch_size, num_threads=self.num_threads)
        
        return stream
    
    def get_dataloader(self) -> Iterator[Dict[str, mx.array]]:
        """Get optimized dataloader iterator."""
        def batch_iterator():
            # Recreate stream for each iteration to allow multiple passes
            fresh_stream = self._create_stream()
            for batch in fresh_stream:
                # The batch is already a dictionary with arrays
                mlx_batch = {}
                
                # Convert numpy arrays to MLX arrays
                for key in ['input_ids', 'attention_mask', 'labels']:
                    if key in batch:
                        # Convert to MLX array
                        mlx_batch[key] = mx.array(batch[key])
                        
                        # Ensure labels are 1D
                        if key == 'labels' and len(mlx_batch[key].shape) > 1:
                            mlx_batch[key] = mlx_batch[key].reshape(-1)
                
                yield mlx_batch
        
        return batch_iterator
    
    def get_num_batches(self) -> int:
        """Get number of batches."""
        return (len(self.texts) + self.batch_size - 1) // self.batch_size
    
    def __len__(self) -> int:
        """Get total number of samples."""
        return len(self.texts)


def create_optimized_dataloaders(
    train_path: str,
    val_path: Optional[str] = None,
    test_path: Optional[str] = None,
    tokenizer_name: str = "answerdotai/ModernBERT-base",
    max_length: int = 256,
    batch_size: int = 32,
    num_workers: int = 4,
    augment_train: bool = True,
) -> Tuple[OptimizedTitanicDataPipeline, Optional[OptimizedTitanicDataPipeline], Optional[OptimizedTitanicDataPipeline]]:
    """Create optimized data loaders for train, validation, and test sets."""
    
    # Training loader with augmentation
    train_loader = OptimizedTitanicDataPipeline(
        data_path=train_path,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        batch_size=batch_size,
        prefetch_size=4,
        num_threads=num_workers,
        is_training=True,
        augment=augment_train,
        cache_size=2000,
    )
    
    # Validation loader (no augmentation)
    val_loader = None
    if val_path:
        val_loader = OptimizedTitanicDataPipeline(
            data_path=val_path,
            tokenizer_name=tokenizer_name,
            max_length=max_length,
            batch_size=batch_size,
            prefetch_size=2,
            num_threads=max(1, num_workers // 2),
            is_training=False,
            augment=False,
            cache_size=500,
        )
    
    # Test loader (no augmentation)
    test_loader = None
    if test_path:
        test_loader = OptimizedTitanicDataPipeline(
            data_path=test_path,
            tokenizer_name=tokenizer_name,
            max_length=max_length,
            batch_size=batch_size,
            prefetch_size=2,
            num_threads=max(1, num_workers // 2),
            is_training=False,
            augment=False,
            cache_size=500,
        )
    
    return train_loader, val_loader, test_loader