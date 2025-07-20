"""MLX-optimized data loader for Apple Silicon.

This module provides a high-performance data loader optimized for MLX by:
- Using direct iteration without complex threading/multiprocessing
- Leveraging unified memory for zero-copy operations
- Supporting lazy evaluation
- Minimizing state management to avoid concurrency issues
"""

from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Any
import random

import mlx.core as mx
from loguru import logger

from ..core.base import KaggleDataset


@dataclass
class MLXLoaderConfig:
    """Configuration for MLX data loader."""
    
    # Batch configuration
    batch_size: int = 32
    shuffle: bool = True
    drop_last: bool = False
    
    # MLX-specific optimization
    use_unified_memory: bool = True
    lazy_evaluation: bool = True
    
    # Tokenization
    max_length: int = 512
    padding: str = "max_length"  # "max_length" or "longest"
    truncation: bool = True


class MLXDataLoader:
    """High-performance data loader optimized for MLX and Apple Silicon.
    
    Features:
    - Direct iteration without complex prefetching (avoids threading issues)
    - Leverages unified memory for zero-copy operations
    - Supports MLX-native tokenization when available
    - Efficient batching with minimal overhead
    """
    
    def __init__(
        self,
        dataset: KaggleDataset,
        config: Optional[MLXLoaderConfig] = None,
        tokenizer=None,
    ):
        """Initialize MLX data loader.
        
        Args:
            dataset: Kaggle dataset instance
            config: Loader configuration
            tokenizer: Optional tokenizer for text processing
        """
        self.dataset = dataset
        self.config = config or MLXLoaderConfig()
        self.tokenizer = tokenizer
        
        # MLX device
        self.device = mx.default_device()
        
        # Initialize indices
        self.indices = list(range(len(self.dataset)))
        if self.config.shuffle:
            random.shuffle(self.indices)
            
        # Calculate number of batches
        if self.config.drop_last:
            self.num_batches = len(self.indices) // self.config.batch_size
        else:
            self.num_batches = (len(self.indices) + self.config.batch_size - 1) // self.config.batch_size
            
        logger.info(
            f"Initialized MLXDataLoader: batch_size={self.config.batch_size}, "
            f"num_batches={self.num_batches}, device={self.device}"
        )
    
    def __len__(self) -> int:
        """Get number of batches."""
        return self.num_batches
    
    def __iter__(self) -> Iterator[Dict[str, mx.array]]:
        """Iterate over batches."""
        logger.debug(f"Starting iteration over {self.num_batches} batches")
        
        # Reshuffle if needed
        if self.config.shuffle:
            random.shuffle(self.indices)
            
        for batch_idx in range(self.num_batches):
            logger.debug(f"Processing batch {batch_idx}/{self.num_batches}")
            
            # Get batch indices
            start_idx = batch_idx * self.config.batch_size
            end_idx = min(start_idx + self.config.batch_size, len(self.indices))
            batch_indices = self.indices[start_idx:end_idx]
            
            # Get samples
            samples = [self.dataset[idx] for idx in batch_indices]
            
            # Collate into batch
            batch = self._collate_samples(samples)
            
            logger.debug(f"Yielding batch {batch_idx} with keys: {list(batch.keys())}")
            yield batch
    
    def _collate_samples(self, samples: list[Dict[str, Any]]) -> Dict[str, mx.array]:
        """Collate samples into a batch.
        
        Args:
            samples: List of sample dictionaries
            
        Returns:
            Dictionary of batched MLX arrays
        """
        if not samples:
            return {}
            
        batch = {}
        
        # Check if we have pre-tokenized data
        has_tokenized_data = ('input_ids' in samples[0] and samples[0]['input_ids'] is not None)
        
        # Handle text data
        if 'text' in samples[0] and not has_tokenized_data:
            texts = [sample['text'] for sample in samples]
            
            if self.tokenizer:
                # Tokenize texts
                tokenized = self._tokenize_batch(texts)
                batch.update(tokenized)
            else:
                raise ValueError("Tokenizer is required for text data")
        
        # Handle pre-tokenized data
        if has_tokenized_data:
            input_ids = self._pad_sequences([sample['input_ids'] for sample in samples])
            batch['input_ids'] = mx.array(input_ids, dtype=mx.int32)
            
        if 'attention_mask' in samples[0] and samples[0]['attention_mask'] is not None:
            attention_masks = self._pad_sequences([sample['attention_mask'] for sample in samples])
            batch['attention_mask'] = mx.array(attention_masks, dtype=mx.int32)
            
        if 'token_type_ids' in samples[0] and samples[0]['token_type_ids'] is not None:
            token_type_ids = self._pad_sequences([sample['token_type_ids'] for sample in samples])
            batch['token_type_ids'] = mx.array(token_type_ids, dtype=mx.int32)
            
        # Handle labels
        if 'labels' in samples[0]:
            labels = [sample['labels'] for sample in samples if sample['labels'] is not None]
            if labels:
                if isinstance(labels[0], (int, float)):
                    # Single label per sample
                    batch['labels'] = mx.array(labels, dtype=mx.float32)
                else:
                    # Multi-label case
                    batch['labels'] = mx.array(labels, dtype=mx.float32)
                    
        # Handle metadata
        if 'metadata' in samples[0]:
            batch['metadata'] = [sample['metadata'] for sample in samples]
            
        return batch
    
    def _tokenize_batch(self, texts: list[str]) -> Dict[str, mx.array]:
        """Tokenize a batch of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Dictionary with tokenized data as MLX arrays
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer required for tokenization")
            
        # Check if we're using MLX embeddings tokenizer
        if hasattr(self.tokenizer, 'backend') and self.tokenizer.backend == 'mlx':
            # Use MLX-native tokenization if available
            encodings = self.tokenizer(
                texts,
                padding=self.config.padding,
                truncation=self.config.truncation,
                max_length=self.config.max_length,
                return_tensors="mlx"  # Return MLX tensors directly
            )
        else:
            # Standard tokenization
            encodings = self.tokenizer(
                texts,
                padding=self.config.padding,
                truncation=self.config.truncation,
                max_length=self.config.max_length,
                return_tensors="np"  # Get numpy arrays first
            )
            
        # Convert to MLX arrays
        tokenized = {}
        for key, values in encodings.items():
            if key in ['input_ids', 'attention_mask', 'token_type_ids']:
                if hasattr(values, 'numpy'):  # If it's already a tensor
                    values = values.numpy()
                tokenized[key] = mx.array(values, dtype=mx.int32)
                
        return tokenized
    
    def _pad_sequences(self, sequences: list[list[int]]) -> list[list[int]]:
        """Pad sequences to uniform length.
        
        Args:
            sequences: List of sequences
            
        Returns:
            Padded sequences
        """
        if not sequences:
            return []
            
        # Determine max length
        if self.config.padding == "max_length":
            max_len = self.config.max_length
        else:  # "longest"
            max_len = max(len(seq) for seq in sequences)
            max_len = min(max_len, self.config.max_length)
            
        # Pad sequences
        padded = []
        for seq in sequences:
            # Convert to list if needed
            if hasattr(seq, 'tolist'):
                seq = seq.tolist()
            elif not isinstance(seq, list):
                seq = list(seq)
                
            if len(seq) > max_len:
                # Truncate
                padded_seq = seq[:max_len]
            else:
                # Pad
                padded_seq = seq + [0] * (max_len - len(seq))
            padded.append(padded_seq)
            
        return padded