"""MLX-optimized data loader for Apple Silicon unified memory.

This module provides high-performance data loading that leverages
Apple Silicon's unified memory architecture for zero-copy operations.
"""

import asyncio
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import mlx.core as mx
from loguru import logger

from ..core.base import DatasetSpec, KaggleDataset


@dataclass
class MLXLoaderConfig:
    """Configuration for MLX data loader."""
    
    # Batch configuration
    batch_size: int = 32
    shuffle: bool = True
    drop_last: bool = False
    
    # Performance optimization
    num_workers: int = 4
    prefetch_size: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    
    # MLX-specific optimization
    use_unified_memory: bool = True
    async_device_transfer: bool = True
    lazy_evaluation: bool = True
    
    # Caching
    enable_caching: bool = True
    cache_size_mb: int = 512
    
    # Tokenization
    max_length: int = 512
    padding: str = "max_length"  # "max_length" or "longest"
    truncation: bool = True


class MLXDataLoader:
    """High-performance data loader optimized for MLX and Apple Silicon.
    
    This loader maximizes throughput by leveraging:
    1. Unified memory architecture for zero-copy operations
    2. Asynchronous prefetching with worker threads
    3. Intelligent batching and caching strategies
    4. MLX-native tensor operations
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
        
        # MLX device and stream management
        self.device = mx.default_device()
        self.stream = mx.default_stream(self.device)
        
        # Internal state
        self._current_epoch = 0
        self._batch_cache: Dict[int, Dict[str, mx.array]] = {}
        self._prefetch_queue = deque(maxlen=self.config.prefetch_size)
        self._worker_pool: Optional[ThreadPoolExecutor] = None
        self._stop_prefetching = threading.Event()
        
        # Performance tracking
        self._start_time = time.time()
        self._samples_processed = 0
        self._batches_processed = 0
        
        # Initialize indices for iteration
        self._initialize_indices()
        
        self.logger = logger.bind(component="MLXDataLoader")
        self.logger.info(
            f"Initialized MLX loader: batch_size={self.config.batch_size}, "
            f"workers={self.config.num_workers}, device={self.device}"
        )
    
    def _initialize_indices(self) -> None:
        """Initialize sample indices for iteration."""
        self.indices = list(range(len(self.dataset)))
        
        if self.config.shuffle:
            import random
            random.shuffle(self.indices)
            
        # Calculate number of batches
        if self.config.drop_last:
            self.num_batches = len(self.indices) // self.config.batch_size
        else:
            self.num_batches = (len(self.indices) + self.config.batch_size - 1) // self.config.batch_size
    
    def __len__(self) -> int:
        """Get number of batches."""
        return self.num_batches
    
    def __iter__(self) -> Iterator[Dict[str, mx.array]]:
        """Iterate over batches."""
        # Start prefetching if using workers
        if self.config.num_workers > 0:
            self._start_prefetching()
            
        try:
            for batch_idx in range(self.num_batches):
                # Get batch indices
                start_idx = batch_idx * self.config.batch_size
                end_idx = min(start_idx + self.config.batch_size, len(self.indices))
                batch_indices = self.indices[start_idx:end_idx]
                
                # Get batch data
                if self.config.num_workers > 0:
                    batch = self._get_prefetched_batch(batch_idx)
                else:
                    batch = self._create_batch(batch_indices)
                
                # Track performance
                self._batches_processed += 1
                self._samples_processed += len(batch_indices)
                
                yield batch
                
        finally:
            # Stop prefetching
            if self.config.num_workers > 0:
                self._stop_prefetching_workers()
                
        # Prepare for next epoch
        self._current_epoch += 1
        if self.config.shuffle:
            import random
            random.shuffle(self.indices)
    
    def _create_batch(self, indices: List[int]) -> Dict[str, mx.array]:
        """Create a batch from sample indices.
        
        Args:
            indices: List of sample indices
            
        Returns:
            Dictionary of batched MLX arrays
        """
        # Check cache first
        cache_key = hash(tuple(indices))
        if self.config.enable_caching and cache_key in self._batch_cache:
            return self._batch_cache[cache_key]
            
        # Get samples
        samples = []
        for idx in indices:
            sample = self.dataset[idx]
            samples.append(sample)
            
        # Convert to batch
        batch = self._collate_samples(samples)
        
        # Cache if enabled
        if self.config.enable_caching:
            self._batch_cache[cache_key] = batch
            
            # Manage cache size
            if len(self._batch_cache) > self.config.cache_size_mb:
                # Remove oldest entries (simple FIFO)
                keys_to_remove = list(self._batch_cache.keys())[:len(self._batch_cache) // 4]
                for key in keys_to_remove:
                    del self._batch_cache[key]
                    
        return batch
    
    def _collate_samples(self, samples: List[Dict[str, Any]]) -> Dict[str, mx.array]:
        """Collate a list of samples into a batch with MLX optimization.
        
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
                # Tokenizer is required for text data when no pre-tokenized data exists
                raise ValueError("Tokenizer is required when loading text data without pre-tokenized input_ids. Please provide a tokenizer in the config.")
        
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
    
    def _tokenize_batch(self, texts: List[str]) -> Dict[str, mx.array]:
        """Tokenize a batch of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Dictionary with tokenized data as MLX arrays
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer required for tokenization")
            
        # Tokenize
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
                tokenized[key] = mx.array(values, dtype=mx.int32)
                
        return tokenized
    
    def _pad_sequences(self, sequences: List[List[int]]) -> List[List[int]]:
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
            # Convert to list if it's a numpy array
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
    
    def _start_prefetching(self) -> None:
        """Start prefetching workers."""
        if self._worker_pool is not None:
            return
            
        self._stop_prefetching.clear()
        self._worker_pool = ThreadPoolExecutor(max_workers=self.config.num_workers)
        
        # Submit prefetch tasks
        for batch_idx in range(min(self.config.prefetch_size, self.num_batches)):
            future = self._worker_pool.submit(self._prefetch_batch, batch_idx)
            self._prefetch_queue.append((batch_idx, future))
            
        self.logger.debug(f"Started prefetching with {self.config.num_workers} workers")
    
    def _prefetch_batch(self, batch_idx: int) -> Dict[str, mx.array]:
        """Prefetch a single batch.
        
        Args:
            batch_idx: Batch index
            
        Returns:
            Batch data
        """
        start_idx = batch_idx * self.config.batch_size
        end_idx = min(start_idx + self.config.batch_size, len(self.indices))
        batch_indices = self.indices[start_idx:end_idx]
        
        return self._create_batch(batch_indices)
    
    def _get_prefetched_batch(self, batch_idx: int) -> Dict[str, mx.array]:
        """Get a prefetched batch.
        
        Args:
            batch_idx: Batch index
            
        Returns:
            Batch data
        """
        # Look for the batch in prefetch queue
        for i, (cached_idx, future) in enumerate(self._prefetch_queue):
            if cached_idx == batch_idx:
                # Get result and remove from queue
                batch = future.result()
                del self._prefetch_queue[i]
                
                # Submit next batch for prefetching
                next_batch_idx = batch_idx + self.config.prefetch_size
                if next_batch_idx < self.num_batches:
                    next_future = self._worker_pool.submit(self._prefetch_batch, next_batch_idx)
                    self._prefetch_queue.append((next_batch_idx, next_future))
                    
                return batch
                
        # Fallback to synchronous creation
        self.logger.warning(f"Batch {batch_idx} not prefetched, creating synchronously")
        start_idx = batch_idx * self.config.batch_size
        end_idx = min(start_idx + self.config.batch_size, len(self.indices))
        batch_indices = self.indices[start_idx:end_idx]
        return self._create_batch(batch_indices)
    
    def _stop_prefetching_workers(self) -> None:
        """Stop prefetching workers."""
        if self._worker_pool is None:
            return
            
        self._stop_prefetching.set()
        
        # Cancel pending futures
        for _, future in self._prefetch_queue:
            future.cancel()
            
        # Shutdown worker pool
        self._worker_pool.shutdown(wait=True)
        self._worker_pool = None
        self._prefetch_queue.clear()
        
        self.logger.debug("Stopped prefetching workers")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        elapsed_time = time.time() - self._start_time
        
        return {
            'batches_processed': self._batches_processed,
            'samples_processed': self._samples_processed,
            'elapsed_time': elapsed_time,
            'batches_per_second': self._batches_processed / elapsed_time if elapsed_time > 0 else 0,
            'samples_per_second': self._samples_processed / elapsed_time if elapsed_time > 0 else 0,
            'current_epoch': self._current_epoch,
            'cache_size': len(self._batch_cache),
            'device': str(self.device),
            'config': {
                'batch_size': self.config.batch_size,
                'num_workers': self.config.num_workers,
                'prefetch_size': self.config.prefetch_size,
                'use_unified_memory': self.config.use_unified_memory,
            }
        }
    
    def clear_cache(self) -> None:
        """Clear the batch cache."""
        self._batch_cache.clear()
        self.logger.info("Cleared batch cache")
    
    def __del__(self):
        """Cleanup when loader is destroyed."""
        if hasattr(self, '_worker_pool') and self._worker_pool is not None:
            self._stop_prefetching_workers()