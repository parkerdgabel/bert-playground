"""MLX implementation of DataLoaderPort."""

from typing import Iterator, List, Any, Dict, Optional
import random
import mlx.core as mx
import numpy as np

from domain.entities.dataset import Dataset, DataBatch, TokenSequence
from domain.ports.data import DataLoaderPort
from adapters.secondary.data.base import BaseDataAdapter
from .dataset import MLXDatasetWrapper
from .transforms import MLXPaddingTransform


class MLXDataLoader(BaseDataAdapter):
    """MLX implementation of DataLoaderPort with efficient batching."""
    
    def __init__(self):
        """Initialize MLX data loader."""
        super().__init__()
        self._dataset_wrapper: Optional[MLXDatasetWrapper] = None
        self._indices: Optional[List[int]] = None
        self._padding_transform = MLXPaddingTransform()
        
    def create_dataloader(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = False,
        num_workers: int = 0,
        prefetch_size: int = 1,
        **kwargs: Any,
    ) -> 'DataLoaderPort':
        """Create a dataloader from dataset.
        
        Args:
            dataset: Source dataset
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of parallel workers (ignored for MLX)
            prefetch_size: Number of batches to prefetch
            **kwargs: Additional loader parameters
            
        Returns:
            Self (for chaining)
        """
        super().create_dataloader(
            dataset, batch_size, shuffle, num_workers, prefetch_size, **kwargs
        )
        
        # Create MLX dataset wrapper
        self._dataset_wrapper = MLXDatasetWrapper(dataset)
        
        # Initialize indices
        self._reset_indices()
        
        return self
    
    def __iter__(self) -> Iterator[DataBatch]:
        """Iterate over batches.
        
        Yields:
            Data batches
        """
        if self._dataset_wrapper is None:
            raise RuntimeError("DataLoader not initialized. Call create_dataloader first.")
        
        # Reset iteration
        self.reset()
        
        # Shuffle if needed
        if self._shuffle:
            self._shuffle_indices()
        
        # Generate batches
        num_samples = len(self._indices)
        for start_idx in range(0, num_samples, self._batch_size):
            end_idx = min(start_idx + self._batch_size, num_samples)
            batch_indices = self._indices[start_idx:end_idx]
            
            # Get batch
            batch = self.get_batch(batch_indices)
            
            # Increment counters
            self._increment_counters(batch)
            
            yield batch
    
    def __len__(self) -> int:
        """Get number of batches.
        
        Returns:
            Total number of batches
        """
        if self._dataset_wrapper is None:
            return 0
        
        num_samples = self._dataset_wrapper.size
        return (num_samples + self._batch_size - 1) // self._batch_size
    
    def get_batch(
        self,
        indices: List[int],
    ) -> DataBatch:
        """Get specific batch by indices.
        
        Args:
            indices: Sample indices
            
        Returns:
            Data batch
        """
        if self._dataset_wrapper is None:
            raise RuntimeError("DataLoader not initialized.")
        
        # Get samples from dataset
        sequences = []
        labels = []
        
        for idx in indices:
            sample = self._dataset_wrapper.get_sample(idx)
            
            # Extract sequence and label
            if isinstance(sample, dict):
                # Handle dictionary format
                seq = self._create_token_sequence(sample)
                sequences.append(seq)
                
                if "label" in sample:
                    labels.append(sample["label"])
                elif "labels" in sample:
                    labels.append(sample["labels"])
            else:
                # Handle tuple format (sequence, label)
                seq, label = sample
                sequences.append(seq)
                labels.append(label)
        
        # Create batch
        batch = DataBatch(
            sequences=sequences,
            labels=labels if labels else None,
        )
        
        # Apply padding to create uniform batch
        batch = self._padding_transform.apply(batch, pad_token_id=self._kwargs.get("pad_token_id", 0))
        
        # Add metadata
        batch.metadata["batch_size"] = len(indices)
        batch.metadata["indices"] = indices
        
        return batch
    
    # Private helper methods
    
    def _reset_indices(self) -> None:
        """Reset indices for iteration."""
        if self._dataset_wrapper is None:
            self._indices = []
        else:
            self._indices = list(range(self._dataset_wrapper.size))
    
    def _shuffle_indices(self) -> None:
        """Shuffle indices based on current epoch."""
        if self._indices is None:
            return
        
        # Use epoch as random seed for reproducibility
        rng = random.Random(self._current_epoch)
        rng.shuffle(self._indices)
    
    def _create_token_sequence(self, sample: Dict[str, Any]) -> TokenSequence:
        """Create TokenSequence from sample dictionary."""
        return TokenSequence(
            input_ids=sample.get("input_ids", []),
            attention_mask=sample.get("attention_mask", []),
            token_type_ids=sample.get("token_type_ids"),
            position_ids=sample.get("position_ids"),
        )


class MLXStreamingDataLoader(MLXDataLoader):
    """Streaming version of MLX data loader for large datasets."""
    
    def __init__(self, buffer_size: int = 10000):
        """Initialize streaming data loader.
        
        Args:
            buffer_size: Size of the streaming buffer
        """
        super().__init__()
        self.buffer_size = buffer_size
        self._buffer: List[Any] = []
    
    def __iter__(self) -> Iterator[DataBatch]:
        """Stream batches with buffering.
        
        Yields:
            Data batches
        """
        if self._dataset_wrapper is None:
            raise RuntimeError("DataLoader not initialized.")
        
        # Reset
        self.reset()
        
        # Stream through dataset
        buffer = []
        
        for i in range(self._dataset_wrapper.size):
            # Add to buffer
            buffer.append(i)
            
            # When buffer is full, shuffle and yield batches
            if len(buffer) >= self.buffer_size or i == self._dataset_wrapper.size - 1:
                if self._shuffle:
                    random.shuffle(buffer)
                
                # Create batches from buffer
                for start_idx in range(0, len(buffer), self._batch_size):
                    end_idx = min(start_idx + self._batch_size, len(buffer))
                    batch_indices = buffer[start_idx:end_idx]
                    
                    if len(batch_indices) == self._batch_size or not self._kwargs.get("drop_last", False):
                        batch = self.get_batch(batch_indices)
                        self._increment_counters(batch)
                        yield batch
                
                # Clear buffer
                buffer = []