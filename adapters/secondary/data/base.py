"""Base data adapter with common functionality."""

from abc import ABC
from typing import Any, Dict, Optional, List, Iterator
from domain.entities.dataset import Dataset, DataBatch, DatasetSplit
from ports.secondary.data import DataLoaderPort


class BaseDataAdapter(ABC, DataLoaderPort):
    """Base implementation of DataLoaderPort with common functionality."""
    
    def __init__(self):
        """Initialize base data adapter."""
        self._dataset: Optional[Dataset] = None
        self._batch_size: int = 1
        self._shuffle: bool = False
        self._num_workers: int = 0
        self._prefetch_size: int = 1
        self._current_epoch: int = 0
        self._iteration_count: int = 0
        self._total_samples_seen: int = 0
        
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
            num_workers: Number of parallel workers
            prefetch_size: Number of batches to prefetch
            **kwargs: Additional loader parameters
            
        Returns:
            Self (for chaining)
        """
        self._dataset = dataset
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._num_workers = num_workers
        self._prefetch_size = prefetch_size
        self._kwargs = kwargs
        
        # Reset counters
        self._iteration_count = 0
        self._total_samples_seen = 0
        
        return self
    
    def reset(self) -> None:
        """Reset dataloader state."""
        self._iteration_count = 0
    
    def set_epoch(self, epoch: int) -> None:
        """Set current epoch for shuffling.
        
        Args:
            epoch: Current epoch number
        """
        self._current_epoch = epoch
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataloader statistics.
        
        Returns:
            Dictionary with statistics
        """
        total_batches = len(self) if self._dataset else 0
        
        return {
            "total_batches": total_batches,
            "samples_seen": self._total_samples_seen,
            "current_epoch": self._current_epoch,
            "iteration_count": self._iteration_count,
            "batch_size": self._batch_size,
            "shuffle": self._shuffle,
            "num_workers": self._num_workers,
            "prefetch_size": self._prefetch_size,
        }
    
    def _increment_counters(self, batch: DataBatch) -> None:
        """Increment internal counters after yielding a batch."""
        self._iteration_count += 1
        self._total_samples_seen += batch.batch_size