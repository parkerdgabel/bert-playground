"""Data ports for dataset and dataloader operations."""

from typing import Protocol, Iterator, Optional, Dict, Any, List, Callable
from domain.entities.dataset import Dataset, DataBatch, DatasetSplit


class DatasetPort(Protocol):
    """Port for dataset operations."""
    
    def load_dataset(
        self,
        name: str,
        split: DatasetSplit,
        **kwargs: Any,
    ) -> Dataset:
        """Load a dataset.
        
        Args:
            name: Dataset name or path
            split: Which split to load
            **kwargs: Additional dataset-specific parameters
            
        Returns:
            Loaded dataset
        """
        ...
    
    def get_dataset_info(
        self,
        name: str,
    ) -> Dict[str, Any]:
        """Get information about a dataset.
        
        Args:
            name: Dataset name
            
        Returns:
            Dictionary with dataset information
        """
        ...
    
    def create_dataset(
        self,
        data: Any,
        split: DatasetSplit,
        name: Optional[str] = None,
    ) -> Dataset:
        """Create a dataset from data.
        
        Args:
            data: Raw data (format depends on implementation)
            split: Dataset split
            name: Optional dataset name
            
        Returns:
            Created dataset
        """
        ...
    
    def save_dataset(
        self,
        dataset: Dataset,
        path: str,
        format: Optional[str] = None,
    ) -> None:
        """Save dataset to disk.
        
        Args:
            dataset: Dataset to save
            path: Save path
            format: Optional format specification
        """
        ...
    
    def filter_dataset(
        self,
        dataset: Dataset,
        predicate: Callable[[Any], bool],
    ) -> Dataset:
        """Filter dataset based on predicate.
        
        Args:
            dataset: Dataset to filter
            predicate: Filter function
            
        Returns:
            Filtered dataset
        """
        ...
    
    def split_dataset(
        self,
        dataset: Dataset,
        splits: Dict[str, float],
        seed: Optional[int] = None,
    ) -> Dict[str, Dataset]:
        """Split dataset into multiple parts.
        
        Args:
            dataset: Dataset to split
            splits: Dictionary of split_name -> fraction
            seed: Random seed
            
        Returns:
            Dictionary of split datasets
        """
        ...


class DataLoaderPort(Protocol):
    """Port for data loading operations."""
    
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
        ...
    
    def __iter__(self) -> Iterator[DataBatch]:
        """Iterate over batches.
        
        Yields:
            Data batches
        """
        ...
    
    def __len__(self) -> int:
        """Get number of batches.
        
        Returns:
            Total number of batches
        """
        ...
    
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
        ...
    
    def reset(
        self,
    ) -> None:
        """Reset dataloader state."""
        ...
    
    def set_epoch(
        self,
        epoch: int,
    ) -> None:
        """Set current epoch for shuffling.
        
        Args:
            epoch: Current epoch number
        """
        ...
    
    def get_stats(
        self,
    ) -> Dict[str, Any]:
        """Get dataloader statistics.
        
        Returns:
            Dictionary with statistics like:
            - 'total_batches': Total number of batches
            - 'samples_seen': Number of samples processed
            - 'average_batch_time': Average time per batch
        """
        ...