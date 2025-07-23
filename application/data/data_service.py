"""Domain service for data processing and management.

This module contains the pure business logic for data processing workflows,
free from any framework dependencies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, TypeVar, Generic, Tuple, Iterator, Callable
from enum import Enum
import math

from .models import (
    CompetitionType, DatasetSpec, DataSample, DataBatch, Dataset,
    DatasetRepository, DataValidationResult, DataProcessor
)
from .data_models import DatasetType, TaskDataType, DataStatistics, DataConfig, DataSplit


TArray = TypeVar('TArray')
TDataset = TypeVar('TDataset')
TDataLoader = TypeVar('TDataLoader')


@dataclass
class DatasetInfo:
    """Information about a dataset."""
    
    name: str
    dataset_type: DatasetType
    task_type: TaskDataType
    num_samples: int
    num_features: Optional[int] = None
    statistics: Optional[DataStatistics] = None
    
    # Storage information
    file_path: Optional[str] = None
    format: Optional[str] = None
    size_mb: Optional[float] = None
    
    # Processing information
    is_tokenized: bool = False
    is_cached: bool = False
    cache_path: Optional[str] = None
    
    def get_info_dict(self) -> Dict[str, Any]:
        """Get dataset information as dictionary."""
        info = {
            "name": self.name,
            "type": self.dataset_type.value,
            "task": self.task_type.value,
            "num_samples": self.num_samples,
            "is_tokenized": self.is_tokenized,
            "is_cached": self.is_cached,
        }
        
        if self.num_features is not None:
            info["num_features"] = self.num_features
        if self.file_path is not None:
            info["file_path"] = self.file_path
        if self.size_mb is not None:
            info["size_mb"] = self.size_mb
        if self.statistics is not None:
            info["statistics"] = self.statistics.get_summary()
            
        return info


class DataService(ABC, Generic[TDataset, TDataLoader]):
    """Abstract service for data processing workflows."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.datasets: Dict[DatasetType, TDataset] = {}
        self.statistics: Dict[DatasetType, DataStatistics] = {}
    
    @abstractmethod
    def load_dataset(
        self,
        file_path: str,
        dataset_type: DatasetType = DatasetType.TRAIN
    ) -> TDataset:
        """Load dataset from file."""
        pass
    
    @abstractmethod
    def create_dataloader(
        self,
        dataset: TDataset,
        batch_size: Optional[int] = None,
        shuffle: Optional[bool] = None
    ) -> TDataLoader:
        """Create dataloader from dataset."""
        pass
    
    @abstractmethod
    def tokenize_dataset(
        self,
        dataset: TDataset,
        tokenizer: Any
    ) -> TDataset:
        """Tokenize dataset."""
        pass
    
    @abstractmethod
    def split_dataset(
        self,
        dataset: TDataset,
        split_config: DataSplit
    ) -> Tuple[TDataset, TDataset, TDataset]:
        """Split dataset into train/val/test."""
        pass
    
    def compute_statistics(self, dataset: TDataset) -> DataStatistics:
        """Compute dataset statistics."""
        # This would be implemented by the adapter
        raise NotImplementedError("Must be implemented by adapter")
    
    def validate_dataset(self, dataset: TDataset) -> List[str]:
        """Validate dataset quality."""
        # This would be implemented by the adapter
        raise NotImplementedError("Must be implemented by adapter")
    
    def get_dataset_info(self, dataset_type: DatasetType) -> Optional[DatasetInfo]:
        """Get information about a loaded dataset."""
        if dataset_type not in self.datasets:
            return None
            
        dataset = self.datasets[dataset_type]
        statistics = self.statistics.get(dataset_type)
        
        # This would be implemented by the adapter
        raise NotImplementedError("Must be implemented by adapter")


class BatchProcessor(ABC, Generic[TArray]):
    """Abstract batch processor for efficient data processing."""
    
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
    
    @abstractmethod
    def process_batch(self, batch: List[Any]) -> TArray:
        """Process a batch of examples."""
        pass
    
    def process_dataset(
        self,
        examples: List[Any],
        show_progress: bool = True
    ) -> List[TArray]:
        """Process entire dataset in batches."""
        results = []
        num_batches = math.ceil(len(examples) / self.batch_size)
        
        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, len(examples))
            batch = examples[start_idx:end_idx]
            
            batch_result = self.process_batch(batch)
            results.append(batch_result)
            
        return results


@dataclass
class CacheConfig:
    """Configuration for data caching."""
    
    enable_cache: bool = True
    cache_dir: str = ".cache"
    cache_format: str = "pickle"  # "pickle", "json", "parquet"
    compression: Optional[str] = None  # "gzip", "bz2", "lz4"
    max_cache_size_gb: Optional[float] = None
    cache_ttl_hours: Optional[float] = None  # Time to live
    
    def get_cache_key(self, dataset_name: str, config_hash: str) -> str:
        """Generate cache key for dataset."""
        return f"{dataset_name}_{config_hash}"
    
    def get_cache_path(self, cache_key: str) -> str:
        """Get full cache file path."""
        extension = self.cache_format
        if self.compression:
            extension = f"{extension}.{self.compression}"
        return f"{self.cache_dir}/{cache_key}.{extension}"


class DataCache(ABC):
    """Abstract data cache."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
    
    @abstractmethod
    def exists(self, cache_key: str) -> bool:
        """Check if cache entry exists."""
        pass
    
    @abstractmethod
    def load(self, cache_key: str) -> Any:
        """Load data from cache."""
        pass
    
    @abstractmethod
    def save(self, cache_key: str, data: Any) -> None:
        """Save data to cache."""
        pass
    
    @abstractmethod
    def clear(self, cache_key: Optional[str] = None) -> None:
        """Clear cache (all or specific entry)."""
        pass
    
    def is_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid."""
        if not self.exists(cache_key):
            return False
            
        if self.config.cache_ttl_hours is not None:
            # Check age of cache file (would be implemented by adapter)
            pass
            
        return True


class DataPipeline:
    """Orchestrates the data processing pipeline."""
    
    def __init__(
        self,
        data_service: DataService,
        cache: Optional[DataCache] = None
    ):
        self.data_service = data_service
        self.cache = cache
        self.pipeline_steps = []
    
    def add_step(self, step_name: str, step_function: Callable) -> None:
        """Add processing step to pipeline."""
        self.pipeline_steps.append((step_name, step_function))
    
    def run(
        self,
        input_path: str,
        dataset_type: DatasetType = DatasetType.TRAIN
    ) -> Any:
        """Run the complete pipeline."""
        # Check cache first
        if self.cache:
            cache_key = self._generate_cache_key(input_path)
            if self.cache.is_valid(cache_key):
                return self.cache.load(cache_key)
        
        # Load dataset
        dataset = self.data_service.load_dataset(input_path, dataset_type)
        
        # Run pipeline steps
        for step_name, step_function in self.pipeline_steps:
            dataset = step_function(dataset)
        
        # Save to cache
        if self.cache:
            self.cache.save(cache_key, dataset)
        
        return dataset
    
    def _generate_cache_key(self, input_path: str) -> str:
        """Generate cache key based on input and pipeline configuration."""
        # This would create a hash of the input path and pipeline steps
        # Implementation would be done by adapter
        raise NotImplementedError("Must be implemented by adapter")


class DatasetIterator(ABC, Generic[TArray]):
    """Abstract iterator for efficient data loading."""
    
    def __init__(
        self,
        dataset: Any,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
    
    @abstractmethod
    def __iter__(self) -> Iterator[Dict[str, TArray]]:
        """Iterate over batches."""
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """Get number of batches."""
        pass
    
    @property
    def num_samples(self) -> int:
        """Get total number of samples."""
        raise NotImplementedError("Must be implemented by adapter")


class StreamingDataService(ABC):
    """Service for handling streaming/large datasets."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.buffer_size = 10000  # Examples to keep in memory
    
    @abstractmethod
    def stream_from_file(
        self,
        file_path: str,
        chunk_size: int = 1000
    ) -> Iterator[List[Any]]:
        """Stream data from file in chunks."""
        pass
    
    @abstractmethod
    def stream_from_url(
        self,
        url: str,
        chunk_size: int = 1000
    ) -> Iterator[List[Any]]:
        """Stream data from URL."""
        pass
    
    def create_streaming_dataset(
        self,
        source: str,
        is_url: bool = False
    ) -> Iterator[Any]:
        """Create streaming dataset from source."""
        if is_url:
            stream = self.stream_from_url(source)
        else:
            stream = self.stream_from_file(source)
            
        # Apply processing to stream
        for chunk in stream:
            processed_chunk = self._process_chunk(chunk)
            yield from processed_chunk
    
    def _process_chunk(self, chunk: List[Any]) -> List[Any]:
        """Process a chunk of streaming data."""
        # This would apply any necessary transformations
        return chunk


@dataclass
class DataQualityReport:
    """Report on data quality issues."""
    
    total_samples: int
    valid_samples: int
    
    # Issues found
    missing_labels: int = 0
    invalid_labels: int = 0
    empty_texts: int = 0
    too_long_texts: int = 0
    too_short_texts: int = 0
    duplicate_samples: int = 0
    
    # Warnings
    class_imbalance_ratio: Optional[float] = None
    rare_classes: List[Tuple[Any, int]] = field(default_factory=list)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    @property
    def error_rate(self) -> float:
        """Calculate overall error rate."""
        invalid = self.total_samples - self.valid_samples
        return invalid / self.total_samples if self.total_samples > 0 else 0.0
    
    def add_recommendation(self, recommendation: str) -> None:
        """Add recommendation for data quality improvement."""
        self.recommendations.append(recommendation)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "summary": {
                "total_samples": self.total_samples,
                "valid_samples": self.valid_samples,
                "error_rate": self.error_rate,
            },
            "issues": {
                "missing_labels": self.missing_labels,
                "invalid_labels": self.invalid_labels,
                "empty_texts": self.empty_texts,
                "too_long_texts": self.too_long_texts,
                "too_short_texts": self.too_short_texts,
                "duplicate_samples": self.duplicate_samples,
            },
            "warnings": {
                "class_imbalance_ratio": self.class_imbalance_ratio,
                "rare_classes": self.rare_classes,
            },
            "recommendations": self.recommendations,
        }