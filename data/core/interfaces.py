"""
Core interfaces and protocols for the modular MLX dataloader system.
Defines the contracts that all components must follow.
"""

from typing import Protocol, Dict, Any, List, Optional, Union, Tuple, Iterator
from abc import ABC, abstractmethod
import mlx.core as mx
from pathlib import Path


class Transform(Protocol):
    """Protocol for data transformations."""
    
    def __call__(self, data: Any) -> Any:
        """Apply transformation to data."""
        ...
    
    def __repr__(self) -> str:
        """String representation of the transform."""
        ...


class Dataset(Protocol):
    """Protocol for dataset implementations."""
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        ...
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single item from the dataset."""
        ...
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return dataset metadata (schema, stats, etc.)."""
        ...


class TextConverter(Protocol):
    """Protocol for converting data to text."""
    
    def convert(self, data: Dict[str, Any]) -> str:
        """Convert a data sample to text representation."""
        ...
    
    def batch_convert(self, data: List[Dict[str, Any]]) -> List[str]:
        """Convert multiple samples to text."""
        ...


class Tokenizer(Protocol):
    """Protocol for tokenization."""
    
    def encode(
        self,
        text: Union[str, List[str]],
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
        return_attention_mask: bool = True,
    ) -> Dict[str, Any]:
        """Encode text to tokens."""
        ...
    
    @property
    def pad_token_id(self) -> int:
        """Get padding token ID."""
        ...


class Cache(Protocol):
    """Protocol for caching mechanisms."""
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve item from cache."""
        ...
    
    def set(self, key: str, value: Any) -> None:
        """Store item in cache."""
        ...
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        ...
    
    def clear(self) -> None:
        """Clear all cached items."""
        ...


class DataLoader(Protocol):
    """Protocol for data loaders."""
    
    def __iter__(self) -> Iterator[Dict[str, mx.array]]:
        """Iterate over batches."""
        ...
    
    def __len__(self) -> int:
        """Return number of batches."""
        ...
    
    @property
    def batch_size(self) -> int:
        """Get batch size."""
        ...
    
    @property
    def num_samples(self) -> int:
        """Get total number of samples."""
        ...


class DataFormat(ABC):
    """Abstract base class for data format handlers."""
    
    @abstractmethod
    def read(self, path: Path, **kwargs) -> Any:
        """Read data from file."""
        pass
    
    @abstractmethod
    def write(self, data: Any, path: Path, **kwargs) -> None:
        """Write data to file."""
        pass
    
    @abstractmethod
    def validate(self, path: Path) -> bool:
        """Validate if file is in correct format."""
        pass
    
    @abstractmethod
    def infer_schema(self, path: Path) -> Dict[str, Any]:
        """Infer data schema from file."""
        pass


class StreamBuilder(Protocol):
    """Protocol for building MLX data streams."""
    
    def build_stream(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = False,
        num_workers: int = 1,
        prefetch_size: int = 1,
    ) -> Any:  # Returns dx.Stream
        """Build MLX data stream from dataset."""
        ...


class BatchProcessor(Protocol):
    """Protocol for batch processing."""
    
    def process_batch(
        self,
        batch: List[Dict[str, Any]]
    ) -> Dict[str, mx.array]:
        """Process a batch of samples into MLX arrays."""
        ...
    
    def collate(
        self,
        samples: List[Dict[str, Any]]
    ) -> Dict[str, mx.array]:
        """Collate samples into a batch."""
        ...


class DataConfig(Protocol):
    """Protocol for data configuration."""
    
    @property
    def dataset_name(self) -> str:
        """Name of the dataset."""
        ...
    
    @property
    def data_format(self) -> str:
        """Format of the data (csv, json, etc.)."""
        ...
    
    @property
    def text_conversion_strategy(self) -> str:
        """Strategy for text conversion."""
        ...
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        ...
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "DataConfig":
        """Create config from dictionary."""
        ...


class OptimizationProfile:
    """Optimization profile for MLX data loading."""
    
    def __init__(
        self,
        name: str,
        prefetch_size: int,
        num_workers: int,
        buffer_size: int,
        cache_size: Optional[int] = None,
        use_memory_mapping: bool = False,
    ):
        self.name = name
        self.prefetch_size = prefetch_size
        self.num_workers = num_workers
        self.buffer_size = buffer_size
        self.cache_size = cache_size
        self.use_memory_mapping = use_memory_mapping
    
    def __repr__(self) -> str:
        return f"OptimizationProfile(name='{self.name}', workers={self.num_workers}, prefetch={self.prefetch_size})"


# Predefined optimization profiles
OPTIMIZATION_PROFILES = {
    "development": OptimizationProfile(
        name="development",
        prefetch_size=2,
        num_workers=2,
        buffer_size=100,
        cache_size=1000,
    ),
    "training": OptimizationProfile(
        name="training",
        prefetch_size=4,
        num_workers=4,
        buffer_size=1000,
        cache_size=10000,
    ),
    "competition": OptimizationProfile(
        name="competition",
        prefetch_size=8,
        num_workers=8,
        buffer_size=2000,
        cache_size=50000,
        use_memory_mapping=True,
    ),
}


class DataSpec:
    """Specification for a dataset."""
    
    def __init__(
        self,
        name: str,
        task_type: str,
        num_classes: Optional[int] = None,
        feature_columns: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        id_column: Optional[str] = None,
        text_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
        numerical_columns: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.task_type = task_type
        self.num_classes = num_classes
        self.feature_columns = feature_columns or []
        self.target_column = target_column
        self.id_column = id_column
        self.text_columns = text_columns or []
        self.categorical_columns = categorical_columns or []
        self.numerical_columns = numerical_columns or []
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert spec to dictionary."""
        return {
            "name": self.name,
            "task_type": self.task_type,
            "num_classes": self.num_classes,
            "feature_columns": self.feature_columns,
            "target_column": self.target_column,
            "id_column": self.id_column,
            "text_columns": self.text_columns,
            "categorical_columns": self.categorical_columns,
            "numerical_columns": self.numerical_columns,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataSpec":
        """Create spec from dictionary."""
        return cls(**data)
    
    def __repr__(self) -> str:
        return f"DataSpec(name='{self.name}', task='{self.task_type}', features={len(self.feature_columns)})"