"""Domain interfaces for data layer - contracts that infrastructure must implement.

This module defines the ports (interfaces) that the domain layer requires
from the infrastructure adapters, following the hexagonal architecture pattern.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Union

from .models import DatasetSpec, DataSample, DataBatch, Dataset, DataValidationResult


class DataRepository(ABC):
    """Interface for data storage and retrieval operations."""
    
    @abstractmethod
    def load_raw_data(self, file_path: str) -> Any:
        """Load raw data from file system."""
        pass
    
    @abstractmethod
    def save_processed_data(self, data: Any, file_path: str) -> None:
        """Save processed data to file system."""
        pass
    
    @abstractmethod
    def exists(self, file_path: str) -> bool:
        """Check if file exists."""
        pass


class DataCache(ABC):
    """Interface for data caching operations."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get cached data by key."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Cache data with optional TTL."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete cached data."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cached data."""
        pass


class TokenizerAdapter(ABC):
    """Interface for text tokenization operations."""
    
    @abstractmethod
    def tokenize(
        self,
        texts: Union[str, List[str]],
        max_length: int = 512,
        padding: bool = True,
        truncation: bool = True
    ) -> Dict[str, Any]:
        """Tokenize text(s) into model inputs."""
        pass
    
    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        pass
    
    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        pass


class DataLoader(ABC):
    """Interface for batch data loading operations."""
    
    @abstractmethod
    def __iter__(self) -> Iterator[DataBatch]:
        """Iterate over data batches."""
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """Get number of batches."""
        pass
    
    @property
    @abstractmethod
    def batch_size(self) -> int:
        """Get batch size."""
        pass


class DataValidatorAdapter(ABC):
    """Interface for data validation operations."""
    
    @abstractmethod
    def validate_file_format(self, file_path: str) -> DataValidationResult:
        """Validate file format and structure."""
        pass
    
    @abstractmethod
    def validate_data_quality(self, data: Any) -> DataValidationResult:
        """Validate data quality and integrity."""
        pass
    
    @abstractmethod
    def validate_schema(self, data: Any, schema: Dict[str, Any]) -> DataValidationResult:
        """Validate data against schema."""
        pass


class TextProcessorAdapter(ABC):
    """Interface for text processing operations."""
    
    @abstractmethod
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        pass
    
    @abstractmethod
    def extract_features(self, text: str) -> Dict[str, Any]:
        """Extract text features."""
        pass


class DataAugmentationAdapter(ABC):
    """Interface for data augmentation operations."""
    
    @abstractmethod
    def augment_sample(self, sample: DataSample) -> DataSample:
        """Augment a single data sample."""
        pass
    
    @abstractmethod
    def augment_batch(self, batch: DataBatch) -> DataBatch:
        """Augment a batch of data samples."""
        pass


class TemplateEngine(ABC):
    """Interface for data-to-text template operations."""
    
    @abstractmethod
    def apply_template(self, template_name: str, data: Dict[str, Any]) -> str:
        """Apply named template to data."""
        pass
    
    @abstractmethod
    def register_template(self, name: str, template: Any) -> None:
        """Register a new template."""
        pass
    
    @abstractmethod
    def list_templates(self) -> List[str]:
        """List available templates."""
        pass


class DatasetFactory(ABC):
    """Interface for dataset creation operations."""
    
    @abstractmethod
    def create_dataset(
        self,
        spec: DatasetSpec,
        split: str = "train",
        cache: Optional[DataCache] = None
    ) -> Dataset:
        """Create dataset from specification."""
        pass
    
    @abstractmethod 
    def create_dataloader(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = False,
        tokenizer: Optional[TokenizerAdapter] = None
    ) -> DataLoader:
        """Create data loader from dataset."""
        pass


class MetricsCollector(ABC):
    """Interface for collecting data processing metrics."""
    
    @abstractmethod
    def record_processing_time(self, operation: str, duration: float) -> None:
        """Record processing time for an operation."""
        pass
    
    @abstractmethod
    def record_data_quality_metrics(self, metrics: Dict[str, Any]) -> None:
        """Record data quality metrics."""
        pass
    
    @abstractmethod
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics."""
        pass


class FileSystemAdapter(ABC):
    """Interface for file system operations."""
    
    @abstractmethod
    def read_file(self, path: str) -> Any:
        """Read file from file system."""
        pass
    
    @abstractmethod
    def write_file(self, path: str, data: Any) -> None:
        """Write file to file system."""
        pass
    
    @abstractmethod
    def list_files(self, directory: str, pattern: Optional[str] = None) -> List[str]:
        """List files in directory matching pattern."""
        pass
    
    @abstractmethod
    def create_directory(self, path: str) -> None:
        """Create directory if it doesn't exist."""
        pass


class ComputeBackendAdapter(ABC):
    """Interface for compute backend operations (MLX, etc.)."""
    
    @abstractmethod
    def create_array(self, data: Any, dtype: Optional[str] = None) -> Any:
        """Create array on compute backend."""
        pass
    
    @abstractmethod
    def concatenate(self, arrays: List[Any], axis: int = 0) -> Any:
        """Concatenate arrays."""
        pass
    
    @abstractmethod
    def pad_sequences(
        self,
        sequences: List[List[int]],
        max_length: int,
        pad_value: int = 0
    ) -> Any:
        """Pad sequences to uniform length."""
        pass