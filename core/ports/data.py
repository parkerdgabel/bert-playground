"""Data processing ports for hexagonal architecture.

These ports define the interfaces between the domain layer and the adapters
for data-related operations.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterator, Protocol

from domain.data.data_models import (
    DataConfig, DatasetType, DataStatistics, TextExample, 
    TabularExample, TokenizedExample, DataSplit, DatasetInfo
)


class DataRepository(Protocol):
    """Port for data storage and retrieval."""
    
    def load_data(self, path: Path, format: str = "csv") -> Any:
        """Load raw data from storage."""
        ...
    
    def save_data(self, data: Any, path: Path, format: str = "csv") -> None:
        """Save data to storage."""
        ...
    
    def exists(self, path: Path) -> bool:
        """Check if data exists at path."""
        ...
    
    def get_file_info(self, path: Path) -> Dict[str, Any]:
        """Get metadata about data file."""
        ...


class TokenizationService(Protocol):
    """Port for tokenization operations."""
    
    def tokenize_text(self, text: str, max_length: int = 512) -> Dict[str, Any]:
        """Tokenize a single text."""
        ...
    
    def tokenize_batch(self, texts: List[str], max_length: int = 512) -> Dict[str, Any]:
        """Tokenize a batch of texts."""
        ...
    
    def decode_tokens(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        ...
    
    def get_vocab_size(self) -> int:
        """Get tokenizer vocabulary size."""
        ...


class CacheService(Protocol):
    """Port for caching operations."""
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached data."""
        ...
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Cache data with optional TTL."""
        ...
    
    def exists(self, key: str) -> bool:
        """Check if cache key exists."""
        ...
    
    def clear(self, pattern: Optional[str] = None) -> None:
        """Clear cache entries matching pattern."""
        ...
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache statistics."""
        ...


class DataTransformationService(Protocol):
    """Port for data transformation operations."""
    
    def tabular_to_text(
        self, 
        tabular_data: TabularExample, 
        template: str = "default"
    ) -> TextExample:
        """Convert tabular data to text."""
        ...
    
    def augment_text(self, example: TextExample, strategy: str) -> TextExample:
        """Apply text augmentation."""
        ...
    
    def normalize_data(self, data: Any, config: DataConfig) -> Any:
        """Normalize data according to configuration."""
        ...
    
    def validate_data_quality(self, data: Any) -> List[str]:
        """Validate data quality and return issues."""
        ...


class DataLoaderService(Protocol):
    """Port for data loading operations."""
    
    def create_dataloader(
        self,
        dataset: Any,
        batch_size: int,
        shuffle: bool = False,
        **kwargs
    ) -> Any:
        """Create a data loader for the dataset."""
        ...
    
    def get_batch(self, dataloader: Any) -> Dict[str, Any]:
        """Get next batch from dataloader."""
        ...
    
    def get_loader_stats(self, dataloader: Any) -> Dict[str, Any]:
        """Get statistics about the dataloader."""
        ...


class DatasetFactory(ABC):
    """Abstract factory for creating datasets."""
    
    @abstractmethod
    def create_text_dataset(
        self,
        data_path: Path,
        config: DataConfig,
        dataset_type: DatasetType = DatasetType.TRAIN
    ) -> Any:
        """Create a text dataset."""
        pass
    
    @abstractmethod
    def create_tabular_dataset(
        self,
        data_path: Path,
        config: DataConfig,
        dataset_type: DatasetType = DatasetType.TRAIN
    ) -> Any:
        """Create a tabular dataset."""
        pass
    
    @abstractmethod
    def create_tokenized_dataset(
        self,
        dataset: Any,
        tokenizer: Any,
        config: DataConfig
    ) -> Any:
        """Create a tokenized dataset from raw dataset."""
        pass


class DataStatisticsCalculator(ABC):
    """Abstract calculator for dataset statistics."""
    
    @abstractmethod
    def calculate_text_stats(self, examples: List[TextExample]) -> DataStatistics:
        """Calculate statistics for text data."""
        pass
    
    @abstractmethod
    def calculate_tabular_stats(self, examples: List[TabularExample]) -> DataStatistics:
        """Calculate statistics for tabular data."""
        pass
    
    @abstractmethod 
    def calculate_label_distribution(self, labels: List[Any]) -> Dict[Any, int]:
        """Calculate label distribution."""
        pass
    
    @abstractmethod
    def detect_data_issues(self, examples: List[Any]) -> List[str]:
        """Detect common data quality issues."""
        pass


class DataSplitter(ABC):
    """Abstract data splitter for train/val/test splits."""
    
    @abstractmethod
    def split_dataset(
        self,
        dataset: Any,
        split_config: DataSplit
    ) -> Tuple[Any, Any, Any]:
        """Split dataset into train/validation/test."""
        pass
    
    @abstractmethod
    def stratified_split(
        self,
        dataset: Any,
        labels: List[Any],
        split_config: DataSplit
    ) -> Tuple[Any, Any, Any]:
        """Perform stratified split maintaining label distribution."""
        pass
    
    @abstractmethod
    def temporal_split(
        self,
        dataset: Any,
        timestamps: List[Any],
        split_config: DataSplit
    ) -> Tuple[Any, Any, Any]:
        """Perform temporal split based on timestamps."""
        pass


class BatchCollator(ABC):
    """Abstract batch collator for efficient batching."""
    
    @abstractmethod
    def collate_text_batch(self, examples: List[TextExample]) -> Dict[str, Any]:
        """Collate text examples into a batch."""
        pass
    
    @abstractmethod  
    def collate_tokenized_batch(self, examples: List[TokenizedExample]) -> Dict[str, Any]:
        """Collate tokenized examples into a batch."""
        pass
    
    @abstractmethod
    def pad_sequences(self, sequences: List[List[int]], max_length: int) -> Any:
        """Pad sequences to uniform length."""
        pass


class DataValidationService(ABC):
    """Abstract service for data validation."""
    
    @abstractmethod
    def validate_schema(self, data: Any, schema: Dict[str, Any]) -> List[str]:
        """Validate data against schema."""
        pass
    
    @abstractmethod
    def check_data_consistency(self, datasets: List[Any]) -> List[str]:
        """Check consistency across multiple datasets."""
        pass
    
    @abstractmethod
    def detect_outliers(self, data: Any, threshold: float = 2.0) -> List[int]:
        """Detect outlier samples."""
        pass
    
    @abstractmethod
    def validate_text_quality(self, examples: List[TextExample]) -> List[str]:
        """Validate text quality (encoding, length, etc.)."""
        pass


class DataPreprocessor(ABC):
    """Abstract data preprocessor."""
    
    @abstractmethod
    def preprocess_text(self, text: str, config: DataConfig) -> str:
        """Preprocess raw text."""
        pass
    
    @abstractmethod
    def preprocess_tabular(self, data: Dict[str, Any], config: DataConfig) -> Dict[str, Any]:
        """Preprocess tabular data."""
        pass
    
    @abstractmethod
    def handle_missing_values(self, data: Any, strategy: str = "drop") -> Any:
        """Handle missing values in data."""
        pass
    
    @abstractmethod
    def encode_categorical(self, data: Any, columns: List[str]) -> Any:
        """Encode categorical variables."""
        pass


class StreamingDataService(ABC):
    """Abstract service for streaming data."""
    
    @abstractmethod
    def create_streaming_dataset(
        self,
        data_path: Path,
        chunk_size: int = 1000
    ) -> Iterator[List[Any]]:
        """Create streaming dataset for large files."""
        pass
    
    @abstractmethod
    def process_streaming_batch(self, batch: List[Any]) -> Dict[str, Any]:
        """Process a streaming batch."""
        pass
    
    @abstractmethod
    def get_streaming_stats(self, stream: Iterator) -> Dict[str, Any]:
        """Get statistics from streaming data."""
        pass


# Convenience type aliases for common combinations
DataPipeline = Tuple[DataRepository, TokenizationService, CacheService]
DataProcessingPipeline = Tuple[DataTransformationService, DataValidationService, DataPreprocessor]