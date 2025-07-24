"""Secondary data port - Data loading and processing backend.

This port defines the data interface that the application core uses
for loading and processing data. It's a driven port implemented by
adapters for different data sources and formats.
"""

from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Protocol, Tuple, runtime_checkable
from typing_extensions import TypeAlias

from infrastructure.di import port

# Type aliases
DataSource: TypeAlias = str | Any  # Path, URL, or data source identifier


class DataFormat(Enum):
    """Supported data formats."""
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"
    TEXT = "text"
    HUGGINGFACE = "huggingface"


class DataSplit(Enum):
    """Data split types."""
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    PREDICTION = "prediction"


@port()
@runtime_checkable
class DataLoaderPort(Protocol):
    """Secondary port for data loading operations.
    
    This interface is implemented by adapters that handle
    specific data sources and formats.
    """
    
    @property
    def supported_formats(self) -> List[DataFormat]:
        """List of data formats this loader supports."""
        ...
    
    def load_data(
        self,
        source: DataSource,
        format: DataFormat,
        split: Optional[DataSplit] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Load data from source.
        
        Args:
            source: Data source (path, URL, identifier)
            format: Data format
            split: Optional data split
            **kwargs: Format-specific options
            
        Returns:
            Dictionary containing loaded data and metadata
        """
        ...
    
    def create_batches(
        self,
        data: Dict[str, Any],
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        **kwargs: Any,
    ) -> Iterator[Dict[str, Any]]:
        """Create batches from loaded data.
        
        Args:
            data: Loaded data dictionary
            batch_size: Size of each batch
            shuffle: Whether to shuffle data
            drop_last: Whether to drop last incomplete batch
            **kwargs: Additional batching options
            
        Returns:
            Iterator over batches
        """
        ...
    
    def preprocess_batch(
        self,
        batch: Dict[str, Any],
        preprocessing_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Preprocess a batch of data.
        
        Args:
            batch: Raw batch data
            preprocessing_config: Preprocessing configuration
            
        Returns:
            Preprocessed batch
        """
        ...
    
    def get_dataset_info(
        self,
        source: DataSource,
        format: DataFormat,
    ) -> Dict[str, Any]:
        """Get information about a dataset without loading it.
        
        Args:
            source: Data source
            format: Data format
            
        Returns:
            Dataset information (size, features, etc.)
        """
        ...
    
    def validate_data(
        self,
        data: Dict[str, Any],
        validation_rules: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        """Validate loaded data.
        
        Args:
            data: Data to validate
            validation_rules: Validation rules to apply
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        ...


@port()
@runtime_checkable
class DataTransformPort(Protocol):
    """Secondary port for data transformation operations."""
    
    def tokenize_text(
        self,
        texts: List[str],
        tokenizer_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Tokenize text data.
        
        Args:
            texts: List of text strings
            tokenizer_config: Tokenizer configuration
            
        Returns:
            Dictionary with tokenized data
        """
        ...
    
    def convert_tabular_to_text(
        self,
        tabular_data: Dict[str, Any],
        template: str,
        column_mapping: Dict[str, str],
    ) -> List[str]:
        """Convert tabular data to text format.
        
        Args:
            tabular_data: Tabular data dictionary
            template: Text template with placeholders
            column_mapping: Mapping of columns to template vars
            
        Returns:
            List of generated text strings
        """
        ...
    
    def augment_data(
        self,
        data: Dict[str, Any],
        augmentation_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply data augmentation.
        
        Args:
            data: Data to augment
            augmentation_config: Augmentation configuration
            
        Returns:
            Augmented data
        """
        ...
    
    def normalize_features(
        self,
        features: Dict[str, Any],
        normalization_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Normalize feature values.
        
        Args:
            features: Feature dictionary
            normalization_config: Normalization configuration
            
        Returns:
            Normalized features
        """
        ...


@port()
@runtime_checkable  
class DataCachePort(Protocol):
    """Secondary port for data caching operations."""
    
    def cache_exists(
        self,
        cache_key: str,
    ) -> bool:
        """Check if cached data exists.
        
        Args:
            cache_key: Cache identifier
            
        Returns:
            True if cache exists
        """
        ...
    
    def load_from_cache(
        self,
        cache_key: str,
    ) -> Optional[Dict[str, Any]]:
        """Load data from cache.
        
        Args:
            cache_key: Cache identifier
            
        Returns:
            Cached data or None if not found
        """
        ...
    
    def save_to_cache(
        self,
        cache_key: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save data to cache.
        
        Args:
            cache_key: Cache identifier
            data: Data to cache
            metadata: Optional metadata about cached data
        """
        ...
    
    def clear_cache(
        self,
        pattern: Optional[str] = None,
    ) -> int:
        """Clear cached data.
        
        Args:
            pattern: Optional pattern to match cache keys
            
        Returns:
            Number of cache entries cleared
        """
        ...
    
    def get_cache_info(
        self,
    ) -> Dict[str, Any]:
        """Get information about cache usage.
        
        Returns:
            Cache statistics and metadata
        """
        ...