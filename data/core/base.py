"""Base classes for Kaggle dataset abstraction.

This module defines the core interfaces and base classes for handling
Kaggle datasets with optimizations for BERT models and MLX framework.
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import mlx.core as mx
import pandas as pd
from loguru import logger


class CompetitionType(Enum):
    """Types of Kaggle competitions."""
    
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"
    REGRESSION = "regression"
    ORDINAL_REGRESSION = "ordinal_regression"
    TIME_SERIES = "time_series"
    RANKING = "ranking"
    STRUCTURED_PREDICTION = "structured_prediction"
    GENERATIVE = "generative"
    UNKNOWN = "unknown"


@dataclass
class DatasetSpec:
    """Specification for a Kaggle dataset.
    
    This class defines the metadata and configuration for a specific
    Kaggle competition dataset, including optimization hints for BERT
    and MLX processing.
    """
    
    # Basic identification
    competition_name: str
    dataset_path: Union[str, Path]
    competition_type: CompetitionType
    
    # Data characteristics
    num_samples: int
    num_features: int
    target_column: Optional[str] = None
    text_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    numerical_columns: List[str] = field(default_factory=list)
    
    # Target characteristics
    num_classes: Optional[int] = None
    class_distribution: Optional[Dict[str, int]] = None
    is_balanced: bool = True
    
    # Performance optimization hints
    recommended_batch_size: int = 32
    recommended_max_length: int = 512
    use_attention_mask: bool = True
    enable_caching: bool = True
    
    # MLX optimization settings
    use_unified_memory: bool = True
    prefetch_size: int = 4
    num_workers: int = 4
    
    # BERT-specific settings
    text_template: Optional[str] = None
    tokenizer_backend: str = "transformers"  # "transformers" or "mlx"
    
    def __post_init__(self):
        """Validate and normalize the dataset specification."""
        self.dataset_path = Path(self.dataset_path)
        
        # Validate competition type and num_classes consistency
        if self.competition_type == CompetitionType.BINARY_CLASSIFICATION:
            if self.num_classes is None:
                self.num_classes = 2
            elif self.num_classes != 2:
                logger.warning(f"Binary classification should have 2 classes, got {self.num_classes}")
                
        elif self.competition_type == CompetitionType.REGRESSION:
            if self.num_classes is None:
                self.num_classes = 1
                
        # Set reasonable defaults based on dataset size
        if self.num_samples > 100000:
            self.recommended_batch_size = min(64, self.recommended_batch_size)
            self.prefetch_size = 8
        elif self.num_samples < 5000:
            self.recommended_batch_size = max(16, self.recommended_batch_size)
            self.prefetch_size = 2


class KaggleDataset(ABC):
    """Abstract base class for Kaggle datasets.
    
    This class provides a unified interface for all Kaggle competition datasets,
    with built-in optimizations for BERT models and MLX framework on Apple Silicon.
    
    The design follows these principles:
    1. Unified interface across all competition types
    2. MLX-optimized data loading with unified memory
    3. BERT-compatible tokenization and batching
    4. Automatic dataset analysis and optimization
    5. Efficient caching and streaming capabilities
    """
    
    def __init__(
        self,
        spec: DatasetSpec,
        split: str = "train",
        transform: Optional[callable] = None,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        """Initialize the Kaggle dataset.
        
        Args:
            spec: Dataset specification containing metadata and configuration
            split: Dataset split ("train", "validation", "test")
            transform: Optional transform function to apply to samples
            cache_dir: Directory for caching processed data
        """
        self.spec = spec
        self.split = split
        self.transform = transform
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Internal state
        self._data: Optional[pd.DataFrame] = None
        self._cached_samples: Optional[List[Dict[str, Any]]] = None
        self._sample_cache: Dict[int, Dict[str, Any]] = {}
        
        # MLX optimization settings
        self._device = mx.default_device()
        self._stream = mx.default_stream()
        
        # Load and validate data
        self._load_data()
        self._validate_data()
        
        logger.info(
            f"Initialized {self.__class__.__name__} for {spec.competition_name} "
            f"({split}) with {len(self)} samples"
        )
    
    @abstractmethod
    def _load_data(self) -> None:
        """Load the raw data from files.
        
        This method should be implemented by subclasses to handle
        the specific data loading logic for different competition types.
        """
        pass
    
    @abstractmethod
    def _validate_data(self) -> None:
        """Validate the loaded data.
        
        This method should check data integrity, column presence,
        and consistency with the dataset specification.
        """
        pass
    
    @abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get a single sample by index.
        
        Args:
            index: Sample index
            
        Returns:
            Dictionary containing sample data with keys:
            - 'text': Text representation of the sample
            - 'input_ids': Tokenized input (if tokenized)
            - 'attention_mask': Attention mask (if applicable)
            - 'labels': Ground truth labels (if available)
            - 'metadata': Additional metadata
        """
        pass
    
    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        if self._data is None:
            return 0
        return len(self._data)
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over dataset samples."""
        for i in range(len(self)):
            yield self[i]
    
    def get_sample_by_id(self, sample_id: str) -> Optional[Dict[str, Any]]:
        """Get a sample by its unique ID.
        
        Args:
            sample_id: Unique identifier for the sample
            
        Returns:
            Sample dictionary or None if not found
        """
        if self._data is None:
            return None
            
        # Try to find by index first (most common case)
        try:
            idx = int(sample_id)
            if 0 <= idx < len(self):
                return self[idx]
        except ValueError:
            pass
            
        # Search by ID column if it exists
        id_columns = ['id', 'Id', 'ID', 'PassengerId', 'test_id']
        for col in id_columns:
            if col in self._data.columns:
                matching_rows = self._data[self._data[col] == sample_id]
                if not matching_rows.empty:
                    return self[matching_rows.index[0]]
                    
        return None
    
    def get_batch(self, indices: List[int]) -> Dict[str, mx.array]:
        """Get a batch of samples as MLX arrays.
        
        Args:
            indices: List of sample indices
            
        Returns:
            Dictionary of batched MLX arrays
        """
        batch_samples = [self[i] for i in indices]
        return self._collate_batch(batch_samples)
    
    def _collate_batch(self, samples: List[Dict[str, Any]]) -> Dict[str, mx.array]:
        """Collate a list of samples into a batch.
        
        Args:
            samples: List of sample dictionaries
            
        Returns:
            Dictionary of batched MLX arrays optimized for unified memory
        """
        if not samples:
            return {}
            
        # Collect all keys from samples
        all_keys = set()
        for sample in samples:
            all_keys.update(sample.keys())
            
        batch = {}
        
        for key in all_keys:
            values = []
            for sample in samples:
                if key in sample:
                    values.append(sample[key])
                else:
                    # Handle missing keys with appropriate defaults
                    if key == 'attention_mask':
                        # Default attention mask of all 1s
                        seq_len = len(sample.get('input_ids', []))
                        values.append([1] * seq_len)
                    elif key == 'labels':
                        values.append(None)
                    else:
                        values.append(None)
            
            # Convert to MLX arrays based on data type
            if key in ['input_ids', 'attention_mask', 'token_type_ids']:
                # Token arrays - pad to max length
                if any(v is not None for v in values):
                    max_len = max(len(v) for v in values if v is not None)
                    padded_values = []
                    for v in values:
                        if v is None:
                            padded_values.append([0] * max_len)
                        else:
                            padded = v + [0] * (max_len - len(v))
                            padded_values.append(padded)
                    batch[key] = mx.array(padded_values, dtype=mx.int32)
                    
            elif key == 'labels':
                # Label arrays
                valid_labels = [v for v in values if v is not None]
                if valid_labels:
                    if isinstance(valid_labels[0], (int, float)):
                        # Single label per sample
                        batch[key] = mx.array(valid_labels, dtype=mx.float32)
                    else:
                        # Multi-label case
                        batch[key] = mx.array(valid_labels, dtype=mx.float32)
                        
            elif key in ['text', 'metadata']:
                # Keep as lists for non-numeric data
                batch[key] = values
                
        return batch
    
    def get_competition_info(self) -> Dict[str, Any]:
        """Get competition information and statistics.
        
        Returns:
            Dictionary containing competition metadata
        """
        info = {
            'competition_name': self.spec.competition_name,
            'competition_type': self.spec.competition_type.value,
            'num_samples': len(self),
            'num_features': self.spec.num_features,
            'split': self.split,
        }
        
        if self.spec.target_column:
            info['target_column'] = self.spec.target_column
            info['num_classes'] = self.spec.num_classes
            info['is_balanced'] = self.spec.is_balanced
            
        if self.spec.class_distribution:
            info['class_distribution'] = self.spec.class_distribution
            
        return info
    
    def get_sample_text(self, index: int) -> str:
        """Get the text representation of a sample.
        
        Args:
            index: Sample index
            
        Returns:
            Text representation of the sample
        """
        sample = self[index]
        return sample.get('text', '')
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about the dataset.
        
        Returns:
            Dictionary containing dataset statistics
        """
        if self._data is None:
            return {}
            
        stats = {
            'total_samples': len(self._data),
            'columns': list(self._data.columns),
            'dtypes': self._data.dtypes.to_dict(),
            'memory_usage': self._data.memory_usage(deep=True).sum(),
            'missing_values': self._data.isnull().sum().to_dict(),
        }
        
        # Add text-specific statistics
        if self.spec.text_columns:
            text_stats = {}
            for col in self.spec.text_columns:
                if col in self._data.columns:
                    text_data = self._data[col].astype(str)
                    text_stats[col] = {
                        'avg_length': text_data.str.len().mean(),
                        'max_length': text_data.str.len().max(),
                        'min_length': text_data.str.len().min(),
                        'unique_count': text_data.nunique(),
                    }
            stats['text_statistics'] = text_stats
            
        # Add label statistics for classification tasks
        if self.spec.target_column and self.spec.target_column in self._data.columns:
            target_stats = {
                'unique_values': self._data[self.spec.target_column].nunique(),
                'value_counts': self._data[self.spec.target_column].value_counts().to_dict(),
            }
            stats['target_statistics'] = target_stats
            
        return stats
    
    def enable_caching(self, cache_dir: Optional[Union[str, Path]] = None) -> None:
        """Enable sample caching for faster access.
        
        Args:
            cache_dir: Directory to store cache files
        """
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Enabled caching in {self.cache_dir}")
        else:
            logger.info("Enabled in-memory caching")
            
        self.spec.enable_caching = True
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._sample_cache.clear()
        self._cached_samples = None
        
        if self.cache_dir and self.cache_dir.exists():
            import shutil
            shutil.rmtree(self.cache_dir)
            logger.info("Cleared disk cache")
        
        logger.info("Cleared memory cache")
    
    def get_mlx_device_info(self) -> Dict[str, Any]:
        """Get MLX device and memory information.
        
        Returns:
            Dictionary containing MLX device info
        """
        return {
            'device': str(self._device),
            'unified_memory': self.spec.use_unified_memory,
            'default_stream': str(self._stream),
            'mlx_available': True,
        }