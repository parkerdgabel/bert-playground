"""Dataset and data batch entities.

Pure domain entities with no framework dependencies.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum


class DatasetSplit(Enum):
    """Dataset split types."""
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    PREDICTION = "prediction"


class DataFormat(Enum):
    """Data format types."""
    TEXT = "text"
    TABULAR = "tabular"
    TOKENIZED = "tokenized"


@dataclass
class DataSample:
    """Single data sample.
    
    This represents one example in a dataset, independent
    of whether it's text, tabular, or already tokenized.
    """
    id: str
    content: Any  # Can be text string, dict of features, or token ids
    label: Optional[Any] = None  # Can be int, float, list, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_labeled(self) -> bool:
        """Check if sample has a label."""
        return self.label is not None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "content": self.content,
            "label": self.label,
            "metadata": self.metadata
        }


@dataclass
class DataBatch:
    """Batch of data samples.
    
    This represents a collection of samples that will be
    processed together, maintaining batch coherence.
    """
    samples: List[DataSample]
    batch_size: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate batch."""
        if self.batch_size != len(self.samples):
            self.batch_size = len(self.samples)
    
    @property
    def is_labeled(self) -> bool:
        """Check if all samples in batch have labels."""
        return all(sample.is_labeled for sample in self.samples)
    
    @property
    def labels(self) -> Optional[List[Any]]:
        """Get labels from all samples if available."""
        if not self.is_labeled:
            return None
        return [sample.label for sample in self.samples]
    
    def split(self, sizes: List[int]) -> List['DataBatch']:
        """Split batch into multiple smaller batches."""
        if sum(sizes) != self.batch_size:
            raise ValueError("Split sizes must sum to batch size")
        
        batches = []
        start = 0
        for size in sizes:
            end = start + size
            batch_samples = self.samples[start:end]
            batches.append(DataBatch(
                samples=batch_samples,
                batch_size=size,
                metadata=self.metadata.copy()
            ))
            start = end
        
        return batches


@dataclass
class DatasetStatistics:
    """Statistics about a dataset."""
    num_samples: int
    num_features: Optional[int] = None
    num_classes: Optional[int] = None
    class_distribution: Optional[Dict[Any, int]] = None
    
    # Text-specific stats
    avg_text_length: Optional[float] = None
    max_text_length: Optional[int] = None
    min_text_length: Optional[int] = None
    vocabulary_size: Optional[int] = None
    
    # Quality metrics
    num_missing_values: int = 0
    num_duplicates: int = 0
    
    @property
    def is_balanced(self) -> bool:
        """Check if dataset is balanced (for classification)."""
        if not self.class_distribution:
            return True
        
        counts = list(self.class_distribution.values())
        if not counts:
            return True
            
        min_count = min(counts)
        max_count = max(counts)
        # Consider balanced if max/min ratio < 2
        return max_count / min_count < 2.0 if min_count > 0 else False
    
    def summary(self) -> Dict[str, Any]:
        """Get summary of statistics."""
        summary = {
            "num_samples": self.num_samples,
            "num_missing_values": self.num_missing_values,
            "num_duplicates": self.num_duplicates,
        }
        
        if self.num_features is not None:
            summary["num_features"] = self.num_features
            
        if self.num_classes is not None:
            summary["num_classes"] = self.num_classes
            summary["is_balanced"] = self.is_balanced
            
        if self.avg_text_length is not None:
            summary["text_stats"] = {
                "avg_length": self.avg_text_length,
                "max_length": self.max_text_length,
                "min_length": self.min_text_length,
                "vocabulary_size": self.vocabulary_size,
            }
            
        return summary


@dataclass
class Dataset:
    """Dataset entity.
    
    This represents a complete dataset with samples, metadata,
    and statistics. It's independent of the storage format or
    loading mechanism.
    """
    id: str
    name: str
    split: DatasetSplit
    format: DataFormat
    samples: List[DataSample]
    statistics: Optional[DatasetStatistics] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Task-specific information
    task_type: Optional[str] = None
    num_classes: Optional[int] = None
    label_names: Optional[List[str]] = None
    feature_names: Optional[List[str]] = None
    
    @property
    def size(self) -> int:
        """Get number of samples in dataset."""
        return len(self.samples)
    
    @property
    def is_empty(self) -> bool:
        """Check if dataset is empty."""
        return len(self.samples) == 0
    
    @property
    def is_labeled(self) -> bool:
        """Check if dataset has labels."""
        return all(sample.is_labeled for sample in self.samples)
    
    def get_sample(self, index: int) -> DataSample:
        """Get sample by index."""
        if 0 <= index < len(self.samples):
            return self.samples[index]
        raise IndexError(f"Index {index} out of range for dataset of size {self.size}")
    
    def create_batch(self, indices: List[int]) -> DataBatch:
        """Create a batch from specific indices."""
        batch_samples = [self.samples[i] for i in indices if 0 <= i < len(self.samples)]
        return DataBatch(
            samples=batch_samples,
            batch_size=len(batch_samples),
            metadata={"dataset_id": self.id, "dataset_name": self.name}
        )
    
    def filter_samples(self, predicate) -> 'Dataset':
        """Create new dataset with filtered samples."""
        filtered_samples = [s for s in self.samples if predicate(s)]
        return Dataset(
            id=f"{self.id}_filtered",
            name=f"{self.name} (filtered)",
            split=self.split,
            format=self.format,
            samples=filtered_samples,
            statistics=None,  # Statistics need recalculation
            metadata=self.metadata.copy(),
            task_type=self.task_type,
            num_classes=self.num_classes,
            label_names=self.label_names,
            feature_names=self.feature_names
        )
    
    def get_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        info = {
            "id": self.id,
            "name": self.name,
            "split": self.split.value,
            "format": self.format.value,
            "size": self.size,
            "is_labeled": self.is_labeled,
            "task_type": self.task_type,
        }
        
        if self.num_classes is not None:
            info["num_classes"] = self.num_classes
            info["label_names"] = self.label_names
            
        if self.feature_names is not None:
            info["num_features"] = len(self.feature_names)
            info["feature_names"] = self.feature_names
            
        if self.statistics:
            info["statistics"] = self.statistics.summary()
            
        return info


@dataclass
class DatasetSpecification:
    """Specification for creating or loading a dataset.
    
    This is used by data services to specify how a dataset
    should be created or loaded, independent of the actual
    loading mechanism.
    """
    source: str  # Path, URL, or identifier
    format: DataFormat
    split: DatasetSplit
    task_type: Optional[str] = None
    
    # Processing options
    max_samples: Optional[int] = None
    shuffle: bool = False
    random_seed: Optional[int] = None
    
    # Text-specific options
    max_text_length: Optional[int] = None
    text_column: Optional[str] = None
    label_column: Optional[str] = None
    
    # Tabular-specific options
    feature_columns: Optional[List[str]] = None
    categorical_columns: Optional[List[str]] = None
    numerical_columns: Optional[List[str]] = None
    
    # Caching options
    use_cache: bool = True
    cache_dir: Optional[str] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)