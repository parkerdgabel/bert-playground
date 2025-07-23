"""Domain models for data handling.

This module contains the core domain models and specifications for data handling,
separate from infrastructure concerns like I/O and caching.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


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
    """Specification for a dataset - pure domain model.
    
    This class defines the metadata and configuration for a specific
    dataset, focusing on business logic rather than technical implementation.
    """
    
    # Basic identification
    competition_name: str
    dataset_path: Union[str, Path]
    competition_type: CompetitionType
    
    # Data characteristics
    num_samples: int
    num_features: int
    target_column: str | None = None
    text_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    numerical_columns: List[str] = field(default_factory=list)
    
    # Target characteristics
    num_classes: int | None = None
    class_distribution: Dict[str, int] | None = None
    is_balanced: bool = True
    
    # Performance optimization hints (domain knowledge)
    recommended_batch_size: int = 32
    recommended_max_length: int = 512
    use_attention_mask: bool = True
    enable_caching: bool = True
    
    # Text template for conversion
    text_template: str | None = None
    
    def __post_init__(self):
        """Validate and normalize the dataset specification."""
        self.dataset_path = Path(self.dataset_path)
        
        # Validate competition type and num_classes consistency
        if self.competition_type == CompetitionType.BINARY_CLASSIFICATION:
            if self.num_classes is None:
                self.num_classes = 2
            elif self.num_classes != 2:
                raise ValueError(f"Binary classification should have 2 classes, got {self.num_classes}")
        
        elif self.competition_type == CompetitionType.REGRESSION:
            if self.num_classes is None:
                self.num_classes = 1
        
        # Set reasonable defaults based on dataset size
        if self.num_samples > 100000:
            self.recommended_batch_size = min(64, self.recommended_batch_size)
        elif self.num_samples < 5000:
            self.recommended_batch_size = max(16, self.recommended_batch_size)


@dataclass
class DataSample:
    """Domain model for a single data sample."""
    
    text: str
    labels: Optional[Union[int, float, List[float]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate the sample."""
        if not isinstance(self.text, str):
            raise ValueError("Text must be a string")
        
        if self.labels is not None:
            if isinstance(self.labels, list):
                if not all(isinstance(x, (int, float)) for x in self.labels):
                    raise ValueError("All labels must be numeric")
            elif not isinstance(self.labels, (int, float)):
                raise ValueError("Labels must be numeric or list of numeric values")


@dataclass
class DataBatch:
    """Domain model for a batch of data samples."""
    
    texts: List[str]
    labels: Optional[List[Union[int, float, List[float]]]] = None
    metadata: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate the batch."""
        if not self.texts:
            raise ValueError("Batch cannot be empty")
        
        if self.labels is not None and len(self.labels) != len(self.texts):
            raise ValueError("Number of labels must match number of texts")
        
        if self.metadata and len(self.metadata) != len(self.texts):
            raise ValueError("Number of metadata entries must match number of texts")
    
    @property
    def size(self) -> int:
        """Get batch size."""
        return len(self.texts)
    
    def __len__(self) -> int:
        """Get batch size."""
        return self.size


class DatasetRepository(ABC):
    """Abstract repository for dataset operations - domain interface."""
    
    @abstractmethod
    def get_by_spec(self, spec: DatasetSpec, split: str = "train") -> "Dataset":
        """Get dataset by specification."""
        pass
    
    @abstractmethod
    def validate_spec(self, spec: DatasetSpec) -> bool:
        """Validate a dataset specification."""
        pass


class Dataset(ABC):
    """Abstract domain model for datasets."""
    
    def __init__(self, spec: DatasetSpec, split: str = "train"):
        """Initialize dataset with specification."""
        self.spec = spec
        self.split = split
    
    @abstractmethod
    def get_sample(self, index: int) -> DataSample:
        """Get a single sample by index."""
        pass
    
    @abstractmethod
    def get_batch(self, indices: List[int]) -> DataBatch:
        """Get a batch of samples."""
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """Get dataset size."""
        pass
    
    def get_competition_info(self) -> Dict[str, Any]:
        """Get competition information."""
        return {
            "competition_name": self.spec.competition_name,
            "competition_type": self.spec.competition_type.value,
            "num_samples": len(self),
            "num_features": self.spec.num_features,
            "split": self.split,
            "target_column": self.spec.target_column,
            "num_classes": self.spec.num_classes,
            "is_balanced": self.spec.is_balanced,
        }


class DataValidationResult:
    """Result of data validation."""
    
    def __init__(self, is_valid: bool, errors: List[str] = None):
        """Initialize validation result."""
        self.is_valid = is_valid
        self.errors = errors or []
    
    def add_error(self, error: str) -> None:
        """Add validation error."""
        self.errors.append(error)
        self.is_valid = False
    
    def __bool__(self) -> bool:
        """Return validation status."""
        return self.is_valid


class DataProcessor(ABC):
    """Abstract base class for data processing operations."""
    
    @abstractmethod
    def process(self, data: Any) -> Any:
        """Process data."""
        pass
    
    @abstractmethod
    def validate(self, data: Any) -> DataValidationResult:
        """Validate data."""
        pass