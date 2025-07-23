"""Domain models for data handling.

This module contains the pure business logic for data processing and handling,
free from any framework dependencies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, TypeVar, Generic, Tuple
from enum import Enum


TArray = TypeVar('TArray')
TTokenizer = TypeVar('TTokenizer')


class DatasetType(Enum):
    """Types of datasets."""
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    PREDICTION = "prediction"


class InputFormat(Enum):
    """Input data formats."""
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"
    TEXT = "text"
    HUGGINGFACE = "huggingface"


class TaskDataType(Enum):
    """Types of data for different tasks."""
    TEXT_CLASSIFICATION = "text_classification"
    TEXT_REGRESSION = "text_regression"
    TOKEN_CLASSIFICATION = "token_classification"
    TEXT_SIMILARITY = "text_similarity"
    TABULAR_CLASSIFICATION = "tabular_classification"
    TABULAR_REGRESSION = "tabular_regression"


@dataclass
class DataConfig:
    """Configuration for data processing."""
    
    # Basic settings
    task_type: TaskDataType
    input_format: InputFormat = InputFormat.CSV
    text_column: str = "text"
    label_column: Optional[str] = "label"
    
    # Tokenization
    max_sequence_length: int = 512
    tokenizer_name: str = "bert-base-uncased"
    padding_strategy: str = "max_length"  # "max_length", "longest", "do_not_pad"
    truncation_strategy: str = "longest_first"  # "longest_first", "only_first", "only_second"
    
    # Processing
    lowercase: bool = False
    remove_special_chars: bool = False
    normalize_unicode: bool = True
    
    # Data loading
    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 0
    prefetch_factor: int = 2
    
    # Caching
    use_cache: bool = True
    cache_dir: Optional[str] = None
    
    # Sampling
    max_samples: Optional[int] = None
    sample_strategy: str = "random"  # "random", "stratified", "sequential"
    
    # Augmentation
    augmentation_prob: float = 0.0
    augmentation_strategies: List[str] = field(default_factory=list)
    
    # Multi-label specific
    label_delimiter: str = ","
    label_threshold: float = 0.5
    
    # Tabular specific
    numerical_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate configuration."""
        if self.task_type in [
            TaskDataType.TEXT_CLASSIFICATION,
            TaskDataType.TEXT_REGRESSION,
            TaskDataType.TABULAR_CLASSIFICATION,
            TaskDataType.TABULAR_REGRESSION
        ] and self.label_column is None:
            raise ValueError(f"label_column required for {self.task_type.value}")
        
        if self.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")
        
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


@dataclass
class DataStatistics:
    """Statistics about a dataset."""
    
    num_samples: int
    num_features: Optional[int] = None
    
    # Label statistics
    num_classes: Optional[int] = None
    class_distribution: Optional[Dict[Any, int]] = None
    label_mean: Optional[float] = None
    label_std: Optional[float] = None
    
    # Text statistics
    avg_sequence_length: Optional[float] = None
    max_sequence_length: Optional[int] = None
    min_sequence_length: Optional[int] = None
    vocabulary_size: Optional[int] = None
    
    # Data quality
    num_missing_values: int = 0
    num_duplicates: int = 0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        summary = {
            "num_samples": self.num_samples,
            "num_missing_values": self.num_missing_values,
            "num_duplicates": self.num_duplicates,
        }
        
        if self.num_features is not None:
            summary["num_features"] = self.num_features
            
        if self.num_classes is not None:
            summary["num_classes"] = self.num_classes
            summary["class_distribution"] = self.class_distribution
            
        if self.label_mean is not None:
            summary["label_statistics"] = {
                "mean": self.label_mean,
                "std": self.label_std,
            }
            
        if self.avg_sequence_length is not None:
            summary["text_statistics"] = {
                "avg_length": self.avg_sequence_length,
                "max_length": self.max_sequence_length,
                "min_length": self.min_sequence_length,
                "vocabulary_size": self.vocabulary_size,
            }
            
        return summary


@dataclass
class TextExample:
    """Single text example."""
    
    text: str
    label: Optional[Any] = None
    example_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {"text": self.text}
        if self.label is not None:
            result["label"] = self.label
        if self.example_id is not None:
            result["id"] = self.example_id
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass
class TabularExample:
    """Single tabular example."""
    
    features: Dict[str, Any]
    label: Optional[Any] = None
    example_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_text(self, template: Optional[str] = None) -> str:
        """Convert tabular data to text representation."""
        if template:
            return template.format(**self.features)
        else:
            # Default template
            parts = []
            for key, value in self.features.items():
                parts.append(f"{key}: {value}")
            return " | ".join(parts)


@dataclass
class TokenizedExample(Generic[TArray]):
    """Tokenized example ready for model input."""
    
    input_ids: TArray
    attention_mask: TArray
    token_type_ids: Optional[TArray] = None
    labels: Optional[TArray] = None
    
    # Additional information
    original_length: Optional[int] = None
    truncated: bool = False
    
    def get_sequence_length(self) -> int:
        """Get actual sequence length (excluding padding)."""
        # This would be implemented by framework adapter
        raise NotImplementedError("Must be implemented by adapter")


class DataProcessor(ABC):
    """Abstract data processor."""
    
    def __init__(self, config: DataConfig):
        self.config = config
    
    @abstractmethod
    def process_example(self, example: Any) -> Any:
        """Process single example."""
        pass
    
    @abstractmethod
    def process_batch(self, examples: List[Any]) -> List[Any]:
        """Process batch of examples."""
        pass
    
    def should_filter(self, example: Any) -> bool:
        """Determine if example should be filtered out."""
        return False


class TextProcessor(DataProcessor):
    """Processor for text data."""
    
    def clean_text(self, text: str) -> str:
        """Clean text according to configuration."""
        if self.config.lowercase:
            text = text.lower()
            
        if self.config.normalize_unicode:
            # Normalize unicode (would be implemented by adapter)
            pass
            
        if self.config.remove_special_chars:
            # Remove special characters (would be implemented by adapter)
            pass
            
        return text.strip()
    
    def process_example(self, example: TextExample) -> TextExample:
        """Process single text example."""
        example.text = self.clean_text(example.text)
        return example
    
    def process_batch(self, examples: List[TextExample]) -> List[TextExample]:
        """Process batch of text examples."""
        return [self.process_example(ex) for ex in examples]


class TabularProcessor(DataProcessor):
    """Processor for tabular data."""
    
    def __init__(self, config: DataConfig):
        super().__init__(config)
        self.text_template = self._create_text_template()
    
    def _create_text_template(self) -> str:
        """Create template for converting tabular to text."""
        # Default template - can be customized
        parts = []
        for col in self.config.categorical_columns:
            parts.append(f"{col}: {{{col}}}")
        for col in self.config.numerical_columns:
            parts.append(f"{col}: {{{col}}}")
        return " | ".join(parts)
    
    def process_example(self, example: TabularExample) -> TextExample:
        """Convert tabular example to text."""
        text = example.to_text(self.text_template)
        return TextExample(
            text=text,
            label=example.label,
            example_id=example.example_id,
            metadata=example.metadata
        )
    
    def process_batch(self, examples: List[TabularExample]) -> List[TextExample]:
        """Process batch of tabular examples."""
        return [self.process_example(ex) for ex in examples]


class DataAugmenter(ABC):
    """Abstract data augmenter."""
    
    @abstractmethod
    def augment(self, example: TextExample) -> TextExample:
        """Augment single example."""
        pass
    
    def should_augment(self, probability: float) -> bool:
        """Determine if augmentation should be applied."""
        # Random sampling (would be implemented by adapter)
        raise NotImplementedError("Must be implemented by adapter")


@dataclass
class DataSplit:
    """Configuration for data splitting."""
    
    train_ratio: float = 0.8
    validation_ratio: float = 0.1
    test_ratio: float = 0.1
    stratify: bool = True
    random_seed: Optional[int] = 42
    
    def __post_init__(self):
        """Validate split ratios."""
        total = self.train_ratio + self.validation_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        
        if any(ratio < 0 for ratio in [self.train_ratio, self.validation_ratio, self.test_ratio]):
            raise ValueError("Split ratios must be non-negative")


class DataValidator:
    """Validates data quality and consistency."""
    
    @staticmethod
    def validate_text_length(
        examples: List[TextExample],
        max_length: int,
        min_length: int = 1
    ) -> List[str]:
        """Validate text length constraints."""
        errors = []
        for i, example in enumerate(examples):
            text_length = len(example.text.split())
            if text_length < min_length:
                errors.append(f"Example {i}: text too short ({text_length} words)")
            elif text_length > max_length:
                errors.append(f"Example {i}: text too long ({text_length} words)")
        return errors
    
    @staticmethod
    def validate_labels(
        examples: List[TextExample],
        valid_labels: Optional[List[Any]] = None
    ) -> List[str]:
        """Validate label values."""
        errors = []
        
        if valid_labels is not None:
            valid_set = set(valid_labels)
            for i, example in enumerate(examples):
                if example.label not in valid_set:
                    errors.append(f"Example {i}: invalid label '{example.label}'")
        
        # Check for missing labels
        for i, example in enumerate(examples):
            if example.label is None:
                errors.append(f"Example {i}: missing label")
                
        return errors
    
    @staticmethod
    def check_data_balance(
        examples: List[TextExample],
        min_samples_per_class: int = 10
    ) -> Dict[str, Any]:
        """Check class balance in dataset."""
        from collections import Counter
        
        label_counts = Counter(ex.label for ex in examples if ex.label is not None)
        
        issues = []
        for label, count in label_counts.items():
            if count < min_samples_per_class:
                issues.append(f"Label '{label}' has only {count} samples")
        
        total = sum(label_counts.values())
        min_ratio = min(label_counts.values()) / total if total > 0 else 0
        max_ratio = max(label_counts.values()) / total if total > 0 else 0
        
        return {
            "label_counts": dict(label_counts),
            "total_samples": total,
            "num_classes": len(label_counts),
            "min_class_ratio": min_ratio,
            "max_class_ratio": max_ratio,
            "imbalance_ratio": max_ratio / min_ratio if min_ratio > 0 else float('inf'),
            "issues": issues
        }