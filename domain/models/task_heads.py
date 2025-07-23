"""Domain models for task-specific heads.

This module contains the pure business logic for various task heads
that can be attached to BERT models, free from framework dependencies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, TypeVar, Generic
from enum import Enum


TArray = TypeVar('TArray')


class TaskType(Enum):
    """Types of machine learning tasks."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    TOKEN_CLASSIFICATION = "token_classification"
    MULTI_LABEL_CLASSIFICATION = "multi_label_classification"
    RANKING = "ranking"
    SIMILARITY = "similarity"


class LossType(Enum):
    """Types of loss functions."""
    CROSS_ENTROPY = "cross_entropy"
    MSE = "mean_squared_error"
    MAE = "mean_absolute_error"
    BCE = "binary_cross_entropy"
    FOCAL = "focal_loss"
    CONTRASTIVE = "contrastive_loss"
    TRIPLET = "triplet_loss"


class PoolingType(Enum):
    """Types of pooling strategies."""
    CLS_TOKEN = "cls"
    MEAN = "mean"
    MAX = "max"
    ATTENTION_WEIGHTED = "attention_weighted"
    FIRST_LAST_AVG = "first_last_avg"


@dataclass
class HeadConfig:
    """Configuration for a task head."""
    
    task_type: TaskType
    input_size: int
    output_size: int
    
    # Architecture
    hidden_sizes: Optional[List[int]] = None
    dropout_probability: float = 0.1
    activation: str = "tanh"
    use_bias: bool = True
    
    # Pooling
    pooling_type: PoolingType = PoolingType.CLS_TOKEN
    
    # Loss
    loss_type: Optional[LossType] = None
    loss_weight: float = 1.0
    
    # Task-specific
    num_labels: Optional[int] = None
    label_smoothing: float = 0.0
    class_weights: Optional[List[float]] = None
    
    def __post_init__(self):
        """Validate configuration."""
        if self.task_type == TaskType.CLASSIFICATION:
            if self.num_labels is None:
                self.num_labels = self.output_size
            if self.loss_type is None:
                self.loss_type = LossType.CROSS_ENTROPY
                
        elif self.task_type == TaskType.REGRESSION:
            if self.num_labels is None:
                self.num_labels = self.output_size
            if self.loss_type is None:
                self.loss_type = LossType.MSE
                
        elif self.task_type == TaskType.MULTI_LABEL_CLASSIFICATION:
            if self.loss_type is None:
                self.loss_type = LossType.BCE
    
    @property
    def total_parameters(self) -> int:
        """Estimate total parameters in the head."""
        params = 0
        
        if self.hidden_sizes:
            # Multi-layer head
            prev_size = self.input_size
            for hidden_size in self.hidden_sizes:
                params += prev_size * hidden_size
                if self.use_bias:
                    params += hidden_size
                prev_size = hidden_size
            
            # Final projection
            params += prev_size * self.output_size
            if self.use_bias:
                params += self.output_size
        else:
            # Single layer head
            params += self.input_size * self.output_size
            if self.use_bias:
                params += self.output_size
        
        return params


class TaskHead(ABC, Generic[TArray]):
    """Abstract base class for task-specific heads."""
    
    def __init__(self, config: HeadConfig):
        self.config = config
        self.validate_config()
    
    def validate_config(self):
        """Validate head configuration."""
        if self.config.output_size <= 0:
            raise ValueError("Output size must be positive")
            
        if self.config.dropout_probability < 0 or self.config.dropout_probability > 1:
            raise ValueError("Dropout probability must be between 0 and 1")
    
    @property
    @abstractmethod
    def requires_labels(self) -> bool:
        """Whether this head requires labels during forward pass."""
        pass
    
    @property
    @abstractmethod
    def output_activation(self) -> Optional[str]:
        """Activation function applied to final output."""
        pass
    
    @abstractmethod
    def compute_metrics(
        self,
        predictions: TArray,
        labels: TArray,
        mask: Optional[TArray] = None
    ) -> Dict[str, float]:
        """Compute task-specific metrics."""
        pass
    
    def get_head_info(self) -> Dict[str, Any]:
        """Get information about the head."""
        return {
            "task_type": self.config.task_type.value,
            "input_size": self.config.input_size,
            "output_size": self.config.output_size,
            "num_parameters": self.config.total_parameters,
            "requires_labels": self.requires_labels,
            "output_activation": self.output_activation,
            "loss_type": self.config.loss_type.value if self.config.loss_type else None,
            "pooling_type": self.config.pooling_type.value,
        }


class ClassificationHead(TaskHead[TArray]):
    """Head for classification tasks."""
    
    @property
    def requires_labels(self) -> bool:
        return True
    
    @property
    def output_activation(self) -> Optional[str]:
        return None  # Logits output, softmax applied in loss
    
    def compute_metrics(
        self,
        predictions: TArray,
        labels: TArray,
        mask: Optional[TArray] = None
    ) -> Dict[str, float]:
        """Compute classification metrics."""
        # Framework-specific implementation would calculate these
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
    
    @property
    def supports_class_weights(self) -> bool:
        """Whether this head supports class weighting."""
        return True
    
    @property
    def supports_label_smoothing(self) -> bool:
        """Whether this head supports label smoothing."""
        return self.config.label_smoothing > 0


class RegressionHead(TaskHead[TArray]):
    """Head for regression tasks."""
    
    @property
    def requires_labels(self) -> bool:
        return True
    
    @property
    def output_activation(self) -> Optional[str]:
        return None  # Linear output for regression
    
    def compute_metrics(
        self,
        predictions: TArray,
        labels: TArray,
        mask: Optional[TArray] = None
    ) -> Dict[str, float]:
        """Compute regression metrics."""
        # Framework-specific implementation would calculate these
        return {
            "mse": 0.0,
            "mae": 0.0,
            "rmse": 0.0,
            "r2": 0.0,
        }
    
    @property
    def output_bounds(self) -> Optional[tuple[float, float]]:
        """Get output bounds if any."""
        # Could be extended to support bounded regression
        return None


class MultiLabelClassificationHead(TaskHead[TArray]):
    """Head for multi-label classification tasks."""
    
    @property
    def requires_labels(self) -> bool:
        return True
    
    @property
    def output_activation(self) -> Optional[str]:
        return "sigmoid"  # Each label is independent
    
    def compute_metrics(
        self,
        predictions: TArray,
        labels: TArray,
        mask: Optional[TArray] = None
    ) -> Dict[str, float]:
        """Compute multi-label metrics."""
        return {
            "accuracy": 0.0,
            "hamming_loss": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_micro": 0.0,
            "f1_macro": 0.0,
        }
    
    @property
    def threshold(self) -> float:
        """Decision threshold for binary predictions."""
        return 0.5


class TokenClassificationHead(TaskHead[TArray]):
    """Head for token classification tasks (e.g., NER)."""
    
    @property
    def requires_labels(self) -> bool:
        return True
    
    @property
    def output_activation(self) -> Optional[str]:
        return None  # Logits output
    
    def compute_metrics(
        self,
        predictions: TArray,
        labels: TArray,
        mask: Optional[TArray] = None
    ) -> Dict[str, float]:
        """Compute token classification metrics."""
        return {
            "token_accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
    
    @property
    def ignore_index(self) -> int:
        """Index to ignore in loss computation (e.g., padding)."""
        return -100


class SimilarityHead(TaskHead[TArray]):
    """Head for similarity/matching tasks."""
    
    def __init__(self, config: HeadConfig):
        super().__init__(config)
        self.similarity_function = "cosine"  # or "euclidean", "dot_product"
    
    @property
    def requires_labels(self) -> bool:
        return False  # Can work without labels for inference
    
    @property
    def output_activation(self) -> Optional[str]:
        return None  # Embeddings output
    
    def compute_metrics(
        self,
        predictions: TArray,
        labels: TArray,
        mask: Optional[TArray] = None
    ) -> Dict[str, float]:
        """Compute similarity metrics."""
        return {
            "cosine_similarity": 0.0,
            "euclidean_distance": 0.0,
        }
    
    @property
    def normalize_embeddings(self) -> bool:
        """Whether to normalize embeddings."""
        return self.similarity_function == "cosine"


class HeadFactory:
    """Factory for creating appropriate task heads."""
    
    @staticmethod
    def create_head(
        task_type: TaskType,
        input_size: int,
        num_labels: int,
        **kwargs
    ) -> TaskHead:
        """Create a task head based on task type."""
        config = HeadConfig(
            task_type=task_type,
            input_size=input_size,
            output_size=num_labels,
            num_labels=num_labels,
            **kwargs
        )
        
        if task_type == TaskType.CLASSIFICATION:
            return ClassificationHead(config)
        elif task_type == TaskType.REGRESSION:
            return RegressionHead(config)
        elif task_type == TaskType.MULTI_LABEL_CLASSIFICATION:
            return MultiLabelClassificationHead(config)
        elif task_type == TaskType.TOKEN_CLASSIFICATION:
            return TokenClassificationHead(config)
        elif task_type == TaskType.SIMILARITY:
            return SimilarityHead(config)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    @staticmethod
    def infer_task_type(num_labels: int, **kwargs) -> TaskType:
        """Infer task type from configuration."""
        if kwargs.get("is_regression", False):
            return TaskType.REGRESSION
        elif kwargs.get("is_multi_label", False):
            return TaskType.MULTI_LABEL_CLASSIFICATION
        elif kwargs.get("is_token_classification", False):
            return TaskType.TOKEN_CLASSIFICATION
        elif num_labels == 1:
            return TaskType.REGRESSION
        else:
            return TaskType.CLASSIFICATION


@dataclass
class HeadSelectionCriteria:
    """Criteria for selecting appropriate head type."""
    
    num_labels: int
    task_description: Optional[str] = None
    output_type: Optional[str] = None  # "probabilities", "scores", "embeddings"
    loss_function: Optional[str] = None
    metric_names: Optional[List[str]] = None
    
    def determine_task_type(self) -> TaskType:
        """Determine task type from criteria."""
        # Check explicit indicators
        if self.task_description:
            desc_lower = self.task_description.lower()
            if any(word in desc_lower for word in ["classify", "classification"]):
                if "multi" in desc_lower or "multiple" in desc_lower:
                    return TaskType.MULTI_LABEL_CLASSIFICATION
                return TaskType.CLASSIFICATION
            elif any(word in desc_lower for word in ["regression", "predict", "score"]):
                return TaskType.REGRESSION
            elif any(word in desc_lower for word in ["ner", "token", "tagging"]):
                return TaskType.TOKEN_CLASSIFICATION
            elif any(word in desc_lower for word in ["similarity", "matching", "retrieval"]):
                return TaskType.SIMILARITY
        
        # Check loss function
        if self.loss_function:
            if self.loss_function in ["mse", "mae", "rmse"]:
                return TaskType.REGRESSION
            elif self.loss_function in ["bce", "binary_crossentropy"]:
                return TaskType.MULTI_LABEL_CLASSIFICATION
        
        # Check metrics
        if self.metric_names:
            if any(metric in self.metric_names for metric in ["mse", "mae", "r2"]):
                return TaskType.REGRESSION
            elif "hamming_loss" in self.metric_names:
                return TaskType.MULTI_LABEL_CLASSIFICATION
        
        # Default based on num_labels
        if self.num_labels == 1:
            return TaskType.REGRESSION
        else:
            return TaskType.CLASSIFICATION