"""Domain model for BERT outputs.

This module contains the pure business logic for BERT model outputs,
free from any framework dependencies.
"""

from dataclasses import dataclass
from typing import List, Optional, TypeVar, Generic
from abc import ABC, abstractmethod


# Generic type for array representation
TArray = TypeVar('TArray')


@dataclass
class BertDomainOutput(Generic[TArray]):
    """Standard output format for BERT domain models.
    
    This represents the business-level output of a BERT model,
    independent of any ML framework implementation.
    """
    
    # Primary outputs
    last_hidden_state: TArray  # [batch_size, seq_len, hidden_size]
    pooler_output: TArray  # [batch_size, hidden_size]
    
    # Optional outputs for interpretability
    hidden_states: Optional[List[TArray]] = None  # List of all hidden states
    attentions: Optional[List[TArray]] = None  # List of attention weights
    
    # Additional pooled representations
    cls_token_output: Optional[TArray] = None  # [batch_size, hidden_size]
    mean_pooled_output: Optional[TArray] = None  # [batch_size, hidden_size]
    max_pooled_output: Optional[TArray] = None  # [batch_size, hidden_size]
    
    # Metadata for downstream processing
    attention_mask: Optional[TArray] = None  # [batch_size, seq_len]
    sequence_lengths: Optional[TArray] = None  # [batch_size]
    
    @property
    def batch_size(self) -> int:
        """Get the batch size from output shape."""
        # This would be implemented by the framework adapter
        raise NotImplementedError("Shape extraction must be implemented by adapter")
    
    @property
    def sequence_length(self) -> int:
        """Get the sequence length from output shape."""
        # This would be implemented by the framework adapter
        raise NotImplementedError("Shape extraction must be implemented by adapter")
    
    @property
    def hidden_size(self) -> int:
        """Get the hidden size from output shape."""
        # This would be implemented by the framework adapter
        raise NotImplementedError("Shape extraction must be implemented by adapter")


class PoolingStrategy(ABC):
    """Abstract strategy for pooling sequence representations."""
    
    @abstractmethod
    def pool(self, 
             hidden_states: TArray, 
             attention_mask: Optional[TArray] = None) -> TArray:
        """Apply pooling to hidden states.
        
        Args:
            hidden_states: Sequence hidden states [batch, seq_len, hidden_size]
            attention_mask: Optional mask [batch, seq_len]
            
        Returns:
            Pooled representation [batch, hidden_size]
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the pooling strategy."""
        pass


@dataclass
class HeadOutput(Generic[TArray]):
    """Output from a task-specific head.
    
    This represents the business-level output of a head component,
    independent of any ML framework implementation.
    """
    
    # Primary outputs
    logits: TArray  # Task-specific shape
    loss: Optional[TArray] = None  # Scalar loss value
    
    # Additional outputs for analysis
    probabilities: Optional[TArray] = None
    predictions: Optional[TArray] = None
    
    # Intermediate representations
    pre_logit_hidden: Optional[TArray] = None  # Hidden state before final projection
    attention_weights: Optional[TArray] = None  # If head uses attention
    
    # Metadata
    num_labels: Optional[int] = None
    task_type: Optional[str] = None  # "classification", "regression", etc.


@dataclass
class TrainingOutput(Generic[TArray]):
    """Output during training phase.
    
    Combines model outputs with training-specific information.
    """
    
    loss: TArray  # Primary training loss
    logits: TArray  # Model predictions
    
    # Optional detailed losses
    primary_loss: Optional[TArray] = None
    auxiliary_losses: Optional[dict[str, TArray]] = None
    
    # Metrics for monitoring
    metrics: Optional[dict[str, TArray]] = None
    
    # Gradients and optimization info
    gradient_norm: Optional[float] = None
    learning_rate: Optional[float] = None
    
    # Model outputs for analysis
    model_output: Optional[BertDomainOutput[TArray]] = None
    head_output: Optional[HeadOutput[TArray]] = None
    
    @property
    def total_loss(self) -> TArray:
        """Compute total loss including auxiliary losses."""
        if self.auxiliary_losses:
            # This would be implemented by the framework adapter
            raise NotImplementedError("Loss combination must be implemented by adapter")
        return self.loss


@dataclass
class InferenceOutput(Generic[TArray]):
    """Output during inference phase.
    
    Optimized output structure for inference without training artifacts.
    """
    
    predictions: TArray  # Final predictions
    probabilities: Optional[TArray] = None  # Prediction probabilities
    
    # Confidence and uncertainty estimates
    confidence_scores: Optional[TArray] = None
    uncertainty_estimates: Optional[TArray] = None
    
    # Feature representations
    embeddings: Optional[TArray] = None  # Pooled embeddings
    hidden_states: Optional[TArray] = None  # Last hidden states
    
    # Metadata
    prediction_time_ms: Optional[float] = None
    batch_size: Optional[int] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        result = {"predictions": self.predictions}
        
        if self.probabilities is not None:
            result["probabilities"] = self.probabilities
        if self.confidence_scores is not None:
            result["confidence_scores"] = self.confidence_scores
        if self.uncertainty_estimates is not None:
            result["uncertainty_estimates"] = self.uncertainty_estimates
        if self.prediction_time_ms is not None:
            result["prediction_time_ms"] = self.prediction_time_ms
            
        return result


class OutputProcessor:
    """Business logic for processing model outputs."""
    
    @staticmethod
    def select_pooling(
        output: BertDomainOutput[TArray],
        pooling_type: str = "cls"
    ) -> TArray:
        """Select appropriate pooled representation.
        
        Args:
            output: BERT model output
            pooling_type: Type of pooling - "cls", "mean", "max", or "pooler"
            
        Returns:
            Pooled representation
            
        Raises:
            ValueError: If pooling type is unknown or output not available
        """
        pooling_map = {
            "cls": output.cls_token_output,
            "mean": output.mean_pooled_output,
            "max": output.max_pooled_output,
            "pooler": output.pooler_output,
        }
        
        if pooling_type not in pooling_map:
            raise ValueError(f"Unknown pooling type: {pooling_type}")
            
        pooled = pooling_map[pooling_type]
        if pooled is None:
            raise ValueError(f"Pooling type '{pooling_type}' not computed in output")
            
        return pooled
    
    @staticmethod
    def merge_outputs(
        outputs: List[BertDomainOutput[TArray]]
    ) -> BertDomainOutput[TArray]:
        """Merge multiple outputs (e.g., from ensemble).
        
        Args:
            outputs: List of model outputs to merge
            
        Returns:
            Merged output
            
        Raises:
            ValueError: If outputs list is empty
        """
        if not outputs:
            raise ValueError("Cannot merge empty outputs list")
            
        # This would be implemented by the framework adapter
        raise NotImplementedError("Output merging must be implemented by adapter")