"""Base classes for k-bert plugins.

This module defines the base classes that custom components should inherit from
to integrate seamlessly with k-bert's architecture.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
from pydantic import BaseModel


class PluginMetadata(BaseModel):
    """Metadata for a plugin component."""
    
    name: str
    version: str = "1.0.0"
    description: Optional[str] = None
    author: Optional[str] = None
    tags: List[str] = []
    requirements: List[str] = []


class BasePlugin(ABC):
    """Base class for all k-bert plugins."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize plugin with configuration.
        
        Args:
            config: Plugin-specific configuration
        """
        self.config = config or {}
        self._metadata = self.get_metadata()
    
    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        pass
    
    @property
    def name(self) -> str:
        """Get plugin name."""
        return self._metadata.name
    
    @property
    def version(self) -> str:
        """Get plugin version."""
        return self._metadata.version


class HeadPlugin(BasePlugin):
    """Base class for custom BERT head implementations.
    
    Custom heads should inherit from this class and implement the required methods
    to work with k-bert's model architecture.
    """
    
    @abstractmethod
    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        **kwargs
    ) -> Dict[str, mx.array]:
        """Forward pass through the head.
        
        Args:
            hidden_states: BERT output hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing:
                - logits: Output logits
                - Any additional outputs (e.g., attention weights)
        """
        pass
    
    @abstractmethod
    def compute_loss(
        self,
        logits: mx.array,
        labels: mx.array,
        **kwargs
    ) -> mx.array:
        """Compute loss for the head.
        
        Args:
            logits: Model output logits
            labels: Ground truth labels
            **kwargs: Additional arguments (e.g., class weights)
            
        Returns:
            Loss value
        """
        pass
    
    @abstractmethod
    def get_output_size(self) -> int:
        """Get the output size of the head."""
        pass
    
    def get_metrics(self) -> List[str]:
        """Get list of metrics this head supports.
        
        Returns:
            List of metric names
        """
        return ["loss"]


class AugmenterPlugin(BasePlugin):
    """Base class for custom data augmentation strategies.
    
    Custom augmenters should inherit from this class to implement
    domain-specific data augmentation techniques.
    """
    
    @abstractmethod
    def augment(
        self,
        data: Dict[str, Any],
        training: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Apply augmentation to a data sample.
        
        Args:
            data: Input data dictionary
            training: Whether in training mode
            **kwargs: Additional arguments
            
        Returns:
            Augmented data dictionary
        """
        pass
    
    def augment_batch(
        self,
        batch: List[Dict[str, Any]],
        training: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Apply augmentation to a batch of samples.
        
        Args:
            batch: List of data dictionaries
            training: Whether in training mode
            **kwargs: Additional arguments
            
        Returns:
            List of augmented data dictionaries
        """
        return [self.augment(sample, training, **kwargs) for sample in batch]
    
    @abstractmethod
    def get_augmentation_params(self) -> Dict[str, Any]:
        """Get current augmentation parameters.
        
        Returns:
            Dictionary of augmentation parameters
        """
        pass


class FeatureExtractorPlugin(BasePlugin):
    """Base class for custom feature extraction.
    
    Feature extractors transform raw data into features suitable
    for BERT models.
    """
    
    @abstractmethod
    def extract_features(
        self,
        data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Extract features from raw data.
        
        Args:
            data: Raw data dictionary
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing extracted features
        """
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get list of feature names this extractor produces.
        
        Returns:
            List of feature names
        """
        pass
    
    def batch_extract(
        self,
        batch: List[Dict[str, Any]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Extract features from a batch of samples.
        
        Args:
            batch: List of raw data dictionaries
            **kwargs: Additional arguments
            
        Returns:
            List of feature dictionaries
        """
        return [self.extract_features(sample, **kwargs) for sample in batch]


class DataLoaderPlugin(BasePlugin):
    """Base class for custom data loading strategies.
    
    Custom data loaders can implement competition-specific
    data loading and preprocessing logic.
    """
    
    @abstractmethod
    def load_data(
        self,
        path: str,
        split: str = "train",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Load data from file.
        
        Args:
            path: Path to data file
            split: Data split (train/val/test)
            **kwargs: Additional arguments
            
        Returns:
            List of data samples
        """
        pass
    
    @abstractmethod
    def get_data_info(self) -> Dict[str, Any]:
        """Get information about the loaded data.
        
        Returns:
            Dictionary containing data statistics and metadata
        """
        pass
    
    def validate_data(
        self,
        data: List[Dict[str, Any]]
    ) -> Tuple[bool, List[str]]:
        """Validate loaded data.
        
        Args:
            data: List of data samples
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        return True, []


class ModelPlugin(BasePlugin):
    """Base class for custom model architectures.
    
    This allows users to implement custom BERT variants or
    entirely different architectures that work with k-bert.
    """
    
    @abstractmethod
    def build_model(self, config: Dict[str, Any]) -> Any:
        """Build the model.
        
        Args:
            config: Model configuration
            
        Returns:
            Model instance
        """
        pass
    
    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        """Get default model configuration.
        
        Returns:
            Dictionary of default configuration values
        """
        pass
    
    def load_pretrained(self, path: str, **kwargs) -> Any:
        """Load pretrained weights.
        
        Args:
            path: Path to pretrained weights
            **kwargs: Additional arguments
            
        Returns:
            Model with loaded weights
        """
        raise NotImplementedError("Pretrained loading not implemented")


class MetricPlugin(BasePlugin):
    """Base class for custom evaluation metrics.
    
    Custom metrics can be implemented for competition-specific
    evaluation requirements.
    """
    
    @abstractmethod
    def compute(
        self,
        predictions: mx.array,
        labels: mx.array,
        **kwargs
    ) -> Dict[str, float]:
        """Compute metric values.
        
        Args:
            predictions: Model predictions
            labels: Ground truth labels
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of metric name to value
        """
        pass
    
    @abstractmethod
    def get_metric_names(self) -> List[str]:
        """Get list of metric names this plugin computes.
        
        Returns:
            List of metric names
        """
        pass
    
    def is_better(self, value1: float, value2: float, metric: str) -> bool:
        """Check if value1 is better than value2 for given metric.
        
        Args:
            value1: First metric value
            value2: Second metric value
            metric: Metric name
            
        Returns:
            True if value1 is better than value2
        """
        # Default: higher is better
        return value1 > value2