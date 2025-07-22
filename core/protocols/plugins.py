"""Plugin system protocols for k-bert.

These protocols define the contracts for custom components that can be
plugged into k-bert's architecture.
"""

from typing import Any, Protocol

from core.ports.compute import Array, Module
from pydantic import BaseModel


class PluginMetadata(BaseModel):
    """Metadata for a plugin component."""
    
    name: str
    version: str = "1.0.0"
    description: str | None = None
    author: str | None = None
    tags: list[str] = []
    requirements: list[str] = []


class Plugin(Protocol):
    """Base protocol for all k-bert plugins."""
    
    @property
    def config(self) -> dict[str, Any] | None:
        """Plugin configuration."""
        ...
    
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        ...
    
    @property
    def name(self) -> str:
        """Get plugin name."""
        ...
    
    @property
    def version(self) -> str:
        """Get plugin version."""
        ...


class HeadPlugin(Plugin):
    """Protocol for custom BERT head implementations."""
    
    def __call__(
        self,
        hidden_states: Array,
        attention_mask: Array | None = None,
        **kwargs
    ) -> dict[str, Array]:
        """Forward pass through the head.
        
        Args:
            hidden_states: BERT output hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing:
                - logits: Output logits
                - Any additional outputs
        """
        ...
    
    def compute_loss(
        self,
        logits: Array,
        labels: Array,
        **kwargs
    ) -> Array:
        """Compute loss for the head.
        
        Args:
            logits: Model output logits
            labels: Ground truth labels
            **kwargs: Additional arguments (e.g., class weights)
            
        Returns:
            Loss value
        """
        ...
    
    def get_output_size(self) -> int:
        """Get the output size of the head."""
        ...
    
    def get_metrics(self) -> list[str]:
        """Get list of metrics this head supports.
        
        Returns:
            List of metric names
        """
        ...


class AugmenterPlugin(Plugin):
    """Protocol for custom data augmentation strategies."""
    
    def augment(
        self,
        data: dict[str, Any],
        training: bool = True,
        **kwargs
    ) -> dict[str, Any]:
        """Apply augmentation to a data sample.
        
        Args:
            data: Input data dictionary
            training: Whether in training mode
            **kwargs: Additional arguments
            
        Returns:
            Augmented data dictionary
        """
        ...
    
    def augment_batch(
        self,
        batch: list[dict[str, Any]],
        training: bool = True,
        **kwargs
    ) -> list[dict[str, Any]]:
        """Apply augmentation to a batch of samples.
        
        Args:
            batch: List of data dictionaries
            training: Whether in training mode
            **kwargs: Additional arguments
            
        Returns:
            List of augmented data dictionaries
        """
        ...
    
    def get_augmentation_params(self) -> dict[str, Any]:
        """Get current augmentation parameters.
        
        Returns:
            Dictionary of augmentation parameters
        """
        ...


class FeatureExtractorPlugin(Plugin):
    """Protocol for custom feature extraction."""
    
    def extract_features(
        self,
        data: dict[str, Any],
        **kwargs
    ) -> dict[str, Any]:
        """Extract features from raw data.
        
        Args:
            data: Raw data dictionary
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing extracted features
        """
        ...
    
    def get_feature_names(self) -> list[str]:
        """Get list of feature names this extractor produces.
        
        Returns:
            List of feature names
        """
        ...
    
    def batch_extract(
        self,
        batch: list[dict[str, Any]],
        **kwargs
    ) -> list[dict[str, Any]]:
        """Extract features from a batch of samples.
        
        Args:
            batch: List of raw data dictionaries
            **kwargs: Additional arguments
            
        Returns:
            List of feature dictionaries
        """
        ...


class DataLoaderPlugin(Plugin):
    """Protocol for custom data loading strategies."""
    
    def load_data(
        self,
        path: str,
        split: str = "train",
        **kwargs
    ) -> list[dict[str, Any]]:
        """Load data from file.
        
        Args:
            path: Path to data file
            split: Data split (train/val/test)
            **kwargs: Additional arguments
            
        Returns:
            List of data samples
        """
        ...
    
    def get_data_info(self) -> dict[str, Any]:
        """Get information about the loaded data.
        
        Returns:
            Dictionary containing data statistics and metadata
        """
        ...
    
    def validate_data(
        self,
        data: list[dict[str, Any]]
    ) -> tuple[bool, list[str]]:
        """Validate loaded data.
        
        Args:
            data: List of data samples
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        ...


class ModelPlugin(Plugin):
    """Protocol for custom model architectures."""
    
    def build_model(self, config: dict[str, Any]) -> Any:
        """Build the model.
        
        Args:
            config: Model configuration
            
        Returns:
            Model instance
        """
        ...
    
    def get_default_config(self) -> dict[str, Any]:
        """Get default model configuration.
        
        Returns:
            Dictionary of default configuration values
        """
        ...
    
    def load_pretrained(self, path: str, **kwargs) -> Any:
        """Load pretrained weights.
        
        Args:
            path: Path to pretrained weights
            **kwargs: Additional arguments
            
        Returns:
            Model with loaded weights
        """
        ...


class MetricPlugin(Plugin):
    """Protocol for custom evaluation metrics."""
    
    def compute(
        self,
        predictions: Array,
        labels: Array,
        **kwargs
    ) -> dict[str, float]:
        """Compute metric values.
        
        Args:
            predictions: Model predictions
            labels: Ground truth labels
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of metric name to value
        """
        ...
    
    def get_metric_names(self) -> list[str]:
        """Get list of metric names this plugin computes.
        
        Returns:
            List of metric names
        """
        ...
    
    def is_better(self, value1: float, value2: float, metric: str) -> bool:
        """Check if value1 is better than value2 for given metric.
        
        Args:
            value1: First metric value
            value2: Second metric value
            metric: Metric name
            
        Returns:
            True if value1 is better than value2
        """
        ...