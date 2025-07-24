"""Plugin port - Contracts for extensibility.

This port defines the protocols for plugin-based extensions to the system.
Plugins allow external actors to extend functionality without modifying core code.
"""

from typing import Any, Protocol, runtime_checkable
from dataclasses import dataclass

from infrastructure.di import port


@dataclass
class PluginMetadata:
    """Metadata for a plugin component."""
    
    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    tags: list[str] = None
    requirements: list[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.requirements is None:
            self.requirements = []


@runtime_checkable
class Plugin(Protocol):
    """Base protocol for all plugins."""
    
    @property
    def config(self) -> dict[str, Any]:
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


@runtime_checkable
class HeadPlugin(Plugin, Protocol):
    """Protocol for custom head implementations."""
    
    def forward(
        self,
        hidden_states: Any,
        attention_mask: Any = None,
        **kwargs
    ) -> dict[str, Any]:
        """Forward pass through the head.
        
        Args:
            hidden_states: Input hidden states
            attention_mask: Attention mask
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing outputs
        """
        ...
    
    def compute_loss(
        self,
        outputs: dict[str, Any],
        labels: Any,
        **kwargs
    ) -> Any:
        """Compute loss for the head.
        
        Args:
            outputs: Model outputs
            labels: Ground truth labels
            **kwargs: Additional arguments
            
        Returns:
            Loss value
        """
        ...
    
    def get_output_size(self) -> int:
        """Get the output size of the head."""
        ...
    
    def get_metrics(self) -> list[str]:
        """Get list of metrics this head supports."""
        ...


@runtime_checkable
class AugmenterPlugin(Plugin, Protocol):
    """Protocol for data augmentation plugins."""
    
    def augment(
        self,
        text: str,
        metadata: dict[str, Any] = None,
        **kwargs
    ) -> list[str]:
        """Augment text data.
        
        Args:
            text: Input text
            metadata: Optional metadata
            **kwargs: Additional arguments
            
        Returns:
            List of augmented texts
        """
        ...
    
    def get_augmentation_types(self) -> list[str]:
        """Get list of augmentation types supported."""
        ...


@runtime_checkable
class FeatureExtractorPlugin(Plugin, Protocol):
    """Protocol for feature extraction plugins."""
    
    def extract_features(
        self,
        data: dict[str, Any],
        **kwargs
    ) -> dict[str, Any]:
        """Extract features from data.
        
        Args:
            data: Input data
            **kwargs: Additional arguments
            
        Returns:
            Extracted features
        """
        ...
    
    def get_feature_names(self) -> list[str]:
        """Get list of feature names produced."""
        ...


@runtime_checkable
class DataLoaderPlugin(Plugin, Protocol):
    """Protocol for custom data loading plugins."""
    
    def load_data(
        self,
        path: str,
        **kwargs
    ) -> Any:
        """Load data from path.
        
        Args:
            path: Data path
            **kwargs: Additional arguments
            
        Returns:
            Loaded data
        """
        ...
    
    def validate_data(
        self,
        data: Any,
        **kwargs
    ) -> tuple[bool, list[str]]:
        """Validate loaded data.
        
        Args:
            data: Data to validate
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        ...


@runtime_checkable
class ModelPlugin(Plugin, Protocol):
    """Protocol for custom model architectures."""
    
    def create_model(
        self,
        config: dict[str, Any],
        **kwargs
    ) -> Any:
        """Create a model instance.
        
        Args:
            config: Model configuration
            **kwargs: Additional arguments
            
        Returns:
            Model instance
        """
        ...
    
    def get_model_type(self) -> str:
        """Get the model type identifier."""
        ...
    
    def get_default_config(self) -> dict[str, Any]:
        """Get default model configuration."""
        ...


@runtime_checkable
class MetricPlugin(Plugin, Protocol):
    """Protocol for custom metrics."""
    
    def compute(
        self,
        predictions: Any,
        targets: Any,
        **kwargs
    ) -> float:
        """Compute metric value.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            **kwargs: Additional arguments
            
        Returns:
            Metric value
        """
        ...
    
    def get_metric_name(self) -> str:
        """Get metric name."""
        ...
    
    def higher_is_better(self) -> bool:
        """Whether higher values are better."""
        ...