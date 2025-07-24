"""Domain model protocols - Core model abstractions.

These protocols define the fundamental model contracts used throughout
the system. They are independent of any specific implementation.
"""

from pathlib import Path
from typing import Any, Protocol

from .compute import Array, Module


class Model(Protocol):
    """Protocol for models compatible with the training system."""

    def __call__(self, inputs: dict[str, Array]) -> dict[str, Array]:
        """Forward pass of the model.
        
        Args:
            inputs: Dictionary of input arrays
            
        Returns:
            Dictionary of output arrays
        """
        ...

    def parameters(self) -> dict[str, Array]:
        """Get model parameters.
        
        Returns:
            Dictionary mapping parameter names to arrays
        """
        ...

    def save_pretrained(self, path: Path) -> None:
        """Save model to disk.
        
        Args:
            path: Directory to save model to
        """
        ...

    @classmethod
    def load_pretrained(cls, path: Path) -> "Model":
        """Load model from disk.
        
        Args:
            path: Directory to load model from
            
        Returns:
            Loaded model
        """
        ...

    @property
    def config(self) -> Any | None:
        """Model configuration."""
        ...


class Head(Protocol):
    """Protocol for task-specific heads."""

    def __call__(
        self,
        hidden_states: Array,
        attention_mask: Array | None = None,
        labels: Array | None = None,
        **kwargs
    ) -> dict[str, Array]:
        """Forward pass through the head.
        
        Args:
            hidden_states: BERT output hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask [batch_size, seq_len]
            labels: Optional labels for computing loss
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing:
                - logits: Output logits
                - loss: Loss value (if labels provided)
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
        """Get the output size of the head.
        
        Returns:
            Number of output units
        """
        ...

    def get_metrics(self) -> list[str]:
        """Get list of metrics this head supports.
        
        Returns:
            List of metric names
        """
        ...


class ModelFactory(Protocol):
    """Protocol for model factories."""
    
    def create_model(self, config: dict[str, Any]) -> Model:
        """Create a model from configuration.
        
        Args:
            config: Model configuration
            
        Returns:
            Created model instance
        """
        ...
    
    def register_model(self, name: str, model_class: type[Model]) -> None:
        """Register a new model type.
        
        Args:
            name: Name to register model under
            model_class: Model class to register
        """
        ...
    
    def list_models(self) -> list[str]:
        """List available model types.
        
        Returns:
            List of registered model names
        """
        ...