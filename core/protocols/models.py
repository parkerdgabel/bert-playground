"""Model-related protocols for k-bert.

These protocols define the contracts for models, heads, and configurations.
"""

from pathlib import Path
from typing import Any, Protocol

import mlx.core as mx


class Model(Protocol):
    """Protocol for models compatible with the training system."""

    def __call__(self, inputs: dict[str, mx.array]) -> dict[str, mx.array]:
        """Forward pass of the model.
        
        Args:
            inputs: Dictionary of input arrays
            
        Returns:
            Dictionary of output arrays
        """
        ...

    def parameters(self) -> dict[str, mx.array]:
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
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        labels: mx.array | None = None,
        **kwargs
    ) -> dict[str, mx.array]:
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


class ModelConfig(Protocol):
    """Protocol for model configuration."""

    @property
    def model_type(self) -> str:
        """Type of model (e.g., 'bert', 'modernbert')."""
        ...

    @property
    def hidden_size(self) -> int:
        """Size of hidden layers."""
        ...

    @property
    def num_hidden_layers(self) -> int:
        """Number of hidden layers."""
        ...

    @property
    def num_attention_heads(self) -> int:
        """Number of attention heads."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Configuration as dictionary
        """
        ...

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "ModelConfig":
        """Create configuration from dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Model configuration instance
        """
        ...

    def validate(self) -> list[str]:
        """Validate configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        ...