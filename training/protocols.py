"""Protocol definitions for training interfaces.

This module defines protocols (structural typing) for various training components
to enable flexible and type-safe interfaces.
"""

from typing import Protocol, Iterator, Optional, Any, runtime_checkable
import mlx.core as mx


@runtime_checkable
class DataLoaderProtocol(Protocol):
    """Protocol defining the interface expected by MLXTrainer for data loaders.
    
    Any data loader that implements this protocol can be used with MLXTrainer,
    regardless of its actual class hierarchy.
    """
    
    def __iter__(self) -> Iterator[dict[str, mx.array]]:
        """Return an iterator over batches.
        
        Each batch must be a dictionary containing:
        - input_ids: MLX array of shape [batch_size, sequence_length]
        - attention_mask: MLX array of shape [batch_size, sequence_length]
        - labels: MLX array of shape [batch_size] or [batch_size, 1]
        """
        ...
    
    def __len__(self) -> int:
        """Return the number of batches in the data loader."""
        ...
    
    # Optional attributes
    dataset_spec: Optional[Any] = None
    batch_size: Optional[int] = None
    max_length: Optional[int] = None


@runtime_checkable
class OptimizerProtocol(Protocol):
    """Protocol for MLX optimizers."""
    
    def update(self, model: Any, gradients: Any) -> None:
        """Update model parameters using gradients."""
        ...
    
    # Optional attributes
    learning_rate: Optional[float] = None
    state: Optional[Any] = None


@runtime_checkable 
class ModelProtocol(Protocol):
    """Protocol for models compatible with MLXTrainer."""
    
    def __call__(self, **kwargs) -> dict[str, mx.array]:
        """Forward pass returning loss and logits."""
        ...
    
    def parameters(self) -> Any:
        """Return model parameters."""
        ...
    
    def save_pretrained(self, path: str) -> None:
        """Save model weights."""
        ...
    
    def load_pretrained(self, path: str) -> None:
        """Load model weights."""
        ...
    
    # Optional attributes
    config: Optional[Any] = None