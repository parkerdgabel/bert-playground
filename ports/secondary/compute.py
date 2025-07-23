"""Secondary compute port - Compute backend that the application depends on.

This port defines the compute interface that the application core uses
to perform ML computations. It's a driven port implemented by adapters
for different ML frameworks (MLX, PyTorch, JAX, etc.).
"""

from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np
from typing_extensions import TypeAlias

# Type aliases for framework-agnostic arrays
Array: TypeAlias = Any  # Framework-specific array type
ArrayLike: TypeAlias = np.ndarray | list | tuple | Any
Device: TypeAlias = str | Any
DType: TypeAlias = Any
Shape: TypeAlias = tuple[int, ...]
Module: TypeAlias = Any  # Framework-specific neural network module


class DataType(Enum):
    """Framework-agnostic data types."""
    
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    INT32 = "int32"
    INT64 = "int64"
    BOOL = "bool"


@runtime_checkable
class ComputeBackend(Protocol):
    """Secondary port for compute operations.
    
    This interface is implemented by adapters for specific ML frameworks.
    The application core depends on this interface for all compute operations.
    """

    @property
    def name(self) -> str:
        """Name of the compute backend (e.g., 'mlx', 'pytorch')."""
        ...

    @property
    def supports_compilation(self) -> bool:
        """Whether this backend supports JIT compilation."""
        ...

    def array(
        self,
        data: ArrayLike,
        dtype: DataType | DType | None = None,
        device: Device | None = None
    ) -> Array:
        """Create an array from data.
        
        Args:
            data: Input data
            dtype: Data type
            device: Target device
            
        Returns:
            Backend-specific array
        """
        ...

    def zeros(
        self,
        shape: Shape,
        dtype: DataType | DType | None = None,
        device: Device | None = None
    ) -> Array:
        """Create array of zeros.
        
        Args:
            shape: Array shape
            dtype: Data type
            device: Target device
            
        Returns:
            Array of zeros
        """
        ...

    def ones(
        self,
        shape: Shape,
        dtype: DataType | DType | None = None,
        device: Device | None = None
    ) -> Array:
        """Create array of ones.
        
        Args:
            shape: Array shape
            dtype: Data type
            device: Target device
            
        Returns:
            Array of ones
        """
        ...

    def randn(
        self,
        shape: Shape,
        dtype: DataType | DType | None = None,
        device: Device | None = None,
        seed: int | None = None
    ) -> Array:
        """Create array with normal random values.
        
        Args:
            shape: Array shape
            dtype: Data type
            device: Target device
            seed: Random seed
            
        Returns:
            Array with random values
        """
        ...

    def to_numpy(self, array: Array) -> np.ndarray:
        """Convert array to numpy.
        
        Args:
            array: Backend array
            
        Returns:
            Numpy array
        """
        ...

    def from_numpy(
        self,
        array: np.ndarray,
        dtype: DataType | DType | None = None,
        device: Device | None = None
    ) -> Array:
        """Create array from numpy.
        
        Args:
            array: Numpy array
            dtype: Target data type
            device: Target device
            
        Returns:
            Backend array
        """
        ...

    def shape(self, array: Array) -> Shape:
        """Get array shape.
        
        Args:
            array: Input array
            
        Returns:
            Array shape
        """
        ...

    def dtype(self, array: Array) -> DType:
        """Get array data type.
        
        Args:
            array: Input array
            
        Returns:
            Array data type
        """
        ...

    def device(self, array: Array) -> Device:
        """Get array device.
        
        Args:
            array: Input array
            
        Returns:
            Array device
        """
        ...

    def compile(
        self,
        function: Callable[..., Any],
        static_argnums: Sequence[int] | None = None,
        static_argnames: Sequence[str] | None = None
    ) -> Callable[..., Any]:
        """Compile a function for faster execution.
        
        Args:
            function: Function to compile
            static_argnums: Indices of static arguments
            static_argnames: Names of static arguments
            
        Returns:
            Compiled function
        """
        ...

    def gradient(
        self,
        function: Callable[..., Array],
        argnums: int | Sequence[int] = 0
    ) -> Callable[..., Array | tuple[Array, ...]]:
        """Create gradient function.
        
        Args:
            function: Function to differentiate
            argnums: Argument indices to differentiate with respect to
            
        Returns:
            Gradient function
        """
        ...

    def value_and_gradient(
        self,
        function: Callable[..., Array],
        argnums: int | Sequence[int] = 0
    ) -> Callable[..., tuple[Array, Array | tuple[Array, ...]]]:
        """Create function that returns both value and gradient.
        
        Args:
            function: Function to differentiate
            argnums: Argument indices to differentiate with respect to
            
        Returns:
            Function returning (value, gradient)
        """
        ...

    def load_weights(self, path: str) -> dict[str, Array]:
        """Load weights from file.
        
        Args:
            path: Path to weights file
            
        Returns:
            Dictionary of parameter names to arrays
        """
        ...

    def tree_unflatten(self, items: list[tuple[str, Array]]) -> dict[str, Any]:
        """Unflatten a list of (key, value) pairs into nested dictionary.
        
        Args:
            items: List of (key, value) pairs with dot-separated keys
            
        Returns:
            Nested dictionary structure
        """
        ...

    def save_arrays(self, path: str, arrays: dict[str, Array]) -> None:
        """Save arrays to file.
        
        Args:
            path: Path to save file
            arrays: Dictionary of arrays to save
        """
        ...

    def load_arrays(self, path: str) -> dict[str, Array]:
        """Load arrays from file.
        
        Args:
            path: Path to load file
            
        Returns:
            Dictionary of loaded arrays
        """
        ...
    
    # Model-specific operations
    
    def forward_pass(
        self,
        model_spec: Dict[str, Any],
        inputs: Dict[str, Array],
        training: bool = False,
    ) -> Dict[str, Array]:
        """Perform forward pass through model.
        
        Args:
            model_spec: Model specification (architecture, weights, etc.)
            inputs: Input data as dictionary of arrays
            training: Whether in training mode
            
        Returns:
            Dictionary containing outputs (logits, loss, hidden_states, etc.)
        """
        ...
    
    def compute_loss(
        self,
        predictions: Array,
        targets: Array,
        loss_type: str = "cross_entropy",
        **kwargs: Any,
    ) -> Array:
        """Compute loss between predictions and targets.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            loss_type: Type of loss function
            **kwargs: Additional loss-specific parameters
            
        Returns:
            Loss value
        """
        ...
    
    def backward_pass(
        self,
        loss: Array,
        model_spec: Dict[str, Any],
        retain_graph: bool = False,
    ) -> Dict[str, Array]:
        """Perform backward pass to compute gradients.
        
        Args:
            loss: Loss value to backpropagate
            model_spec: Model specification
            retain_graph: Whether to retain computation graph
            
        Returns:
            Dictionary of gradients
        """
        ...
    
    def optimize_step(
        self,
        model_params: Dict[str, Array],
        gradients: Dict[str, Array],
        optimizer_state: Dict[str, Any],
        learning_rate: float,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Array], Dict[str, Any]]:
        """Perform optimization step.
        
        Args:
            model_params: Current model parameters
            gradients: Computed gradients
            optimizer_state: Current optimizer state
            learning_rate: Learning rate
            **kwargs: Additional optimizer parameters
            
        Returns:
            Tuple of (updated_parameters, updated_optimizer_state)
        """
        ...
    
    def count_parameters(
        self,
        model_spec: Dict[str, Any],
        trainable_only: bool = False,
    ) -> int:
        """Count model parameters.
        
        Args:
            model_spec: Model specification
            trainable_only: Whether to count only trainable parameters
            
        Returns:
            Number of parameters
        """
        ...


@runtime_checkable
class NeuralOps(Protocol):
    """Neural network operations - a specialized compute port.
    
    This provides higher-level neural network operations that
    the application core uses. Implemented by framework-specific adapters.
    """

    def linear(
        self,
        input: Array,
        weight: Array,
        bias: Array | None = None
    ) -> Array:
        """Linear transformation (fully connected layer).
        
        Args:
            input: Input array [*, in_features]
            weight: Weight matrix [out_features, in_features]
            bias: Optional bias [out_features]
            
        Returns:
            Output array [*, out_features]
        """
        ...

    def embedding(
        self,
        input: Array,
        weight: Array,
        padding_idx: int | None = None
    ) -> Array:
        """Embedding lookup.
        
        Args:
            input: Input indices [*]
            weight: Embedding matrix [num_embeddings, embedding_dim]
            padding_idx: Optional padding index
            
        Returns:
            Embeddings [*, embedding_dim]
        """
        ...

    def layer_norm(
        self,
        input: Array,
        normalized_shape: Shape,
        weight: Array | None = None,
        bias: Array | None = None,
        eps: float = 1e-5
    ) -> Array:
        """Layer normalization.
        
        Args:
            input: Input array
            normalized_shape: Shape to normalize over
            weight: Optional scale parameter
            bias: Optional shift parameter
            eps: Epsilon for numerical stability
            
        Returns:
            Normalized array
        """
        ...

    def dropout(
        self,
        input: Array,
        p: float = 0.5,
        training: bool = True,
        seed: int | None = None
    ) -> Array:
        """Dropout regularization.
        
        Args:
            input: Input array
            p: Dropout probability
            training: Whether in training mode
            seed: Random seed
            
        Returns:
            Array with dropout applied
        """
        ...

    def softmax(
        self,
        input: Array,
        dim: int = -1
    ) -> Array:
        """Softmax activation.
        
        Args:
            input: Input array
            dim: Dimension to apply softmax over
            
        Returns:
            Softmax output
        """
        ...

    def cross_entropy(
        self,
        input: Array,
        target: Array,
        reduction: str = "mean",
        ignore_index: int = -100
    ) -> Array:
        """Cross entropy loss.
        
        Args:
            input: Predictions [batch_size, num_classes]
            target: Targets [batch_size]
            reduction: Reduction method ('none', 'mean', 'sum')
            ignore_index: Index to ignore
            
        Returns:
            Loss value
        """
        ...

    def attention(
        self,
        query: Array,
        key: Array,
        value: Array,
        mask: Array | None = None,
        dropout_p: float = 0.0,
        scale: float | None = None,
        training: bool = True,
    ) -> tuple[Array, Array | None]:
        """Scaled dot-product attention.
        
        Args:
            query: Query tensor [batch, seq_len, d_k]
            key: Key tensor [batch, seq_len, d_k]
            value: Value tensor [batch, seq_len, d_v]
            mask: Optional attention mask
            dropout_p: Dropout probability
            scale: Optional scale factor (default: 1/sqrt(d_k))
            training: Whether in training mode
            
        Returns:
            Tuple of (attention output, attention weights)
        """
        ...

    def gelu(self, input: Array, approximate: bool = False) -> Array:
        """GELU activation function.
        
        Args:
            input: Input array
            approximate: Whether to use approximate version
            
        Returns:
            GELU output
        """
        ...

    def swiglu(self, input: Array, gate: Array) -> Array:
        """SwiGLU activation function.
        
        Args:
            input: Input array
            gate: Gate array
            
        Returns:
            SwiGLU output
        """
        ...