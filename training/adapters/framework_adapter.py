"""Comprehensive framework adapter implementation for k-bert.

This module provides the full implementation of the FrameworkAdapter protocol
from core/protocols/training.py, bridging the gap between the abstract protocol
and concrete framework implementations.
"""

from typing import Any, Dict, Tuple, Optional

from core.protocols.training import FrameworkAdapter as IFrameworkAdapter, Model, Optimizer
from core.ports.compute import ComputeBackend, NeuralOps
from training.adapters.mlx_adapter import MLXFrameworkAdapter
from loguru import logger


class FrameworkAdapter(IFrameworkAdapter):
    """Comprehensive implementation of the FrameworkAdapter protocol.
    
    This adapter provides a complete abstraction layer over framework-specific
    operations, delegating to the appropriate backend implementation.
    """
    
    def __init__(self, backend: str = "mlx"):
        """Initialize framework adapter.
        
        Args:
            backend: Name of the backend to use ("mlx", "pytorch", etc.)
        """
        self.backend_name = backend.lower()
        self._adapter = self._create_adapter(backend)
        self._compute_backend: Optional[ComputeBackend] = None
        self._neural_ops: Optional[NeuralOps] = None
        
        # Try to get compute backend and neural ops if available
        if backend == "mlx":
            try:
                from core.adapters.mlx_adapter import MLXComputeAdapter, MLXNeuralOpsAdapter
                self._compute_backend = MLXComputeAdapter()
                self._neural_ops = MLXNeuralOpsAdapter(self._compute_backend)
            except ImportError:
                logger.warning("Could not import MLX compute adapters")
    
    def _create_adapter(self, backend: str):
        """Create the appropriate backend adapter.
        
        Args:
            backend: Backend name
            
        Returns:
            Backend-specific adapter
            
        Raises:
            ValueError: If backend is not supported
        """
        if backend.lower() == "mlx":
            return MLXFrameworkAdapter()
        else:
            raise ValueError(f"Unsupported backend: {backend}")
    
    @property
    def name(self) -> str:
        """Framework name."""
        return self._adapter.name
    
    @property
    def available(self) -> bool:
        """Whether framework is available."""
        return self._adapter.available
    
    @property
    def supports_compilation(self) -> bool:
        """Whether this backend supports JIT compilation."""
        # Check if compute backend supports compilation
        if self._compute_backend:
            return self._compute_backend.supports_compilation
        # Fall back to checking if compile_function does anything
        return hasattr(self._adapter, 'compile_function')
    
    def to_tensor(self, data: Any) -> Any:
        """Convert data to framework tensor.
        
        Args:
            data: Input data
            
        Returns:
            Framework-specific tensor
        """
        return self._adapter.to_tensor(data)
    
    def to_python(self, tensor: Any) -> float | int | list:
        """Convert tensor to Python types.
        
        Args:
            tensor: Framework tensor
            
        Returns:
            Python scalar or list
        """
        return self._adapter.to_python(tensor)
    
    def compute_gradient_norm(self, gradients: dict[str, Any]) -> float:
        """Compute norm of gradients.
        
        Args:
            gradients: Dictionary of gradients
            
        Returns:
            Gradient norm as float
        """
        return self._adapter.compute_gradient_norm(gradients)
    
    def clip_gradients_by_norm(
        self, 
        gradients: dict[str, Any], 
        max_norm: float
    ) -> tuple[dict[str, Any], float]:
        """Clip gradients by norm.
        
        Args:
            gradients: Dictionary of gradients
            max_norm: Maximum allowed norm
            
        Returns:
            Tuple of (clipped_gradients, actual_norm)
        """
        return self._adapter.clip_gradients_by_norm(gradients, max_norm)
    
    def scale_gradients(self, gradients: dict[str, Any], scale: float) -> dict[str, Any]:
        """Scale gradients by factor.
        
        Args:
            gradients: Dictionary of gradients
            scale: Scaling factor
            
        Returns:
            Scaled gradients
        """
        return self._adapter.scale_gradients(gradients, scale)
    
    def update_model_parameters(
        self, 
        model: Model, 
        optimizer: Optimizer, 
        gradients: dict[str, Any]
    ) -> None:
        """Update model parameters with optimizer.
        
        Args:
            model: Model to update
            optimizer: Optimizer instance
            gradients: Gradients to apply
        """
        self._adapter.update_model_parameters(model, optimizer, gradients)
    
    def get_learning_rate(self, optimizer: Optimizer) -> float:
        """Get current learning rate from optimizer.
        
        Args:
            optimizer: Optimizer instance
            
        Returns:
            Current learning rate
        """
        return self._adapter.get_learning_rate(optimizer)
    
    def apply_mixed_precision(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Apply mixed precision to inputs if configured.
        
        Args:
            inputs: Model inputs
            
        Returns:
            Inputs with mixed precision applied
        """
        return self._adapter.apply_mixed_precision(inputs)
    
    def evaluate_tensors(self, *tensors: Any) -> None:
        """Force evaluation of lazy tensors.
        
        Args:
            *tensors: Tensors to evaluate
        """
        self._adapter.evaluate_tensors(*tensors)
    
    def create_value_and_grad_fn(self, model: Any, loss_fn: Any) -> Any:
        """Create value and gradient function.
        
        Args:
            model: Model instance
            loss_fn: Loss function
            
        Returns:
            Function that computes value and gradients
        """
        return self._adapter.create_value_and_grad_fn(model, loss_fn)
    
    def compile_function(self, fn: Any, **kwargs) -> Any:
        """Compile function for optimization.
        
        Args:
            fn: Function to compile
            **kwargs: Backend-specific compilation options
            
        Returns:
            Compiled function
        """
        return self._adapter.compile_function(fn, **kwargs)
    
    def set_random_seed(self, seed: int) -> None:
        """Set random seed for reproducibility.
        
        Args:
            seed: Random seed
        """
        self._adapter.set_random_seed(seed)
    
    @property
    def compute_backend(self) -> Optional[ComputeBackend]:
        """Get compute backend if available."""
        return self._compute_backend
    
    @property
    def neural_ops(self) -> Optional[NeuralOps]:
        """Get neural operations if available."""
        return self._neural_ops
    
    def accumulate_gradients(
        self, 
        accumulated: dict[str, Any], 
        current: dict[str, Any]
    ) -> dict[str, Any]:
        """Accumulate gradients.
        
        Args:
            accumulated: Previously accumulated gradients
            current: Current gradients to add
            
        Returns:
            Combined gradients
        """
        return self._adapter.accumulate_gradients(accumulated, current)
    
    def get_model_parameters(self, model: Any) -> dict[str, Any]:
        """Get model parameters.
        
        Args:
            model: Model instance
            
        Returns:
            Dictionary of parameters
        """
        return self._adapter.get_model_parameters(model)
    
    def tensor_multiply(self, tensor: Any, scalar: float) -> Any:
        """Multiply tensor by scalar.
        
        Args:
            tensor: Input tensor
            scalar: Scalar multiplier
            
        Returns:
            Scaled tensor
        """
        return self._adapter.tensor_multiply(tensor, scalar)