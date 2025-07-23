"""Compute port for neural network operations."""

from typing import Protocol, Any, Dict, Tuple, Optional, List
from domain.entities.model import BertModel
from domain.entities.dataset import DataBatch


class ComputePort(Protocol):
    """Port for neural network computations."""
    
    def forward(
        self,
        model: BertModel,
        batch: DataBatch,
        training: bool = False,
    ) -> Dict[str, Any]:
        """Perform forward pass through model.
        
        Args:
            model: The BERT model
            batch: Input data batch
            training: Whether in training mode
            
        Returns:
            Dictionary containing:
            - 'logits': Model predictions
            - 'loss': Computed loss (if labels provided)
            - 'hidden_states': Optional hidden states
            - 'attentions': Optional attention weights
        """
        ...
    
    def backward(
        self,
        loss: Any,
        retain_graph: bool = False,
    ) -> None:
        """Perform backward pass to compute gradients.
        
        Args:
            loss: Loss value to backpropagate
            retain_graph: Whether to retain computation graph
        """
        ...
    
    def optimize_step(
        self,
        model: BertModel,
        optimizer_state: Dict[str, Any],
        learning_rate: float,
        max_grad_norm: Optional[float] = None,
    ) -> Tuple[Dict[str, Any], float]:
        """Perform optimization step.
        
        Args:
            model: The model to optimize
            optimizer_state: Current optimizer state
            learning_rate: Current learning rate
            max_grad_norm: Maximum gradient norm for clipping
            
        Returns:
            Tuple of (updated_optimizer_state, gradient_norm)
        """
        ...
    
    def compile_model(
        self,
        model: BertModel,
        backend: Optional[str] = None,
    ) -> BertModel:
        """Compile model for optimized execution.
        
        Args:
            model: Model to compile
            backend: Optional backend specification
            
        Returns:
            Compiled model
        """
        ...
    
    def mixed_precision_context(
        self,
        enabled: bool = True,
    ) -> Any:
        """Context manager for mixed precision training.
        
        Args:
            enabled: Whether to enable mixed precision
            
        Returns:
            Context manager
        """
        ...
    
    def get_device_info(
        self,
    ) -> Dict[str, Any]:
        """Get information about compute device.
        
        Returns:
            Dictionary with device information:
            - 'device_type': Type of device (cpu, gpu, etc.)
            - 'device_name': Name of the device
            - 'memory_total': Total memory in bytes
            - 'memory_available': Available memory in bytes
        """
        ...
    
    def synchronize(
        self,
    ) -> None:
        """Synchronize compute operations."""
        ...
    
    def create_optimizer(
        self,
        model: BertModel,
        optimizer_type: str,
        learning_rate: float,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Create optimizer state.
        
        Args:
            model: Model to optimize
            optimizer_type: Type of optimizer
            learning_rate: Learning rate
            **kwargs: Additional optimizer parameters
            
        Returns:
            Optimizer state dictionary
        """
        ...
    
    def get_model_parameters(
        self,
        model: BertModel,
    ) -> List[Tuple[str, Any]]:
        """Get model parameters.
        
        Args:
            model: The model
            
        Returns:
            List of (name, parameter) tuples
        """
        ...
    
    def count_parameters(
        self,
        model: BertModel,
        trainable_only: bool = False,
    ) -> int:
        """Count model parameters.
        
        Args:
            model: The model
            trainable_only: Whether to count only trainable parameters
            
        Returns:
            Number of parameters
        """
        ...