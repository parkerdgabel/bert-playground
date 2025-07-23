"""MLX implementation of the ComputePort."""

from typing import Any, Dict, Tuple, Optional, List
from contextlib import contextmanager

import mlx.core as mx
import mlx.nn as nn

from domain.entities.model import BertModel
from domain.entities.dataset import DataBatch
from domain.ports.compute import ComputePort
from adapters.secondary.compute.base import BaseComputeAdapter
from .model_adapter import MLXModelAdapter
from .optimization import MLXOptimizer, MLXOptimizerState
from .utils import convert_to_mlx_array, get_mlx_device_info


class MLXComputeAdapter(BaseComputeAdapter):
    """MLX implementation of the ComputePort for neural network operations."""
    
    def __init__(self):
        """Initialize MLX compute adapter."""
        super().__init__()
        self._model_adapters: Dict[int, MLXModelAdapter] = {}
        self._optimizers: Dict[int, MLXOptimizer] = {}
        
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
        # Get or create MLX model adapter
        model_adapter = self._get_or_create_model_adapter(model)
        
        # Convert batch to MLX arrays
        mlx_batch = self._convert_batch_to_mlx(batch)
        
        # Set training mode
        model_adapter.train(training)
        
        # Forward pass
        outputs = model_adapter.forward(
            input_ids=mlx_batch["input_ids"],
            attention_mask=mlx_batch.get("attention_mask"),
            token_type_ids=mlx_batch.get("token_type_ids"),
            position_ids=mlx_batch.get("position_ids"),
            labels=mlx_batch.get("labels"),
        )
        
        return outputs
    
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
        # MLX computes gradients through mx.grad or mx.value_and_grad
        # The actual backward pass is handled by the optimizer
        # This is a no-op for MLX as gradients are computed in optimize_step
        pass
    
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
        model_adapter = self._get_or_create_model_adapter(model)
        
        # Get or create optimizer
        optimizer_id = id(model)
        if optimizer_id not in self._optimizers:
            optimizer_type = optimizer_state.get("type", "adamw")
            self._optimizers[optimizer_id] = self._create_optimizer(
                model_adapter, optimizer_type, learning_rate, **optimizer_state
            )
        
        optimizer = self._optimizers[optimizer_id]
        
        # Update learning rate
        optimizer.learning_rate = learning_rate
        
        # Get gradients and update weights
        grad_norm = optimizer.update(model_adapter, max_grad_norm)
        
        # Update optimizer state
        updated_state = optimizer.get_state()
        updated_state["learning_rate"] = learning_rate
        
        return updated_state, grad_norm
    
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
        # Check if already compiled
        cached = self._get_cached_compiled_model(model)
        if cached is not None:
            return model
            
        # Get MLX model adapter
        model_adapter = self._get_or_create_model_adapter(model)
        
        # Compile the forward function
        model_adapter.compile()
        
        # Cache the compiled model
        self._cache_compiled_model(model, model_adapter)
        
        return model
    
    @contextmanager
    def mixed_precision_context(
        self,
        enabled: bool = True,
    ):
        """Context manager for mixed precision training.
        
        Args:
            enabled: Whether to enable mixed precision
            
        Yields:
            Context for mixed precision operations
        """
        # MLX handles mixed precision through dtype specifications
        # This is primarily managed at model creation time
        yield
    
    def get_device_info(
        self,
    ) -> Dict[str, Any]:
        """Get information about compute device.
        
        Returns:
            Dictionary with device information
        """
        if self._device_cache is None:
            self._device_cache = get_mlx_device_info()
        return self._device_cache
    
    def synchronize(
        self,
    ) -> None:
        """Synchronize compute operations."""
        # Force evaluation of any pending operations
        mx.eval()
    
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
        state = {
            "type": optimizer_type,
            "learning_rate": learning_rate,
        }
        state.update(kwargs)
        return state
    
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
        model_adapter = self._get_or_create_model_adapter(model)
        return list(model_adapter.named_parameters())
    
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
        model_adapter = self._get_or_create_model_adapter(model)
        return model_adapter.count_parameters(trainable_only)
    
    # Private helper methods
    
    def _get_or_create_model_adapter(self, model: BertModel) -> MLXModelAdapter:
        """Get or create MLX model adapter for a BertModel."""
        model_id = id(model)
        if model_id not in self._model_adapters:
            self._model_adapters[model_id] = MLXModelAdapter(model)
        return self._model_adapters[model_id]
    
    def _convert_batch_to_mlx(self, batch: DataBatch) -> Dict[str, mx.array]:
        """Convert DataBatch to MLX arrays."""
        mlx_batch = {}
        
        # Convert sequences to arrays
        if batch.sequences:
            # Stack sequences into batch tensors
            input_ids = []
            attention_mask = []
            token_type_ids = []
            position_ids = []
            
            for seq in batch.sequences:
                input_ids.append(seq.input_ids)
                attention_mask.append(seq.attention_mask)
                if seq.token_type_ids is not None:
                    token_type_ids.append(seq.token_type_ids)
                if seq.position_ids is not None:
                    position_ids.append(seq.position_ids)
            
            mlx_batch["input_ids"] = convert_to_mlx_array(input_ids, dtype=mx.int32)
            mlx_batch["attention_mask"] = convert_to_mlx_array(attention_mask, dtype=mx.int32)
            
            if token_type_ids:
                mlx_batch["token_type_ids"] = convert_to_mlx_array(token_type_ids, dtype=mx.int32)
            
            if position_ids:
                mlx_batch["position_ids"] = convert_to_mlx_array(position_ids, dtype=mx.int32)
        
        # Convert labels if present
        if batch.labels is not None:
            mlx_batch["labels"] = convert_to_mlx_array(batch.labels)
        
        return mlx_batch
    
    def _create_optimizer(
        self,
        model_adapter: MLXModelAdapter,
        optimizer_type: str,
        learning_rate: float,
        **kwargs: Any,
    ) -> MLXOptimizer:
        """Create MLX optimizer."""
        return MLXOptimizer(
            model_adapter,
            optimizer_type=optimizer_type,
            learning_rate=learning_rate,
            **kwargs
        )