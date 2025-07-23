"""Base compute adapter with common functionality."""

from abc import ABC
from typing import Any, Dict, Tuple, Optional
from domain.entities.model import BertModel
from domain.entities.dataset import DataBatch
from ports.secondary.compute import ComputeBackend


class BaseComputeAdapter(ABC, ComputeBackend):
    """Base implementation of ComputePort with common functionality."""
    
    def __init__(self):
        """Initialize base compute adapter."""
        self._device_cache: Optional[Dict[str, Any]] = None
        self._compiled_models: Dict[int, Any] = {}
        
    def forward_backward(
        self,
        model: BertModel,
        batch: DataBatch,
        training: bool = False,
    ) -> Tuple[Dict[str, Any], Any]:
        """Perform forward and backward pass in one operation.
        
        This is a convenience method that combines forward and backward passes.
        
        Args:
            model: The BERT model
            batch: Input data batch  
            training: Whether in training mode
            
        Returns:
            Tuple of (forward_output, loss_value)
        """
        # First perform forward pass
        output = self.forward(model, batch, training)
        
        # If we have a loss, perform backward pass
        if "loss" in output and output["loss"] is not None and training:
            self.backward(output["loss"])
            
        return output, output.get("loss")
    
    def get_device_memory(self) -> Dict[str, int]:
        """Get device memory information.
        
        Returns:
            Dictionary with 'total' and 'available' memory in bytes
        """
        device_info = self.get_device_info()
        return {
            "total": device_info.get("memory_total", 0),
            "available": device_info.get("memory_available", 0),
        }
    
    def clear_cache(self) -> None:
        """Clear any cached data."""
        self._device_cache = None
        self._compiled_models.clear()
        
    def _cache_compiled_model(self, model: BertModel, compiled: Any) -> None:
        """Cache a compiled model."""
        model_id = id(model)
        self._compiled_models[model_id] = compiled
        
    def _get_cached_compiled_model(self, model: BertModel) -> Optional[Any]:
        """Get cached compiled model if available."""
        model_id = id(model)
        return self._compiled_models.get(model_id)