"""Model wrapper to handle dictionary inputs for training."""

from typing import Dict, Any
import mlx.core as mx
import mlx.nn as nn


class ModelWrapper(nn.Module):
    """Wrapper to handle dictionary inputs from data loaders."""
    
    def __init__(self, model):
        """Initialize the wrapper.
        
        Args:
            model: The actual model to wrap
        """
        super().__init__()
        self.model = model
    
    def __call__(self, batch: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Forward pass handling dictionary input.
        
        Args:
            batch: Dictionary with keys like 'input_ids', 'attention_mask', etc.
            
        Returns:
            Dictionary with model outputs
        """
        # Extract inputs from batch
        input_ids = batch.get('input_ids')
        attention_mask = batch.get('attention_mask')
        token_type_ids = batch.get('token_type_ids')
        labels = batch.get('labels')
        
        # Call the model with proper arguments
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )
    
    def save_pretrained(self, path):
        """Save the wrapped model."""
        return self.model.save_pretrained(path)
    
    def load_pretrained(self, path):
        """Load the wrapped model."""
        return self.model.load_pretrained(path)
    
    @property
    def config(self):
        """Get model config."""
        return self.model.config if hasattr(self.model, 'config') else None
    
    def parameters(self):
        """Get model parameters."""
        return self.model.parameters()