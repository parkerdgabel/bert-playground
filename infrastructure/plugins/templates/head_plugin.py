"""Template for creating a head plugin.

This template shows how to create a custom task head plugin.
"""

from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
from loguru import logger

from infrastructure.plugins import PluginBase, PluginContext, PluginError
from ports.secondary.plugins import HeadPlugin


class CustomHeadPlugin(PluginBase, HeadPlugin):
    """Custom task head plugin.
    
    This plugin provides a custom head implementation
    for specific task types.
    """
    
    NAME = "custom_head"
    VERSION = "1.0.0"
    DESCRIPTION = "Custom task head for k-bert"
    CATEGORY = "head"
    TAGS = ["head", "custom", "task"]
    
    PROVIDES = ["custom_head_v1"]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize head plugin.
        
        Args:
            config: Head configuration
        """
        super().__init__(config)
        
        # Head configuration
        self.head_config = {
            "input_size": 768,
            "output_size": 2,
            "hidden_size": 256,
            "dropout": 0.1,
            "activation": "gelu",
            **self.config,
        }
        
        # Head layers
        self.layers = None
        self.loss_fn = None
        self.metrics = ["accuracy", "f1"]
    
    def _initialize(self, context: PluginContext) -> None:
        """Initialize the head plugin.
        
        Args:
            context: Plugin context
        """
        # Create head layers
        self.layers = self._create_head_layers()
        
        # Create loss function
        self.loss_fn = self._create_loss_function()
        
        logger.info(f"{self.NAME}: Head layers and loss function created")
    
    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        **kwargs
    ) -> Dict[str, mx.array]:
        """Forward pass through the head.
        
        Args:
            hidden_states: BERT output hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing logits and other outputs
        """
        if self.layers is None:
            raise PluginError(
                "Head not initialized",
                plugin_name=self.NAME
            )
        
        # Pool the hidden states (use [CLS] token)
        pooled_output = hidden_states[:, 0]  # [batch_size, hidden_size]
        
        # Apply head layers
        logits = self.layers(pooled_output)  # [batch_size, output_size]
        
        outputs = {
            "logits": logits,
            "pooled_output": pooled_output,
        }
        
        # Add attention mask if provided
        if attention_mask is not None:
            outputs["attention_mask"] = attention_mask
        
        return outputs
    
    def compute_loss(
        self,
        logits: mx.array,
        labels: mx.array,
        **kwargs
    ) -> mx.array:
        """Compute loss for the head.
        
        Args:
            logits: Model output logits [batch_size, num_classes]
            labels: Ground truth labels [batch_size] or [batch_size, num_classes]
            **kwargs: Additional arguments (e.g., class weights)
            
        Returns:
            Loss value
        """
        if self.loss_fn is None:
            raise PluginError(
                "Loss function not initialized",
                plugin_name=self.NAME
            )
        
        # Compute loss
        loss = self.loss_fn(logits, labels)
        
        # Apply class weights if provided
        if "class_weights" in kwargs:
            weights = kwargs["class_weights"]
            # Apply weights based on labels
            weighted_loss = loss * weights[labels]
            loss = mx.mean(weighted_loss)
        
        return loss
    
    def get_output_size(self) -> int:
        """Get the output size of the head.
        
        Returns:
            Output size
        """
        return self.head_config["output_size"]
    
    def get_metrics(self) -> List[str]:
        """Get list of metrics this head supports.
        
        Returns:
            List of metric names
        """
        return self.metrics.copy()
    
    def _create_head_layers(self) -> nn.Module:
        """Create the head layer architecture.
        
        Returns:
            Head layers as nn.Module
        """
        input_size = self.head_config["input_size"]
        hidden_size = self.head_config["hidden_size"]
        output_size = self.head_config["output_size"]
        dropout = self.head_config["dropout"]
        activation = self.head_config["activation"]
        
        # Get activation function
        activation_fn = self._get_activation(activation)
        
        # Create layers
        layers = [
            nn.Linear(input_size, hidden_size),
            activation_fn,
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        ]
        
        return nn.Sequential(*layers)
    
    def _create_loss_function(self) -> callable:
        """Create the loss function.
        
        Returns:
            Loss function
        """
        # For classification tasks, use cross-entropy
        # You can extend this to support other loss types
        task_type = self.config.get("task_type", "classification")
        
        if task_type == "classification":
            return nn.losses.cross_entropy
        elif task_type == "regression":
            return nn.losses.mse_loss
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name.
        
        Args:
            name: Activation function name
            
        Returns:
            Activation function
        """
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
        }
        
        if name.lower() not in activations:
            raise ValueError(f"Unsupported activation: {name}")
        
        return activations[name.lower()]
    
    # Additional methods for specific functionality
    
    def predict(self, hidden_states: mx.array, **kwargs) -> mx.array:
        """Make predictions.
        
        Args:
            hidden_states: Input hidden states
            **kwargs: Additional arguments
            
        Returns:
            Predictions
        """
        outputs = self(hidden_states, **kwargs)
        logits = outputs["logits"]
        
        # Convert logits to predictions
        task_type = self.config.get("task_type", "classification")
        
        if task_type == "classification":
            predictions = mx.argmax(logits, axis=-1)
        else:  # regression
            predictions = logits
        
        return predictions
    
    def predict_proba(self, hidden_states: mx.array, **kwargs) -> mx.array:
        """Predict class probabilities.
        
        Args:
            hidden_states: Input hidden states
            **kwargs: Additional arguments
            
        Returns:
            Class probabilities
        """
        outputs = self(hidden_states, **kwargs)
        logits = outputs["logits"]
        
        # Apply softmax to get probabilities
        probabilities = nn.softmax(logits, axis=-1)
        
        return probabilities


# Factory function for entry point
def create_plugin(config: Optional[Dict[str, Any]] = None) -> CustomHeadPlugin:
    """Create plugin instance.
    
    Args:
        config: Plugin configuration
        
    Returns:
        Plugin instance
    """
    return CustomHeadPlugin(config=config)