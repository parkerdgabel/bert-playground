"""Example custom BERT head implementation.

This example shows how to create a custom head that can be used with k-bert models.
"""

from typing import Dict, Optional

import mlx.core as mx
import mlx.nn as nn

from k_bert.plugins import HeadPlugin, PluginMetadata, register_component


@register_component
class CustomBinaryHead(HeadPlugin):
    """Custom binary classification head with additional features.
    
    This example head includes:
    - Dropout for regularization
    - Hidden layer for additional capacity
    - Custom activation function
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the custom head.
        
        Args:
            config: Configuration dictionary with keys:
                - hidden_size: BERT hidden size (default: 768)
                - dropout_prob: Dropout probability (default: 0.1)
                - hidden_dim: Hidden layer dimension (default: 256)
                - activation: Activation function (default: "gelu")
        """
        super().__init__(config)
        
        # Extract configuration
        hidden_size = self.config.get("hidden_size", 768)
        dropout_prob = self.config.get("dropout_prob", 0.1)
        hidden_dim = self.config.get("hidden_dim", 256)
        activation = self.config.get("activation", "gelu")
        
        # Build layers
        self.dropout = nn.Dropout(dropout_prob)
        self.hidden = nn.Linear(hidden_size, hidden_dim)
        self.activation = self._get_activation(activation)
        self.output = nn.Linear(hidden_dim, 1)
    
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="CustomBinaryHead",
            version="1.0.0",
            description="Custom binary classification head with hidden layer",
            author="Your Name",
            tags=["classification", "binary", "custom"],
        )
    
    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        **kwargs
    ) -> Dict[str, mx.array]:
        """Forward pass through the head.
        
        Args:
            hidden_states: BERT output [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Dictionary with logits
        """
        # Get [CLS] token representation (first token)
        cls_hidden = hidden_states[:, 0, :]
        
        # Apply layers
        x = self.dropout(cls_hidden)
        x = self.hidden(x)
        x = self.activation(x)
        x = self.dropout(x)
        logits = self.output(x)
        
        return {"logits": logits}
    
    def compute_loss(
        self,
        logits: mx.array,
        labels: mx.array,
        **kwargs
    ) -> mx.array:
        """Compute binary cross-entropy loss.
        
        Args:
            logits: Model output logits [batch_size, 1]
            labels: Binary labels [batch_size]
            
        Returns:
            Loss value
        """
        # Ensure labels are float32
        labels = labels.astype(mx.float32)
        
        # Reshape if needed
        if labels.ndim == 1:
            labels = labels.reshape(-1, 1)
        
        # Binary cross-entropy with logits
        loss = mx.mean(
            mx.maximum(logits, 0) - logits * labels + 
            mx.log(1 + mx.exp(-mx.abs(logits)))
        )
        
        return loss
    
    def get_output_size(self) -> int:
        """Get the output size of the head."""
        return 1
    
    def get_metrics(self) -> List[str]:
        """Get list of metrics this head supports."""
        return ["loss", "accuracy", "auc", "f1"]
    
    def _get_activation(self, name: str):
        """Get activation function by name."""
        activations = {
            "relu": nn.relu,
            "gelu": nn.gelu,
            "silu": nn.silu,
            "tanh": mx.tanh,
        }
        
        if name not in activations:
            raise ValueError(f"Unknown activation: {name}")
        
        return activations[name]


@register_component
class CustomMulticlassHead(HeadPlugin):
    """Custom multiclass classification head.
    
    This example shows a multiclass head with label smoothing support.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the multiclass head."""
        super().__init__(config)
        
        hidden_size = self.config.get("hidden_size", 768)
        num_classes = self.config.get("num_classes", 10)
        dropout_prob = self.config.get("dropout_prob", 0.1)
        self.label_smoothing = self.config.get("label_smoothing", 0.0)
        
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.num_classes = num_classes
    
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="CustomMulticlassHead",
            version="1.0.0",
            description="Custom multiclass classification head with label smoothing",
            tags=["classification", "multiclass", "label-smoothing"],
        )
    
    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        **kwargs
    ) -> Dict[str, mx.array]:
        """Forward pass through the head."""
        # Pool using attention mask if provided
        if attention_mask is not None:
            # Masked mean pooling
            mask = attention_mask.astype(mx.float32)
            mask = mx.expand_dims(mask, -1)
            
            masked_hidden = hidden_states * mask
            sum_hidden = mx.sum(masked_hidden, axis=1)
            sum_mask = mx.sum(mask, axis=1)
            pooled = sum_hidden / mx.maximum(sum_mask, 1e-9)
        else:
            # Simple mean pooling
            pooled = mx.mean(hidden_states, axis=1)
        
        # Apply classifier
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        
        return {"logits": logits}
    
    def compute_loss(
        self,
        logits: mx.array,
        labels: mx.array,
        **kwargs
    ) -> mx.array:
        """Compute cross-entropy loss with optional label smoothing."""
        # One-hot encode labels
        num_classes = logits.shape[-1]
        labels_one_hot = mx.one_hot(labels, num_classes)
        
        # Apply label smoothing if configured
        if self.label_smoothing > 0:
            labels_one_hot = (
                (1 - self.label_smoothing) * labels_one_hot +
                self.label_smoothing / num_classes
            )
        
        # Compute cross-entropy
        log_probs = nn.log_softmax(logits, axis=-1)
        loss = -mx.sum(labels_one_hot * log_probs, axis=-1)
        
        return mx.mean(loss)
    
    def get_output_size(self) -> int:
        """Get the output size of the head."""
        return self.num_classes