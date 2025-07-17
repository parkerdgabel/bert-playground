"""
Classification Heads

Pure classification head implementations that operate on embeddings.
These heads are independent of the embedding models and can be used with any embedding source.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional


class BinaryClassificationHead(nn.Module):
    """
    Binary classification head for two-class problems.
    
    This head takes embeddings as input and outputs logits for binary classification.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        dropout_prob: float = 0.1,
        use_layer_norm: bool = False,
        activation: str = "relu",
    ):
        """
        Initialize binary classification head.
        
        Args:
            input_dim: Input embedding dimension
            hidden_dim: Hidden dimension (None for single layer)
            dropout_prob: Dropout probability
            use_layer_norm: Whether to use layer normalization
            activation: Activation function name
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob
        self.use_layer_norm = use_layer_norm
        
        # Choose activation function
        activations = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "mish": nn.Mish,
        }
        activation_fn = activations.get(activation, nn.ReLU)
        
        if hidden_dim is None:
            # Simple linear classifier
            layers = []
            if use_layer_norm:
                layers.append(nn.LayerNorm(input_dim))
            layers.extend([
                nn.Dropout(dropout_prob),
                nn.Linear(input_dim, 2)
            ])
            self.classifier = nn.Sequential(*layers)
        else:
            # Two-layer classifier with hidden dimension
            layers = []
            if use_layer_norm:
                layers.append(nn.LayerNorm(input_dim))
            layers.extend([
                nn.Dropout(dropout_prob),
                nn.Linear(input_dim, hidden_dim),
                activation_fn(),
                nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_dim, 2),
            ])
            self.classifier = nn.Sequential(*layers)
    
    def __call__(self, embeddings: mx.array) -> mx.array:
        """
        Forward pass through classification head.
        
        Args:
            embeddings: Input embeddings [batch, embedding_dim]
            
        Returns:
            Logits [batch, 2]
        """
        return self.classifier(embeddings)


class MultiClassificationHead(nn.Module):
    """
    Multi-class classification head for problems with more than 2 classes.
    
    This head takes embeddings as input and outputs logits for multi-class classification.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: Optional[int] = None,
        dropout_prob: float = 0.1,
        use_layer_norm: bool = False,
        activation: str = "relu",
    ):
        """
        Initialize multi-class classification head.
        
        Args:
            input_dim: Input embedding dimension
            num_classes: Number of output classes
            hidden_dim: Hidden dimension (None for single layer)
            dropout_prob: Dropout probability
            use_layer_norm: Whether to use layer normalization
            activation: Activation function name
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob
        self.use_layer_norm = use_layer_norm
        
        # Choose activation function
        activations = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "mish": nn.Mish,
        }
        activation_fn = activations.get(activation, nn.ReLU)
        
        if hidden_dim is None:
            # Simple linear classifier
            layers = []
            if use_layer_norm:
                layers.append(nn.LayerNorm(input_dim))
            layers.extend([
                nn.Dropout(dropout_prob),
                nn.Linear(input_dim, num_classes)
            ])
            self.classifier = nn.Sequential(*layers)
        else:
            # Two-layer classifier with hidden dimension
            layers = []
            if use_layer_norm:
                layers.append(nn.LayerNorm(input_dim))
            layers.extend([
                nn.Dropout(dropout_prob),
                nn.Linear(input_dim, hidden_dim),
                activation_fn(),
                nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_dim, num_classes),
            ])
            self.classifier = nn.Sequential(*layers)
    
    def __call__(self, embeddings: mx.array) -> mx.array:
        """
        Forward pass through classification head.
        
        Args:
            embeddings: Input embeddings [batch, embedding_dim]
            
        Returns:
            Logits [batch, num_classes]
        """
        return self.classifier(embeddings)


class RegressionHead(nn.Module):
    """
    Regression head for continuous value prediction.
    
    This head takes embeddings as input and outputs continuous values.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dim: Optional[int] = None,
        dropout_prob: float = 0.1,
        use_layer_norm: bool = False,
        activation: str = "relu",
    ):
        """
        Initialize regression head.
        
        Args:
            input_dim: Input embedding dimension
            output_dim: Output dimension (1 for single value regression)
            hidden_dim: Hidden dimension (None for single layer)
            dropout_prob: Dropout probability
            use_layer_norm: Whether to use layer normalization
            activation: Activation function name
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob
        self.use_layer_norm = use_layer_norm
        
        # Choose activation function
        activations = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "mish": nn.Mish,
        }
        activation_fn = activations.get(activation, nn.ReLU)
        
        if hidden_dim is None:
            # Simple linear regressor
            layers = []
            if use_layer_norm:
                layers.append(nn.LayerNorm(input_dim))
            layers.extend([
                nn.Dropout(dropout_prob),
                nn.Linear(input_dim, output_dim)
            ])
            self.regressor = nn.Sequential(*layers)
        else:
            # Two-layer regressor with hidden dimension
            layers = []
            if use_layer_norm:
                layers.append(nn.LayerNorm(input_dim))
            layers.extend([
                nn.Dropout(dropout_prob),
                nn.Linear(input_dim, hidden_dim),
                activation_fn(),
                nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_dim, output_dim),
            ])
            self.regressor = nn.Sequential(*layers)
    
    def __call__(self, embeddings: mx.array) -> mx.array:
        """
        Forward pass through regression head.
        
        Args:
            embeddings: Input embeddings [batch, embedding_dim]
            
        Returns:
            Regression outputs [batch, output_dim]
        """
        return self.regressor(embeddings)