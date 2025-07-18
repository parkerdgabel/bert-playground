"""
Advanced activation functions for ModernBERT.

This module implements GeGLU and other advanced activation functions
used in modern transformer architectures.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional


class GeGLU(nn.Module):
    """
    GeGLU activation function.
    
    GeGLU (Gated Linear Unit with GELU) is an improvement over standard GELU
    that uses a gating mechanism to control information flow.
    
    GeGLU(x) = GELU(xW + b) ⊙ (xV + c)
    
    Where ⊙ is element-wise multiplication and W, V are different linear layers.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        use_bias: bool = True,
        gate_limit: Optional[float] = None,
    ):
        """
        Initialize GeGLU activation.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden features
            use_bias: Whether to use bias in linear layers
            gate_limit: Optional limit for gate values (for stability)
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gate_limit = gate_limit
        
        # Two linear layers for gating
        self.gate_proj = nn.Linear(input_size, hidden_size, bias=use_bias)
        self.up_proj = nn.Linear(input_size, hidden_size, bias=use_bias)
        
        # GELU activation
        self.gelu = nn.GELU()
    
    def forward(self, x: mx.array) -> mx.array:
        """
        Forward pass through GeGLU.
        
        Args:
            x: Input tensor [..., input_size]
            
        Returns:
            Output tensor [..., hidden_size]
        """
        # Gate path: GELU(xW + b)
        gate = self.gate_proj(x)
        gate = self.gelu(gate)
        
        # Up path: xV + c
        up = self.up_proj(x)
        
        # Apply gate limit if specified
        if self.gate_limit is not None:
            gate = mx.clip(gate, -self.gate_limit, self.gate_limit)
        
        # GeGLU: gate ⊙ up
        return gate * up
    
    def __call__(self, x: mx.array) -> mx.array:
        """Make the module callable."""
        return self.forward(x)


class GeGLUMLP(nn.Module):
    """
    Complete MLP block with GeGLU activation.
    
    This implements the full feed-forward network used in ModernBERT:
    - Input projection with GeGLU
    - Output projection back to hidden size
    - Optional dropout
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        use_bias: bool = True,
        dropout_prob: float = 0.0,
        gate_limit: Optional[float] = None,
    ):
        """
        Initialize GeGLU MLP.
        
        Args:
            hidden_size: Hidden size of the model
            intermediate_size: Intermediate size of the MLP
            use_bias: Whether to use bias in linear layers
            dropout_prob: Dropout probability
            gate_limit: Optional limit for gate values
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # GeGLU activation
        self.geglu = GeGLU(
            input_size=hidden_size,
            hidden_size=intermediate_size,
            use_bias=use_bias,
            gate_limit=gate_limit,
        )
        
        # Output projection
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=use_bias)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, x: mx.array) -> mx.array:
        """
        Forward pass through GeGLU MLP.
        
        Args:
            x: Input tensor [..., hidden_size]
            
        Returns:
            Output tensor [..., hidden_size]
        """
        # GeGLU activation
        x = self.geglu(x)
        
        # Dropout
        x = self.dropout(x)
        
        # Output projection
        x = self.down_proj(x)
        
        return x
    
    def __call__(self, x: mx.array) -> mx.array:
        """Make the module callable."""
        return self.forward(x)


class SwiGLU(nn.Module):
    """
    SwiGLU activation function.
    
    SwiGLU is another variant of gated linear units that uses Swish (SiLU)
    instead of GELU for the gating mechanism.
    
    SwiGLU(x) = Swish(xW + b) ⊙ (xV + c)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        use_bias: bool = True,
        gate_limit: Optional[float] = None,
    ):
        """
        Initialize SwiGLU activation.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden features
            use_bias: Whether to use bias in linear layers
            gate_limit: Optional limit for gate values
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gate_limit = gate_limit
        
        # Two linear layers for gating
        self.gate_proj = nn.Linear(input_size, hidden_size, bias=use_bias)
        self.up_proj = nn.Linear(input_size, hidden_size, bias=use_bias)
        
        # SiLU activation (Swish)
        self.silu = nn.SiLU()
    
    def forward(self, x: mx.array) -> mx.array:
        """
        Forward pass through SwiGLU.
        
        Args:
            x: Input tensor [..., input_size]
            
        Returns:
            Output tensor [..., hidden_size]
        """
        # Gate path: SiLU(xW + b)
        gate = self.gate_proj(x)
        gate = self.silu(gate)
        
        # Up path: xV + c
        up = self.up_proj(x)
        
        # Apply gate limit if specified
        if self.gate_limit is not None:
            gate = mx.clip(gate, -self.gate_limit, self.gate_limit)
        
        # SwiGLU: gate ⊙ up
        return gate * up
    
    def __call__(self, x: mx.array) -> mx.array:
        """Make the module callable."""
        return self.forward(x)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    RMSNorm is a simpler alternative to LayerNorm that normalizes
    using only the root mean square, without centering.
    
    RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma
    """
    
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        use_bias: bool = False,
    ):
        """
        Initialize RMSNorm.
        
        Args:
            hidden_size: Hidden size of the model
            eps: Small value to avoid division by zero
            use_bias: Whether to use bias (typically False for RMSNorm)
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.eps = eps
        
        # Learnable scale parameter
        self.weight = mx.ones((hidden_size,))
        
        # Optional bias (typically not used in RMSNorm)
        if use_bias:
            self.bias = mx.zeros((hidden_size,))
        else:
            self.bias = None
    
    def forward(self, x: mx.array) -> mx.array:
        """
        Forward pass through RMSNorm.
        
        Args:
            x: Input tensor [..., hidden_size]
            
        Returns:
            Normalized tensor [..., hidden_size]
        """
        # Compute root mean square
        variance = mx.mean(x * x, axis=-1, keepdims=True)
        rms = mx.sqrt(variance + self.eps)
        
        # Normalize
        x = x / rms
        
        # Scale
        x = x * self.weight
        
        # Add bias if present
        if self.bias is not None:
            x = x + self.bias
        
        return x
    
    def __call__(self, x: mx.array) -> mx.array:
        """Make the module callable."""
        return self.forward(x)


def get_activation_function(activation_name: str, **kwargs) -> nn.Module:
    """
    Get an activation function by name.
    
    Args:
        activation_name: Name of the activation function
        **kwargs: Additional arguments for the activation function
        
    Returns:
        Activation function module
    """
    activation_name = activation_name.lower()
    
    if activation_name == "gelu":
        return nn.GELU()
    elif activation_name == "relu":
        return nn.ReLU()
    elif activation_name == "silu" or activation_name == "swish":
        return nn.SiLU()
    elif activation_name == "tanh":
        return nn.Tanh()
    elif activation_name == "sigmoid":
        return nn.Sigmoid()
    elif activation_name == "geglu":
        # GeGLU requires specific parameters
        input_size = kwargs.get("input_size")
        hidden_size = kwargs.get("hidden_size")
        if input_size is None or hidden_size is None:
            raise ValueError("GeGLU requires input_size and hidden_size parameters")
        return GeGLU(input_size, hidden_size, **kwargs)
    elif activation_name == "swiglu":
        # SwiGLU requires specific parameters
        input_size = kwargs.get("input_size")
        hidden_size = kwargs.get("hidden_size")
        if input_size is None or hidden_size is None:
            raise ValueError("SwiGLU requires input_size and hidden_size parameters")
        return SwiGLU(input_size, hidden_size, **kwargs)
    else:
        raise ValueError(f"Unknown activation function: {activation_name}")


def get_normalization_function(
    norm_name: str, 
    hidden_size: int,
    **kwargs
) -> nn.Module:
    """
    Get a normalization function by name.
    
    Args:
        norm_name: Name of the normalization function
        hidden_size: Hidden size of the model
        **kwargs: Additional arguments for the normalization function
        
    Returns:
        Normalization function module
    """
    norm_name = norm_name.lower()
    
    if norm_name == "layer_norm" or norm_name == "layernorm":
        eps = kwargs.get("eps", 1e-12)
        return nn.LayerNorm(hidden_size, eps=eps)
    elif norm_name == "rms_norm" or norm_name == "rmsnorm":
        eps = kwargs.get("eps", 1e-6)
        use_bias = kwargs.get("use_bias", False)
        return RMSNorm(hidden_size, eps=eps, use_bias=use_bias)
    else:
        raise ValueError(f"Unknown normalization function: {norm_name}")