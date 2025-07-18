"""
Activation and normalization functions for BERT models.

This module contains specialized activation functions and normalization layers
used in both classic BERT and ModernBERT models, including:
- RMSNorm (Root Mean Square Layer Normalization)
- Factory functions for creating activation and normalization layers
- Utility functions for activation selection
"""

import mlx.core as mx
import mlx.nn as nn

# ============================================================================
# Advanced Normalization Functions
# ============================================================================


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


# ============================================================================
# Factory Functions
# ============================================================================


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
    elif activation_name == "leaky_relu":
        negative_slope = kwargs.get("negative_slope", 0.01)
        return nn.LeakyReLU(negative_slope)
    elif activation_name == "elu":
        alpha = kwargs.get("alpha", 1.0)
        return nn.ELU(alpha)
    elif activation_name == "softmax":
        return nn.Softmax()
    elif activation_name == "log_softmax":
        return nn.LogSoftmax()
    else:
        raise ValueError(f"Unknown activation function: {activation_name}")


def get_normalization_function(norm_name: str, hidden_size: int, **kwargs) -> nn.Module:
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
    elif norm_name == "batch_norm" or norm_name == "batchnorm":
        momentum = kwargs.get("momentum", 0.1)
        eps = kwargs.get("eps", 1e-5)
        return nn.BatchNorm(hidden_size, eps=eps, momentum=momentum)
    elif norm_name == "group_norm" or norm_name == "groupnorm":
        num_groups = kwargs.get("num_groups", 32)
        eps = kwargs.get("eps", 1e-5)
        return nn.GroupNorm(num_groups, hidden_size, eps=eps)
    else:
        raise ValueError(f"Unknown normalization function: {norm_name}")


# ============================================================================
# Utility Functions
# ============================================================================


def create_activation_layer(
    activation_name: str, hidden_size: int | None = None, **kwargs
) -> nn.Module:
    """
    Create an activation layer with optional parameters.

    Args:
        activation_name: Name of the activation function
        hidden_size: Hidden size (needed for some activations)
        **kwargs: Additional arguments

    Returns:
        Activation layer module
    """
    return get_activation_function(activation_name, **kwargs)


def create_normalization_layer(norm_name: str, hidden_size: int, **kwargs) -> nn.Module:
    """
    Create a normalization layer with optional parameters.

    Args:
        norm_name: Name of the normalization function
        hidden_size: Hidden size of the model
        **kwargs: Additional arguments

    Returns:
        Normalization layer module
    """
    return get_normalization_function(norm_name, hidden_size, **kwargs)


def get_available_activations() -> list:
    """
    Get a list of available activation functions.

    Returns:
        List of activation function names
    """
    return [
        "gelu",
        "relu",
        "silu",
        "swish",
        "tanh",
        "sigmoid",
        "leaky_relu",
        "elu",
        "softmax",
        "log_softmax",
    ]


def get_available_normalizations() -> list:
    """
    Get a list of available normalization functions.

    Returns:
        List of normalization function names
    """
    return [
        "layer_norm",
        "layernorm",
        "rms_norm",
        "rmsnorm",
        "batch_norm",
        "batchnorm",
        "group_norm",
        "groupnorm",
    ]


def is_gated_activation(activation_name: str) -> bool:
    """
    Check if an activation function is gated (like GeGLU, SwiGLU).

    Args:
        activation_name: Name of the activation function

    Returns:
        True if the activation is gated, False otherwise
    """
    gated_activations = ["geglu", "swiglu"]
    return activation_name.lower() in gated_activations


def get_activation_output_size(activation_name: str, input_size: int, **kwargs) -> int:
    """
    Get the output size of an activation function.

    Args:
        activation_name: Name of the activation function
        input_size: Input size
        **kwargs: Additional arguments

    Returns:
        Output size of the activation function
    """
    activation_name = activation_name.lower()

    if is_gated_activation(activation_name):
        # Gated activations like GeGLU may change the output size
        return kwargs.get("hidden_size", input_size)
    else:
        # Most activations preserve the input size
        return input_size
