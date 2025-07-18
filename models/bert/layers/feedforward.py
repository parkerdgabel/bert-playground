"""
Unified feedforward network components for BERT models.

This module contains all feedforward network layers for both classic BERT
and ModernBERT models, including:
- Standard BERT FFN with GELU activation
- ModernBERT FFN with GeGLU activation
- Advanced activation functions (GeGLU, SwiGLU)
- Factory functions for creating appropriate FFN layers
"""

import mlx.core as mx
import mlx.nn as nn

from ..config import BertConfig

# ============================================================================
# Standard BERT Feedforward Components
# ============================================================================


class BertIntermediate(nn.Module):
    """BERT intermediate layer (first part of FFN).

    Applies linear transformation and GELU activation.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

        # Use GELU activation as per BERT paper
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states: mx.array) -> mx.array:
        """Forward pass through intermediate layer.

        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]

        Returns:
            Output tensor [batch_size, seq_len, intermediate_size]
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Make the intermediate layer callable."""
        return self.forward(hidden_states)


class BertOutput(nn.Module):
    """BERT FFN output layer (second part of FFN).

    Applies linear transformation, dropout, and layer normalization.
    Note: This class is named BertOutput to match the original BERT naming,
    but it's specifically for the FFN output (distinct from model output).
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: mx.array, input_tensor: mx.array) -> mx.array:
        """Forward pass through FFN output layer.

        Args:
            hidden_states: Output from intermediate layer [batch_size, seq_len, intermediate_size]
            input_tensor: Input tensor for residual connection [batch_size, seq_len, hidden_size]

        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

    def __call__(self, hidden_states: mx.array, input_tensor: mx.array) -> mx.array:
        """Make the FFN output layer callable."""
        return self.forward(hidden_states, input_tensor)


class BertFeedForward(nn.Module):
    """Complete BERT feedforward network.

    Combines the intermediate layer and output layer into a single module.
    This is a convenience wrapper that encapsulates the full FFN.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states: mx.array) -> mx.array:
        """Forward pass through complete FFN.

        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]

        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        input_tensor = hidden_states

        # First part: hidden_size -> intermediate_size with GELU
        hidden_states = self.intermediate(hidden_states)

        # Second part: intermediate_size -> hidden_size with residual connection
        hidden_states = self.output(hidden_states, input_tensor)

        return hidden_states

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Make the feedforward network callable."""
        return self.forward(hidden_states)


# ============================================================================
# Advanced Activation Functions
# ============================================================================


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
        gate_limit: float | None = None,
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
        gate_limit: float | None = None,
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


# ============================================================================
# ModernBERT Feedforward Components
# ============================================================================


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
        gate_limit: float | None = None,
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


class SwiGLUMLP(nn.Module):
    """
    Complete MLP block with SwiGLU activation.

    This implements a feed-forward network with SwiGLU activation,
    used in neoBERT and other modern architectures:
    - Input projection with SwiGLU
    - Output projection back to hidden size
    - Optional dropout
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        use_bias: bool = True,
        dropout_prob: float = 0.0,
        gate_limit: float | None = None,
    ):
        """
        Initialize SwiGLU MLP.

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

        # SwiGLU activation
        self.swiglu = SwiGLU(
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
        Forward pass through SwiGLU MLP.

        Args:
            x: Input tensor [..., hidden_size]

        Returns:
            Output tensor [..., hidden_size]
        """
        # SwiGLU activation
        x = self.swiglu(x)

        # Dropout
        x = self.dropout(x)

        # Output projection
        x = self.down_proj(x)

        return x

    def __call__(self, x: mx.array) -> mx.array:
        """Make the module callable."""
        return self.forward(x)


class ModernBertFeedForward(nn.Module):
    """
    ModernBERT feedforward network.

    Uses GeGLU activation instead of standard GELU and supports
    streamlined architecture without bias terms.
    """

    def __init__(self, config):
        """
        Initialize ModernBERT feedforward network.

        Args:
            config: ModernBERT configuration
        """
        super().__init__()
        self.config = config

        # Use GeGLU MLP
        self.mlp = GeGLUMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            use_bias=getattr(config, "use_bias", False),
            dropout_prob=getattr(config, "hidden_dropout_prob", 0.0),
            gate_limit=getattr(config, "gate_limit", None),
        )

        # Layer normalization (pre-norm in ModernBERT)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Dropout for residual connection
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: mx.array) -> mx.array:
        """
        Forward pass through ModernBERT feedforward network.

        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]

        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        # Store input for residual connection
        residual = hidden_states

        # Pre-normalization
        hidden_states = self.layer_norm(hidden_states)

        # GeGLU MLP
        hidden_states = self.mlp(hidden_states)

        # Dropout
        hidden_states = self.dropout(hidden_states)

        # Residual connection
        hidden_states = residual + hidden_states

        return hidden_states

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Make the feedforward network callable."""
        return self.forward(hidden_states)


class NeoBertFeedForward(nn.Module):
    """
    neoBERT feedforward network.

    Uses SwiGLU activation and RMSNorm with pre-normalization pattern.
    Designed for efficient computation without bias terms.
    """

    def __init__(self, config):
        """
        Initialize neoBERT feedforward network.

        Args:
            config: neoBERT configuration
        """
        super().__init__()
        self.config = config

        # Use SwiGLU MLP
        self.mlp = SwiGLUMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            use_bias=getattr(config, "use_bias", False),
            dropout_prob=getattr(config, "hidden_dropout_prob", 0.0),
            gate_limit=getattr(config, "gate_limit", None),
        )

        # Import RMSNorm from activations module
        from .activations import RMSNorm

        # RMSNorm for pre-normalization in neoBERT
        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            use_bias=False,
        )

        # Dropout for residual connection
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: mx.array) -> mx.array:
        """
        Forward pass through neoBERT feedforward network.

        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]

        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        # Store input for residual connection
        residual = hidden_states

        # Pre-normalization with RMSNorm
        hidden_states = self.norm(hidden_states)

        # SwiGLU MLP
        hidden_states = self.mlp(hidden_states)

        # Dropout
        hidden_states = self.dropout(hidden_states)

        # Residual connection
        hidden_states = residual + hidden_states

        return hidden_states

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Make the feedforward network callable."""
        return self.forward(hidden_states)


# ============================================================================
# Factory Functions
# ============================================================================


def create_feedforward_layer(config) -> nn.Module:
    """
    Create appropriate feedforward layer based on configuration.

    Args:
        config: Model configuration

    Returns:
        Feedforward layer module
    """
    # Check if using neoBERT architecture with SwiGLU
    if hasattr(config, "use_swiglu") and config.use_swiglu:
        return NeoBertFeedForward(config)
    # Check if using ModernBERT architecture with GeGLU
    elif hasattr(config, "use_geglu") and config.use_geglu:
        return ModernBertFeedForward(config)
    else:
        return BertFeedForward(config)


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


# ============================================================================
# Utility Functions
# ============================================================================


def get_intermediate_size(config) -> int:
    """Get the intermediate size for feedforward layers."""
    return config.intermediate_size


def get_hidden_size(config) -> int:
    """Get the hidden size for feedforward layers."""
    return config.hidden_size
