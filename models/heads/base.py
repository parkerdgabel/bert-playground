"""Base class for BERT heads.

This module provides the minimal base class for all head implementations,
following the clean design patterns from the BERT module.
"""

from abc import ABC, abstractmethod

import mlx.core as mx
import mlx.nn as nn

from .config import HeadConfig
from .layers.pooling import create_pooling_layer


class BaseHead(nn.Module, ABC):
    """Abstract base class for all BERT heads.

    This class provides the minimal interface and common functionality
    for all head types, following the BERT module's design principles.
    """

    def __init__(self, config: HeadConfig):
        """Initialize the base head.

        Args:
            config: Head configuration
        """
        super().__init__()
        self.config = config

        # Initialize pooling layer
        self._build_pooling_layer()

        # Initialize projection layers
        self._build_projection_layers()

        # Build output layer (implemented by subclasses)
        self._build_output_layer()

    def _build_pooling_layer(self):
        """Build the pooling layer based on configuration."""
        if (
            self.config.pooling_type == "attention"
            or self.config.pooling_type == "weighted_mean"
        ):
            # These pooling types need the hidden size
            self.pooling = create_pooling_layer(
                self.config.pooling_type, hidden_size=self.config.input_size
            )
        else:
            self.pooling = create_pooling_layer(self.config.pooling_type)

    def _build_projection_layers(self):
        """Build intermediate projection layers."""
        layers = []
        input_size = self.config.input_size

        # Build hidden layers
        for hidden_size in self.config.hidden_sizes:
            # Linear layer
            layers.append(nn.Linear(input_size, hidden_size, bias=self.config.use_bias))

            # Layer normalization
            if self.config.use_layer_norm:
                layers.append(nn.LayerNorm(hidden_size, eps=self.config.layer_norm_eps))

            # Activation
            if self.config.activation != "none":
                layers.append(self._get_activation())

            # Dropout
            if self.config.dropout_prob > 0:
                layers.append(nn.Dropout(self.config.dropout_prob))

            input_size = hidden_size

        # Create sequential module or identity if no layers
        if layers:
            self.projection = nn.Sequential(*layers)
            self.projection_output_size = input_size
        else:
            self.projection = nn.Identity()
            self.projection_output_size = self.config.input_size

    def _get_activation(self) -> nn.Module:
        """Get activation module based on configuration."""
        activation_map = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "silu": nn.SiLU(),
            "leaky_relu": nn.LeakyReLU(0.1),
        }

        activation = activation_map.get(self.config.activation)
        if activation is None:
            raise ValueError(f"Unknown activation: {self.config.activation}")

        return activation

    @abstractmethod
    def _build_output_layer(self):
        """Build the output layer. Must be implemented by subclasses."""
        pass

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        **kwargs,
    ) -> dict[str, mx.array]:
        """Make the head callable.

        Args:
            hidden_states: Input hidden states from BERT [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask [batch_size, seq_len]
            **kwargs: Additional arguments

        Returns:
            Dictionary containing output tensors
        """
        return self.forward(hidden_states, attention_mask, **kwargs)

    def forward(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        **kwargs,
    ) -> dict[str, mx.array]:
        """Forward pass through the head.

        Args:
            hidden_states: Input hidden states from BERT [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask [batch_size, seq_len]
            **kwargs: Additional arguments

        Returns:
            Dictionary containing output tensors
        """
        # Apply pooling
        pooled_output = self.pooling(hidden_states, attention_mask)

        # Apply projection layers
        projected = self.projection(pooled_output)

        # Apply output layer (implemented by subclasses)
        output = self._forward_output(projected)

        return output

    @abstractmethod
    def _forward_output(self, features: mx.array) -> dict[str, mx.array]:
        """Forward pass through the output layer.

        Args:
            features: Features after projection [batch_size, projection_output_size]

        Returns:
            Dictionary containing output tensors
        """
        pass


__all__ = ["BaseHead"]
