"""Pooling layers for BERT heads.

This module provides various pooling strategies for converting sequence
representations into fixed-size vectors suitable for classification and
regression tasks.
"""

import mlx.core as mx
import mlx.nn as nn


class MeanPooling(nn.Module):
    """Mean pooling over sequence dimension."""

    def __call__(
        self, hidden_states: mx.array, attention_mask: mx.array | None = None
    ) -> mx.array:
        """Apply mean pooling.

        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional mask of shape [batch_size, seq_len]

        Returns:
            Pooled tensor of shape [batch_size, hidden_size]
        """
        return self.forward(hidden_states, attention_mask)

    def forward(
        self, hidden_states: mx.array, attention_mask: mx.array | None = None
    ) -> mx.array:
        """Forward pass for mean pooling."""
        if attention_mask is not None:
            # Mask out padding tokens
            attention_mask = attention_mask.astype(mx.float32)
            attention_mask = attention_mask[..., None]  # Add feature dimension
            hidden_states = hidden_states * attention_mask

            # Compute mean over non-padded tokens
            seq_lengths = attention_mask.sum(axis=1)
            return hidden_states.sum(axis=1) / seq_lengths
        else:
            return hidden_states.mean(axis=1)


class MaxPooling(nn.Module):
    """Max pooling over sequence dimension."""

    def __call__(
        self, hidden_states: mx.array, attention_mask: mx.array | None = None
    ) -> mx.array:
        """Apply max pooling.

        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional mask of shape [batch_size, seq_len]

        Returns:
            Pooled tensor of shape [batch_size, hidden_size]
        """
        return self.forward(hidden_states, attention_mask)

    def forward(
        self, hidden_states: mx.array, attention_mask: mx.array | None = None
    ) -> mx.array:
        """Forward pass for max pooling."""
        if attention_mask is not None:
            # Mask out padding tokens with large negative values
            attention_mask = attention_mask.astype(mx.float32)
            attention_mask = attention_mask[..., None]  # Add feature dimension
            hidden_states = hidden_states + (1.0 - attention_mask) * -1e9

        return hidden_states.max(axis=1)


class AttentionPooling(nn.Module):
    """Attention-based pooling over sequence dimension."""

    def __init__(self, hidden_size: int):
        """Initialize attention pooling.

        Args:
            hidden_size: Size of hidden dimension
        """
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def __call__(
        self, hidden_states: mx.array, attention_mask: mx.array | None = None
    ) -> mx.array:
        """Apply attention pooling.

        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional mask of shape [batch_size, seq_len]

        Returns:
            Pooled tensor of shape [batch_size, hidden_size]
        """
        return self.forward(hidden_states, attention_mask)

    def forward(
        self, hidden_states: mx.array, attention_mask: mx.array | None = None
    ) -> mx.array:
        """Forward pass for attention pooling."""
        # Compute attention scores
        attention_scores = self.attention(hidden_states).squeeze(
            -1
        )  # [batch_size, seq_len]

        if attention_mask is not None:
            # Mask out padding tokens
            attention_mask = attention_mask.astype(mx.float32)
            attention_scores = attention_scores + (1.0 - attention_mask) * -1e9

        # Apply softmax
        attention_weights = mx.softmax(
            attention_scores, axis=-1
        )  # [batch_size, seq_len]

        # Apply weighted sum
        attention_weights = attention_weights[..., None]  # Add feature dimension
        return (hidden_states * attention_weights).sum(axis=1)


class WeightedMeanPooling(nn.Module):
    """Weighted mean pooling with learned weights."""

    def __init__(self, hidden_size: int):
        """Initialize weighted mean pooling.

        Args:
            hidden_size: Size of hidden dimension
        """
        super().__init__()
        self.weights = nn.Linear(hidden_size, 1)

    def __call__(
        self, hidden_states: mx.array, attention_mask: mx.array | None = None
    ) -> mx.array:
        """Apply weighted mean pooling.

        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional mask of shape [batch_size, seq_len]

        Returns:
            Pooled tensor of shape [batch_size, hidden_size]
        """
        return self.forward(hidden_states, attention_mask)

    def forward(
        self, hidden_states: mx.array, attention_mask: mx.array | None = None
    ) -> mx.array:
        """Forward pass for weighted mean pooling."""
        # Compute position weights
        position_weights = mx.sigmoid(self.weights(hidden_states)).squeeze(
            -1
        )  # [batch_size, seq_len]

        if attention_mask is not None:
            # Mask out padding tokens
            attention_mask = attention_mask.astype(mx.float32)
            position_weights = position_weights * attention_mask

        # Normalize weights
        weight_sum = position_weights.sum(axis=1, keepdims=True)
        position_weights = position_weights / (weight_sum + 1e-9)

        # Apply weighted sum
        position_weights = position_weights[..., None]  # Add feature dimension
        return (hidden_states * position_weights).sum(axis=1)


class LastTokenPooling(nn.Module):
    """Use the last non-padding token."""

    def __call__(
        self, hidden_states: mx.array, attention_mask: mx.array | None = None
    ) -> mx.array:
        """Apply last token pooling.

        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional mask of shape [batch_size, seq_len]

        Returns:
            Pooled tensor of shape [batch_size, hidden_size]
        """
        return self.forward(hidden_states, attention_mask)

    def forward(
        self, hidden_states: mx.array, attention_mask: mx.array | None = None
    ) -> mx.array:
        """Forward pass for last token pooling."""
        if attention_mask is not None:
            # Find the last non-padding token for each sequence
            seq_lengths = attention_mask.sum(axis=1) - 1  # 0-indexed
            batch_size = hidden_states.shape[0]
            batch_indices = mx.arange(batch_size)
            return hidden_states[batch_indices, seq_lengths]
        else:
            # Use the last token
            return hidden_states[:, -1, :]


class CLSTokenPooling(nn.Module):
    """Use the [CLS] token (first token) for pooling."""

    def __call__(
        self, hidden_states: mx.array, attention_mask: mx.array | None = None
    ) -> mx.array:
        """Apply CLS token pooling.

        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional mask (ignored for CLS pooling)

        Returns:
            Pooled tensor of shape [batch_size, hidden_size]
        """
        return hidden_states[:, 0, :]

    def forward(
        self, hidden_states: mx.array, attention_mask: mx.array | None = None
    ) -> mx.array:
        """Forward pass for CLS token pooling."""
        return hidden_states[:, 0, :]


# Factory function for creating pooling layers
def create_pooling_layer(pooling_type: str, hidden_size: int = None) -> nn.Module:
    """Create a pooling layer based on the specified type.

    Args:
        pooling_type: Type of pooling ('cls', 'mean', 'max', 'attention', 'weighted_mean', 'last')
        hidden_size: Hidden size (required for attention and weighted_mean pooling)

    Returns:
        Pooling layer module

    Raises:
        ValueError: If pooling type is unknown or hidden_size not provided when required
    """
    pooling_type = pooling_type.lower()

    if pooling_type == "cls":
        return CLSTokenPooling()
    elif pooling_type == "mean":
        return MeanPooling()
    elif pooling_type == "max":
        return MaxPooling()
    elif pooling_type == "attention":
        if hidden_size is None:
            raise ValueError("hidden_size must be provided for attention pooling")
        return AttentionPooling(hidden_size)
    elif pooling_type == "weighted_mean":
        if hidden_size is None:
            raise ValueError("hidden_size must be provided for weighted mean pooling")
        return WeightedMeanPooling(hidden_size)
    elif pooling_type == "last":
        return LastTokenPooling()
    else:
        raise ValueError(f"Unknown pooling type: {pooling_type}")


__all__ = [
    # Pooling layers
    "MeanPooling",
    "MaxPooling",
    "AttentionPooling",
    "WeightedMeanPooling",
    "LastTokenPooling",
    "CLSTokenPooling",
    # Factory function
    "create_pooling_layer",
]
