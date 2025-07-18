"""Layers module for BERT heads.

This module provides reusable layer components for building heads.
"""

from .pooling import (
    AttentionPooling,
    CLSTokenPooling,
    LastTokenPooling,
    MaxPooling,
    # Pooling layers
    MeanPooling,
    WeightedMeanPooling,
    # Factory function
    create_pooling_layer,
)

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
