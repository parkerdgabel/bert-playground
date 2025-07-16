"""Compatibility wrapper for classification_head.py - redirects to unified implementation."""

import warnings
from .classification import *

warnings.warn(
    "classification_head.py is deprecated. Please import from models.classification instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export all symbols for backward compatibility
__all__ = [
    'BinaryClassificationHead',
    'TitanicClassifier',
    'create_classifier'
]