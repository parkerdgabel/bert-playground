"""
Classification Module

Dedicated classification logic separated from embeddings.
Provides various classification heads and task-specific classifiers.
"""

from .heads import BinaryClassificationHead, MultiClassificationHead
from .titanic_classifier import TitanicClassifier
from .factory import create_classifier, create_titanic_classifier

__all__ = [
    "BinaryClassificationHead",
    "MultiClassificationHead", 
    "TitanicClassifier",
    "create_classifier",
    "create_titanic_classifier",
]