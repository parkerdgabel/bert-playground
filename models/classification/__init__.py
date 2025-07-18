"""
Classification Module

Dedicated classification logic separated from embeddings.
Provides various classification heads and task-specific classifiers.
"""

from .heads import BinaryClassificationHead, MultiClassificationHead, RegressionHead
from .advanced_heads import (
    MultilabelClassificationHead,
    OrdinalRegressionHead,
    HierarchicalClassificationHead,
    EnsembleClassificationHead,
)
from .base_classifier import BaseClassifier, AttentionPooling, WeightedPooling, LearnedPooling
from .generic_classifier import GenericClassifier, ClassificationTask
from .titanic_classifier import TitanicClassifier
from .factory import (
    create_classifier,
    create_titanic_classifier,
    create_multilabel_classifier,
    create_ordinal_classifier,
    create_hierarchical_classifier,
    create_ensemble_classifier,
)

__all__ = [
    # Basic heads
    "BinaryClassificationHead",
    "MultiClassificationHead",
    "RegressionHead",
    # Advanced heads
    "MultilabelClassificationHead",
    "OrdinalRegressionHead", 
    "HierarchicalClassificationHead",
    "EnsembleClassificationHead",
    # Base classes
    "BaseClassifier",
    "GenericClassifier",
    "ClassificationTask",
    # Pooling strategies
    "AttentionPooling",
    "WeightedPooling",
    "LearnedPooling",
    # Specific classifiers
    "TitanicClassifier",
    # Factory functions
    "create_classifier",
    "create_titanic_classifier",
    "create_multilabel_classifier",
    "create_ordinal_classifier",
    "create_hierarchical_classifier",
    "create_ensemble_classifier",
]