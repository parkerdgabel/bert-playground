"""Data interfaces - re-exported from core.protocols for backward compatibility."""

# Re-export all data protocols from the centralized location
from core.protocols.data import (
    Augmenter,
    BatchProcessor,
    DataFormat,
    DataLoader,
    Dataset,
    FeatureAugmenter,
)

__all__ = [
    "Dataset",
    "DataLoader",
    "DataFormat",
    "BatchProcessor",
    "Augmenter",
    "FeatureAugmenter",
]