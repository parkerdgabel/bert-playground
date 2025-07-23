"""Domain data protocols - Core data abstractions.

These protocols define the fundamental data contracts used throughout
the system. They are independent of any specific implementation.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol

from .compute import Array


class Dataset(Protocol):
    """Protocol for dataset implementations."""

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        ...

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single item from the dataset.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Dictionary containing the data sample
        """
        ...

    def get_metadata(self) -> dict[str, Any]:
        """Return dataset metadata (schema, stats, etc.).
        
        Returns:
            Dictionary containing metadata about the dataset
        """
        ...


class DataLoader(Protocol):
    """Protocol for data loaders."""

    def __iter__(self) -> Iterator[dict[str, Array]]:
        """Iterate over batches.
        
        Returns:
            Iterator yielding batches as dictionaries of arrays
        """
        ...

    def __len__(self) -> int:
        """Return number of batches."""
        ...

    @property
    def batch_size(self) -> int:
        """Get batch size."""
        ...

    @property
    def num_samples(self) -> int:
        """Get total number of samples."""
        ...


class Tokenizer(Protocol):
    """Protocol for tokenizers."""
    
    def tokenize(self, text: str) -> list[int]:
        """Tokenize text into token IDs.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of token IDs
        """
        ...
    
    def detokenize(self, tokens: list[int]) -> str:
        """Convert token IDs back to text.
        
        Args:
            tokens: List of token IDs
            
        Returns:
            Decoded text
        """
        ...
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        ...
    
    @property
    def pad_token_id(self) -> int:
        """Get padding token ID."""
        ...
    
    @property
    def eos_token_id(self) -> int:
        """Get end-of-sequence token ID."""
        ...