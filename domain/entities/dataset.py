"""Dataset and data batch entities."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum


class DatasetSplit(Enum):
    """Dataset split types."""
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class TokenizerType(Enum):
    """Tokenizer types."""
    WORDPIECE = "wordpiece"
    BPE = "bpe"
    SENTENCEPIECE = "sentencepiece"


@dataclass
class TokenSequence:
    """Represents a tokenized sequence."""
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    position_ids: Optional[List[int]] = None
    
    @property
    def length(self) -> int:
        """Get sequence length."""
        return len(self.input_ids)
    
    @property
    def num_tokens(self) -> int:
        """Get number of actual tokens (excluding padding)."""
        return sum(self.attention_mask)
    
    def truncate(self, max_length: int) -> 'TokenSequence':
        """Truncate sequence to maximum length."""
        return TokenSequence(
            input_ids=self.input_ids[:max_length],
            attention_mask=self.attention_mask[:max_length],
            token_type_ids=self.token_type_ids[:max_length] if self.token_type_ids else None,
            position_ids=self.position_ids[:max_length] if self.position_ids else None,
        )
    
    def pad(self, max_length: int, pad_token_id: int = 0) -> 'TokenSequence':
        """Pad sequence to maximum length."""
        padding_length = max_length - self.length
        if padding_length <= 0:
            return self
        
        return TokenSequence(
            input_ids=self.input_ids + [pad_token_id] * padding_length,
            attention_mask=self.attention_mask + [0] * padding_length,
            token_type_ids=(
                self.token_type_ids + [0] * padding_length 
                if self.token_type_ids else None
            ),
            position_ids=(
                self.position_ids + list(range(self.length, max_length))
                if self.position_ids else None
            ),
        )


@dataclass
class DataBatch:
    """Represents a batch of data for training/inference."""
    sequences: List[TokenSequence]
    labels: Optional[Any] = None  # Can be List[int], List[float], List[List[float]], etc.
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def batch_size(self) -> int:
        """Get batch size."""
        return len(self.sequences)
    
    @property
    def max_sequence_length(self) -> int:
        """Get maximum sequence length in batch."""
        if not self.sequences:
            return 0
        return max(seq.length for seq in self.sequences)
    
    def collate(self, pad_token_id: int = 0) -> 'DataBatch':
        """Collate sequences to same length."""
        max_length = self.max_sequence_length
        padded_sequences = [
            seq.pad(max_length, pad_token_id) for seq in self.sequences
        ]
        return DataBatch(
            sequences=padded_sequences,
            labels=self.labels,
            metadata=self.metadata,
        )
    
    def split(self, sizes: List[int]) -> List['DataBatch']:
        """Split batch into multiple smaller batches."""
        if sum(sizes) != self.batch_size:
            raise ValueError("Split sizes must sum to batch size")
        
        batches = []
        start = 0
        for size in sizes:
            end = start + size
            batch_sequences = self.sequences[start:end]
            batch_labels = None
            if self.labels is not None:
                if isinstance(self.labels, list):
                    batch_labels = self.labels[start:end]
                else:
                    # Handle other label types as needed
                    batch_labels = self.labels
            
            batches.append(DataBatch(
                sequences=batch_sequences,
                labels=batch_labels,
                metadata=self.metadata.copy(),
            ))
            start = end
        
        return batches


@dataclass
class Dataset:
    """Represents a dataset."""
    name: str
    split: DatasetSplit
    size: int
    features: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Statistics
    num_classes: Optional[int] = None
    class_distribution: Optional[Dict[Any, int]] = None
    sequence_length_stats: Optional[Dict[str, float]] = None
    
    @property
    def is_classification(self) -> bool:
        """Check if dataset is for classification."""
        return self.num_classes is not None and self.num_classes > 0
    
    @property
    def is_regression(self) -> bool:
        """Check if dataset is for regression."""
        return self.num_classes is None or self.num_classes == 0
    
    @property
    def is_balanced(self) -> bool:
        """Check if dataset is balanced (for classification)."""
        if not self.is_classification or not self.class_distribution:
            return True
        
        counts = list(self.class_distribution.values())
        if not counts:
            return True
            
        min_count = min(counts)
        max_count = max(counts)
        # Consider balanced if max/min ratio < 2
        return max_count / min_count < 2.0 if min_count > 0 else False
    
    def get_split_info(self) -> Dict[str, Any]:
        """Get information about this dataset split."""
        return {
            "name": self.name,
            "split": self.split.value,
            "size": self.size,
            "num_classes": self.num_classes,
            "is_balanced": self.is_balanced,
            "features": list(self.features.keys()),
        }