"""Data transformations for MLX data loading."""

from typing import Any, List, Optional, Dict
import mlx.core as mx
import numpy as np

from domain.entities.dataset import DataBatch, TokenSequence


class MLXTransform:
    """Base class for MLX data transforms."""
    
    def apply(self, data: Any) -> Any:
        """Apply transform to data."""
        raise NotImplementedError


class MLXTokenTransform(MLXTransform):
    """Transform for tokenization operations."""
    
    def __init__(self, tokenizer: Optional[Any] = None):
        """Initialize token transform.
        
        Args:
            tokenizer: Optional tokenizer to use
        """
        self.tokenizer = tokenizer
    
    def apply(self, text: str) -> TokenSequence:
        """Apply tokenization to text.
        
        Args:
            text: Input text
            
        Returns:
            TokenSequence
        """
        if self.tokenizer is None:
            # Simple whitespace tokenization as fallback
            tokens = text.split()
            input_ids = [hash(token) % 30000 for token in tokens]  # Simple hash
            attention_mask = [1] * len(input_ids)
            
            return TokenSequence(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        
        # Use actual tokenizer
        encoded = self.tokenizer.encode(text)
        
        return TokenSequence(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            token_type_ids=encoded.get("token_type_ids"),
        )


class MLXPaddingTransform(MLXTransform):
    """Transform for padding sequences in a batch."""
    
    def apply(
        self,
        batch: DataBatch,
        pad_token_id: int = 0,
        max_length: Optional[int] = None
    ) -> DataBatch:
        """Apply padding to batch.
        
        Args:
            batch: Input batch
            pad_token_id: Token ID to use for padding
            max_length: Maximum sequence length (if None, use batch max)
            
        Returns:
            Padded batch
        """
        if not batch.sequences:
            return batch
        
        # Determine target length
        if max_length is None:
            max_length = batch.max_sequence_length
        
        # Pad sequences
        padded_sequences = []
        for seq in batch.sequences:
            if seq.length < max_length:
                padded_seq = seq.pad(max_length, pad_token_id)
            elif seq.length > max_length:
                padded_seq = seq.truncate(max_length)
            else:
                padded_seq = seq
            
            padded_sequences.append(padded_seq)
        
        # Create new batch with padded sequences
        return DataBatch(
            sequences=padded_sequences,
            labels=batch.labels,
            metadata=batch.metadata,
        )


class MLXTruncationTransform(MLXTransform):
    """Transform for truncating sequences."""
    
    def __init__(self, max_length: int):
        """Initialize truncation transform.
        
        Args:
            max_length: Maximum sequence length
        """
        self.max_length = max_length
    
    def apply(self, sequence: TokenSequence) -> TokenSequence:
        """Apply truncation to sequence.
        
        Args:
            sequence: Input sequence
            
        Returns:
            Truncated sequence
        """
        if sequence.length <= self.max_length:
            return sequence
        
        return sequence.truncate(self.max_length)


class MLXNormalizationTransform(MLXTransform):
    """Transform for normalizing numerical features."""
    
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        """Initialize normalization transform.
        
        Args:
            mean: Mean for normalization
            std: Standard deviation for normalization
        """
        self.mean = mean
        self.std = std
    
    def apply(self, data: mx.array) -> mx.array:
        """Apply normalization to data.
        
        Args:
            data: Input data
            
        Returns:
            Normalized data
        """
        return (data - self.mean) / self.std


class MLXAugmentationTransform(MLXTransform):
    """Transform for data augmentation."""
    
    def __init__(
        self,
        mask_prob: float = 0.15,
        mask_token_id: int = 103,  # [MASK] token
        vocab_size: int = 30522
    ):
        """Initialize augmentation transform.
        
        Args:
            mask_prob: Probability of masking tokens
            mask_token_id: Token ID for mask token
            vocab_size: Size of vocabulary
        """
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
    
    def apply(self, sequence: TokenSequence) -> TokenSequence:
        """Apply masking augmentation to sequence.
        
        Args:
            sequence: Input sequence
            
        Returns:
            Augmented sequence
        """
        # Create mask for positions to modify
        mask = np.random.random(len(sequence.input_ids)) < self.mask_prob
        
        # Copy input IDs
        augmented_ids = sequence.input_ids.copy()
        
        # Apply masking
        for i, should_mask in enumerate(mask):
            if should_mask and sequence.attention_mask[i] == 1:  # Only mask real tokens
                rand = np.random.random()
                if rand < 0.8:
                    # 80% of time, replace with [MASK]
                    augmented_ids[i] = self.mask_token_id
                elif rand < 0.9:
                    # 10% of time, replace with random token
                    augmented_ids[i] = np.random.randint(0, self.vocab_size)
                # 10% of time, keep original token
        
        return TokenSequence(
            input_ids=augmented_ids,
            attention_mask=sequence.attention_mask,
            token_type_ids=sequence.token_type_ids,
            position_ids=sequence.position_ids,
        )


class MLXComposeTransform(MLXTransform):
    """Compose multiple transforms."""
    
    def __init__(self, transforms: List[MLXTransform]):
        """Initialize compose transform.
        
        Args:
            transforms: List of transforms to apply in order
        """
        self.transforms = transforms
    
    def apply(self, data: Any) -> Any:
        """Apply all transforms in sequence.
        
        Args:
            data: Input data
            
        Returns:
            Transformed data
        """
        for transform in self.transforms:
            data = transform.apply(data)
        return data


class MLXCacheTransform(MLXTransform):
    """Transform that caches results."""
    
    def __init__(self, transform: MLXTransform, cache_size: int = 10000):
        """Initialize cache transform.
        
        Args:
            transform: Transform to cache
            cache_size: Maximum cache size
        """
        self.transform = transform
        self.cache_size = cache_size
        self._cache: Dict[int, Any] = {}
    
    def apply(self, data: Any) -> Any:
        """Apply transform with caching.
        
        Args:
            data: Input data
            
        Returns:
            Transformed data
        """
        # Create cache key (simple hash)
        cache_key = hash(str(data)) % (self.cache_size * 10)
        
        # Check cache
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Apply transform
        result = self.transform.apply(data)
        
        # Update cache if not full
        if len(self._cache) < self.cache_size:
            self._cache[cache_key] = result
        
        return result