"""Token sequence data structure for tokenizer adapters."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TokenSequence:
    """Represents a tokenized sequence with associated metadata.
    
    This is an adapter-specific data structure used by tokenizer implementations.
    """
    
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    position_ids: Optional[List[int]] = None
    special_tokens_mask: Optional[List[int]] = None
    
    @property
    def length(self) -> int:
        """Get sequence length."""
        return len(self.input_ids)
    
    def truncate(self, max_length: int) -> "TokenSequence":
        """Truncate sequence to max length.
        
        Args:
            max_length: Maximum sequence length
            
        Returns:
            Truncated sequence
        """
        if self.length <= max_length:
            return self
            
        return TokenSequence(
            input_ids=self.input_ids[:max_length],
            attention_mask=self.attention_mask[:max_length],
            token_type_ids=self.token_type_ids[:max_length] if self.token_type_ids else None,
            position_ids=self.position_ids[:max_length] if self.position_ids else None,
            special_tokens_mask=self.special_tokens_mask[:max_length] if self.special_tokens_mask else None,
        )
    
    def pad(self, max_length: int, pad_token_id: int = 0) -> "TokenSequence":
        """Pad sequence to max length.
        
        Args:
            max_length: Target sequence length
            pad_token_id: Token ID to use for padding
            
        Returns:
            Padded sequence
        """
        if self.length >= max_length:
            return self
            
        pad_length = max_length - self.length
        
        return TokenSequence(
            input_ids=self.input_ids + [pad_token_id] * pad_length,
            attention_mask=self.attention_mask + [0] * pad_length,
            token_type_ids=(self.token_type_ids + [0] * pad_length) if self.token_type_ids else None,
            position_ids=(self.position_ids + list(range(self.length, max_length))) if self.position_ids else None,
            special_tokens_mask=(self.special_tokens_mask + [0] * pad_length) if self.special_tokens_mask else None,
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'input_ids': self.input_ids,
            'attention_mask': self.attention_mask,
            'token_type_ids': self.token_type_ids,
            'position_ids': self.position_ids,
            'special_tokens_mask': self.special_tokens_mask,
        }