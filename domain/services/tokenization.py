"""Tokenization service - pure business logic for text processing.

This service contains only the business logic for tokenization,
without any dependencies on specific tokenizer implementations.
"""

from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
from domain.entities.dataset import DataBatch, DataSample
from infrastructure.di import service


@service
class TokenizationService:
    """Service for text tokenization business logic.
    
    This service defines the business rules for tokenization
    without depending on specific tokenizer implementations.
    """
    
    def calculate_token_statistics(
        self,
        texts: List[str],
        tokenizer_vocab_size: int,
    ) -> Dict[str, Any]:
        """Calculate tokenization statistics for texts.
        
        Args:
            texts: List of text strings
            tokenizer_vocab_size: Size of tokenizer vocabulary
            
        Returns:
            Dictionary with statistics
        """
        if not texts:
            return {
                "num_texts": 0,
                "avg_length": 0,
                "max_length": 0,
                "min_length": 0,
                "total_chars": 0,
            }
        
        lengths = [len(text) for text in texts]
        
        return {
            "num_texts": len(texts),
            "avg_length": sum(lengths) / len(lengths),
            "max_length": max(lengths),
            "min_length": min(lengths),
            "total_chars": sum(lengths),
            "vocab_size": tokenizer_vocab_size,
        }
    
    def validate_text_length(
        self,
        text: str,
        max_length: int,
        tokenizer_type: str = "wordpiece",
    ) -> bool:
        """Validate if text is within acceptable length.
        
        Args:
            text: Text to validate
            max_length: Maximum allowed length in tokens
            tokenizer_type: Type of tokenizer
            
        Returns:
            True if text is valid length
        """
        # Estimate token count based on tokenizer type
        if tokenizer_type == "wordpiece":
            # WordPiece typically creates ~1.3 tokens per word
            estimated_tokens = len(text.split()) * 1.3
        elif tokenizer_type == "bpe":
            # BPE typically creates ~1.1 tokens per word
            estimated_tokens = len(text.split()) * 1.1
        else:
            # Conservative estimate
            estimated_tokens = len(text.split()) * 1.5
        
        return estimated_tokens <= max_length
    
    def prepare_text_for_tokenization(
        self,
        text: str,
        lowercase: bool = False,
        remove_extra_spaces: bool = True,
        max_length_chars: Optional[int] = None,
    ) -> str:
        """Prepare text for tokenization.
        
        Args:
            text: Raw text
            lowercase: Whether to lowercase
            remove_extra_spaces: Whether to normalize whitespace
            max_length_chars: Maximum character length
            
        Returns:
            Prepared text
        """
        # Basic text cleaning
        if lowercase:
            text = text.lower()
        
        if remove_extra_spaces:
            # Normalize whitespace
            text = ' '.join(text.split())
        
        # Truncate if needed
        if max_length_chars and len(text) > max_length_chars:
            text = text[:max_length_chars]
            # Try to truncate at word boundary
            last_space = text.rfind(' ')
            if last_space > max_length_chars * 0.8:  # Don't truncate too much
                text = text[:last_space]
        
        return text.strip()
    
    def create_token_type_segments(
        self,
        text_a: str,
        text_b: Optional[str] = None,
        sep_token: str = "[SEP]",
    ) -> List[int]:
        """Create token type IDs for BERT-style input.
        
        Args:
            text_a: First text segment
            text_b: Optional second text segment
            sep_token: Separator token
            
        Returns:
            List of token type IDs (0 for text_a, 1 for text_b)
        """
        # Estimate token counts
        tokens_a = len(text_a.split()) + 2  # +2 for [CLS] and [SEP]
        
        if text_b:
            tokens_b = len(text_b.split()) + 1  # +1 for final [SEP]
            return [0] * tokens_a + [1] * tokens_b
        else:
            return [0] * tokens_a
    
    def calculate_attention_mask(
        self,
        sequence_length: int,
        pad_length: int,
    ) -> List[int]:
        """Calculate attention mask for padded sequence.
        
        Args:
            sequence_length: Actual sequence length
            pad_length: Total padded length
            
        Returns:
            Attention mask (1 for real tokens, 0 for padding)
        """
        if sequence_length > pad_length:
            raise ValueError(
                f"Sequence length ({sequence_length}) cannot exceed "
                f"pad length ({pad_length})"
            )
        
        return [1] * sequence_length + [0] * (pad_length - sequence_length)
    
    def merge_subword_tokens(
        self,
        tokens: List[str],
        subword_prefix: str = "##",
    ) -> List[str]:
        """Merge subword tokens back into words.
        
        Args:
            tokens: List of tokens
            subword_prefix: Prefix indicating subword tokens
            
        Returns:
            List of merged words
        """
        if not tokens:
            return []
        
        words = []
        current_word = []
        
        for token in tokens:
            if token.startswith(subword_prefix) and current_word:
                # Continue building current word
                current_word.append(token[len(subword_prefix):])
            else:
                # Start new word
                if current_word:
                    words.append(''.join(current_word))
                current_word = [token]
        
        # Don't forget last word
        if current_word:
            words.append(''.join(current_word))
        
        return words
    
    def truncate_sequences(
        self,
        tokens_a: List[Any],
        tokens_b: Optional[List[Any]],
        max_length: int,
        truncation_strategy: str = "longest_first",
    ) -> tuple[List[Any], Optional[List[Any]]]:
        """Truncate sequences to fit within max_length.
        
        Args:
            tokens_a: First token sequence
            tokens_b: Optional second token sequence
            max_length: Maximum total length
            truncation_strategy: How to truncate
            
        Returns:
            Truncated sequences
        """
        if tokens_b is None:
            # Single sequence
            return tokens_a[:max_length], None
        
        # Account for special tokens: [CLS] + tokens_a + [SEP] + tokens_b + [SEP]
        special_tokens_count = 3
        max_tokens = max_length - special_tokens_count
        
        total_length = len(tokens_a) + len(tokens_b)
        
        if total_length <= max_tokens:
            return tokens_a, tokens_b
        
        if truncation_strategy == "only_first":
            tokens_a = tokens_a[:max_tokens]
        elif truncation_strategy == "only_second":
            tokens_b = tokens_b[:max_tokens]
        else:  # longest_first or equal
            while total_length > max_tokens:
                if len(tokens_a) > len(tokens_b):
                    tokens_a = tokens_a[:-1]
                else:
                    tokens_b = tokens_b[:-1]
                total_length = len(tokens_a) + len(tokens_b)
        
        return tokens_a, tokens_b