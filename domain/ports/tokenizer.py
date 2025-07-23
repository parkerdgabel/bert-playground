"""Tokenizer port for text processing."""

from typing import Protocol, List, Dict, Optional, Union, Any
from domain.entities.dataset import TokenSequence


class TokenizerPort(Protocol):
    """Port for tokenization operations."""
    
    def tokenize(
        self,
        text: Union[str, List[str]],
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
        return_attention_mask: bool = True,
        return_token_type_ids: bool = False,
    ) -> Union[TokenSequence, List[TokenSequence]]:
        """Tokenize text input.
        
        Args:
            text: Single text or list of texts
            max_length: Maximum sequence length
            padding: Whether to pad sequences
            truncation: Whether to truncate long sequences
            return_attention_mask: Whether to return attention mask
            return_token_type_ids: Whether to return token type IDs
            
        Returns:
            TokenSequence or list of TokenSequences
        """
        ...
    
    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        """Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            clean_up_tokenization_spaces: Whether to clean up spaces
            
        Returns:
            Decoded text
        """
        ...
    
    def batch_decode(
        self,
        token_ids_batch: List[List[int]],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> List[str]:
        """Decode batch of token IDs.
        
        Args:
            token_ids_batch: Batch of token ID lists
            skip_special_tokens: Whether to skip special tokens
            clean_up_tokenization_spaces: Whether to clean up spaces
            
        Returns:
            List of decoded texts
        """
        ...
    
    def get_vocab_size(
        self,
    ) -> int:
        """Get vocabulary size.
        
        Returns:
            Size of vocabulary
        """
        ...
    
    def get_special_tokens(
        self,
    ) -> Dict[str, int]:
        """Get special token mappings.
        
        Returns:
            Dictionary mapping special token names to IDs
        """
        ...
    
    def convert_tokens_to_ids(
        self,
        tokens: List[str],
    ) -> List[int]:
        """Convert tokens to IDs.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of token IDs
        """
        ...
    
    def convert_ids_to_tokens(
        self,
        ids: List[int],
    ) -> List[str]:
        """Convert IDs to tokens.
        
        Args:
            ids: List of token IDs
            
        Returns:
            List of tokens
        """
        ...
    
    def save(
        self,
        path: str,
    ) -> None:
        """Save tokenizer to disk.
        
        Args:
            path: Save directory path
        """
        ...
    
    def load(
        self,
        path: str,
    ) -> 'TokenizerPort':
        """Load tokenizer from disk.
        
        Args:
            path: Load directory path
            
        Returns:
            Loaded tokenizer
        """
        ...
    
    @property
    def pad_token_id(self) -> int:
        """Get padding token ID."""
        ...
    
    @property
    def cls_token_id(self) -> int:
        """Get CLS token ID."""
        ...
    
    @property
    def sep_token_id(self) -> int:
        """Get SEP token ID."""
        ...
    
    @property
    def mask_token_id(self) -> int:
        """Get MASK token ID."""
        ...
    
    @property
    def unk_token_id(self) -> int:
        """Get UNK token ID."""
        ...