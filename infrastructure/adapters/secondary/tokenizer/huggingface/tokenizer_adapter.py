from ..token_sequence import TokenSequence
"""HuggingFace tokenizer adapter implementation."""

from typing import List, Dict, Optional, Union, Any
from pathlib import Path

from infrastructure.di import adapter, Scope
from application.ports.secondary.tokenizer import TokenizerPort
from infrastructure.adapters.secondary.tokenizer.base import BaseTokenizerAdapter


@adapter(TokenizerPort, scope=Scope.SINGLETON)
class HuggingFaceTokenizerAdapter(BaseTokenizerAdapter):
    """HuggingFace implementation of TokenizerPort."""
    
    def __init__(self, model_name_or_path: str, **kwargs):
        """Initialize HuggingFace tokenizer adapter.
        
        Args:
            model_name_or_path: HuggingFace model name or path
            **kwargs: Additional tokenizer configuration
        """
        super().__init__(model_name_or_path, **kwargs)
        
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers library required for HuggingFace tokenizer. "
                "Install with: pip install transformers"
            )
        
        # Initialize tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            **kwargs
        )
        
        # Set vocabulary size
        self._vocab_size = self._tokenizer.vocab_size
        
        # Set special tokens
        self._update_special_tokens()
    
    def _update_special_tokens(self) -> None:
        """Update special tokens from tokenizer."""
        self._special_tokens = {
            'pad_token': self._tokenizer.pad_token,
            'unk_token': self._tokenizer.unk_token,
            'cls_token': self._tokenizer.cls_token,
            'sep_token': self._tokenizer.sep_token,
            'mask_token': self._tokenizer.mask_token,
        }
        
        # Remove None values
        self._special_tokens = {k: v for k, v in self._special_tokens.items() if v is not None}
    
    def _tokenize_single(
        self,
        text: str,
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
        return_attention_mask: bool = True,
        return_token_type_ids: bool = False,
    ) -> TokenSequence:
        """Tokenize a single text using HuggingFace tokenizer.
        
        Args:
            text: Text to tokenize
            max_length: Maximum sequence length
            padding: Whether to pad sequences
            truncation: Whether to truncate long sequences
            return_attention_mask: Whether to return attention mask
            return_token_type_ids: Whether to return token type IDs
            
        Returns:
            TokenSequence
        """
        # Use HuggingFace tokenizer
        encoding = self._tokenizer(
            text,
            max_length=max_length,
            padding='max_length' if padding and max_length else padding,
            truncation=truncation,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_tensors=None,  # Return Python lists
        )
        
        # Create TokenSequence
        return TokenSequence(
            input_ids=encoding['input_ids'],
            attention_mask=encoding.get('attention_mask', [1] * len(encoding['input_ids'])),
            token_type_ids=encoding.get('token_type_ids'),
            position_ids=None,  # HuggingFace doesn't return position_ids by default
        )
    
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
        return self._tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert tokens to IDs.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of token IDs
        """
        return self._tokenizer.convert_tokens_to_ids(tokens)
    
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert IDs to tokens.
        
        Args:
            ids: List of token IDs
            
        Returns:
            List of tokens
        """
        return self._tokenizer.convert_ids_to_tokens(ids)
    
    @property
    def pad_token_id(self) -> int:
        """Get padding token ID."""
        return self._tokenizer.pad_token_id or 0
    
    @property
    def cls_token_id(self) -> int:
        """Get CLS token ID."""
        return self._tokenizer.cls_token_id or self._tokenizer.bos_token_id or 0
    
    @property
    def sep_token_id(self) -> int:
        """Get SEP token ID."""
        return self._tokenizer.sep_token_id or self._tokenizer.eos_token_id or 0
    
    @property
    def mask_token_id(self) -> int:
        """Get MASK token ID."""
        return self._tokenizer.mask_token_id or self._tokenizer.unk_token_id or 0
    
    @property
    def unk_token_id(self) -> int:
        """Get UNK token ID."""
        return self._tokenizer.unk_token_id or 0
    
    def _save_tokenizer_files(self, path: Path) -> None:
        """Save HuggingFace tokenizer files.
        
        Args:
            path: Directory to save files
        """
        self._tokenizer.save_pretrained(str(path))
    
    def _load_tokenizer_files(self, path: Path) -> None:
        """Load HuggingFace tokenizer files.
        
        Args:
            path: Directory to load files from
        """
        from transformers import AutoTokenizer
        
        self._tokenizer = AutoTokenizer.from_pretrained(str(path))
        self._vocab_size = self._tokenizer.vocab_size
        self._update_special_tokens()
    
    def get_tokenizer(self) -> Any:
        """Get the underlying HuggingFace tokenizer.
        
        Returns:
            HuggingFace tokenizer instance
        """
        return self._tokenizer
    
    def enable_padding(
        self,
        length: Optional[int] = None,
        direction: str = 'right',
        pad_to_multiple_of: Optional[int] = None,
    ) -> None:
        """Enable padding configuration.
        
        Args:
            length: Padding length
            direction: Padding direction ('right' or 'left')
            pad_to_multiple_of: Pad to multiple of this value
        """
        self._tokenizer.padding_side = direction
        if length:
            self._tokenizer.model_max_length = length
        if pad_to_multiple_of:
            self._tokenizer.pad_to_multiple_of = pad_to_multiple_of
    
    def enable_truncation(
        self,
        max_length: int,
        stride: int = 0,
        strategy: str = 'longest_first',
    ) -> None:
        """Enable truncation configuration.
        
        Args:
            max_length: Maximum length
            stride: Stride for sliding window
            strategy: Truncation strategy
        """
        self._tokenizer.model_max_length = max_length
        self._tokenizer.truncation_side = 'right' if strategy == 'longest_first' else 'left'