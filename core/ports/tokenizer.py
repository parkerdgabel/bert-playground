"""Tokenizer port interface for text processing operations.

This port abstracts tokenization operations, allowing the core domain
to be independent of specific tokenizer implementations (HuggingFace, SentencePiece, etc.).
"""

from typing import Any, Dict, List, Protocol, Union, runtime_checkable

from typing_extensions import TypeAlias

# Type aliases for tokenizer operations
TokenizerOutput: TypeAlias = Dict[str, Any]  # Framework-specific tokenizer output
TextInput: TypeAlias = Union[str, List[str]]


@runtime_checkable
class TokenizerPort(Protocol):
    """Port for tokenization operations."""

    @property
    def name(self) -> str:
        """Name of the tokenizer."""
        ...

    @property
    def vocab_size(self) -> int:
        """Size of the vocabulary."""
        ...

    @property
    def max_length(self) -> int:
        """Maximum sequence length supported."""
        ...

    def encode(
        self,
        text: TextInput,
        add_special_tokens: bool = True,
        padding: Union[bool, str] = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_attention_mask: bool = True,
        return_tensors: str | None = None,
        **kwargs
    ) -> TokenizerOutput:
        """Encode text to tokens.
        
        Args:
            text: Input text or list of texts
            add_special_tokens: Whether to add special tokens
            padding: Padding strategy
            truncation: Whether to truncate sequences
            max_length: Maximum sequence length
            return_attention_mask: Whether to return attention mask
            return_tensors: Format for returned tensors
            **kwargs: Additional tokenizer arguments
            
        Returns:
            Dictionary containing encoded tokens and metadata
        """
        ...

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
        **kwargs
    ) -> str:
        """Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs to decode
            skip_special_tokens: Whether to skip special tokens
            **kwargs: Additional decode arguments
            
        Returns:
            Decoded text string
        """
        ...

    def batch_decode(
        self,
        sequences: List[List[int]],
        skip_special_tokens: bool = True,
        **kwargs
    ) -> List[str]:
        """Decode multiple sequences of token IDs.
        
        Args:
            sequences: List of token ID sequences
            skip_special_tokens: Whether to skip special tokens
            **kwargs: Additional decode arguments
            
        Returns:
            List of decoded text strings
        """
        ...

    def get_vocab(self) -> Dict[str, int]:
        """Get the tokenizer vocabulary.
        
        Returns:
            Dictionary mapping tokens to IDs
        """
        ...

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into individual tokens.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of token strings
        """
        ...

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert tokens to their corresponding IDs.
        
        Args:
            tokens: List of token strings
            
        Returns:
            List of token IDs
        """
        ...

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert token IDs to their corresponding tokens.
        
        Args:
            ids: List of token IDs
            
        Returns:
            List of token strings
        """
        ...


@runtime_checkable
class TokenizerFactory(Protocol):
    """Factory for creating tokenizers."""

    def create_tokenizer(
        self,
        model_name: str,
        max_length: int = 512,
        **kwargs
    ) -> TokenizerPort:
        """Create a tokenizer instance.
        
        Args:
            model_name: Name/path of the tokenizer model
            max_length: Maximum sequence length
            **kwargs: Additional tokenizer arguments
            
        Returns:
            Tokenizer instance
        """
        ...

    def list_available_tokenizers(self) -> List[str]:
        """List available tokenizer models.
        
        Returns:
            List of available tokenizer names
        """
        ...