"""Secondary tokenizer port - Tokenization services that the application depends on.

This port defines the tokenizer interface that the application core uses
for text tokenization. It's a driven port implemented by adapters for
different tokenizer libraries (HuggingFace, SentencePiece, etc.).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable, Optional

from typing_extensions import TypeAlias
from infrastructure.di import port

# Type aliases
TokenIds: TypeAlias = list[int]
TokenizedBatch: TypeAlias = dict[str, Any]


@dataclass
class TokenizerConfig:
    """Configuration for tokenizers."""
    
    vocab_size: int
    model_max_length: int
    padding_token: str = "[PAD]"
    unk_token: str = "[UNK]"
    sep_token: str = "[SEP]"
    cls_token: str = "[CLS]"
    mask_token: str = "[MASK]"
    add_prefix_space: bool = False
    lowercase: bool = False
    strip_accents: bool = False


@dataclass
class TokenizerOutput:
    """Output from tokenization."""
    
    input_ids: list[TokenIds]
    attention_mask: list[list[int]]
    token_type_ids: list[list[int]] | None = None
    special_tokens_mask: list[list[int]] | None = None
    offsets: list[list[tuple[int, int]]] | None = None
    
    def __getitem__(self, key: str) -> Any:
        """Dict-like access for compatibility."""
        return getattr(self, key)
    
    def keys(self) -> list[str]:
        """Get available keys."""
        keys = ["input_ids", "attention_mask"]
        if self.token_type_ids is not None:
            keys.append("token_type_ids")
        if self.special_tokens_mask is not None:
            keys.append("special_tokens_mask")
        if self.offsets is not None:
            keys.append("offsets")
        return keys


@dataclass
class TokenizerVocabulary:
    """Tokenizer vocabulary information."""
    
    token_to_id: dict[str, int]
    id_to_token: dict[int, str]
    special_tokens: dict[str, str]
    vocab_size: int


@port()
@runtime_checkable
class TokenizerPort(Protocol):
    """Secondary port for tokenization operations.
    
    This interface is implemented by adapters for specific tokenizer
    libraries. The application core depends on this for all tokenization.
    """

    @property
    def vocab_size(self) -> int:
        """Size of the vocabulary."""
        ...

    @property
    def model_max_length(self) -> int:
        """Maximum sequence length."""
        ...

    @property
    def pad_token_id(self) -> int:
        """ID of padding token."""
        ...

    @property
    def unk_token_id(self) -> int:
        """ID of unknown token."""
        ...

    @property
    def cls_token_id(self) -> int:
        """ID of CLS token."""
        ...

    @property
    def sep_token_id(self) -> int:
        """ID of SEP token."""
        ...

    @property
    def mask_token_id(self) -> int:
        """ID of MASK token."""
        ...

    def tokenize(
        self,
        text: str | list[str],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = True,
        padding: bool | str = False,
        return_tensors: Optional[str] = None,
        return_offsets: bool = False,
    ) -> TokenizerOutput:
        """Tokenize text or batch of texts.
        
        Args:
            text: Text or list of texts to tokenize
            add_special_tokens: Whether to add special tokens
            max_length: Maximum sequence length
            truncation: Whether to truncate sequences
            padding: Padding strategy (True, False, 'max_length', 'longest')
            return_tensors: Return format ('np', 'pt', 'mlx', None)
            return_offsets: Whether to return character offsets
            
        Returns:
            Tokenization output
        """
        ...

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = True,
    ) -> TokenIds:
        """Encode text to token IDs.
        
        Args:
            text: Text to encode
            add_special_tokens: Whether to add special tokens
            max_length: Maximum sequence length
            truncation: Whether to truncate
            
        Returns:
            List of token IDs
        """
        ...

    def decode(
        self,
        token_ids: TokenIds,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        """Decode token IDs to text.
        
        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens
            clean_up_tokenization_spaces: Whether to clean up spaces
            
        Returns:
            Decoded text
        """
        ...

    def batch_encode(
        self,
        texts: list[str],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = True,
        padding: bool | str = True,
    ) -> list[TokenIds]:
        """Batch encode multiple texts.
        
        Args:
            texts: List of texts to encode
            add_special_tokens: Whether to add special tokens
            max_length: Maximum sequence length
            truncation: Whether to truncate
            padding: Padding strategy
            
        Returns:
            List of token ID lists
        """
        ...

    def batch_decode(
        self,
        token_ids_batch: list[TokenIds],
        skip_special_tokens: bool = True,
    ) -> list[str]:
        """Batch decode multiple token ID sequences.
        
        Args:
            token_ids_batch: List of token ID lists
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            List of decoded texts
        """
        ...

    def get_vocab(self) -> dict[str, int]:
        """Get the vocabulary mapping.
        
        Returns:
            Token to ID mapping
        """
        ...

    def save_pretrained(self, save_directory: Path) -> None:
        """Save tokenizer to directory.
        
        Args:
            save_directory: Directory to save to
        """
        ...

    def convert_tokens_to_ids(self, tokens: list[str]) -> TokenIds:
        """Convert tokens to IDs.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of token IDs
        """
        ...

    def convert_ids_to_tokens(self, ids: TokenIds) -> list[str]:
        """Convert IDs to tokens.
        
        Args:
            ids: List of token IDs
            
        Returns:
            List of tokens
        """
        ...

    def get_special_tokens_mask(
        self,
        token_ids: TokenIds,
        already_has_special_tokens: bool = False,
    ) -> list[int]:
        """Get mask of special tokens.
        
        Args:
            token_ids: Token IDs
            already_has_special_tokens: Whether IDs already have special tokens
            
        Returns:
            Binary mask (1 for special tokens)
        """
        ...

    def create_token_type_ids_from_sequences(
        self,
        token_ids_0: TokenIds,
        token_ids_1: Optional[TokenIds] = None,
    ) -> list[int]:
        """Create token type IDs for sequence pair.
        
        Args:
            token_ids_0: First sequence
            token_ids_1: Optional second sequence
            
        Returns:
            Token type IDs
        """
        ...

    def truncate_sequences(
        self,
        ids: TokenIds,
        pair_ids: Optional[TokenIds] = None,
        max_length: Optional[int] = None,
        truncation_strategy: str = "longest_first",
    ) -> tuple[TokenIds, Optional[TokenIds]]:
        """Truncate sequences to max length.
        
        Args:
            ids: First sequence
            pair_ids: Optional second sequence
            max_length: Maximum total length
            truncation_strategy: How to truncate ('longest_first', 'only_first', etc.)
            
        Returns:
            Truncated sequences
        """
        ...