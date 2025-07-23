"""HuggingFace tokenizer adapter implementation.

This module provides HuggingFace implementations of the tokenizer port,
enabling integration with the vast ecosystem of HuggingFace tokenizers.
"""

from typing import Any

from ports.secondary.tokenizer import TokenEncoding, TokenizerPort


class HuggingFaceTokenizerAdapter(TokenizerPort):
    """HuggingFace implementation of the TokenizerPort."""

    def __init__(self, model_name: str, **kwargs):
        """Initialize HuggingFace tokenizer adapter.
        
        Args:
            model_name: Name/path of the tokenizer model
            **kwargs: Additional tokenizer arguments
        """
        self.model_name = model_name
        self.kwargs = kwargs
        self._tokenizer = None
        self._initialize_tokenizer()

    def _initialize_tokenizer(self):
        """Initialize the HuggingFace tokenizer."""
        try:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                **self.kwargs
            )
        except ImportError as e:
            raise ImportError(
                "HuggingFace transformers library is required for HuggingFaceTokenizerAdapter. "
                "Install it with: pip install transformers"
            ) from e

    @property
    def vocab_size(self) -> int:
        """Size of the vocabulary."""
        return self._tokenizer.vocab_size

    @property
    def pad_token_id(self) -> int | None:
        """ID of the padding token."""
        return self._tokenizer.pad_token_id

    @property
    def eos_token_id(self) -> int | None:
        """ID of the end-of-sequence token."""
        return self._tokenizer.eos_token_id

    @property
    def bos_token_id(self) -> int | None:
        """ID of the beginning-of-sequence token."""
        return self._tokenizer.bos_token_id

    @property
    def unk_token_id(self) -> int | None:
        """ID of the unknown token."""
        return self._tokenizer.unk_token_id

    def encode(
        self,
        text: str | list[str],
        add_special_tokens: bool = True,
        padding: bool | str = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_attention_mask: bool = True,
        **kwargs
    ) -> TokenEncoding:
        """Encode text to tokens.
        
        Args:
            text: Input text or list of texts
            add_special_tokens: Whether to add special tokens
            padding: Padding strategy
            truncation: Whether to truncate
            max_length: Maximum length
            return_attention_mask: Whether to return attention mask
            **kwargs: Additional encoding arguments
            
        Returns:
            TokenEncoding with input IDs and optional attention mask
        """
        # Handle the return_tensors parameter
        return_tensors = kwargs.pop("return_tensors", None)
        
        # Encode using HuggingFace tokenizer
        encoded = self._tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_attention_mask=return_attention_mask,
            return_tensors=return_tensors,
            **kwargs
        )
        
        # Create TokenEncoding from HuggingFace output
        result = TokenEncoding(
            input_ids=encoded["input_ids"],
            attention_mask=encoded.get("attention_mask"),
            token_type_ids=encoded.get("token_type_ids"),
        )
        
        # Add any additional fields from the encoding
        for key, value in encoded.items():
            if key not in ["input_ids", "attention_mask", "token_type_ids"]:
                setattr(result, key, value)
        
        return result

    def decode(
        self,
        token_ids: list[int] | Any,
        skip_special_tokens: bool = True,
        **kwargs
    ) -> str:
        """Decode token IDs back to text.
        
        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens
            **kwargs: Additional decoding arguments
            
        Returns:
            Decoded text
        """
        return self._tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            **kwargs
        )

    def batch_decode(
        self,
        sequences: list[list[int]] | Any,
        skip_special_tokens: bool = True,
        **kwargs
    ) -> list[str]:
        """Decode multiple sequences of token IDs.
        
        Args:
            sequences: List of token ID sequences
            skip_special_tokens: Whether to skip special tokens
            **kwargs: Additional decoding arguments
            
        Returns:
            List of decoded texts
        """
        return self._tokenizer.batch_decode(
            sequences,
            skip_special_tokens=skip_special_tokens,
            **kwargs
        )

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into tokens (not IDs).
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of token strings
        """
        return self._tokenizer.tokenize(text)

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        """Convert tokens to their IDs.
        
        Args:
            tokens: List of token strings
            
        Returns:
            List of token IDs
        """
        return self._tokenizer.convert_tokens_to_ids(tokens)

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        """Convert token IDs to tokens.
        
        Args:
            ids: List of token IDs
            
        Returns:
            List of token strings
        """
        return self._tokenizer.convert_ids_to_tokens(ids)

    def save(self, path: str) -> None:
        """Save tokenizer to disk.
        
        Args:
            path: Directory path to save tokenizer
        """
        self._tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: str, **kwargs) -> "HuggingFaceTokenizerAdapter":
        """Load tokenizer from disk.
        
        Args:
            path: Directory path to load tokenizer from
            **kwargs: Additional loading arguments
            
        Returns:
            Loaded tokenizer adapter
        """
        return cls(path, **kwargs)


class HuggingFaceTokenizerFactory:
    """Factory for creating HuggingFace tokenizers."""

    @staticmethod
    def create_tokenizer(
        model_name: str,
        **kwargs
    ) -> TokenizerPort:
        """Create a HuggingFace tokenizer instance.
        
        Args:
            model_name: Model name or path
            **kwargs: Additional tokenizer arguments
            
        Returns:
            Tokenizer instance
        """
        return HuggingFaceTokenizerAdapter(
            model_name=model_name,
            **kwargs
        )

    @staticmethod
    def list_available_tokenizers() -> list[str]:
        """List some common available tokenizer models.
        
        Returns:
            List of common tokenizer model names
        """
        return [
            "bert-base-uncased",
            "bert-base-cased", 
            "bert-large-uncased",
            "bert-large-cased",
            "roberta-base",
            "roberta-large",
            "distilbert-base-uncased",
            "answerdotai/ModernBERT-base",
            "answerdotai/ModernBERT-large",
            "xlm-roberta-base",
            "xlm-roberta-large",
            "albert-base-v2",
            "albert-large-v2",
            "google/electra-base-discriminator",
            "microsoft/deberta-v3-base",
            "microsoft/deberta-v3-large",
        ]