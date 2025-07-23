"""HuggingFace tokenizer adapter implementation."""

from typing import Any, Dict, List, Union

from ports.secondary.tokenizer import TokenizerPort, TokenizerOutput


class HuggingFaceTokenizerAdapter:
    """HuggingFace implementation of the TokenizerPort."""

    def __init__(self, model_name: str, max_length: int = 512, **kwargs):
        """Initialize HuggingFace tokenizer adapter.
        
        Args:
            model_name: Name/path of the tokenizer model
            max_length: Maximum sequence length
            **kwargs: Additional tokenizer arguments
        """
        self.model_name = model_name
        self._max_length = max_length
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
    def name(self) -> str:
        """Name of the tokenizer."""
        return self.model_name

    @property
    def vocab_size(self) -> int:
        """Size of the vocabulary."""
        return self._tokenizer.vocab_size

    @property
    def max_length(self) -> int:
        """Maximum sequence length supported."""
        return self._max_length

    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        padding: Union[bool, str] = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_attention_mask: bool = True,
        return_tensors: str | None = None,
        **kwargs
    ) -> TokenizerOutput:
        """Encode text to tokens using HuggingFace tokenizer."""
        max_len = max_length or self._max_length
        
        return self._tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_len,
            return_attention_mask=return_attention_mask,
            return_tensors=return_tensors,
            **kwargs
        )

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
        **kwargs
    ) -> str:
        """Decode token IDs back to text."""
        return self._tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            **kwargs
        )

    def batch_decode(
        self,
        sequences: List[List[int]],
        skip_special_tokens: bool = True,
        **kwargs
    ) -> List[str]:
        """Decode multiple sequences of token IDs."""
        return self._tokenizer.batch_decode(
            sequences,
            skip_special_tokens=skip_special_tokens,
            **kwargs
        )

    def get_vocab(self) -> Dict[str, int]:
        """Get the tokenizer vocabulary."""
        return self._tokenizer.get_vocab()

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into individual tokens."""
        return self._tokenizer.tokenize(text)

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert tokens to their corresponding IDs."""
        return self._tokenizer.convert_tokens_to_ids(tokens)

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert token IDs to their corresponding tokens."""
        return self._tokenizer.convert_ids_to_tokens(ids)


class HuggingFaceTokenizerFactory:
    """Factory for creating HuggingFace tokenizers."""

    def create_tokenizer(
        self,
        model_name: str,
        max_length: int = 512,
        **kwargs
    ) -> TokenizerPort:
        """Create a HuggingFace tokenizer instance."""
        return HuggingFaceTokenizerAdapter(
            model_name=model_name,
            max_length=max_length,
            **kwargs
        )

    def list_available_tokenizers(self) -> List[str]:
        """List some common available tokenizer models."""
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
        ]