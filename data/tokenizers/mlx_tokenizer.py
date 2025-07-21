"""MLX-native tokenizer wrapper for efficient tokenization on Apple Silicon."""

from typing import Dict, List, Optional, Union, Any
import mlx.core as mx
from loguru import logger


class MLXTokenizer:
    """Tokenizer wrapper that supports MLX-native operations.
    
    This wrapper provides a unified interface for tokenization that can
    leverage MLX embeddings when available, falling back to standard
    tokenizers when needed.
    """
    
    def __init__(
        self,
        tokenizer_name: str = "bert-base-uncased",
        backend: str = "auto",
        max_length: int = 512,
    ):
        """Initialize MLX tokenizer.
        
        Args:
            tokenizer_name: Name of the tokenizer to use
            backend: Backend to use ("mlx", "huggingface", or "auto")
            max_length: Maximum sequence length
        """
        self.tokenizer_name = tokenizer_name
        self.backend = backend
        self.max_length = max_length
        self._tokenizer = None
        
        # Initialize the appropriate backend
        self._initialize_backend()
        
    def _initialize_backend(self) -> None:
        """Initialize the tokenization backend."""
        if self.backend == "mlx" or (self.backend == "auto" and self._check_mlx_available()):
            try:
                self._initialize_mlx_backend()
                self.backend = "mlx"
                logger.info(f"Initialized MLX tokenizer backend for {self.tokenizer_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize MLX backend: {e}, falling back to HuggingFace")
                self._initialize_hf_backend()
                self.backend = "huggingface"
        else:
            self._initialize_hf_backend()
            self.backend = "huggingface"
            
    def _check_mlx_available(self) -> bool:
        """Check if MLX embeddings is available."""
        try:
            import mlx_embeddings
            return True
        except ImportError:
            return False
            
    def _initialize_mlx_backend(self) -> None:
        """Initialize MLX embeddings backend."""
        from mlx_embeddings import get_model
        
        # Get the embedding model which includes tokenizer
        self._model = get_model(self.tokenizer_name)
        self._tokenizer = self._model.tokenizer
        
    def _initialize_hf_backend(self) -> None:
        """Initialize HuggingFace tokenizer backend."""
        from transformers import AutoTokenizer
        
        self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        
    def __call__(
        self,
        texts: Union[str, List[str]],
        padding: Union[bool, str] = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = "mlx",
        **kwargs
    ) -> Dict[str, Any]:
        """Tokenize texts.
        
        Args:
            texts: Text or list of texts to tokenize
            padding: Padding strategy
            truncation: Whether to truncate
            max_length: Maximum length (uses self.max_length if None)
            return_tensors: Format of returned tensors ("mlx", "np", "pt")
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with tokenized outputs
        """
        if isinstance(texts, str):
            texts = [texts]
            
        max_length = max_length or self.max_length
        
        if self.backend == "mlx":
            return self._tokenize_mlx(
                texts, padding, truncation, max_length, return_tensors, **kwargs
            )
        else:
            return self._tokenize_hf(
                texts, padding, truncation, max_length, return_tensors, **kwargs
            )
            
    def _tokenize_mlx(
        self,
        texts: List[str],
        padding: Union[bool, str],
        truncation: bool,
        max_length: int,
        return_tensors: str,
        **kwargs
    ) -> Dict[str, mx.array]:
        """Tokenize using MLX backend."""
        # MLX embeddings models typically have a encode method
        if hasattr(self._model, 'encode'):
            # Get embeddings directly (includes tokenization)
            embeddings = self._model.encode(texts)
            
            # For compatibility, we need to return tokenized format
            # This is a simplified approach - in practice you might need
            # to access the actual tokens
            batch_size = len(texts)
            
            # Create dummy input_ids and attention_mask
            # In a real implementation, you'd extract these from the model
            input_ids = mx.zeros((batch_size, max_length), dtype=mx.int32)
            attention_mask = mx.ones((batch_size, max_length), dtype=mx.int32)
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "embeddings": embeddings,  # Include the actual embeddings
            }
        else:
            # Fallback to standard tokenization
            return self._tokenize_hf(
                texts, padding, truncation, max_length, return_tensors, **kwargs
            )
            
    def _tokenize_hf(
        self,
        texts: List[str],
        padding: Union[bool, str],
        truncation: bool,
        max_length: int,
        return_tensors: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Tokenize using HuggingFace backend."""
        # Map return_tensors format
        if return_tensors == "mlx":
            hf_return_tensors = "np"  # Get numpy then convert
        else:
            hf_return_tensors = return_tensors
            
        # Tokenize
        encodings = self._tokenizer(
            texts,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=hf_return_tensors,
            **kwargs
        )
        
        # Convert to MLX if needed
        if return_tensors == "mlx":
            mlx_encodings = {}
            for key, values in encodings.items():
                if key in ['input_ids', 'attention_mask', 'token_type_ids']:
                    mlx_encodings[key] = mx.array(values, dtype=mx.int32)
                else:
                    mlx_encodings[key] = values
            return mlx_encodings
        
        return encodings
        
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        if hasattr(self._tokenizer, 'vocab_size'):
            return self._tokenizer.vocab_size
        elif hasattr(self._tokenizer, 'get_vocab'):
            return len(self._tokenizer.get_vocab())
        else:
            return 30522  # Default BERT vocab size
            
    @property
    def pad_token_id(self) -> int:
        """Get padding token ID."""
        if hasattr(self._tokenizer, 'pad_token_id'):
            return self._tokenizer.pad_token_id
        return 0
        
    @property
    def cls_token_id(self) -> int:
        """Get CLS token ID."""
        if hasattr(self._tokenizer, 'cls_token_id'):
            return self._tokenizer.cls_token_id
        return 101  # Default BERT CLS token
        
    @property
    def sep_token_id(self) -> int:
        """Get SEP token ID."""
        if hasattr(self._tokenizer, 'sep_token_id'):
            return self._tokenizer.sep_token_id
        return 102  # Default BERT SEP token
        
    @property
    def name_or_path(self) -> str:
        """Get tokenizer name or path for compatibility."""
        return self.tokenizer_name