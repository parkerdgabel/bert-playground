"""
Tokenizer Wrapper

Provides a unified interface for both HuggingFace and mlx-embeddings tokenizers,
ensuring backward compatibility while enabling migration to mlx-embeddings.
"""

from typing import Dict, List, Optional, Union
import mlx.core as mx
from loguru import logger

from embeddings.mlx_adapter import MLXEmbeddingsAdapter

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not available")


class TokenizerWrapper:
    """
    Unified tokenizer interface supporting both HuggingFace and mlx-embeddings.
    
    This wrapper provides a consistent API regardless of the backend used,
    making it easy to switch between implementations.
    """
    
    def __init__(
        self,
        model_name: str = "answerdotai/ModernBERT-base",
        backend: str = "auto",
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize tokenizer wrapper.
        
        Args:
            model_name: Name of the model/tokenizer to load
            backend: Backend to use ("huggingface", "mlx", "auto")
            cache_dir: Directory for caching models
        """
        self.model_name = model_name
        self.backend = backend
        self.cache_dir = cache_dir
        
        self._tokenizer = None
        self._mlx_adapter = None
        
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize the appropriate backend based on configuration."""
        if self.backend == "auto":
            # Try MLX first, fall back to HuggingFace
            if self._try_mlx_backend():
                self.backend = "mlx"
                logger.info("Using MLX embeddings tokenizer")
            elif TRANSFORMERS_AVAILABLE:
                self._load_huggingface_tokenizer()
                self.backend = "huggingface"
                logger.info("Using HuggingFace tokenizer")
            else:
                raise RuntimeError("No tokenizer backend available")
        elif self.backend == "mlx":
            if not self._try_mlx_backend():
                raise RuntimeError("MLX backend requested but not available")
        elif self.backend == "huggingface":
            if not TRANSFORMERS_AVAILABLE:
                raise RuntimeError("HuggingFace backend requested but not available")
            self._load_huggingface_tokenizer()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _try_mlx_backend(self) -> bool:
        """Try to initialize MLX backend."""
        try:
            self._mlx_adapter = MLXEmbeddingsAdapter(
                model_name=self.model_name,
                use_mlx_embeddings=True,
                cache_dir=self.cache_dir
            )
            return self._mlx_adapter.is_available
        except Exception as e:
            logger.debug(f"Failed to initialize MLX backend: {e}")
            return False
    
    def _load_huggingface_tokenizer(self):
        """Load HuggingFace tokenizer."""
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=True
        )
        # Ensure we don't use PyTorch tensors by default
        self._tokenizer.model_max_length = 512  # Set reasonable default
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        return_tensors: Optional[str] = "mlx",
        **kwargs
    ) -> Union[List[int], mx.array]:
        """
        Encode a single text string.
        
        Args:
            text: Text to encode
            add_special_tokens: Whether to add special tokens
            return_tensors: Format of output ("mlx", "list", None)
            **kwargs: Additional tokenizer arguments
            
        Returns:
            Encoded text as list or MLX array
        """
        if self.backend == "mlx":
            result = self._mlx_adapter.encode_text(
                text,
                return_tensors="mlx",
                add_special_tokens=add_special_tokens,
                **kwargs
            )
            if return_tensors == "mlx":
                return result["input_ids"][0]  # Remove batch dimension
            else:
                return result["input_ids"][0].tolist()
        else:
            # HuggingFace backend
            encoded = self._tokenizer.encode(
                text,
                add_special_tokens=add_special_tokens,
                **kwargs
            )
            if return_tensors == "mlx":
                return mx.array(encoded)
            return encoded
    
    def batch_encode_plus(
        self,
        texts: List[str],
        padding: Union[bool, str] = True,
        truncation: bool = True,
        max_length: Optional[int] = 512,
        return_tensors: Optional[str] = "mlx",
        **kwargs
    ) -> Dict[str, mx.array]:
        """
        Encode multiple texts with padding and truncation.
        
        Args:
            texts: List of texts to encode
            padding: Padding strategy
            truncation: Whether to truncate
            max_length: Maximum sequence length
            return_tensors: Format of output tensors
            **kwargs: Additional tokenizer arguments
            
        Returns:
            Dictionary with input_ids, attention_mask, etc.
        """
        if self.backend == "mlx":
            return self._mlx_adapter.encode_text(
                texts,
                return_tensors=return_tensors,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                **kwargs
            )
        else:
            # HuggingFace backend - avoid PyTorch tensors completely
            try:
                # Try using numpy first
                encoded = self._tokenizer.batch_encode_plus(
                    texts,
                    padding=padding,
                    truncation=truncation,
                    max_length=max_length,
                    return_tensors="np",
                    **kwargs
                )
            except ImportError:
                # Fall back to Python lists if numpy isn't available
                encoded = self._tokenizer.batch_encode_plus(
                    texts,
                    padding=padding,
                    truncation=truncation,
                    max_length=max_length,
                    return_tensors=None,
                    **kwargs
                )
            
            # Convert to MLX if needed
            if return_tensors == "mlx":
                result = {}
                for key, value in encoded.items():
                    if hasattr(value, 'numpy'):
                        result[key] = mx.array(value.numpy())
                    elif isinstance(value, (list, tuple)):
                        result[key] = mx.array(value)
                    else:
                        result[key] = mx.array(value)
                return result
            
            return encoded
    
    def decode(
        self,
        token_ids: Union[List[int], mx.array],
        skip_special_tokens: bool = True,
        **kwargs
    ) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens
            **kwargs: Additional decoder arguments
            
        Returns:
            Decoded text string
        """
        # Convert MLX array to list if needed
        if isinstance(token_ids, mx.array):
            token_ids = token_ids.tolist()
        
        if self.backend == "mlx" and self._mlx_adapter.tokenizer:
            return self._mlx_adapter.tokenizer.decode(
                token_ids,
                skip_special_tokens=skip_special_tokens,
                **kwargs
            )
        else:
            return self._tokenizer.decode(
                token_ids,
                skip_special_tokens=skip_special_tokens,
                **kwargs
            )
    
    def batch_decode(
        self,
        sequences: Union[List[List[int]], mx.array],
        skip_special_tokens: bool = True,
        **kwargs
    ) -> List[str]:
        """
        Decode multiple sequences of token IDs.
        
        Args:
            sequences: List of token ID sequences
            skip_special_tokens: Whether to skip special tokens
            **kwargs: Additional decoder arguments
            
        Returns:
            List of decoded text strings
        """
        # Convert MLX array to list if needed
        if isinstance(sequences, mx.array):
            sequences = sequences.tolist()
        
        if self.backend == "mlx" and self._mlx_adapter.tokenizer:
            return self._mlx_adapter.tokenizer.batch_decode(
                sequences,
                skip_special_tokens=skip_special_tokens,
                **kwargs
            )
        else:
            return self._tokenizer.batch_decode(
                sequences,
                skip_special_tokens=skip_special_tokens,
                **kwargs
            )
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        if self.backend == "mlx":
            return self._mlx_adapter.vocab_size
        else:
            return len(self._tokenizer)
    
    @property
    def pad_token_id(self) -> int:
        """Get padding token ID."""
        if self.backend == "mlx" and self._mlx_adapter.tokenizer:
            return self._mlx_adapter.tokenizer.pad_token_id
        else:
            return self._tokenizer.pad_token_id
    
    @property
    def pad_token(self) -> str:
        """Get padding token."""
        if self.backend == "mlx" and self._mlx_adapter.tokenizer:
            return self._mlx_adapter.tokenizer.pad_token
        else:
            return self._tokenizer.pad_token
    
    @pad_token.setter
    def pad_token(self, value: str):
        """Set padding token."""
        if self.backend == "mlx" and self._mlx_adapter.tokenizer:
            self._mlx_adapter.tokenizer.pad_token = value
        else:
            self._tokenizer.pad_token = value
    
    @property
    def eos_token(self) -> Optional[str]:
        """Get end of sequence token."""
        if self.backend == "mlx" and self._mlx_adapter.tokenizer:
            return getattr(self._mlx_adapter.tokenizer, 'eos_token', None)
        else:
            return getattr(self._tokenizer, 'eos_token', None)
    
    @property
    def eos_token_id(self) -> Optional[int]:
        """Get end of sequence token ID."""
        if self.backend == "mlx" and self._mlx_adapter.tokenizer:
            return getattr(self._mlx_adapter.tokenizer, 'eos_token_id', None)
        else:
            return getattr(self._tokenizer, 'eos_token_id', None)
    
    @property
    def cls_token_id(self) -> Optional[int]:
        """Get CLS token ID."""
        if self.backend == "mlx" and self._mlx_adapter.tokenizer:
            return getattr(self._mlx_adapter.tokenizer, 'cls_token_id', None)
        else:
            return getattr(self._tokenizer, 'cls_token_id', None)
    
    @property
    def sep_token_id(self) -> Optional[int]:
        """Get SEP token ID."""
        if self.backend == "mlx" and self._mlx_adapter.tokenizer:
            return getattr(self._mlx_adapter.tokenizer, 'sep_token_id', None)
        else:
            return getattr(self._tokenizer, 'sep_token_id', None)
    
    @property
    def unk_token_id(self) -> Optional[int]:
        """Get UNK token ID."""
        if self.backend == "mlx" and self._mlx_adapter.tokenizer:
            return getattr(self._mlx_adapter.tokenizer, 'unk_token_id', None)
        else:
            return getattr(self._tokenizer, 'unk_token_id', None)
    
    @property
    def mask_token_id(self) -> Optional[int]:
        """Get mask token ID."""
        if self.backend == "mlx" and self._mlx_adapter.tokenizer:
            return getattr(self._mlx_adapter.tokenizer, 'mask_token_id', None)
        else:
            return getattr(self._tokenizer, 'mask_token_id', None)
    
    def __len__(self) -> int:
        """Get vocabulary size."""
        return self.vocab_size
    
    def __call__(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        padding: Union[bool, str] = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        **kwargs
    ) -> Union[Dict[str, mx.array], Dict[str, List]]:
        """
        Make the tokenizer callable like HuggingFace tokenizers.
        
        Args:
            text: Single text or list of texts
            add_special_tokens: Whether to add special tokens
            padding: Padding strategy
            truncation: Whether to truncate
            max_length: Maximum sequence length
            return_tensors: Format of output tensors
            **kwargs: Additional tokenizer arguments
            
        Returns:
            Dictionary with tokenized outputs
        """
        # Handle single text vs batch
        if isinstance(text, str):
            texts = [text]
            single_text = True
        else:
            texts = text
            single_text = False
        
        # Use batch_encode_plus for consistency
        result = self.batch_encode_plus(
            texts,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            **kwargs
        )
        
        # If single text input, remove batch dimension for some keys
        if single_text and return_tensors is None:
            # For non-tensor output, unwrap lists
            for key in ['input_ids', 'attention_mask', 'token_type_ids']:
                if key in result and isinstance(result[key], list):
                    result[key] = result[key][0] if result[key] else []
        
        return result
    
    def save_pretrained(self, save_directory: str):
        """Save tokenizer to directory."""
        if self.backend == "huggingface":
            self._tokenizer.save_pretrained(save_directory)
        else:
            logger.warning("Saving MLX tokenizer not yet implemented")
    
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        backend: str = "auto",
        **kwargs
    ) -> "TokenizerWrapper":
        """
        Load tokenizer from pretrained model.
        
        Args:
            model_name: Model name or path
            backend: Backend to use
            **kwargs: Additional arguments
            
        Returns:
            TokenizerWrapper instance
        """
        return cls(model_name=model_name, backend=backend, **kwargs)