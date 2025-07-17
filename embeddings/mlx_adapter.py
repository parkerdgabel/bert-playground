"""
MLX Embeddings Adapter

Provides an adapter layer between our codebase and the mlx-embeddings library.
Handles model loading, tokenization, and embedding generation.
"""

from typing import Dict, List, Optional, Tuple, Union
import mlx.core as mx
from loguru import logger

try:
    from mlx_embeddings.utils import load as mlx_load
    MLX_EMBEDDINGS_AVAILABLE = True
except ImportError:
    MLX_EMBEDDINGS_AVAILABLE = False
    logger.warning("mlx-embeddings not available. Install with: pip install mlx-embeddings")


class MLXEmbeddingsAdapter:
    """Adapter for mlx-embeddings library integration."""
    
    def __init__(
        self,
        model_name: str = "mlx-community/answerdotai-ModernBERT-base-4bit",
        use_mlx_embeddings: bool = True,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the MLX embeddings adapter.
        
        Args:
            model_name: Name of the model to load (HuggingFace or mlx-community)
            use_mlx_embeddings: Whether to use mlx-embeddings library
            cache_dir: Directory for caching models
        """
        self.model_name = model_name
        self.use_mlx_embeddings = use_mlx_embeddings and MLX_EMBEDDINGS_AVAILABLE
        self.cache_dir = cache_dir
        
        self.model = None
        self.tokenizer = None
        
        if self.use_mlx_embeddings:
            self._load_mlx_embeddings()
        else:
            logger.info("MLX embeddings disabled, using fallback implementation")
    
    def _load_mlx_embeddings(self):
        """Load model and tokenizer using mlx-embeddings."""
        try:
            # Convert HuggingFace model names to mlx-community if available
            mlx_model_name = self._convert_to_mlx_model_name(self.model_name)
            
            logger.info(f"Loading MLX embeddings model: {mlx_model_name}")
            self.model, self.tokenizer = mlx_load(mlx_model_name)
            
            logger.info("Successfully loaded MLX embeddings model")
        except Exception as e:
            logger.error(f"Failed to load MLX embeddings: {e}")
            self.use_mlx_embeddings = False
    
    def _convert_to_mlx_model_name(self, model_name: str) -> str:
        """Convert HuggingFace model names to mlx-community equivalents."""
        # Mapping of common HuggingFace models to mlx-community versions
        model_mapping = {
            "answerdotai/ModernBERT-base": "mlx-community/answerdotai-ModernBERT-base-4bit",
            "answerdotai/ModernBERT-large": "mlx-community/answerdotai-ModernBERT-large-4bit",
            "bert-base-uncased": "mlx-community/bert-base-uncased-4bit",
            "sentence-transformers/all-MiniLM-L6-v2": "mlx-community/all-MiniLM-L6-v2-4bit",
        }
        
        # Return mapped name if available, otherwise return original
        return model_mapping.get(model_name, model_name)
    
    def encode_text(
        self,
        texts: Union[str, List[str]],
        return_tensors: str = "mlx",
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 512,
        **kwargs
    ) -> Dict[str, mx.array]:
        """
        Encode text using the mlx-embeddings tokenizer.
        
        Args:
            texts: Single text or list of texts to encode
            return_tensors: Format of output tensors ("mlx" or "np")
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            max_length: Maximum sequence length
            **kwargs: Additional tokenizer arguments
            
        Returns:
            Dictionary with input_ids, attention_mask, etc.
        """
        if not self.use_mlx_embeddings or self.tokenizer is None:
            raise RuntimeError("MLX embeddings not available or not loaded")
        
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]
        
        # Use batch encoding if available
        if hasattr(self.tokenizer, 'batch_encode_plus'):
            encoded = self.tokenizer.batch_encode_plus(
                texts,
                return_tensors="pt",  # mlx-embeddings uses PyTorch format internally
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                **kwargs
            )
        else:
            # Fallback to single encoding
            encoded = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                **kwargs
            )
        
        # Convert to MLX arrays
        result = {}
        for key, value in encoded.items():
            if hasattr(value, 'numpy'):
                result[key] = mx.array(value.numpy())
            else:
                result[key] = mx.array(value)
        
        return result
    
    def get_embeddings(
        self,
        texts: Union[str, List[str]],
        pooling: str = "mean",
        normalize: bool = True,
        **kwargs
    ) -> mx.array:
        """
        Get embeddings for texts using mlx-embeddings model.
        
        Args:
            texts: Single text or list of texts
            pooling: Pooling strategy ("mean", "cls", "max")
            normalize: Whether to normalize embeddings
            **kwargs: Additional model arguments
            
        Returns:
            Embeddings as MLX array
        """
        if not self.use_mlx_embeddings or self.model is None:
            raise RuntimeError("MLX embeddings model not available")
        
        # Encode texts
        inputs = self.encode_text(texts, **kwargs)
        
        # Get model outputs
        outputs = self.model(
            inputs["input_ids"],
            attention_mask=inputs.get("attention_mask")
        )
        
        # Extract embeddings based on pooling strategy
        if hasattr(outputs, 'text_embeds'):
            # Use pre-computed text embeddings if available
            embeddings = outputs.text_embeds
        elif pooling == "cls":
            embeddings = outputs.last_hidden_state[:, 0, :]
        elif pooling == "mean":
            # Mean pooling with attention mask
            mask = inputs.get("attention_mask", mx.ones_like(inputs["input_ids"]))
            mask = mask.astype(mx.float32)
            mask_expanded = mx.expand_dims(mask, -1)
            
            sum_embeddings = mx.sum(outputs.last_hidden_state * mask_expanded, axis=1)
            sum_mask = mx.sum(mask, axis=1, keepdims=True)
            embeddings = sum_embeddings / mx.maximum(sum_mask, 1e-9)
        elif pooling == "max":
            embeddings = mx.max(outputs.last_hidden_state, axis=1)
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling}")
        
        # Normalize if requested
        if normalize:
            embeddings = embeddings / mx.maximum(
                mx.linalg.norm(embeddings, axis=-1, keepdims=True), 1e-9
            )
        
        return embeddings
    
    def get_model_config(self) -> Dict:
        """Get the model configuration."""
        if self.model is None:
            return {}
        
        if hasattr(self.model, 'config'):
            return self.model.config
        return {}
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size from tokenizer."""
        if self.tokenizer is None:
            return 0
        return len(self.tokenizer)
    
    @property
    def is_available(self) -> bool:
        """Check if mlx-embeddings is available and loaded."""
        return self.use_mlx_embeddings and self.model is not None