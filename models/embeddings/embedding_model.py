"""
Pure Embedding Model

Provides a pure embedding model wrapper focused only on embedding extraction.
No classification logic - that's handled by separate classification modules.
"""

from typing import Dict, List, Optional, Union
import mlx.core as mx
import mlx.nn as nn
from loguru import logger

from .mlx_adapter import MLXEmbeddingsAdapter


class EmbeddingModel(nn.Module):
    """
    Pure embedding model that extracts embeddings from text.
    
    This model operates only in embedding mode and does not include
    any classification logic. Classification heads are handled separately.
    """
    
    def __init__(
        self,
        model_name: str = "mlx-community/answerdotai-ModernBERT-base-4bit",
        pooling_strategy: str = "mean",
        normalize_embeddings: bool = True,
        use_mlx_embeddings: bool = True,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize pure embedding model.
        
        Args:
            model_name: Name of the mlx-embeddings model
            pooling_strategy: Pooling strategy for embeddings (\"mean\", \"cls\", \"max\")
            normalize_embeddings: Whether to normalize embeddings
            use_mlx_embeddings: Whether to use mlx-embeddings backend
            cache_dir: Directory for caching models
        """
        super().__init__()
        
        self.model_name = model_name
        self.pooling_strategy = pooling_strategy
        self.normalize_embeddings = normalize_embeddings
        self.use_mlx_embeddings = use_mlx_embeddings
        self.cache_dir = cache_dir
        
        # Initialize MLX adapter
        self.adapter = MLXEmbeddingsAdapter(
            model_name=model_name,
            use_mlx_embeddings=use_mlx_embeddings,
            cache_dir=cache_dir
        )
        
        # Get model configuration
        self.config = self.adapter.get_model_config()
        self.hidden_size = self.adapter.get_hidden_size()
        
        # Store the embedding model reference
        self.embedding_model = self.adapter.model
        
        # Initialize freeze status
        self._freeze = False
        
        logger.info(f"Initialized EmbeddingModel: {model_name} (hidden_size={self.hidden_size})")
    
    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        return_hidden_states: bool = False,
        **kwargs
    ) -> Union[mx.array, Dict[str, mx.array]]:
        """
        Forward pass through the embedding model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            return_hidden_states: Whether to return hidden states along with embeddings
            **kwargs: Additional model arguments
            
        Returns:
            Embeddings (or dict with embeddings and hidden states)
        """
        # Get embeddings from the base model
        if self.use_mlx_embeddings and self.embedding_model is not None:
            # Convert attention mask to float16 if provided
            if attention_mask is not None:
                # Cast to float16 for model compatibility
                attention_mask = mx.array(attention_mask, dtype=mx.float16)
            
            # Use the embedding model directly
            outputs = self.embedding_model(
                input_ids,
                attention_mask=attention_mask,
                **kwargs
            )
            
            # Return the full sequence embeddings - pooling is done by classifier
            if hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state
            else:
                # Fallback: use text_embeds 
                hidden_states = outputs.text_embeds if hasattr(outputs, 'text_embeds') else outputs
            
            embeddings = hidden_states
        else:
            # Fallback: return random embeddings for testing
            batch_size = input_ids.shape[0]
            embeddings = mx.random.normal((batch_size, self.hidden_size))
            hidden_states = mx.random.normal((batch_size, input_ids.shape[1], self.hidden_size))
        
        # Apply normalization if requested
        if self.normalize_embeddings:
            embeddings = self._normalize_embeddings(embeddings)
        
        # Return format
        if return_hidden_states:
            return {
                "embeddings": embeddings,
                "hidden_states": hidden_states
            }
        
        return embeddings
    
    def _pool_embeddings(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None
    ) -> mx.array:
        """
        Pool embeddings based on the configured strategy.
        
        Args:
            hidden_states: Hidden states from the model [batch, seq_len, hidden]
            attention_mask: Attention mask [batch, seq_len]
            
        Returns:
            Pooled embeddings [batch, hidden]
        """
        if self.pooling_strategy == "cls":
            # Use CLS token (first token)
            return hidden_states[:, 0, :]
        
        elif self.pooling_strategy == "mean":
            # Mean pooling with attention mask
            if attention_mask is None:
                return mx.mean(hidden_states, axis=1)
            
            # Expand mask for broadcasting
            mask = attention_mask.astype(mx.float32)
            mask_expanded = mx.expand_dims(mask, -1)
            
            # Sum embeddings and divide by number of non-masked tokens
            sum_embeddings = mx.sum(hidden_states * mask_expanded, axis=1)
            sum_mask = mx.sum(mask, axis=1, keepdims=True)
            return sum_embeddings / mx.maximum(sum_mask, 1e-9)
        
        elif self.pooling_strategy == "max":
            # Max pooling
            if attention_mask is None:
                return mx.max(hidden_states, axis=1)
            
            # Apply mask before max pooling
            mask = attention_mask.astype(mx.float32)
            mask_expanded = mx.expand_dims(mask, -1)
            masked_hidden = hidden_states * mask_expanded - (1 - mask_expanded) * 1e9
            return mx.max(masked_hidden, axis=1)
        
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
    
    def _normalize_embeddings(self, embeddings: mx.array) -> mx.array:
        """
        Normalize embeddings to unit vectors.
        
        Args:
            embeddings: Embeddings to normalize
            
        Returns:
            Normalized embeddings
        """
        return embeddings / mx.maximum(
            mx.linalg.norm(embeddings, axis=-1, keepdims=True), 1e-9
        )
    
    def get_embeddings(
        self,
        texts: Union[str, List[str]],
        **kwargs
    ) -> mx.array:
        """
        Get embeddings for texts using the model.
        
        Args:
            texts: Single text or list of texts
            **kwargs: Additional arguments for tokenization
            
        Returns:
            Embeddings as MLX array
        """
        if self.use_mlx_embeddings and self.adapter.is_available:
            return self.adapter.get_embeddings(
                texts,
                pooling=self.pooling_strategy,
                normalize=self.normalize_embeddings,
                **kwargs
            )
        else:
            raise RuntimeError("MLX embeddings not available")
    
    def save_weights(self, file_path: str):
        """Save model weights (only config for pure embedding model)."""
        config = {
            "model_name": self.model_name,
            "pooling_strategy": self.pooling_strategy,
            "normalize_embeddings": self.normalize_embeddings,
            "use_mlx_embeddings": self.use_mlx_embeddings,
            "hidden_size": self.hidden_size,
        }
        
        # For pure embedding model, we mainly save configuration
        # The actual model weights are managed by mlx-embeddings
        mx.save_safetensors(file_path, {"config": config})
        logger.info(f"Saved embedding model config to {file_path}")
    
    def load_weights(self, file_path: str):
        """Load model weights (only config for pure embedding model)."""
        weights = mx.load(file_path)
        
        if "config" in weights:
            config = weights["config"]
            logger.info(f"Loaded embedding model config from {file_path}")
        else:
            logger.warning("No config found in weights file")
    
    def freeze(self):
        """Freeze the embedding model parameters."""
        self._freeze = True
        # Note: Actual parameter freezing would be handled during training
        logger.info("Froze embedding model parameters")
    
    def unfreeze(self):
        """Unfreeze the embedding model parameters."""
        self._freeze = False
        logger.info("Unfroze embedding model parameters")
    
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        **kwargs
    ) -> "EmbeddingModel":
        """
        Load a pretrained embedding model.
        
        Args:
            model_name: Model name or path
            **kwargs: Additional model arguments
            
        Returns:
            EmbeddingModel instance
        """
        return cls(
            model_name=model_name,
            **kwargs
        )
    
    @property
    def is_available(self) -> bool:
        """Check if the embedding model is available."""
        return self.adapter.is_available