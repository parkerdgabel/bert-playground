"""
MLX Embedding Model Wrapper

Provides a wrapper around mlx-embeddings models that integrates with our
existing ModernBERT architecture and training pipeline.
"""

from typing import Dict, List, Optional, Tuple, Union
import mlx.core as mx
import mlx.nn as nn
from loguru import logger

from embeddings.mlx_adapter import MLXEmbeddingsAdapter
from models.classification import BinaryClassificationHead


class MLXEmbeddingModel(nn.Module):
    """
    Wrapper for mlx-embeddings models that provides compatibility with our training pipeline.
    
    This model can operate in two modes:
    1. Embedding mode: Uses pre-trained mlx-embeddings models for feature extraction
    2. Classification mode: Adds a classification head for fine-tuning
    """
    
    def __init__(
        self,
        model_name: str = "mlx-community/answerdotai-ModernBERT-base-4bit",
        num_labels: Optional[int] = None,
        hidden_size: int = 768,
        dropout_rate: float = 0.1,
        freeze_embeddings: bool = False,
        pooling_strategy: str = "mean",
        use_mlx_embeddings: bool = True,
    ):
        """
        Initialize MLX embedding model wrapper.
        
        Args:
            model_name: Name of the mlx-embeddings model
            num_labels: Number of classification labels (None for embedding-only mode)
            hidden_size: Hidden size of the model
            dropout_rate: Dropout rate for classification head
            freeze_embeddings: Whether to freeze embedding model parameters
            pooling_strategy: Pooling strategy for embeddings ("mean", "cls", "max")
            use_mlx_embeddings: Whether to use mlx-embeddings backend
        """
        super().__init__()
        
        self.model_name = model_name
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.freeze_embeddings = freeze_embeddings
        self.pooling_strategy = pooling_strategy
        self.use_mlx_embeddings = use_mlx_embeddings
        
        # Initialize MLX adapter
        self.adapter = MLXEmbeddingsAdapter(
            model_name=model_name,
            use_mlx_embeddings=use_mlx_embeddings
        )
        
        # Get model config
        config = self.adapter.get_model_config()
        if config:
            # Handle both dict and object configs
            if hasattr(config, "hidden_size"):
                self.hidden_size = config.hidden_size
            elif isinstance(config, dict):
                self.hidden_size = config.get("hidden_size", hidden_size)
            else:
                self.hidden_size = hidden_size
        
        # Initialize classification head if needed
        self.classification_head = None
        if num_labels is not None:
            self.classification_head = BinaryClassificationHead(
                input_dim=self.hidden_size,
                dropout_prob=dropout_rate  # Note: BinaryClassificationHead uses dropout_prob
            )
        
        # Store the embedding model reference
        self.embedding_model = self.adapter.model
    
    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        token_type_ids: Optional[mx.array] = None,
        labels: Optional[mx.array] = None,
        return_embeddings: bool = False,
        **kwargs
    ) -> Union[mx.array, Tuple[mx.array, mx.array]]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs (for BERT-style models)
            return_embeddings: Whether to return embeddings along with logits
            **kwargs: Additional model arguments
            
        Returns:
            Logits (if classification head) or embeddings
            Optionally returns tuple of (logits, embeddings) if return_embeddings=True
        """
        # Get embeddings from the base model
        if self.use_mlx_embeddings and self.embedding_model is not None:
            # Use the embedding model directly
            # MLX embeddings don't use token_type_ids
            outputs = self.embedding_model(
                input_ids,
                attention_mask=attention_mask,
                **kwargs
            )
            
            # Extract embeddings based on pooling strategy
            if hasattr(outputs, 'text_embeds'):
                embeddings = outputs.text_embeds
            else:
                embeddings = self._pool_embeddings(
                    outputs.last_hidden_state,
                    attention_mask
                )
        else:
            # Fallback: return random embeddings for testing
            batch_size = input_ids.shape[0]
            embeddings = mx.random.normal((batch_size, self.hidden_size))
        
        # Apply classification head if available
        if self.classification_head is not None:
            logits = self.classification_head(embeddings)
            
            # Return dictionary format expected by trainer
            outputs = {"logits": logits}
            
            # Calculate loss if labels are provided
            if labels is not None:
                import mlx.nn as nn
                loss = mx.mean(nn.losses.cross_entropy(logits, labels, reduction="none"))
                outputs["loss"] = loss
            
            if return_embeddings:
                outputs["embeddings"] = embeddings
                
            return outputs
        
        # Return embeddings for embedding-only mode
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
    
    def get_embeddings(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
        **kwargs
    ) -> mx.array:
        """
        Get embeddings for texts using the model.
        
        Args:
            texts: Single text or list of texts
            normalize: Whether to normalize embeddings
            **kwargs: Additional arguments for tokenization
            
        Returns:
            Embeddings as MLX array
        """
        if self.use_mlx_embeddings and self.adapter.is_available:
            return self.adapter.get_embeddings(
                texts,
                pooling=self.pooling_strategy,
                normalize=normalize,
                **kwargs
            )
        else:
            raise RuntimeError("MLX embeddings not available")
    
    def freeze_embedding_model(self):
        """Freeze the embedding model parameters."""
        if self.embedding_model is not None:
            for param in self.embedding_model.parameters():
                param.freeze()
        self.freeze_embeddings = True
    
    def unfreeze_embedding_model(self):
        """Unfreeze the embedding model parameters."""
        if self.embedding_model is not None:
            for param in self.embedding_model.parameters():
                param.unfreeze()
        self.freeze_embeddings = False
    
    def save_weights(self, file_path: str):
        """Save model weights."""
        weights = {}
        
        # Save classification head weights if available
        if self.classification_head is not None:
            weights["classification_head"] = self.classification_head.state_dict()
        
        # Save configuration
        weights["config"] = {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "hidden_size": self.hidden_size,
            "dropout_rate": self.dropout_rate,
            "pooling_strategy": self.pooling_strategy,
            "use_mlx_embeddings": self.use_mlx_embeddings,
        }
        
        mx.save_safetensors(file_path, weights)
        logger.info(f"Saved model weights to {file_path}")
    
    def load_weights(self, file_path: str):
        """Load model weights."""
        weights = mx.load(file_path)
        
        # Load classification head weights if available
        if "classification_head" in weights and self.classification_head is not None:
            self.classification_head.load_state_dict(weights["classification_head"])
        
        logger.info(f"Loaded model weights from {file_path}")
    
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        num_labels: Optional[int] = None,
        **kwargs
    ) -> "MLXEmbeddingModel":
        """
        Load a pretrained model.
        
        Args:
            model_name: Model name or path
            num_labels: Number of classification labels
            **kwargs: Additional model arguments
            
        Returns:
            MLXEmbeddingModel instance
        """
        return cls(
            model_name=model_name,
            num_labels=num_labels,
            **kwargs
        )