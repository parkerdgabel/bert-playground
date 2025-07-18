"""
Base classifier class for all BERT-based classification models.
Provides common functionality and interfaces for task-specific classifiers.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Literal
import mlx.core as mx
import mlx.nn as nn
from loguru import logger
import numpy as np
from models.embeddings.embedding_model import EmbeddingModel as BertEmbeddingModel
import mlx.core as mx


PoolingType = Literal["mean", "max", "cls", "attention", "weighted", "learned"]
ActivationType = Literal["relu", "gelu", "silu", "mish", "tanh", "swish"]


class BaseClassifier(nn.Module, ABC):
    """Abstract base class for BERT-based classifiers."""
    
    def __init__(
        self,
        embedding_model: BertEmbeddingModel,
        num_classes: int,
        pooling_type: PoolingType = "mean",
        hidden_dim: Optional[int] = None,
        activation: ActivationType = "gelu",
        dropout_rate: float = 0.1,
        use_layer_norm: bool = True,
        use_batch_norm: bool = False,
        freeze_embeddings: bool = False,
        head_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize base classifier.
        
        Args:
            embedding_model: Pre-trained BERT embedding model
            num_classes: Number of output classes
            pooling_type: Type of pooling to use
            hidden_dim: Hidden dimension for classification head
            activation: Activation function type
            dropout_rate: Dropout rate
            use_layer_norm: Whether to use layer normalization
            use_batch_norm: Whether to use batch normalization
            freeze_embeddings: Whether to freeze embedding weights
            head_config: Additional configuration for classification head
        """
        super().__init__()
        
        self.embedding_model = embedding_model
        self.num_classes = num_classes
        self.pooling_type = pooling_type
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.use_batch_norm = use_batch_norm
        self.head_config = head_config or {}
        
        # Get embedding dimension
        if hasattr(embedding_model, 'config') and hasattr(embedding_model.config, 'hidden_size'):
            self.embedding_dim = embedding_model.config.hidden_size
        elif hasattr(embedding_model, 'hidden_size'):
            self.embedding_dim = embedding_model.hidden_size
        else:
            # Try to get from the model itself
            self.embedding_dim = embedding_model.get_hidden_size() if hasattr(embedding_model, 'get_hidden_size') else 768
        
        # Initialize pooling layer
        self.pooling_layer = self._create_pooling_layer()
        
        # Initialize classification head
        self.classification_head = self._create_classification_head()
        
        # Freeze embeddings if requested
        if freeze_embeddings:
            self.freeze_embeddings()
    
    def _create_pooling_layer(self) -> nn.Module:
        """Create pooling layer based on pooling type."""
        if self.pooling_type == "attention":
            return AttentionPooling(self.embedding_dim)
        elif self.pooling_type == "weighted":
            return WeightedPooling(self.embedding_dim)
        elif self.pooling_type == "learned":
            return LearnedPooling(self.embedding_dim)
        else:
            # For mean, max, cls, we'll handle in forward pass
            return nn.Identity()
    
    @abstractmethod
    def _create_classification_head(self) -> nn.Module:
        """Create task-specific classification head."""
        pass
    
    def pool_embeddings(
        self,
        embeddings: mx.array,
        attention_mask: Optional[mx.array] = None
    ) -> mx.array:
        """
        Pool sequence embeddings into a fixed-size representation.
        
        Args:
            embeddings: Sequence embeddings [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Pooled embeddings [batch_size, hidden_dim]
        """
        if self.pooling_type in ["attention", "weighted", "learned"]:
            return self.pooling_layer(embeddings, attention_mask)
        elif self.pooling_type == "mean":
            if attention_mask is not None:
                mask_expanded = mx.expand_dims(attention_mask, axis=-1)
                sum_embeddings = mx.sum(embeddings * mask_expanded, axis=1)
                sum_mask = mx.sum(mask_expanded, axis=1)
                return sum_embeddings / mx.maximum(sum_mask, 1e-9)
            else:
                return mx.mean(embeddings, axis=1)
        elif self.pooling_type == "max":
            if attention_mask is not None:
                mask_expanded = mx.expand_dims(attention_mask, axis=-1)
                # Set padding tokens to large negative value
                masked_embeddings = mx.where(
                    mask_expanded > 0,
                    embeddings,
                    mx.full_like(embeddings, -1e9)
                )
                return mx.max(masked_embeddings, axis=1)
            else:
                return mx.max(embeddings, axis=1)
        elif self.pooling_type == "cls":
            return embeddings[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")
    
    def forward(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        return_embeddings: bool = False
    ) -> Union[mx.array, Tuple[mx.array, mx.array]]:
        """
        Forward pass through the classifier.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            return_embeddings: Whether to return embeddings along with logits
            
        Returns:
            Logits or (logits, pooled_embeddings) if return_embeddings=True
        """
        # Get embeddings
        embeddings = self.embedding_model(input_ids, attention_mask)
        
        # Pool embeddings
        pooled = self.pool_embeddings(embeddings, attention_mask)
        
        # Pass through classification head
        logits = self.classification_head(pooled)
        
        if return_embeddings:
            return logits, pooled
        return logits
    
    @abstractmethod
    def compute_loss(
        self,
        logits: mx.array,
        labels: mx.array,
        class_weights: Optional[mx.array] = None,
        **kwargs
    ) -> mx.array:
        """Compute task-specific loss."""
        pass
    
    def freeze_embeddings(self):
        """Freeze embedding model parameters."""
        self.embedding_model.freeze()
        logger.info("Froze embedding model parameters")
    
    def unfreeze_embeddings(self):
        """Unfreeze embedding model parameters."""
        self.embedding_model.unfreeze()
        logger.info("Unfroze embedding model parameters")
    
    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        # Simple approach: count parameters recursively
        def count_params(module):
            total = 0
            if hasattr(module, 'weight') and module.weight is not None:
                w_shape = module.weight.shape if hasattr(module.weight, 'shape') else ()
                total += mx.prod(mx.array(w_shape)).item() if w_shape else 0
            if hasattr(module, 'bias') and module.bias is not None:
                b_shape = module.bias.shape if hasattr(module.bias, 'shape') else ()
                total += mx.prod(mx.array(b_shape)).item() if b_shape else 0
            
            # Recursively count in child modules
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, nn.Module) and attr is not module:
                    total += count_params(attr)
            
            return total
        
        # Count classification head parameters
        head_params = count_params(self.classification_head)
        
        # Count embedding parameters if not frozen (simplified - just report 0 for now)
        embed_params = 0 if self.embedding_model._freeze else 0
        
        return head_params + embed_params
    
    def save_weights(self, save_path: Path):
        """Save model weights."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save classifier weights
        classifier_path = save_path / "classifier.safetensors"
        mx.save_safetensors(
            classifier_path,
            {"classification_head": self.classification_head.state_dict()}
        )
        
        # Save embedding weights if not frozen
        if not self.embedding_model._freeze:
            embedding_path = save_path / "embeddings.safetensors"
            mx.save_safetensors(
                embedding_path,
                {"embedding_model": self.embedding_model.state_dict()}
            )
        
        # Save configuration
        config = {
            "num_classes": self.num_classes,
            "pooling_type": self.pooling_type,
            "hidden_dim": self.hidden_dim,
            "dropout_rate": self.dropout_rate,
            "use_layer_norm": self.use_layer_norm,
            "use_batch_norm": self.use_batch_norm,
            "head_config": self.head_config,
            "embedding_frozen": self.embedding_model._freeze,
        }
        
        import json
        config_path = save_path / "classifier_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved classifier weights to {save_path}")
    
    def load_weights(self, load_path: Path):
        """Load model weights."""
        load_path = Path(load_path)
        
        # Load classifier weights
        classifier_path = load_path / "classifier.safetensors"
        if classifier_path.exists():
            weights = mx.load(str(classifier_path))
            self.classification_head.load_state_dict(weights["classification_head"])
            logger.info("Loaded classifier weights")
        
        # Load embedding weights if they exist
        embedding_path = load_path / "embeddings.safetensors"
        if embedding_path.exists():
            weights = mx.load(str(embedding_path))
            self.embedding_model.load_state_dict(weights["embedding_model"])
            logger.info("Loaded embedding weights")
    
    @abstractmethod
    def predict(self, input_ids: mx.array, attention_mask: Optional[mx.array] = None) -> mx.array:
        """Make predictions for the given inputs."""
        pass
    
    @abstractmethod
    def predict_proba(self, input_ids: mx.array, attention_mask: Optional[mx.array] = None) -> mx.array:
        """Predict class probabilities."""
        pass


class AttentionPooling(nn.Module):
    """Attention-based pooling layer."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention_weights = nn.Linear(hidden_size, 1)
    
    def __call__(self, embeddings: mx.array, attention_mask: Optional[mx.array] = None) -> mx.array:
        # embeddings shape: [batch_size, seq_len, hidden_size]
        # Compute attention scores: [batch_size, seq_len, 1]
        scores = self.attention_weights(embeddings)
        # Squeeze to [batch_size, seq_len]
        scores = scores.squeeze(-1)
        
        # Apply attention mask
        if attention_mask is not None:
            # attention_mask shape: [batch_size, seq_len]
            scores = mx.where(attention_mask > 0, scores, -1e9)
        
        # Compute attention weights
        weights = mx.softmax(scores, axis=1)
        # Expand for broadcasting: [batch_size, seq_len, 1]
        weights = mx.expand_dims(weights, axis=-1)
        
        # Weighted sum: [batch_size, hidden_size]
        pooled = mx.sum(embeddings * weights, axis=1)
        return pooled


class WeightedPooling(nn.Module):
    """Learned weighted pooling across sequence positions."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.position_weights = nn.Linear(hidden_size, hidden_size)
    
    def __call__(self, embeddings: mx.array, attention_mask: Optional[mx.array] = None) -> mx.array:
        # Apply position-specific weights
        weighted = self.position_weights(embeddings)
        
        # Apply mask and average
        if attention_mask is not None:
            mask_expanded = mx.expand_dims(attention_mask, axis=-1)
            sum_embeddings = mx.sum(weighted * mask_expanded, axis=1)
            sum_mask = mx.sum(mask_expanded, axis=1)
            return sum_embeddings / mx.maximum(sum_mask, 1e-9)
        else:
            return mx.mean(weighted, axis=1)


class LearnedPooling(nn.Module):
    """Fully learned pooling with a pooling token."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.pooling_token = mx.zeros((1, 1, hidden_size))
        self.pooling_attention = nn.MultiHeadAttention(
            hidden_size, num_heads=8, bias=True
        )
    
    def __call__(self, embeddings: mx.array, attention_mask: Optional[mx.array] = None) -> mx.array:
        batch_size = embeddings.shape[0]
        
        # Expand pooling token for batch
        pooling_tokens = mx.broadcast_to(
            self.pooling_token, (batch_size, 1, embeddings.shape[-1])
        )
        
        # Apply attention between pooling token and sequence
        pooled, _ = self.pooling_attention(
            pooling_tokens, embeddings, embeddings,
            mask=attention_mask
        )
        
        return pooled.squeeze(1)