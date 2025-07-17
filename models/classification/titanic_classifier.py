"""
Titanic Classifier

Task-specific classifier for the Titanic dataset that combines embedding models
with classification heads in a clean, separated architecture.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Dict, Optional, Union
from loguru import logger

from models.embeddings import EmbeddingModel
from .heads import BinaryClassificationHead


class TitanicClassifier(nn.Module):
    """
    Titanic classifier that combines embedding extraction and classification.
    
    This classifier uses the new separated architecture where:
    - Embedding extraction is handled by EmbeddingModel
    - Classification is handled by BinaryClassificationHead
    """
    
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        hidden_dim: Optional[int] = None,
        dropout_prob: float = 0.1,
        use_layer_norm: bool = False,
        activation: str = "relu",
        freeze_embeddings: bool = False,
    ):
        """
        Initialize Titanic classifier.
        
        Args:
            embedding_model: Pre-initialized embedding model
            hidden_dim: Hidden dimension for classification head
            dropout_prob: Dropout probability
            use_layer_norm: Whether to use layer normalization
            activation: Activation function name
            freeze_embeddings: Whether to freeze embedding model parameters
        """
        super().__init__()
        
        self.embedding_model = embedding_model
        self.freeze_embeddings = freeze_embeddings
        
        # Initialize classification head
        self.classification_head = BinaryClassificationHead(
            input_dim=embedding_model.hidden_size,
            hidden_dim=hidden_dim,
            dropout_prob=dropout_prob,
            use_layer_norm=use_layer_norm,
            activation=activation,
        )
        
        # Freeze embedding model if requested
        if freeze_embeddings:
            self.freeze_embedding_model()
        
        logger.info(f"Initialized TitanicClassifier with {embedding_model.model_name}")
    
    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        labels: Optional[mx.array] = None,
        return_embeddings: bool = False,
    ) -> Dict[str, mx.array]:
        """
        Forward pass through the classifier.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Ground truth labels (optional)
            return_embeddings: Whether to return embeddings
            
        Returns:
            Dictionary with logits, loss (if labels provided), and optionally embeddings
        """
        # Get embeddings from embedding model
        embeddings = self.embedding_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get classification logits
        logits = self.classification_head(embeddings)
        
        # Prepare outputs
        outputs = {"logits": logits}
        
        # Calculate loss if labels are provided
        if labels is not None:
            loss = mx.mean(nn.losses.cross_entropy(logits, labels, reduction="none"))
            outputs["loss"] = loss
        
        # Add embeddings if requested
        if return_embeddings:
            outputs["embeddings"] = embeddings
        
        return outputs
    
    def predict(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None
    ) -> mx.array:
        """
        Make predictions (class indices).
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Predicted class indices
        """
        outputs = self(input_ids, attention_mask)
        predictions = mx.argmax(outputs["logits"], axis=-1)
        return predictions
    
    def predict_proba(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None
    ) -> mx.array:
        """
        Get prediction probabilities.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Prediction probabilities
        """
        outputs = self(input_ids, attention_mask)
        probabilities = mx.softmax(outputs["logits"], axis=-1)
        return probabilities
    
    def freeze_embedding_model(self):
        """Freeze the embedding model parameters."""
        if hasattr(self.embedding_model, 'embedding_model') and self.embedding_model.embedding_model is not None:
            for param in self.embedding_model.embedding_model.parameters():
                param.freeze()
        self.freeze_embeddings = True
        logger.info("Froze embedding model parameters")
    
    def unfreeze_embedding_model(self):
        """Unfreeze the embedding model parameters."""
        if hasattr(self.embedding_model, 'embedding_model') and self.embedding_model.embedding_model is not None:
            for param in self.embedding_model.embedding_model.parameters():
                param.unfreeze()
        self.freeze_embeddings = False
        logger.info("Unfroze embedding model parameters")
    
    def save_weights(self, file_path: str):
        """Save model weights."""
        weights = {}
        
        # Save embedding model config
        weights["embedding_config"] = {
            "model_name": self.embedding_model.model_name,
            "pooling_strategy": self.embedding_model.pooling_strategy,
            "normalize_embeddings": self.embedding_model.normalize_embeddings,
            "hidden_size": self.embedding_model.hidden_size,
        }
        
        # Save classification head weights
        weights["classification_head"] = self.classification_head.state_dict()
        
        # Save classifier config
        weights["classifier_config"] = {
            "freeze_embeddings": self.freeze_embeddings,
        }
        
        mx.save_safetensors(file_path, weights)
        logger.info(f"Saved TitanicClassifier weights to {file_path}")
    
    def load_weights(self, file_path: str):
        """Load model weights."""
        weights = mx.load(file_path)
        
        # Load classification head weights
        if "classification_head" in weights:
            self.classification_head.load_state_dict(weights["classification_head"])
        
        # Load classifier config
        if "classifier_config" in weights:
            config = weights["classifier_config"]
            if config.get("freeze_embeddings", False):
                self.freeze_embedding_model()
        
        logger.info(f"Loaded TitanicClassifier weights from {file_path}")
    
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        **kwargs
    ) -> "TitanicClassifier":
        """
        Create a TitanicClassifier from a pretrained embedding model.
        
        Args:
            model_name: Name of the embedding model to load
            **kwargs: Additional arguments for the classifier
            
        Returns:
            TitanicClassifier instance
        """
        # Create embedding model
        embedding_model = EmbeddingModel.from_pretrained(model_name)
        
        # Create classifier
        return cls(
            embedding_model=embedding_model,
            **kwargs
        )
    
    @property
    def is_available(self) -> bool:
        """Check if the model is available."""
        return self.embedding_model.is_available