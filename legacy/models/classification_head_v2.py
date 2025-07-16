"""
Enhanced classification head with support for custom loss functions.
Designed to handle class imbalance and prevent model collapse.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Dict, Union
from loguru import logger

from utils.loss_functions import (
    get_loss_function, 
    get_titanic_loss,
    AdaptiveLoss
)


class BinaryClassificationHeadV2(nn.Module):
    """Enhanced binary classification head with better regularization."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        dropout_prob: float = 0.1,
        use_layer_norm: bool = True,
        activation: str = 'gelu'
    ):
        super().__init__()
        
        self.use_layer_norm = use_layer_norm
        
        # Choose activation function
        activations = {
            'relu': nn.ReLU,
            'gelu': nn.GELU,
            'silu': nn.SiLU,
            'mish': nn.Mish
        }
        activation_fn = activations.get(activation, nn.GELU)
        
        if hidden_dim is None:
            # Simple linear classifier with optional layer norm
            layers = []
            if use_layer_norm:
                layers.append(nn.LayerNorm(input_dim))
            layers.extend([
                nn.Dropout(dropout_prob),
                nn.Linear(input_dim, 2)
            ])
            self.classifier = nn.Sequential(*layers)
        else:
            # Two-layer classifier with hidden dimension
            layers = []
            if use_layer_norm:
                layers.append(nn.LayerNorm(input_dim))
            layers.extend([
                nn.Dropout(dropout_prob),
                nn.Linear(input_dim, hidden_dim),
                activation_fn(),
                nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_dim, 2)
            ])
            self.classifier = nn.Sequential(*layers)
    
    def __call__(self, pooled_output: mx.array) -> mx.array:
        return self.classifier(pooled_output)


class TitanicClassifierV2(nn.Module):
    """
    Enhanced Titanic classifier with support for various loss functions
    to handle class imbalance and prevent model collapse.
    """
    
    def __init__(
        self,
        bert_model: nn.Module,
        hidden_dim: Optional[int] = None,
        dropout_prob: float = 0.1,
        freeze_bert: bool = False,
        loss_type: str = 'focal',
        loss_kwargs: Optional[Dict] = None,
        use_layer_norm: bool = True,
        activation: str = 'gelu'
    ):
        super().__init__()
        self.bert = bert_model
        self.classifier = BinaryClassificationHeadV2(
            input_dim=bert_model.config.hidden_size,
            hidden_dim=hidden_dim,
            dropout_prob=dropout_prob,
            use_layer_norm=use_layer_norm,
            activation=activation
        )
        
        # Loss function configuration
        self.loss_type = loss_type
        if loss_kwargs is None:
            # Use optimal settings for Titanic dataset
            self.loss_fn = get_titanic_loss(loss_type)
        else:
            self.loss_fn = get_loss_function(loss_type, **loss_kwargs)
        
        # Track if loss function is adaptive
        self.is_adaptive_loss = isinstance(self.loss_fn, AdaptiveLoss)
        
        # Optionally freeze BERT parameters
        if freeze_bert:
            logger.warning("Parameter freezing in MLX requires custom gradient handling")
        
        # Initialize metrics tracking
        self.metrics_history = {
            'loss': [],
            'focal_term': [],
            'ce_term': [],
            'confidence': []
        }
        
        logger.info(f"Initialized TitanicClassifierV2 with {loss_type} loss")
    
    def compute_loss_with_diagnostics(
        self,
        logits: mx.array,
        labels: mx.array
    ) -> Dict[str, mx.array]:
        """
        Compute loss with additional diagnostic information.
        Useful for monitoring training stability.
        """
        # Basic loss computation
        loss = self.loss_fn(logits, labels)
        
        # Compute additional diagnostics
        probs = mx.softmax(logits, axis=-1)
        
        # Get probability of true class
        batch_indices = mx.arange(labels.shape[0])
        pt = probs[batch_indices, labels]
        
        # Average confidence in predictions
        max_prob = mx.max(probs, axis=-1)
        avg_confidence = mx.mean(max_prob)
        
        # Entropy of predictions (uncertainty)
        entropy = -mx.sum(probs * mx.log(probs + 1e-8), axis=-1)
        avg_entropy = mx.mean(entropy)
        
        # Class-wise accuracy
        predictions = mx.argmax(logits, axis=-1)
        correct = predictions == labels
        
        # Separate accuracy for each class
        class_0_mask = labels == 0
        class_1_mask = labels == 1
        
        acc_class_0 = mx.sum(correct * class_0_mask) / (mx.sum(class_0_mask) + 1e-8)
        acc_class_1 = mx.sum(correct * class_1_mask) / (mx.sum(class_1_mask) + 1e-8)
        
        diagnostics = {
            'loss': loss,
            'avg_confidence': avg_confidence,
            'avg_entropy': avg_entropy,
            'avg_pt': mx.mean(pt),  # Average probability of true class
            'acc_class_0': acc_class_0,
            'acc_class_1': acc_class_1,
            'min_pt': mx.min(pt),  # Worst prediction
            'max_pt': mx.max(pt)   # Best prediction
        }
        
        return diagnostics
    
    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        labels: Optional[mx.array] = None,
        return_diagnostics: bool = False
    ) -> Dict[str, mx.array]:
        """
        Forward pass with optional diagnostic information.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            labels: Ground truth labels
            return_diagnostics: Whether to return detailed diagnostics
        
        Returns:
            Dictionary with logits, loss (if labels provided), and diagnostics
        """
        # Get BERT outputs
        bert_outputs = self.bert(input_ids, attention_mask)
        pooled_output = bert_outputs['pooled_output']
        
        # Classification
        logits = self.classifier(pooled_output)
        
        outputs = {'logits': logits}
        
        # Calculate loss if labels provided
        if labels is not None:
            # Ensure labels have the correct shape
            if labels.ndim == 0:  # Scalar label
                labels = labels.reshape(1)
            elif labels.ndim == 2:  # Already has batch dimension
                labels = labels.squeeze()
            
            # Ensure we have a batch dimension
            if logits.shape[0] != labels.shape[0]:
                logger.warning(f"Shape mismatch: logits {logits.shape} vs labels {labels.shape}")
            
            if return_diagnostics:
                # Get detailed diagnostics
                diagnostics = self.compute_loss_with_diagnostics(logits, labels)
                outputs.update(diagnostics)
            else:
                # Just compute loss
                loss = self.loss_fn(logits, labels)
                outputs['loss'] = loss
        
        return outputs
    
    def predict(self, input_ids: mx.array, attention_mask: Optional[mx.array] = None) -> mx.array:
        """Make predictions (class indices)."""
        outputs = self(input_ids, attention_mask)
        predictions = mx.argmax(outputs['logits'], axis=-1)
        return predictions
    
    def predict_proba(self, input_ids: mx.array, attention_mask: Optional[mx.array] = None) -> mx.array:
        """Get prediction probabilities."""
        outputs = self(input_ids, attention_mask)
        probabilities = mx.softmax(outputs['logits'], axis=-1)
        return probabilities
    
    def reset_adaptive_loss(self):
        """Reset adaptive loss counter if using adaptive loss."""
        if self.is_adaptive_loss:
            self.loss_fn.reset()
            logger.info("Reset adaptive loss counter")
    
    def update_loss_params(self, **kwargs):
        """
        Update loss function parameters dynamically.
        Useful for curriculum learning or dynamic adjustments.
        """
        if hasattr(self.loss_fn, '__dict__'):
            for key, value in kwargs.items():
                if hasattr(self.loss_fn, key):
                    setattr(self.loss_fn, key, value)
                    logger.info(f"Updated loss parameter {key} to {value}")
    
    def get_loss_info(self) -> Dict[str, Union[str, float]]:
        """Get information about the current loss function configuration."""
        info = {
            'loss_type': self.loss_type,
            'is_adaptive': self.is_adaptive_loss
        }
        
        if self.is_adaptive_loss:
            info['current_step'] = self.loss_fn.current_step
            progress = min(self.loss_fn.current_step / self.loss_fn.warmup_steps, 1.0)
            info['warmup_progress'] = progress
            info['current_alpha'] = (self.loss_fn.initial_alpha + 
                                   (self.loss_fn.final_alpha - self.loss_fn.initial_alpha) * progress)
            info['current_gamma'] = (self.loss_fn.initial_gamma + 
                                   (self.loss_fn.final_gamma - self.loss_fn.initial_gamma) * progress)
        
        return info


def create_enhanced_classifier(
    bert_model: nn.Module,
    loss_type: str = 'focal',
    hidden_dim: Optional[int] = 768,
    dropout_prob: float = 0.1,
    **kwargs
) -> TitanicClassifierV2:
    """
    Factory function to create an enhanced classifier with optimal settings.
    
    Args:
        bert_model: Pre-trained BERT model
        loss_type: Type of loss function ('focal', 'weighted_ce', etc.)
        hidden_dim: Hidden dimension for classification head
        dropout_prob: Dropout probability
        **kwargs: Additional arguments for the classifier
    
    Returns:
        Configured TitanicClassifierV2 instance
    """
    # Default configurations for different loss types
    default_configs = {
        'focal': {
            'dropout_prob': 0.1,
            'use_layer_norm': True,
            'activation': 'gelu'
        },
        'weighted_ce': {
            'dropout_prob': 0.2,
            'use_layer_norm': True,
            'activation': 'relu'
        },
        'focal_smooth': {
            'dropout_prob': 0.15,
            'use_layer_norm': True,
            'activation': 'gelu'
        },
        'adaptive': {
            'dropout_prob': 0.1,
            'use_layer_norm': True,
            'activation': 'gelu'
        }
    }
    
    # Get default config for loss type
    config = default_configs.get(loss_type, default_configs['focal'])
    
    # Override with provided kwargs
    config.update(kwargs)
    
    # Create classifier
    classifier = TitanicClassifierV2(
        bert_model=bert_model,
        hidden_dim=hidden_dim,
        dropout_prob=config.get('dropout_prob', dropout_prob),
        loss_type=loss_type,
        use_layer_norm=config.get('use_layer_norm', True),
        activation=config.get('activation', 'gelu'),
        **{k: v for k, v in config.items() 
           if k not in ['dropout_prob', 'use_layer_norm', 'activation']}
    )
    
    logger.info(f"Created enhanced classifier with {loss_type} loss and config: {config}")
    
    return classifier