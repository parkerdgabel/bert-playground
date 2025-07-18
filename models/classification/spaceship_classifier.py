"""
Spaceship Titanic Classifier

Binary classification model for predicting whether passengers were transported
to an alternate dimension.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import mlx.core as mx
import mlx.nn as nn
from loguru import logger

from models.embeddings import EmbeddingModel
from models.classification.generic_classifier import GenericClassifier
from models.classification.factory import create_classifier


class SpaceshipTitanicClassifier(GenericClassifier):
    """
    Specialized classifier for Spaceship Titanic competition.
    
    Inherits from GenericClassifier with optimized settings for this task.
    """
    
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        hidden_dim: int = 256,
        dropout_rate: float = 0.2,
        pooling_type: str = "attention",
        activation: str = "gelu",
        use_layer_norm: bool = True,
        freeze_embeddings: bool = False,
        label_smoothing: float = 0.1,
    ):
        """
        Initialize Spaceship Titanic classifier.
        
        Args:
            embedding_model: Pre-trained embedding model
            hidden_dim: Hidden dimension for classification head
            dropout_rate: Dropout rate
            pooling_type: Pooling strategy
            activation: Activation function
            use_layer_norm: Whether to use layer normalization
            freeze_embeddings: Whether to freeze embeddings
            label_smoothing: Label smoothing factor
        """
        # Initialize as binary classifier
        super().__init__(
            embedding_model=embedding_model,
            num_classes=2,  # Binary: transported or not
            task_type="binary",
            pooling_type=pooling_type,
            hidden_dims=hidden_dim,
            activation=activation,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm,
            freeze_embeddings=freeze_embeddings,
            label_smoothing=label_smoothing,
            class_names=["Not Transported", "Transported"],
        )
        
        logger.info(
            f"Initialized SpaceshipTitanicClassifier with "
            f"hidden_dim={hidden_dim}, pooling={pooling_type}, "
            f"dropout={dropout_rate}"
        )
    
    def predict_transported(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        threshold: float = 0.5
    ) -> mx.array:
        """
        Predict whether passengers were transported.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            threshold: Classification threshold
            
        Returns:
            Boolean array of predictions
        """
        probs = self.predict_proba(input_ids, attention_mask)
        # Get probability of being transported (class 1)
        transported_probs = probs[:, 1]
        return transported_probs > threshold
    
    def get_transported_probabilities(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None
    ) -> mx.array:
        """
        Get probabilities of being transported.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Array of transported probabilities
        """
        probs = self.predict_proba(input_ids, attention_mask)
        return probs[:, 1]  # Probability of class 1 (Transported)


def create_spaceship_classifier(
    model_name: str = "mlx-community/answerdotai-ModernBERT-base-4bit",
    hidden_dim: int = 256,
    dropout_rate: float = 0.2,
    pooling_type: str = "attention",
    activation: str = "gelu",
    use_layer_norm: bool = True,
    freeze_embeddings: bool = False,
    label_smoothing: float = 0.1,
    **kwargs
) -> SpaceshipTitanicClassifier:
    """
    Factory function to create Spaceship Titanic classifier.
    
    Args:
        model_name: Name of embedding model
        hidden_dim: Hidden dimension
        dropout_rate: Dropout rate
        pooling_type: Pooling strategy
        activation: Activation function
        use_layer_norm: Whether to use layer norm
        freeze_embeddings: Whether to freeze embeddings
        label_smoothing: Label smoothing
        **kwargs: Additional arguments
        
    Returns:
        Configured SpaceshipTitanicClassifier
    """
    # Create embedding model
    embedding_model = EmbeddingModel(model_name=model_name)
    
    # Create classifier
    classifier = SpaceshipTitanicClassifier(
        embedding_model=embedding_model,
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate,
        pooling_type=pooling_type,
        activation=activation,
        use_layer_norm=use_layer_norm,
        freeze_embeddings=freeze_embeddings,
        label_smoothing=label_smoothing,
    )
    
    return classifier


def create_ensemble_spaceship_classifier(
    model_name: str = "mlx-community/answerdotai-ModernBERT-base-4bit",
    num_heads: int = 5,
    ensemble_method: str = "attention",
    base_hidden_dim: int = 256,
    **kwargs
) -> GenericClassifier:
    """
    Create an ensemble classifier for Spaceship Titanic.
    
    Uses multiple classification heads with different architectures
    for improved robustness.
    
    Args:
        model_name: Name of embedding model
        num_heads: Number of ensemble heads
        ensemble_method: How to combine predictions
        base_hidden_dim: Base hidden dimension
        **kwargs: Additional arguments
        
    Returns:
        Ensemble classifier
    """
    # Different configurations for each head
    hidden_dims = [
        base_hidden_dim,
        int(base_hidden_dim * 1.5),
        base_hidden_dim * 2,
        int(base_hidden_dim * 0.75),
        base_hidden_dim
    ][:num_heads]
    
    activations = ["gelu", "relu", "silu", "mish", "gelu"][:num_heads]
    dropout_rates = [0.1, 0.2, 0.3, 0.15, 0.25][:num_heads]
    
    # Create ensemble
    return create_classifier(
        task_type="ensemble",
        model_name=model_name,
        num_classes=2,
        pooling_type="mean",
        head_config={
            "num_heads": num_heads,
            "ensemble_method": ensemble_method,
            "hidden_dims": hidden_dims,
            "activations": activations,
            "dropout_rates": dropout_rates,
            "temperature": 1.0,
        },
        class_names=["Not Transported", "Transported"],
        **kwargs
    )


def create_advanced_spaceship_classifier(
    model_name: str = "mlx-community/answerdotai-ModernBERT-base-4bit",
    use_auxiliary_tasks: bool = True,
    **kwargs
) -> GenericClassifier:
    """
    Create an advanced classifier with auxiliary tasks.
    
    Uses multi-task learning with auxiliary objectives to improve
    the main transported prediction task.
    
    Args:
        model_name: Name of embedding model
        use_auxiliary_tasks: Whether to add auxiliary tasks
        **kwargs: Additional arguments
        
    Returns:
        Advanced classifier with auxiliary heads
    """
    auxiliary_heads = None
    
    if use_auxiliary_tasks:
        auxiliary_heads = {
            # Predict if passenger used cryosleep (highly correlated)
            "cryosleep": {
                "task_type": "binary",
                "hidden_dim": 128,
                "activation": "relu",
                "dropout_rate": 0.1,
            },
            # Predict spending level (luxury vs economy)
            "spending_level": {
                "task_type": "multiclass",
                "num_classes": 3,  # Low, Medium, High
                "hidden_dim": 192,
                "activation": "gelu",
                "dropout_rate": 0.15,
            },
            # Predict home planet (demographic info)
            "home_planet": {
                "task_type": "multiclass",
                "num_classes": 3,  # Earth, Europa, Mars
                "hidden_dim": 128,
                "activation": "silu",
                "dropout_rate": 0.2,
            }
        }
    
    return create_classifier(
        task_type="binary",
        model_name=model_name,
        num_classes=2,
        hidden_dim=384,
        pooling_type="attention",
        activation="gelu",
        dropout_rate=0.2,
        use_layer_norm=True,
        label_smoothing=0.1,
        auxiliary_heads=auxiliary_heads,
        class_names=["Not Transported", "Transported"],
        **kwargs
    )


if __name__ == "__main__":
    # Test the classifiers
    import mlx.core as mx
    
    # Create different classifier variants
    print("Creating classifiers...")
    
    # Standard classifier
    standard = create_spaceship_classifier()
    print(f"Standard classifier parameters: {standard.get_num_trainable_params():,}")
    
    # Ensemble classifier
    ensemble = create_ensemble_spaceship_classifier(num_heads=3)
    print(f"Ensemble classifier parameters: {ensemble.get_num_trainable_params():,}")
    
    # Advanced classifier
    advanced = create_advanced_spaceship_classifier()
    print(f"Advanced classifier parameters: {advanced.get_num_trainable_params():,}")
    
    # Test forward pass
    batch_size = 4
    seq_len = 128
    dummy_input = mx.random.randint(0, 1000, (batch_size, seq_len))
    dummy_mask = mx.ones((batch_size, seq_len))
    
    print("\nTesting forward pass...")
    output = standard(dummy_input, dummy_mask)
    print(f"Standard output shape: {output.shape}")
    
    probs = standard.get_transported_probabilities(dummy_input, dummy_mask)
    print(f"Transported probabilities: {probs}")