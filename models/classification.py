"""Unified classification heads for ModernBERT models with advanced loss functions support."""

import json
import mlx.core as mx
import mlx.nn as nn
from pathlib import Path
from typing import Optional, Dict, Union
from loguru import logger
import warnings


class BinaryClassificationHead(nn.Module):
    """Universal binary classification head with configurable architecture."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        dropout_prob: float = 0.1,
        use_layer_norm: bool = False,
        activation: str = "relu",
    ):
        super().__init__()

        self.use_layer_norm = use_layer_norm

        # Choose activation function
        activations = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "mish": nn.Mish,
        }
        activation_fn = activations.get(activation, nn.ReLU)

        if hidden_dim is None:
            # Simple linear classifier
            layers = []
            if use_layer_norm:
                layers.append(nn.LayerNorm(input_dim))
            layers.extend([nn.Dropout(dropout_prob), nn.Linear(input_dim, 2)])
            self.classifier = nn.Sequential(*layers)
        else:
            # Two-layer classifier with hidden dimension
            layers = []
            if use_layer_norm:
                layers.append(nn.LayerNorm(input_dim))
            layers.extend(
                [
                    nn.Dropout(dropout_prob),
                    nn.Linear(input_dim, hidden_dim),
                    activation_fn(),
                    nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
                    nn.Dropout(dropout_prob),
                    nn.Linear(hidden_dim, 2),
                ]
            )
            self.classifier = nn.Sequential(*layers)

    def __call__(self, pooled_output: mx.array) -> mx.array:
        return self.classifier(pooled_output)


class TitanicClassifier(nn.Module):
    """
    Unified Titanic classifier supporting both basic and advanced loss functions.
    Combines features from both V1 and V2 implementations.
    """

    def __init__(
        self,
        bert_model: nn.Module,
        hidden_dim: Optional[int] = None,
        dropout_prob: float = 0.1,
        freeze_bert: bool = False,
        loss_type: str = "cross_entropy",
        loss_kwargs: Optional[Dict] = None,
        use_layer_norm: bool = False,
        activation: str = "relu",
        enable_diagnostics: bool = False,
    ):
        super().__init__()
        self.bert = bert_model

        # Check if the model already has a classifier (CNN hybrid case)
        self.has_built_in_classifier = hasattr(bert_model, "classifier")

        if not self.has_built_in_classifier:
            # Get the actual output dimension from the model
            # Try multiple ways to get hidden size for compatibility
            output_dim = None
            
            # Try output_hidden_size first (for CNN hybrid models)
            if hasattr(bert_model, "output_hidden_size"):
                output_dim = bert_model.output_hidden_size
            # Try config.hidden_size (for standard models)
            elif hasattr(bert_model, "config") and hasattr(bert_model.config, "hidden_size"):
                output_dim = bert_model.config.hidden_size
            # Try hidden_size directly (fallback)
            elif hasattr(bert_model, "hidden_size"):
                output_dim = bert_model.hidden_size
            # Default to 768 (standard ModernBERT base size)
            else:
                output_dim = 768
                logger.warning(f"Could not determine hidden size for {type(bert_model).__name__}, using default 768")

            self.classifier = BinaryClassificationHead(
                input_dim=output_dim,
                hidden_dim=hidden_dim,
                dropout_prob=dropout_prob,
                use_layer_norm=use_layer_norm,
                activation=activation,
            )
        else:
            # CNN hybrid model already has classifier
            self.classifier = None

        # Loss function configuration
        self.loss_type = loss_type
        self.enable_diagnostics = enable_diagnostics

        # Set up loss function
        if loss_type == "cross_entropy":
            self.loss_fn = self._cross_entropy_loss
        else:
            # Import advanced loss functions only if needed
            try:
                from utils.loss_functions import (
                    get_loss_function,
                    get_titanic_loss,
                    AdaptiveLoss,
                )

                if loss_kwargs is None and loss_type in [
                    "focal",
                    "weighted_ce",
                    "focal_smooth",
                    "adaptive",
                ]:
                    self.loss_fn = get_titanic_loss(loss_type)
                else:
                    self.loss_fn = get_loss_function(loss_type, **(loss_kwargs or {}))
                self.is_adaptive_loss = isinstance(self.loss_fn, AdaptiveLoss)
            except ImportError:
                logger.warning(
                    "Advanced loss functions not available, falling back to cross_entropy"
                )
                self.loss_fn = self._cross_entropy_loss
                self.is_adaptive_loss = False

        # Optionally freeze BERT parameters
        if freeze_bert:
            logger.warning(
                "Parameter freezing in MLX requires custom gradient handling"
            )

        # Initialize metrics tracking if diagnostics enabled
        if enable_diagnostics:
            self.metrics_history = {
                "loss": [],
                "confidence": [],
                "entropy": [],
                "acc_class_0": [],
                "acc_class_1": [],
            }

        logger.info(f"Initialized TitanicClassifier with {loss_type} loss")

    def _cross_entropy_loss(self, logits: mx.array, labels: mx.array) -> mx.array:
        """Standard cross-entropy loss."""
        return mx.mean(nn.losses.cross_entropy(logits, labels, reduction="none"))

    def compute_loss_with_diagnostics(
        self, logits: mx.array, labels: mx.array
    ) -> Dict[str, mx.array]:
        """Compute loss with additional diagnostic information."""
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
            "loss": loss,
            "avg_confidence": avg_confidence,
            "avg_entropy": avg_entropy,
            "avg_pt": mx.mean(pt),
            "acc_class_0": acc_class_0,
            "acc_class_1": acc_class_1,
            "min_pt": mx.min(pt),
            "max_pt": mx.max(pt),
        }

        return diagnostics

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        labels: Optional[mx.array] = None,
        return_diagnostics: bool = None,
    ) -> Dict[str, mx.array]:
        """
        Forward pass with optional diagnostic information.

        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            labels: Ground truth labels
            return_diagnostics: Whether to return detailed diagnostics (overrides self.enable_diagnostics)

        Returns:
            Dictionary with logits, loss (if labels provided), and diagnostics
        """
        # Determine if we should return diagnostics
        if return_diagnostics is None:
            return_diagnostics = self.enable_diagnostics

        # Handle CNN hybrid case
        if self.has_built_in_classifier:
            # CNN hybrid model already produces logits and loss
            bert_outputs = self.bert(input_ids, attention_mask, labels=labels)
            return bert_outputs

        # Standard BERT model - need to add classification
        bert_outputs = self.bert(input_ids, attention_mask)

        # Handle different output formats
        if isinstance(bert_outputs, dict):
            pooled_output = bert_outputs.get(
                "pooled_output", bert_outputs.get("pooler_output")
            )
        else:
            # Some models might return tuple
            pooled_output = (
                bert_outputs[1] if isinstance(bert_outputs, tuple) else bert_outputs
            )

        # Classification
        logits = self.classifier(pooled_output)

        outputs = {"logits": logits}

        # Calculate loss if labels provided
        if labels is not None:
            # Ensure labels have the correct shape
            if labels.ndim == 0:  # Scalar label
                labels = labels.reshape(1)
            elif labels.ndim == 2:  # Has batch dimension
                labels = labels.squeeze()

            # Ensure we have a batch dimension
            if logits.shape[0] != labels.shape[0]:
                logger.warning(
                    f"Shape mismatch: logits {logits.shape} vs labels {labels.shape}"
                )

            if return_diagnostics:
                # Get detailed diagnostics
                diagnostics = self.compute_loss_with_diagnostics(logits, labels)
                outputs.update(diagnostics)
            else:
                # Just compute loss
                loss = self.loss_fn(logits, labels)
                outputs["loss"] = loss

        return outputs

    def predict(
        self, input_ids: mx.array, attention_mask: Optional[mx.array] = None
    ) -> mx.array:
        """Make predictions (class indices)."""
        outputs = self(input_ids, attention_mask)
        predictions = mx.argmax(outputs["logits"], axis=-1)
        return predictions

    def predict_proba(
        self, input_ids: mx.array, attention_mask: Optional[mx.array] = None
    ) -> mx.array:
        """Get prediction probabilities."""
        outputs = self(input_ids, attention_mask)
        probabilities = mx.softmax(outputs["logits"], axis=-1)
        return probabilities

    # Advanced methods (only available with advanced loss functions)
    def reset_adaptive_loss(self):
        """Reset adaptive loss counter if using adaptive loss."""
        if hasattr(self, "is_adaptive_loss") and self.is_adaptive_loss:
            self.loss_fn.reset()
            logger.info("Reset adaptive loss counter")

    def update_loss_params(self, **kwargs):
        """Update loss function parameters dynamically."""
        if hasattr(self.loss_fn, "__dict__"):
            for key, value in kwargs.items():
                if hasattr(self.loss_fn, key):
                    setattr(self.loss_fn, key, value)
                    logger.info(f"Updated loss parameter {key} to {value}")

    def get_loss_info(self) -> Dict[str, Union[str, float]]:
        """Get information about the current loss function configuration."""
        info = {
            "loss_type": self.loss_type,
            "is_adaptive": getattr(self, "is_adaptive_loss", False),
        }

        if getattr(self, "is_adaptive_loss", False):
            info["current_step"] = self.loss_fn.current_step
            progress = min(self.loss_fn.current_step / self.loss_fn.warmup_steps, 1.0)
            info["warmup_progress"] = progress

        return info

    def save_pretrained(self, save_path: Union[str, Path]) -> None:
        """
        Save the model to a directory, including both the BERT model and classification head.
        
        Args:
            save_path: Directory path to save the model
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_dict = {
            "bert_model_type": self.bert.__class__.__name__,
            "has_built_in_classifier": self.has_built_in_classifier,
            "loss_type": self.loss_type,
            "enable_diagnostics": self.enable_diagnostics,
        }
        
        # Save classifier config if we have one
        if not self.has_built_in_classifier and self.classifier:
            classifier_config = {
                "hidden_dim": None,  # Will be determined from the layers
                "dropout_prob": 0.1,
                "use_layer_norm": self.classifier.use_layer_norm,
                "activation": "relu",  # Default, could be extracted from layers
            }
            
            # Try to extract hidden_dim from classifier layers
            if hasattr(self.classifier, "classifier") and isinstance(
                self.classifier.classifier, nn.Sequential
            ):
                for layer in self.classifier.classifier:
                    if isinstance(layer, nn.Linear) and layer.dims[1] != 2:
                        classifier_config["hidden_dim"] = layer.dims[1]
                        break
            
            config_dict["classifier_config"] = classifier_config
        
        # Save BERT config if available
        if hasattr(self.bert, "config"):
            config_dict["bert_config"] = (
                self.bert.config.to_dict()
                if hasattr(self.bert.config, "to_dict")
                else self.bert.config.__dict__
            )
        
        with open(save_path / "titanic_classifier_config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        
        # Save the BERT model
        if hasattr(self.bert, "save_pretrained"):
            bert_save_path = save_path / "bert_model"
            self.bert.save_pretrained(bert_save_path)
        else:
            # Fallback: save BERT weights directly
            from mlx.utils import tree_flatten
            
            bert_save_path = save_path / "bert_model"
            bert_save_path.mkdir(exist_ok=True)
            
            flat_params = tree_flatten(self.bert.parameters())
            bert_weights = {k: v for k, v in flat_params}
            mx.save_safetensors(str(bert_save_path / "model.safetensors"), bert_weights)
            
            # Save BERT config if available
            if hasattr(self.bert, "config"):
                with open(bert_save_path / "config.json", "w") as f:
                    bert_config = (
                        self.bert.config.to_dict()
                        if hasattr(self.bert.config, "to_dict")
                        else self.bert.config.__dict__
                    )
                    json.dump(bert_config, f, indent=2)
        
        # Save classifier weights if we have a separate classifier
        if not self.has_built_in_classifier and self.classifier:
            from mlx.utils import tree_flatten
            
            flat_params = tree_flatten(self.classifier.parameters())
            classifier_weights = {k: v for k, v in flat_params}
            mx.save_safetensors(
                str(save_path / "classifier.safetensors"), classifier_weights
            )
        
        logger.info(f"TitanicClassifier saved to {save_path}")

    @classmethod
    def from_pretrained(
        cls, load_path: Union[str, Path], **kwargs
    ) -> "TitanicClassifier":
        """
        Load a TitanicClassifier from a saved directory.
        
        Args:
            load_path: Directory path containing the saved model
            **kwargs: Additional arguments to override loaded configuration
            
        Returns:
            Loaded TitanicClassifier instance
        """
        load_path = Path(load_path)
        
        # Load configuration
        config_path = load_path / "titanic_classifier_config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found at {config_path}"
            )
        
        with open(config_path) as f:
            config_dict = json.load(f)
        
        # Load BERT model
        bert_path = load_path / "bert_model"
        if bert_path.exists():
            # Try to load using the BERT model's from_pretrained if available
            bert_model_type = config_dict.get("bert_model_type", "ModernBertModel")
            
            # Import the appropriate model class
            if bert_model_type == "CNNEnhancedModernBERT":
                from models.modernbert_cnn_hybrid import CNNEnhancedModernBERT
                bert_model = CNNEnhancedModernBERT.from_pretrained(bert_path)
            elif bert_model_type == "ModernBertModel":
                from models.modernbert import ModernBertModel
                bert_model = ModernBertModel.from_pretrained(bert_path)
            else:
                # Fallback: try to load weights directly
                from models.factory import create_model
                
                bert_model = create_model("standard")
                if (bert_path / "model.safetensors").exists():
                    weights = mx.load(str(bert_path / "model.safetensors"))
                    bert_model.load_weights(list(weights.items()))
        else:
            raise FileNotFoundError(f"BERT model not found at {bert_path}")
        
        # Create the classifier with loaded configuration
        classifier_kwargs = {
            "loss_type": config_dict.get("loss_type", "cross_entropy"),
            "enable_diagnostics": config_dict.get("enable_diagnostics", False),
        }
        
        # Add classifier config if available
        if "classifier_config" in config_dict:
            classifier_kwargs.update(config_dict["classifier_config"])
        
        # Override with any provided kwargs
        classifier_kwargs.update(kwargs)
        
        # Create the TitanicClassifier instance
        model = cls(bert_model=bert_model, **classifier_kwargs)
        
        # Load classifier weights if we have a separate classifier
        if not model.has_built_in_classifier and model.classifier:
            classifier_path = load_path / "classifier.safetensors"
            if classifier_path.exists():
                weights = mx.load(str(classifier_path))
                model.classifier.load_weights(list(weights.items()))
        
        logger.info(f"TitanicClassifier loaded from {load_path}")
        return model

    def load_pretrained(self, load_path: Union[str, Path]) -> None:
        """
        Load pretrained weights into the current model instance.
        This is for compatibility with the test that expects an instance method.
        
        Args:
            load_path: Directory path containing the saved model
        """
        # Load a new model and copy its state
        loaded_model = self.from_pretrained(load_path)
        
        # Copy the loaded model's parameters to this instance
        self.bert = loaded_model.bert
        self.classifier = loaded_model.classifier
        self.has_built_in_classifier = loaded_model.has_built_in_classifier
        self.loss_type = loaded_model.loss_type
        self.loss_fn = loaded_model.loss_fn
        self.enable_diagnostics = loaded_model.enable_diagnostics
        if hasattr(loaded_model, "is_adaptive_loss"):
            self.is_adaptive_loss = loaded_model.is_adaptive_loss
        
        logger.info(f"Loaded pretrained weights from {load_path}")


# Factory functions
def create_classifier(
    bert_model: nn.Module,
    loss_type: str = "cross_entropy",
    hidden_dim: Optional[int] = None,
    dropout_prob: float = 0.1,
    **kwargs,
) -> TitanicClassifier:
    """
    Factory function to create a classifier with optimal settings.

    Args:
        bert_model: Pre-trained BERT model
        loss_type: Type of loss function ('cross_entropy', 'focal', 'weighted_ce', etc.)
        hidden_dim: Hidden dimension for classification head
        dropout_prob: Dropout probability
        **kwargs: Additional arguments for the classifier

    Returns:
        Configured TitanicClassifier instance
    """
    # Default configurations for different loss types
    default_configs = {
        "cross_entropy": {
            "dropout_prob": 0.1,
            "use_layer_norm": False,
            "activation": "relu",
        },
        "focal": {
            "dropout_prob": 0.1,
            "use_layer_norm": True,
            "activation": "gelu",
            "enable_diagnostics": True,
        },
        "weighted_ce": {
            "dropout_prob": 0.2,
            "use_layer_norm": True,
            "activation": "relu",
            "enable_diagnostics": True,
        },
        "adaptive": {
            "dropout_prob": 0.1,
            "use_layer_norm": True,
            "activation": "gelu",
            "enable_diagnostics": True,
        },
    }

    # Get default config for loss type
    config = default_configs.get(loss_type, default_configs["cross_entropy"])

    # Override with provided kwargs
    config.update(kwargs)

    # Create classifier
    classifier = TitanicClassifier(
        bert_model=bert_model,
        hidden_dim=hidden_dim,
        dropout_prob=config.get("dropout_prob", dropout_prob),
        loss_type=loss_type,
        use_layer_norm=config.get("use_layer_norm", False),
        activation=config.get("activation", "relu"),
        enable_diagnostics=config.get("enable_diagnostics", False),
        **{
            k: v
            for k, v in config.items()
            if k
            not in [
                "dropout_prob",
                "use_layer_norm",
                "activation",
                "enable_diagnostics",
            ]
        },
    )

    return classifier


# Backward compatibility aliases
BinaryClassificationHeadV2 = BinaryClassificationHead
TitanicClassifierV2 = TitanicClassifier
create_enhanced_classifier = create_classifier


# Deprecation warnings for old names
def __getattr__(name):
    if name in [
        "BinaryClassificationHeadV2",
        "TitanicClassifierV2",
        "create_enhanced_classifier",
    ]:
        warnings.warn(
            f"{name} is deprecated. Use the new unified names instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
