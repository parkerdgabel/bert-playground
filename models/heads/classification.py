"""Classification heads for BERT models.

This module provides various classification heads following the clean
architecture patterns from the BERT module.
"""

import mlx.core as mx
import mlx.nn as nn

from .base import BaseHead
from .config import ClassificationConfig, HeadConfig
from .utils.losses import (
    FocalLoss,
    LabelSmoothingLoss,
    cross_entropy_loss,
    multilabel_bce_loss,
)


class BinaryClassificationHead(BaseHead):
    """Binary classification head for 2-class problems.

    This head uses a single output with sigmoid activation for binary classification.
    """

    def __init__(self, config: ClassificationConfig | HeadConfig):
        """Initialize binary classification head.

        Args:
            config: Head configuration
        """
        # Ensure binary classification settings
        if hasattr(config, "num_classes"):
            config.num_classes = 2
        config.output_size = 2
        config.head_type = "binary_classification"

        super().__init__(config)

        # Initialize loss function
        self._init_loss()

    def _build_output_layer(self):
        """Build the output layer for binary classification."""
        # Two output units for standard softmax classification
        self.classifier = nn.Linear(
            self.projection_output_size, 2, bias=self.config.use_bias
        )

    def _init_loss(self):
        """Initialize loss function based on configuration."""
        if hasattr(self.config, "use_focal_loss") and self.config.use_focal_loss:
            self.loss_fn = FocalLoss(
                alpha=getattr(self.config, "focal_loss_alpha", None),
                gamma=getattr(self.config, "focal_loss_gamma", 2.0),
                num_classes=2,
            )
        else:
            self.loss_fn = None  # Use standard BCE

    def _forward_output(self, features: mx.array) -> dict[str, mx.array]:
        """Forward pass through the output layer.

        Args:
            features: Features after projection [batch_size, projection_output_size]

        Returns:
            Dictionary containing logits, probabilities, and predictions
        """
        # Get logits for both classes
        logits = self.classifier(features)  # [batch_size, 2]

        # Get probabilities using softmax
        probs = mx.softmax(logits, axis=-1)

        # Get predictions
        predictions = mx.argmax(logits, axis=-1)

        return {
            "logits": logits,  # [batch_size, 2]
            "probabilities": probs[
                :, 1
            ],  # [batch_size] - probability of positive class
            "probabilities_2class": probs,  # [batch_size, 2]
            "predictions": predictions,  # [batch_size]
        }

    def compute_loss(
        self, predictions: dict[str, mx.array], targets: mx.array, **kwargs
    ) -> mx.array:
        """Compute loss for binary classification.

        Args:
            predictions: Output from forward pass
            targets: Ground truth labels [batch_size]
            **kwargs: Additional arguments

        Returns:
            Loss value
        """
        logits = predictions["logits"]

        if self.loss_fn is not None:
            # Use focal loss
            return self.loss_fn(logits, targets)
        else:
            # Use standard cross-entropy loss for 2-class classification
            return cross_entropy_loss(logits, targets)


class MulticlassClassificationHead(BaseHead):
    """Multiclass classification head for N-class problems.

    This head uses softmax activation for multiclass classification.
    """

    def __init__(self, config: ClassificationConfig | HeadConfig):
        """Initialize multiclass classification head.

        Args:
            config: Head configuration
        """
        config.head_type = "multiclass_classification"
        super().__init__(config)

        # Initialize loss function
        self._init_loss()

    def _build_output_layer(self):
        """Build the output layer for multiclass classification."""
        self.classifier = nn.Linear(
            self.projection_output_size,
            self.config.output_size,
            bias=self.config.use_bias,
        )

    def _init_loss(self):
        """Initialize loss function based on configuration."""
        if hasattr(self.config, "label_smoothing") and self.config.label_smoothing > 0:
            self.loss_fn = LabelSmoothingLoss(
                smoothing=self.config.label_smoothing,
                num_classes=self.config.output_size,
            )
        elif hasattr(self.config, "use_focal_loss") and self.config.use_focal_loss:
            self.loss_fn = FocalLoss(
                alpha=getattr(self.config, "focal_loss_alpha", None),
                gamma=getattr(self.config, "focal_loss_gamma", 2.0),
                num_classes=self.config.output_size,
            )
        else:
            self.loss_fn = None  # Use standard cross-entropy

    def _forward_output(self, features: mx.array) -> dict[str, mx.array]:
        """Forward pass through the output layer.

        Args:
            features: Features after projection [batch_size, projection_output_size]

        Returns:
            Dictionary containing logits, probabilities, and predictions
        """
        # Get logits
        logits = self.classifier(features)  # [batch_size, num_classes]

        # Get probabilities
        probs = mx.softmax(logits, axis=-1)

        # Predictions
        predictions = mx.argmax(logits, axis=-1)

        return {
            "logits": logits,  # [batch_size, num_classes]
            "probabilities": probs,  # [batch_size, num_classes]
            "predictions": predictions,  # [batch_size]
        }

    def compute_loss(
        self, predictions: dict[str, mx.array], targets: mx.array, **kwargs
    ) -> mx.array:
        """Compute loss for multiclass classification.

        Args:
            predictions: Output from forward pass
            targets: Ground truth labels [batch_size]
            **kwargs: Additional arguments

        Returns:
            Loss value
        """
        logits = predictions["logits"]

        if self.loss_fn is not None:
            # Use configured loss function
            return self.loss_fn(logits, targets)
        else:
            # Use standard cross-entropy
            return cross_entropy_loss(
                logits, targets, num_classes=self.config.output_size
            )


class MultilabelClassificationHead(BaseHead):
    """Multilabel classification head.

    This head treats each label as an independent binary classification problem.
    """

    def __init__(self, config: ClassificationConfig | HeadConfig):
        """Initialize multilabel classification head.

        Args:
            config: Head configuration
        """
        config.head_type = "multilabel_classification"
        super().__init__(config)

        # Initialize thresholds
        self.thresholds = mx.ones(self.config.output_size) * 0.5

        # Initialize loss function
        self._init_loss()

    def _build_output_layer(self):
        """Build the output layer for multilabel classification."""
        self.classifier = nn.Linear(
            self.projection_output_size,
            self.config.output_size,
            bias=self.config.use_bias,
        )

    def _init_loss(self):
        """Initialize loss function based on configuration."""
        if hasattr(self.config, "use_focal_loss") and self.config.use_focal_loss:
            self.loss_fn = FocalLoss(
                alpha=getattr(self.config, "focal_loss_alpha", None),
                gamma=getattr(self.config, "focal_loss_gamma", 2.0),
                num_classes=None,  # Binary mode for each label
            )
        else:
            self.loss_fn = None  # Use standard multilabel BCE

    def _forward_output(self, features: mx.array) -> dict[str, mx.array]:
        """Forward pass through the output layer.

        Args:
            features: Features after projection [batch_size, projection_output_size]

        Returns:
            Dictionary containing logits, probabilities, and predictions
        """
        # Get logits
        logits = self.classifier(features)  # [batch_size, num_labels]

        # Get probabilities (independent sigmoid per label)
        probs = mx.sigmoid(logits)

        # Binary predictions per label
        predictions = (probs > self.thresholds).astype(mx.int32)

        return {
            "logits": logits,  # [batch_size, num_labels]
            "probabilities": probs,  # [batch_size, num_labels]
            "predictions": predictions,  # [batch_size, num_labels]
            "thresholds": self.thresholds,  # [num_labels]
        }

    def compute_loss(
        self, predictions: dict[str, mx.array], targets: mx.array, **kwargs
    ) -> mx.array:
        """Compute loss for multilabel classification.

        Args:
            predictions: Output from forward pass
            targets: Ground truth labels [batch_size, num_labels]
            **kwargs: Additional arguments

        Returns:
            Loss value
        """
        logits = predictions["logits"]

        if self.loss_fn is not None:
            # Apply focal loss per label and average
            total_loss = 0
            for i in range(logits.shape[-1]):
                label_logits = logits[:, i : i + 1]
                label_targets = targets[:, i]
                total_loss = total_loss + self.loss_fn(label_logits, label_targets)
            return total_loss / logits.shape[-1]
        else:
            # Use standard multilabel BCE
            return multilabel_bce_loss(logits, targets)

    def update_thresholds(self, new_thresholds: mx.array):
        """Update the decision thresholds for each label.

        Args:
            new_thresholds: New threshold values [num_labels]
        """
        self.thresholds = new_thresholds


# Factory functions


def create_classification_head(
    config: ClassificationConfig | HeadConfig, head_type: str | None = None
) -> BaseHead:
    """Create a classification head based on configuration.

    Args:
        config: Head configuration
        head_type: Optional override for head type

    Returns:
        Classification head instance

    Raises:
        ValueError: If head type is unknown
    """
    # Determine head type
    if head_type is None:
        head_type = config.head_type

    # Create appropriate head
    if head_type == "binary_classification":
        return BinaryClassificationHead(config)
    elif head_type == "multiclass_classification":
        return MulticlassClassificationHead(config)
    elif head_type == "multilabel_classification":
        return MultilabelClassificationHead(config)
    else:
        raise ValueError(f"Unknown classification head type: {head_type}")


def create_binary_classification_head(
    input_size: int, **kwargs
) -> BinaryClassificationHead:
    """Create a binary classification head with sensible defaults.

    Args:
        input_size: Size of input features
        **kwargs: Additional configuration parameters

    Returns:
        Binary classification head
    """
    from .config import get_binary_classification_config

    config = get_binary_classification_config(input_size)

    # Update with any provided kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return BinaryClassificationHead(config)


def create_multiclass_classification_head(
    input_size: int, num_classes: int, **kwargs
) -> MulticlassClassificationHead:
    """Create a multiclass classification head with sensible defaults.

    Args:
        input_size: Size of input features
        num_classes: Number of output classes
        **kwargs: Additional configuration parameters

    Returns:
        Multiclass classification head
    """
    from .config import get_multiclass_classification_config

    config = get_multiclass_classification_config(input_size, num_classes)

    # Update with any provided kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return MulticlassClassificationHead(config)


def create_multilabel_classification_head(
    input_size: int, num_labels: int, **kwargs
) -> MultilabelClassificationHead:
    """Create a multilabel classification head with sensible defaults.

    Args:
        input_size: Size of input features
        num_labels: Number of output labels
        **kwargs: Additional configuration parameters

    Returns:
        Multilabel classification head
    """
    from .config import get_multilabel_classification_config

    config = get_multilabel_classification_config(input_size, num_labels)

    # Update with any provided kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return MultilabelClassificationHead(config)


__all__ = [
    # Head classes
    "BinaryClassificationHead",
    "MulticlassClassificationHead",
    "MultilabelClassificationHead",
    # Factory functions
    "create_classification_head",
    "create_binary_classification_head",
    "create_multiclass_classification_head",
    "create_multilabel_classification_head",
]
