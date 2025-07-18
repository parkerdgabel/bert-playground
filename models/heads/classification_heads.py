"""Classification heads for BERT models in Kaggle competitions.

This module implements binary, multiclass, and multilabel classification heads
with competition-specific optimizations and loss functions.
"""

from typing import Dict, Optional, List, Any
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .base_head import BaseKaggleHead, HeadType, HeadConfig, ActivationType
from .head_registry import register_head_class, CompetitionType
from loguru import logger


@register_head_class(
    name="binary_classification",
    head_type=HeadType.BINARY_CLASSIFICATION,
    competition_types=[CompetitionType.BINARY_CLASSIFICATION],
    priority=10,
    description="Binary classification head with sigmoid activation and BCE loss"
)
class BinaryClassificationHead(BaseKaggleHead):
    """Binary classification head for 2-class problems.
    
    Features:
    - Sigmoid activation for probability output
    - Binary cross-entropy loss
    - Support for class weights and focal loss
    - Competition-specific optimizations (AUC, F1, etc.)
    """
    
    def __init__(self, config: HeadConfig):
        """Initialize binary classification head.
        
        Args:
            config: Head configuration
        """
        # Ensure output size is correct for binary classification
        if config.output_size != 2:
            config.output_size = 2
        
        # Competition-specific parameters (must be set before super().__init__())
        self.use_focal_loss = config.use_competition_tricks
        self.focal_alpha = 0.25
        self.focal_gamma = 2.0
        self.class_weights = None
            
        super().__init__(config)
        
    def _build_output_layer(self):
        """Build the output layer for binary classification."""
        # Single output with sigmoid activation
        self.classifier = nn.Linear(self.projection_output_size, 1)
        
        # Optional temperature scaling for calibration
        if self.config.use_competition_tricks:
            self.temperature = mx.ones(1)  # Will be learnable parameter
        else:
            self.temperature = None
    
    def _build_loss_function(self):
        """Build loss function for binary classification."""
        # Standard binary cross-entropy (custom implementation)
        self.bce_loss = self._binary_cross_entropy_loss
        
        # Focal loss for imbalanced datasets
        if self.use_focal_loss:
            self.focal_loss = FocalLoss(alpha=self.focal_alpha, gamma=self.focal_gamma)
    
    def forward(self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None, **kwargs) -> Dict[str, mx.array]:
        """Forward pass through binary classification head.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask [batch_size, seq_len]
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with predictions and probabilities
        """
        # Apply pooling
        pooled_output = self._apply_pooling(hidden_states, attention_mask)
        
        # Apply projection layers
        projected = self.projection(pooled_output)
        
        # Get logits
        logits = self.classifier(projected)  # [batch_size, 1]
        
        # Apply temperature scaling if enabled
        if self.temperature is not None:
            logits = logits / self.temperature
        
        # Get probabilities
        probs = mx.sigmoid(logits)
        
        # Create binary predictions
        predictions = (probs > 0.5).astype(mx.int32)
        
        # Also provide 2-class probabilities for compatibility
        probs_2class = mx.concatenate([1 - probs, probs], axis=-1)
        
        return {
            "logits": logits,
            "probabilities": probs,
            "probabilities_2class": probs_2class,
            "predictions": predictions,
        }
    
    def compute_loss(self, predictions: Dict[str, mx.array], targets: mx.array, **kwargs) -> mx.array:
        """Compute loss for binary classification.
        
        Args:
            predictions: Predictions from forward pass
            targets: Ground truth targets [batch_size] (0 or 1)
            **kwargs: Additional arguments
            
        Returns:
            Loss value
        """
        logits = predictions["logits"].squeeze(-1)  # [batch_size]
        targets = targets.astype(mx.float32)
        
        if self.use_focal_loss and hasattr(self, 'focal_loss'):
            loss = self.focal_loss(logits, targets)
        else:
            loss = self.bce_loss(logits, targets)
        
        return loss
    
    def compute_metrics(self, predictions: Dict[str, mx.array], targets: mx.array, **kwargs) -> Dict[str, float]:
        """Compute evaluation metrics for binary classification.
        
        Args:
            predictions: Predictions from forward pass
            targets: Ground truth targets
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of metric values
        """
        probs = predictions["probabilities"].squeeze(-1)  # [batch_size]
        preds = predictions["predictions"].squeeze(-1)    # [batch_size]
        
        # Convert to numpy for metric computation
        probs_np = np.array(probs)
        preds_np = np.array(preds)
        targets_np = np.array(targets)
        
        # Basic metrics
        accuracy = (preds_np == targets_np).mean()
        
        # Precision, Recall, F1
        tp = ((preds_np == 1) & (targets_np == 1)).sum()
        fp = ((preds_np == 1) & (targets_np == 0)).sum()
        fn = ((preds_np == 0) & (targets_np == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # AUC (simplified approximation)
        # For full AUC, would need sklearn or similar
        auc_approx = self._compute_auc_approximation(probs_np, targets_np)
        
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc": float(auc_approx),
        }
    
    def _compute_auc_approximation(self, probs: np.ndarray, targets: np.ndarray) -> float:
        """Compute AUC approximation using trapezoidal rule."""
        if len(np.unique(targets)) < 2:
            return 0.5  # No positive or negative examples
        
        # Sort by probability
        sorted_indices = np.argsort(probs)
        sorted_probs = probs[sorted_indices]
        sorted_targets = targets[sorted_indices]
        
        # Compute TPR and FPR at different thresholds
        n_pos = (targets == 1).sum()
        n_neg = (targets == 0).sum()
        
        if n_pos == 0 or n_neg == 0:
            return 0.5
        
        # Simple approximation: sample 100 thresholds
        thresholds = np.linspace(0, 1, 100)
        tpr_values = []
        fpr_values = []
        
        for threshold in thresholds:
            pred_pos = (sorted_probs >= threshold)
            tp = (pred_pos & (sorted_targets == 1)).sum()
            fp = (pred_pos & (sorted_targets == 0)).sum()
            
            tpr = tp / n_pos
            fpr = fp / n_neg
            
            tpr_values.append(tpr)
            fpr_values.append(fpr)
        
        # Compute AUC using trapezoidal rule
        tpr_values = np.array(tpr_values)
        fpr_values = np.array(fpr_values)
        
        # Sort by FPR
        sorted_indices = np.argsort(fpr_values)
        fpr_sorted = fpr_values[sorted_indices]
        tpr_sorted = tpr_values[sorted_indices]
        
        # Trapezoidal integration
        auc = np.trapz(tpr_sorted, fpr_sorted)
        
        return max(0.0, min(1.0, auc))
    
    def _binary_cross_entropy_loss(self, logits: mx.array, targets: mx.array) -> mx.array:
        """Custom binary cross-entropy loss implementation."""
        # Apply sigmoid to logits
        sigmoid_logits = mx.sigmoid(logits)
        
        # Compute binary cross-entropy
        targets = targets.astype(mx.float32)
        loss = -(targets * mx.log(sigmoid_logits + 1e-8) + (1 - targets) * mx.log(1 - sigmoid_logits + 1e-8))
        
        return loss.mean()


@register_head_class(
    name="multiclass_classification",
    head_type=HeadType.MULTICLASS_CLASSIFICATION,
    competition_types=[CompetitionType.MULTICLASS_CLASSIFICATION],
    priority=10,
    description="Multiclass classification head with softmax activation and cross-entropy loss"
)
class MulticlassClassificationHead(BaseKaggleHead):
    """Multiclass classification head for N-class problems.
    
    Features:
    - Softmax activation for probability distribution
    - Cross-entropy loss with label smoothing
    - Support for class weights and focal loss
    - Competition-specific optimizations
    """
    
    def __init__(self, config: HeadConfig):
        """Initialize multiclass classification head.
        
        Args:
            config: Head configuration
        """
        # Competition-specific parameters (must be set before super().__init__())
        self.use_focal_loss = config.use_competition_tricks
        self.focal_alpha = 1.0
        self.focal_gamma = 2.0
        self.label_smoothing = 0.1 if config.use_competition_tricks else 0.0
        self.class_weights = None
        
        super().__init__(config)
        
    def _build_output_layer(self):
        """Build the output layer for multiclass classification."""
        self.classifier = nn.Linear(self.projection_output_size, self.config.output_size)
        
        # Optional temperature scaling for calibration
        if self.config.use_competition_tricks:
            self.temperature = mx.ones(1)  # Will be learnable parameter
        else:
            self.temperature = None
    
    def _build_loss_function(self):
        """Build loss function for multiclass classification."""
        # Cross-entropy with label smoothing (custom implementation)
        self.ce_loss = self._cross_entropy_loss
        
        # Focal loss for imbalanced datasets
        if self.use_focal_loss:
            self.focal_loss = MulticlassFocalLoss(alpha=self.focal_alpha, gamma=self.focal_gamma)
    
    def forward(self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None, **kwargs) -> Dict[str, mx.array]:
        """Forward pass through multiclass classification head.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask [batch_size, seq_len]
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with predictions and probabilities
        """
        # Apply pooling
        pooled_output = self._apply_pooling(hidden_states, attention_mask)
        
        # Apply projection layers
        projected = self.projection(pooled_output)
        
        # Get logits
        logits = self.classifier(projected)  # [batch_size, num_classes]
        
        # Apply temperature scaling if enabled
        if self.temperature is not None:
            logits = logits / self.temperature
        
        # Get probabilities
        probs = mx.softmax(logits, axis=-1)
        
        # Get predictions
        predictions = mx.argmax(logits, axis=-1)
        
        return {
            "logits": logits,
            "probabilities": probs,
            "predictions": predictions,
        }
    
    def compute_loss(self, predictions: Dict[str, mx.array], targets: mx.array, **kwargs) -> mx.array:
        """Compute loss for multiclass classification.
        
        Args:
            predictions: Predictions from forward pass
            targets: Ground truth targets [batch_size] (class indices)
            **kwargs: Additional arguments
            
        Returns:
            Loss value
        """
        logits = predictions["logits"]
        targets = targets.astype(mx.int32)
        
        if self.use_focal_loss and hasattr(self, 'focal_loss'):
            loss = self.focal_loss(logits, targets)
        else:
            loss = self.ce_loss(logits, targets)
        
        return loss
    
    def compute_metrics(self, predictions: Dict[str, mx.array], targets: mx.array, **kwargs) -> Dict[str, float]:
        """Compute evaluation metrics for multiclass classification.
        
        Args:
            predictions: Predictions from forward pass
            targets: Ground truth targets
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of metric values
        """
        probs = predictions["probabilities"]  # [batch_size, num_classes]
        preds = predictions["predictions"]    # [batch_size]
        
        # Convert to numpy for metric computation
        probs_np = np.array(probs)
        preds_np = np.array(preds)
        targets_np = np.array(targets)
        
        # Basic metrics
        accuracy = (preds_np == targets_np).mean()
        
        # Top-k accuracy (for k=3 if num_classes >= 3)
        top3_accuracy = 0.0
        if self.config.output_size >= 3:
            top3_preds = np.argsort(probs_np, axis=-1)[:, -3:]
            top3_accuracy = np.mean([targets_np[i] in top3_preds[i] for i in range(len(targets_np))])
        
        # Per-class metrics (simplified)
        num_classes = self.config.output_size
        per_class_f1 = []
        
        for class_idx in range(num_classes):
            class_mask = (targets_np == class_idx)
            pred_mask = (preds_np == class_idx)
            
            tp = (class_mask & pred_mask).sum()
            fp = (~class_mask & pred_mask).sum()
            fn = (class_mask & ~pred_mask).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            per_class_f1.append(f1)
        
        macro_f1 = np.mean(per_class_f1)
        
        return {
            "accuracy": float(accuracy),
            "top3_accuracy": float(top3_accuracy),
            "macro_f1": float(macro_f1),
        }
    
    def _cross_entropy_loss(self, logits: mx.array, targets: mx.array) -> mx.array:
        """Custom cross-entropy loss implementation."""
        # Apply softmax to logits
        log_probs = mx.log_softmax(logits, axis=-1)
        
        # Select the log probabilities for the correct classes
        targets = targets.astype(mx.int32)
        batch_size = logits.shape[0]
        batch_indices = mx.arange(batch_size)
        
        # Gather the log probabilities for the true labels
        selected_log_probs = log_probs[batch_indices, targets]
        
        # Compute negative log likelihood
        loss = -selected_log_probs.mean()
        
        return loss


@register_head_class(
    name="multilabel_classification", 
    head_type=HeadType.MULTILABEL_CLASSIFICATION,
    competition_types=[CompetitionType.MULTILABEL_CLASSIFICATION],
    priority=10,
    description="Multilabel classification head with sigmoid activation and BCE loss"
)
class MultilabelClassificationHead(BaseKaggleHead):
    """Multilabel classification head for problems with multiple labels per sample.
    
    Features:
    - Sigmoid activation for independent label probabilities
    - Binary cross-entropy loss for each label
    - Support for label-specific thresholds
    - Competition-specific optimizations
    """
    
    def __init__(self, config: HeadConfig):
        """Initialize multilabel classification head.
        
        Args:
            config: Head configuration
        """
        # Competition-specific parameters (must be set before super().__init__())
        self.use_focal_loss = config.use_competition_tricks
        self.focal_alpha = 0.25
        self.focal_gamma = 2.0
        self.label_weights = None
        
        # Adaptive thresholds for each label
        self.adaptive_thresholds = mx.ones(config.output_size) * 0.5
        
        super().__init__(config)
    
    def _build_output_layer(self):
        """Build the output layer for multilabel classification."""
        self.classifier = nn.Linear(self.projection_output_size, self.config.output_size)
        
        # Optional temperature scaling for calibration
        if self.config.use_competition_tricks:
            self.temperature = mx.ones(1)  # Will be learnable parameter
        else:
            self.temperature = None
    
    def _build_loss_function(self):
        """Build loss function for multilabel classification."""
        # Binary cross-entropy for each label (custom implementation)
        self.bce_loss = self._multilabel_bce_loss
        
        # Focal loss for imbalanced datasets
        if self.use_focal_loss:
            self.focal_loss = MultilabelFocalLoss(alpha=self.focal_alpha, gamma=self.focal_gamma)
    
    def forward(self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None, **kwargs) -> Dict[str, mx.array]:
        """Forward pass through multilabel classification head.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask [batch_size, seq_len]
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with predictions and probabilities
        """
        # Apply pooling
        pooled_output = self._apply_pooling(hidden_states, attention_mask)
        
        # Apply projection layers
        projected = self.projection(pooled_output)
        
        # Get logits
        logits = self.classifier(projected)  # [batch_size, num_labels]
        
        # Apply temperature scaling if enabled
        if self.temperature is not None:
            logits = logits / self.temperature
        
        # Get probabilities
        probs = mx.sigmoid(logits)
        
        # Get predictions using adaptive thresholds
        predictions = (probs > self.adaptive_thresholds).astype(mx.int32)
        
        return {
            "logits": logits,
            "probabilities": probs,
            "predictions": predictions,
        }
    
    def compute_loss(self, predictions: Dict[str, mx.array], targets: mx.array, **kwargs) -> mx.array:
        """Compute loss for multilabel classification.
        
        Args:
            predictions: Predictions from forward pass
            targets: Ground truth targets [batch_size, num_labels] (binary)
            **kwargs: Additional arguments
            
        Returns:
            Loss value
        """
        logits = predictions["logits"]
        targets = targets.astype(mx.float32)
        
        if self.use_focal_loss and hasattr(self, 'focal_loss'):
            loss = self.focal_loss(logits, targets)
        else:
            loss = self.bce_loss(logits, targets)
        
        return loss
    
    def compute_metrics(self, predictions: Dict[str, mx.array], targets: mx.array, **kwargs) -> Dict[str, float]:
        """Compute evaluation metrics for multilabel classification.
        
        Args:
            predictions: Predictions from forward pass
            targets: Ground truth targets
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of metric values
        """
        probs = predictions["probabilities"]  # [batch_size, num_labels]
        preds = predictions["predictions"]    # [batch_size, num_labels]
        
        # Convert to numpy for metric computation
        probs_np = np.array(probs)
        preds_np = np.array(preds)
        targets_np = np.array(targets)
        
        # Subset accuracy (exact match)
        subset_accuracy = np.mean(np.all(preds_np == targets_np, axis=1))
        
        # Hamming loss
        hamming_loss = np.mean(preds_np != targets_np)
        
        # Macro F1 (average F1 across labels)
        label_f1_scores = []
        for label_idx in range(self.config.output_size):
            y_true = targets_np[:, label_idx]
            y_pred = preds_np[:, label_idx]
            
            tp = (y_true * y_pred).sum()
            fp = ((1 - y_true) * y_pred).sum()
            fn = (y_true * (1 - y_pred)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            label_f1_scores.append(f1)
        
        macro_f1 = np.mean(label_f1_scores)
        
        # Micro F1 (global F1)
        tp_total = (targets_np * preds_np).sum()
        fp_total = ((1 - targets_np) * preds_np).sum()
        fn_total = (targets_np * (1 - preds_np)).sum()
        
        micro_precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        micro_recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
        
        return {
            "subset_accuracy": float(subset_accuracy),
            "hamming_loss": float(hamming_loss),
            "macro_f1": float(macro_f1),
            "micro_f1": float(micro_f1),
        }
    
    def _multilabel_bce_loss(self, logits: mx.array, targets: mx.array) -> mx.array:
        """Custom multilabel binary cross-entropy loss implementation."""
        # Apply sigmoid to logits
        sigmoid_logits = mx.sigmoid(logits)
        
        # Compute binary cross-entropy for each label
        targets = targets.astype(mx.float32)
        loss = -(targets * mx.log(sigmoid_logits + 1e-8) + (1 - targets) * mx.log(1 - sigmoid_logits + 1e-8))
        
        return loss.mean()


# Loss function implementations
class FocalLoss(nn.Module):
    """Focal loss for binary classification."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def __call__(self, logits: mx.array, targets: mx.array) -> mx.array:
        # Convert to probabilities
        probs = mx.sigmoid(logits)
        
        # Compute focal loss
        pt = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_weight = alpha_t * mx.power(1 - pt, self.gamma)
        
        # Binary cross-entropy
        bce = -(targets * mx.log(probs + 1e-8) + (1 - targets) * mx.log(1 - probs + 1e-8))
        
        return (focal_weight * bce).mean()


class MulticlassFocalLoss(nn.Module):
    """Focal loss for multiclass classification."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def __call__(self, logits: mx.array, targets: mx.array) -> mx.array:
        # Convert to probabilities
        probs = mx.softmax(logits, axis=-1)
        
        # Get probabilities for true classes
        batch_size = logits.shape[0]
        batch_indices = mx.arange(batch_size)
        pt = probs[batch_indices, targets]
        
        # Compute focal loss
        focal_weight = self.alpha * mx.power(1 - pt, self.gamma)
        
        # Cross-entropy
        ce = -mx.log(pt + 1e-8)
        
        return (focal_weight * ce).mean()


class MultilabelFocalLoss(nn.Module):
    """Focal loss for multilabel classification."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def __call__(self, logits: mx.array, targets: mx.array) -> mx.array:
        # Convert to probabilities
        probs = mx.sigmoid(logits)
        
        # Compute focal loss for each label
        pt = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_weight = alpha_t * mx.power(1 - pt, self.gamma)
        
        # Binary cross-entropy for each label
        bce = -(targets * mx.log(probs + 1e-8) + (1 - targets) * mx.log(1 - probs + 1e-8))
        
        return (focal_weight * bce).mean()