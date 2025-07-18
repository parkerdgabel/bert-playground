"""
Generic classifier implementation that can adapt to various classification tasks.
Provides a flexible architecture that can be configured for different problem types.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Literal
import mlx.core as mx
import mlx.nn as nn
from loguru import logger
import numpy as np

from models.embeddings.embedding_model import EmbeddingModel as BertEmbeddingModel
from models.classification.base_classifier import BaseClassifier, PoolingType, ActivationType
from models.classification.heads import (
    BinaryClassificationHead,
    MultiClassificationHead,
    RegressionHead
)
from models.classification.advanced_heads import (
    MultilabelClassificationHead,
    OrdinalRegressionHead,
    HierarchicalClassificationHead,
    EnsembleClassificationHead
)


ClassificationTask = Literal[
    "binary", "multiclass", "regression", "multilabel", 
    "ordinal", "hierarchical", "ensemble"
]


class GenericClassifier(BaseClassifier):
    """
    Generic classifier that can be configured for various classification tasks.
    Automatically adapts architecture based on task type and configuration.
    """
    
    def __init__(
        self,
        embedding_model: BertEmbeddingModel,
        num_classes: int,
        task_type: ClassificationTask = "multiclass",
        pooling_type: PoolingType = "mean",
        hidden_dims: Optional[Union[int, List[int]]] = None,
        activation: ActivationType = "gelu",
        dropout_rate: float = 0.1,
        use_layer_norm: bool = True,
        use_batch_norm: bool = False,
        freeze_embeddings: bool = False,
        head_config: Optional[Dict[str, Any]] = None,
        class_names: Optional[List[str]] = None,
        label_smoothing: float = 0.0,
        auxiliary_heads: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """
        Initialize generic classifier.
        
        Args:
            embedding_model: Pre-trained BERT embedding model
            num_classes: Number of output classes/labels
            task_type: Type of classification task
            pooling_type: Type of pooling to use
            hidden_dims: Hidden dimensions (int or list for multiple layers)
            activation: Activation function type
            dropout_rate: Dropout rate
            use_layer_norm: Whether to use layer normalization
            use_batch_norm: Whether to use batch normalization
            freeze_embeddings: Whether to freeze embedding weights
            head_config: Additional configuration for classification head
            class_names: Optional names for classes
            label_smoothing: Label smoothing factor
            auxiliary_heads: Additional heads for multi-task learning
        """
        self.task_type = task_type
        self.class_names = class_names
        self.label_smoothing = label_smoothing
        self.auxiliary_heads = auxiliary_heads or {}
        
        # Process hidden dimensions
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        self.hidden_dims = hidden_dims
        
        # Initialize base classifier
        super().__init__(
            embedding_model=embedding_model,
            num_classes=num_classes,
            pooling_type=pooling_type,
            hidden_dim=hidden_dims[0] if hidden_dims else None,
            activation=activation,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm,
            use_batch_norm=use_batch_norm,
            freeze_embeddings=freeze_embeddings,
            head_config=head_config,
        )
        
        # Create auxiliary heads if specified
        self.aux_heads = {}
        if auxiliary_heads:
            for head_name, head_config in auxiliary_heads.items():
                aux_head = self._create_auxiliary_head(head_name, head_config)
                self.aux_heads[head_name] = aux_head
                # Register as attribute for MLX
                setattr(self, f"aux_head_{head_name}", aux_head)
    
    def _create_classification_head(self) -> nn.Module:
        """Create task-specific classification head."""
        head_config = self.head_config or {}
        
        if self.task_type == "binary":
            return BinaryClassificationHead(
                input_dim=self.embedding_dim,
                hidden_dim=self.hidden_dim,
                activation=self.activation,
                dropout_prob=self.dropout_rate,  # BinaryClassificationHead uses dropout_prob
                use_layer_norm=self.use_layer_norm,
            )
        
        elif self.task_type == "multiclass":
            return MultiClassificationHead(
                input_dim=self.embedding_dim,
                num_classes=self.num_classes,
                hidden_dim=self.hidden_dim,
                activation=self.activation,
                dropout_prob=self.dropout_rate,  # MultiClassificationHead uses dropout_prob
                use_layer_norm=self.use_layer_norm,
            )
        
        elif self.task_type == "regression":
            return RegressionHead(
                input_dim=self.embedding_dim,
                output_dim=self.num_classes,  # num_classes represents output_dim for regression
                hidden_dim=self.hidden_dim,
                activation=self.activation,
                dropout_prob=self.dropout_rate,  # RegressionHead uses dropout_prob
                use_layer_norm=self.use_layer_norm,
            )
        
        elif self.task_type == "multilabel":
            return MultilabelClassificationHead(
                input_dim=self.embedding_dim,
                num_labels=self.num_classes,
                hidden_dim=self.hidden_dim,
                activation=self.activation,
                dropout_rate=self.dropout_rate,
                use_layer_norm=self.use_layer_norm,
                label_smoothing=self.label_smoothing,
            )
        
        elif self.task_type == "ordinal":
            return OrdinalRegressionHead(
                input_dim=self.embedding_dim,
                num_classes=self.num_classes,
                hidden_dim=self.hidden_dim,
                activation=self.activation,
                dropout_rate=self.dropout_rate,
                use_layer_norm=self.use_layer_norm,
                temperature=head_config.get("temperature", 1.0),
            )
        
        elif self.task_type == "hierarchical":
            if "hierarchy" not in head_config:
                raise ValueError("Hierarchical classification requires 'hierarchy' in head_config")
            if "label_to_idx" not in head_config:
                raise ValueError("Hierarchical classification requires 'label_to_idx' in head_config")
            
            return HierarchicalClassificationHead(
                input_dim=self.embedding_dim,
                hierarchy=head_config["hierarchy"],
                label_to_idx=head_config["label_to_idx"],
                hidden_dim=self.hidden_dim,
                activation=self.activation,
                dropout_rate=self.dropout_rate,
                use_layer_norm=self.use_layer_norm,
                consistency_weight=head_config.get("consistency_weight", 1.0),
            )
        
        elif self.task_type == "ensemble":
            return EnsembleClassificationHead(
                input_dim=self.embedding_dim,
                num_classes=self.num_classes,
                num_heads=head_config.get("num_heads", 3),
                hidden_dims=head_config.get("hidden_dims", None),
                activations=head_config.get("activations", None),
                dropout_rates=head_config.get("dropout_rates", None),
                ensemble_method=head_config.get("ensemble_method", "average"),
                temperature=head_config.get("temperature", 1.0),
            )
        
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
    
    def _create_auxiliary_head(self, head_name: str, head_config: Dict[str, Any]) -> nn.Module:
        """Create auxiliary head for multi-task learning."""
        task_type = head_config.get("task_type", "multiclass")
        num_classes = head_config.get("num_classes", 2)
        
        # Create appropriate head based on task type
        if task_type == "binary":
            return BinaryClassificationHead(
                input_dim=self.embedding_dim,
                hidden_dim=head_config.get("hidden_dim", None),
                activation=head_config.get("activation", "gelu"),
                dropout_prob=head_config.get("dropout_rate", 0.1),  # Convert to dropout_prob
            )
        elif task_type == "multiclass":
            return MultiClassificationHead(
                input_dim=self.embedding_dim,
                num_classes=num_classes,
                hidden_dim=head_config.get("hidden_dim", None),
                activation=head_config.get("activation", "gelu"),
                dropout_prob=head_config.get("dropout_rate", 0.1),  # Convert to dropout_prob
            )
        elif task_type == "regression":
            return RegressionHead(
                input_dim=self.embedding_dim,
                num_outputs=num_classes,
                hidden_dim=head_config.get("hidden_dim", None),
                activation=head_config.get("activation", "gelu"),
                dropout_prob=head_config.get("dropout_rate", 0.1),  # Convert to dropout_prob
            )
        else:
            raise ValueError(f"Unknown auxiliary task type: {task_type}")
    
    def forward(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        return_embeddings: bool = False,
        compute_auxiliary: bool = True,
    ) -> Union[mx.array, Tuple[mx.array, ...], Dict[str, mx.array]]:
        """
        Forward pass through the classifier.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            return_embeddings: Whether to return embeddings
            compute_auxiliary: Whether to compute auxiliary head outputs
            
        Returns:
            Logits, or tuple of (logits, embeddings), or dict with all outputs
        """
        # Get embeddings
        embeddings = self.embedding_model(input_ids, attention_mask)
        
        # Pool embeddings
        pooled = self.pool_embeddings(embeddings, attention_mask)
        
        # Main task prediction
        if self.task_type == "hierarchical":
            # Hierarchical returns dict of level logits
            main_logits = self.classification_head(pooled)
        else:
            main_logits = self.classification_head(pooled)
        
        # Auxiliary predictions
        aux_outputs = {}
        if compute_auxiliary and self.aux_heads:
            for head_name, aux_head in self.aux_heads.items():
                aux_outputs[head_name] = aux_head(pooled)
        
        # Return based on requirements
        if aux_outputs:
            outputs = {
                "main": main_logits,
                "pooled_embeddings": pooled if return_embeddings else None,
                **aux_outputs
            }
            return outputs
        elif return_embeddings:
            return main_logits, pooled
        else:
            return main_logits
    
    def compute_loss(
        self,
        logits: Union[mx.array, Dict[str, mx.array]],
        labels: mx.array,
        class_weights: Optional[mx.array] = None,
        auxiliary_labels: Optional[Dict[str, mx.array]] = None,
        auxiliary_weights: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> Union[mx.array, Dict[str, mx.array]]:
        """
        Compute task-specific loss.
        
        Args:
            logits: Model outputs (array or dict for hierarchical)
            labels: True labels
            class_weights: Weights for each class
            auxiliary_labels: Labels for auxiliary tasks
            auxiliary_weights: Loss weights for auxiliary tasks
            **kwargs: Additional arguments for specific loss functions
            
        Returns:
            Total loss or dict of losses
        """
        losses = {}
        
        # Main task loss
        if self.task_type in ["binary", "multiclass"]:
            losses["main"] = nn.losses.cross_entropy(
                logits if not isinstance(logits, dict) else logits["main"],
                labels,
                reduction="none"
            )
            if class_weights is not None:
                sample_weights = class_weights[labels]
                losses["main"] = losses["main"] * sample_weights
            losses["main"] = mx.mean(losses["main"])
            
        elif self.task_type == "regression":
            predictions = logits if not isinstance(logits, dict) else logits["main"]
            losses["main"] = mx.mean((predictions - labels) ** 2)
            
        elif self.task_type == "multilabel":
            losses["main"] = self.classification_head.compute_loss(
                logits if not isinstance(logits, dict) else logits["main"],
                labels,
                class_weights=class_weights,
                pos_weight=kwargs.get("pos_weight", None)
            )
            
        elif self.task_type == "ordinal":
            losses["main"] = self.classification_head.compute_loss(
                logits if not isinstance(logits, dict) else logits["main"],
                labels,
                class_weights=class_weights
            )
            
        elif self.task_type == "hierarchical":
            level_logits = logits if isinstance(logits, dict) else {"0": logits}
            losses["main"] = self.classification_head.compute_loss(
                level_logits,
                labels,
                class_weights=class_weights
            )
            
        elif self.task_type == "ensemble":
            losses["main"] = self.classification_head.compute_loss(
                logits if not isinstance(logits, dict) else logits["main"],
                labels,
                class_weights=class_weights,
                diversity_weight=kwargs.get("diversity_weight", 0.01)
            )
        
        # Auxiliary task losses
        if auxiliary_labels and isinstance(logits, dict):
            auxiliary_weights = auxiliary_weights or {}
            
            for task_name, task_labels in auxiliary_labels.items():
                if task_name in logits and task_name != "main":
                    aux_head = self.aux_heads[task_name]
                    task_config = self.auxiliary_heads[task_name]
                    
                    if task_config["task_type"] == "binary":
                        task_loss = nn.losses.binary_cross_entropy(
                            logits[task_name], task_labels, with_logits=True
                        )
                    elif task_config["task_type"] == "multiclass":
                        task_loss = nn.losses.cross_entropy(
                            logits[task_name], task_labels
                        )
                    elif task_config["task_type"] == "regression":
                        task_loss = mx.mean((logits[task_name] - task_labels) ** 2)
                    
                    weight = auxiliary_weights.get(task_name, 1.0)
                    losses[task_name] = weight * mx.mean(task_loss)
        
        # Return total loss or dict of losses
        if len(losses) == 1:
            return losses["main"]
        else:
            losses["total"] = sum(losses.values())
            return losses
    
    def predict(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        return_auxiliary: bool = False
    ) -> Union[mx.array, Dict[str, mx.array]]:
        """
        Make predictions for the given inputs.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            return_auxiliary: Whether to return auxiliary predictions
            
        Returns:
            Predictions (array or dict)
        """
        outputs = self.forward(
            input_ids, 
            attention_mask, 
            return_embeddings=False,
            compute_auxiliary=return_auxiliary
        )
        
        if isinstance(outputs, dict):
            predictions = {}
            
            # Main task predictions
            if self.task_type in ["binary", "multilabel"]:
                predictions["main"] = mx.sigmoid(outputs["main"]) > 0.5
            elif self.task_type == "multiclass":
                predictions["main"] = mx.argmax(outputs["main"], axis=-1)
            elif self.task_type == "regression":
                predictions["main"] = outputs["main"]
            elif self.task_type == "ordinal":
                predictions["main"] = self.classification_head.predict_class(outputs["main"])
            elif self.task_type == "hierarchical":
                # Return predictions for each level
                level_preds = {}
                for level, logits in outputs["main"].items():
                    level_preds[level] = mx.argmax(logits, axis=-1)
                predictions["main"] = level_preds
            elif self.task_type == "ensemble":
                predictions["main"] = mx.argmax(outputs["main"], axis=-1)
            
            # Auxiliary predictions
            if return_auxiliary:
                for task_name in self.aux_heads:
                    if task_name in outputs:
                        task_config = self.auxiliary_heads[task_name]
                        if task_config["task_type"] == "binary":
                            predictions[task_name] = mx.sigmoid(outputs[task_name]) > 0.5
                        elif task_config["task_type"] == "multiclass":
                            predictions[task_name] = mx.argmax(outputs[task_name], axis=-1)
                        elif task_config["task_type"] == "regression":
                            predictions[task_name] = outputs[task_name]
            
            return predictions
        else:
            # Single output
            if self.task_type in ["binary", "multilabel"]:
                return mx.sigmoid(outputs) > 0.5
            elif self.task_type == "multiclass":
                return mx.argmax(outputs, axis=-1)
            elif self.task_type == "regression":
                return outputs
            elif self.task_type == "ordinal":
                return self.classification_head.predict_class(outputs)
            elif self.task_type == "ensemble":
                return mx.argmax(outputs, axis=-1)
    
    def predict_proba(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        return_auxiliary: bool = False
    ) -> Union[mx.array, Dict[str, mx.array]]:
        """
        Predict class probabilities.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            return_auxiliary: Whether to return auxiliary probabilities
            
        Returns:
            Probabilities (array or dict)
        """
        outputs = self.forward(
            input_ids,
            attention_mask,
            return_embeddings=False,
            compute_auxiliary=return_auxiliary
        )
        
        if isinstance(outputs, dict):
            probabilities = {}
            
            # Main task probabilities
            if self.task_type == "binary":
                probabilities["main"] = mx.sigmoid(outputs["main"])
            elif self.task_type == "multiclass":
                probabilities["main"] = mx.softmax(outputs["main"], axis=-1)
            elif self.task_type == "multilabel":
                probabilities["main"] = mx.sigmoid(outputs["main"])
            elif self.task_type == "ordinal":
                probabilities["main"] = mx.sigmoid(outputs["main"])
            elif self.task_type == "hierarchical":
                level_probs = {}
                for level, logits in outputs["main"].items():
                    level_probs[level] = mx.softmax(logits, axis=-1)
                probabilities["main"] = level_probs
            elif self.task_type == "ensemble":
                probabilities["main"] = mx.softmax(outputs["main"], axis=-1)
            elif self.task_type == "regression":
                probabilities["main"] = outputs["main"]  # No probabilities for regression
            
            # Auxiliary probabilities
            if return_auxiliary:
                for task_name in self.aux_heads:
                    if task_name in outputs:
                        task_config = self.auxiliary_heads[task_name]
                        if task_config["task_type"] == "binary":
                            probabilities[task_name] = mx.sigmoid(outputs[task_name])
                        elif task_config["task_type"] == "multiclass":
                            probabilities[task_name] = mx.softmax(outputs[task_name], axis=-1)
                        elif task_config["task_type"] == "regression":
                            probabilities[task_name] = outputs[task_name]
            
            return probabilities
        else:
            # Single output
            if self.task_type == "binary":
                return mx.sigmoid(outputs)
            elif self.task_type == "multiclass":
                return mx.softmax(outputs, axis=-1)
            elif self.task_type == "multilabel":
                return mx.sigmoid(outputs)
            elif self.task_type == "ordinal":
                return mx.sigmoid(outputs)
            elif self.task_type == "ensemble":
                return mx.softmax(outputs, axis=-1)
            elif self.task_type == "regression":
                return outputs
    
    def get_feature_importance(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        method: str = "gradient"
    ) -> mx.array:
        """
        Compute feature importance scores.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            method: Method for computing importance ("gradient", "integrated_gradient")
            
        Returns:
            Importance scores [batch_size, seq_len]
        """
        # Enable gradient computation
        def loss_fn(input_embeds):
            # Get embeddings from token embeddings
            pooled = self.pool_embeddings(input_embeds, attention_mask)
            logits = self.classification_head(pooled)
            
            # Use max logit as proxy for importance
            if self.task_type in ["binary", "regression"]:
                return mx.sum(logits)
            else:
                return mx.sum(mx.max(logits, axis=-1))
        
        # Get token embeddings
        token_embeds = self.embedding_model.embeddings.token_embeddings(input_ids)
        
        if method == "gradient":
            # Simple gradient-based importance
            grad_fn = mx.grad(loss_fn)
            grads = grad_fn(token_embeds)
            importance = mx.sum(mx.abs(grads), axis=-1)
            
        elif method == "integrated_gradient":
            # Integrated gradients (simplified version)
            steps = 10
            baseline = mx.zeros_like(token_embeds)
            
            importance = mx.zeros((input_ids.shape[0], input_ids.shape[1]))
            for i in range(steps):
                alpha = i / steps
                interpolated = baseline + alpha * (token_embeds - baseline)
                
                grad_fn = mx.grad(loss_fn)
                grads = grad_fn(interpolated)
                importance = importance + mx.sum(mx.abs(grads), axis=-1) / steps
        else:
            raise ValueError(f"Unknown importance method: {method}")
        
        # Mask out padding tokens
        if attention_mask is not None:
            importance = importance * attention_mask
        
        return importance
    
    def get_config(self) -> Dict[str, Any]:
        """Get classifier configuration."""
        config = {
            "task_type": self.task_type,
            "num_classes": self.num_classes,
            "pooling_type": self.pooling_type,
            "hidden_dims": self.hidden_dims,
            "activation": self.activation,
            "dropout_rate": self.dropout_rate,
            "use_layer_norm": self.use_layer_norm,
            "use_batch_norm": self.use_batch_norm,
            "class_names": self.class_names,
            "label_smoothing": self.label_smoothing,
            "head_config": self.head_config,
            "auxiliary_heads": self.auxiliary_heads,
        }
        return config