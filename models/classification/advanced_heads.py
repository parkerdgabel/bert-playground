"""
Advanced classification heads for complex classification tasks.
Includes multilabel, ordinal regression, hierarchical, and ensemble heads.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from loguru import logger


class MultilabelClassificationHead(nn.Module):
    """
    Multilabel classification head for problems with multiple labels per sample.
    Each output neuron represents an independent binary classification.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        hidden_dim: Optional[int] = None,
        activation: str = "gelu",
        dropout_rate: float = 0.1,
        use_layer_norm: bool = True,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        
        self.num_labels = num_labels
        self.label_smoothing = label_smoothing
        
        # Build layers
        layers = []
        
        if hidden_dim is not None:
            # Hidden layer
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                self._get_activation(activation),
                nn.Dropout(dropout_rate),
            ])
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            # Output layer
            layers.append(nn.Linear(hidden_dim, num_labels))
        else:
            # Direct mapping
            layers.append(nn.Linear(input_dim, num_labels))
        
        self.mlp = nn.Sequential(*layers)
    
    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "mish": nn.Mish(),
            "tanh": nn.Tanh(),
            "swish": nn.SiLU(),  # Swish is equivalent to SiLU
        }
        return activations.get(name, nn.GELU())
    
    def forward(self, x: mx.array) -> mx.array:
        """Forward pass returning logits."""
        return self.mlp(x)
    
    def compute_loss(
        self,
        logits: mx.array,
        labels: mx.array,
        class_weights: Optional[mx.array] = None,
        pos_weight: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Compute binary cross-entropy loss for multilabel classification.
        
        Args:
            logits: Model outputs [batch_size, num_labels]
            labels: Binary labels [batch_size, num_labels]
            class_weights: Weights for each class
            pos_weight: Weight for positive class in each label
        """
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            labels = labels * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # Compute BCE loss
        loss = nn.losses.binary_cross_entropy(logits, labels, with_logits=True)
        
        # Apply class weights if provided
        if class_weights is not None:
            loss = loss * class_weights
        
        # Apply positive class weights if provided
        if pos_weight is not None:
            pos_mask = labels > 0.5
            loss = mx.where(pos_mask, loss * pos_weight, loss)
        
        return mx.mean(loss)


class OrdinalRegressionHead(nn.Module):
    """
    Ordinal regression head for ordered categorical targets.
    Uses the "all-threshold" variant where we predict P(y >= k) for each threshold k.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: Optional[int] = None,
        activation: str = "gelu",
        dropout_rate: float = 0.1,
        use_layer_norm: bool = True,
        temperature: float = 1.0,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1
        self.temperature = temperature
        
        # Build shared layers
        layers = []
        
        if hidden_dim is not None:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                self._get_activation(activation),
                nn.Dropout(dropout_rate),
            ])
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            final_input_dim = hidden_dim
        else:
            final_input_dim = input_dim
        
        self.shared_layers = nn.Sequential(*layers) if layers else nn.Identity()
        
        # Threshold predictors (one per threshold)
        self.threshold_layers = []
        for i in range(self.num_thresholds):
            layer = nn.Linear(final_input_dim, 1)
            self.threshold_layers.append(layer)
            setattr(self, f"threshold_layer_{i}", layer)
    
    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "mish": nn.Mish(),
            "tanh": nn.Tanh(),
        }
        return activations.get(name, nn.GELU())
    
    def forward(self, x: mx.array) -> mx.array:
        """
        Forward pass returning ordinal logits.
        
        Returns:
            Logits for P(y >= k) for each threshold k
        """
        # Shared representation
        h = self.shared_layers(x)
        
        # Compute threshold logits
        threshold_logits = []
        for threshold_layer in self.threshold_layers:
            logit = threshold_layer(h) / self.temperature
            threshold_logits.append(logit)
        
        # Stack threshold logits
        logits = mx.concatenate(threshold_logits, axis=-1)
        return logits
    
    def compute_loss(
        self,
        logits: mx.array,
        labels: mx.array,
        class_weights: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Compute ordinal regression loss.
        
        Args:
            logits: Threshold logits [batch_size, num_thresholds]
            labels: Ordinal labels (0 to num_classes-1) [batch_size]
            class_weights: Weights for each class
        """
        batch_size = labels.shape[0]
        
        # Create binary targets for each threshold
        # For label k, thresholds 0..k-1 should be 1, k..end should be 0
        threshold_targets = mx.zeros((batch_size, self.num_thresholds))
        for i in range(batch_size):
            label = int(labels[i])
            if label > 0:
                threshold_targets[i, :label] = 1
        
        # Compute binary cross-entropy for each threshold
        loss = nn.losses.binary_cross_entropy(logits, threshold_targets, with_logits=True)
        
        # Apply class weights if provided
        if class_weights is not None:
            # Map class weights to threshold weights
            sample_weights = class_weights[labels]
            loss = loss * mx.expand_dims(sample_weights, axis=-1)
        
        return mx.mean(loss)
    
    def predict_class(self, logits: mx.array) -> mx.array:
        """Convert threshold logits to class predictions."""
        # Apply sigmoid to get probabilities
        probs = mx.sigmoid(logits)
        
        # Count how many thresholds are exceeded
        # This gives us the predicted class
        predictions = mx.sum(probs > 0.5, axis=-1)
        return predictions


class HierarchicalClassificationHead(nn.Module):
    """
    Hierarchical classification head for problems with hierarchical label structure.
    Supports tree-structured labels with parent-child relationships.
    """
    
    def __init__(
        self,
        input_dim: int,
        hierarchy: Dict[str, List[str]],
        label_to_idx: Dict[str, int],
        hidden_dim: Optional[int] = None,
        activation: str = "gelu",
        dropout_rate: float = 0.1,
        use_layer_norm: bool = True,
        consistency_weight: float = 1.0,
    ):
        """
        Args:
            input_dim: Input dimension
            hierarchy: Dict mapping parent labels to list of children
            label_to_idx: Mapping from label names to indices
            hidden_dim: Hidden dimension
            activation: Activation function
            dropout_rate: Dropout rate
            use_layer_norm: Whether to use layer normalization
            consistency_weight: Weight for hierarchical consistency loss
        """
        super().__init__()
        
        self.hierarchy = hierarchy
        self.label_to_idx = label_to_idx
        self.idx_to_label = {v: k for k, v in label_to_idx.items()}
        self.num_labels = len(label_to_idx)
        self.consistency_weight = consistency_weight
        
        # Build hierarchy levels
        self._build_hierarchy_structure()
        
        # Shared layers
        layers = []
        if hidden_dim is not None:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                self._get_activation(activation),
                nn.Dropout(dropout_rate),
            ])
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            final_input_dim = hidden_dim
        else:
            final_input_dim = input_dim
        
        self.shared_layers = nn.Sequential(*layers) if layers else nn.Identity()
        
        # Level-specific classifiers
        self.level_classifiers = {}
        for level, labels in self.level_labels.items():
            classifier = nn.Linear(final_input_dim, len(labels))
            self.level_classifiers[str(level)] = classifier
            # Register as attribute for MLX
            setattr(self, f"level_classifier_{level}", classifier)
    
    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "mish": nn.Mish(),
        }
        return activations.get(name, nn.GELU())
    
    def _build_hierarchy_structure(self):
        """Build hierarchy structure for efficient computation."""
        # Find root nodes and build levels
        all_children = set()
        for children in self.hierarchy.values():
            all_children.update(children)
        
        roots = set(self.hierarchy.keys()) - all_children
        
        # Build level structure
        self.level_labels = {}
        self.label_to_level = {}
        self.parent_child_matrix = mx.zeros((self.num_labels, self.num_labels))
        
        # BFS to assign levels
        current_level = 0
        current_nodes = list(roots)
        
        while current_nodes:
            self.level_labels[current_level] = current_nodes.copy()
            for node in current_nodes:
                self.label_to_level[node] = current_level
            
            next_nodes = []
            for node in current_nodes:
                if node in self.hierarchy:
                    children = self.hierarchy[node]
                    next_nodes.extend(children)
                    
                    # Fill parent-child matrix
                    parent_idx = self.label_to_idx[node]
                    for child in children:
                        child_idx = self.label_to_idx[child]
                        self.parent_child_matrix[parent_idx, child_idx] = 1
            
            current_nodes = next_nodes
            current_level += 1
    
    def forward(self, x: mx.array) -> Dict[str, mx.array]:
        """
        Forward pass returning hierarchical logits.
        
        Returns:
            Dict mapping level to logits for that level
        """
        # Shared representation
        h = self.shared_layers(x)
        
        # Compute logits for each level
        level_logits = {}
        for level, classifier in self.level_classifiers.items():
            level_logits[level] = classifier(h)
        
        return level_logits
    
    def compute_loss(
        self,
        level_logits: Dict[str, mx.array],
        labels: mx.array,
        class_weights: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Compute hierarchical classification loss with consistency constraints.
        
        Args:
            level_logits: Dict mapping level to logits
            labels: True labels [batch_size]
            class_weights: Weights for each class
        """
        total_loss = 0.0
        
        # Standard classification loss for each level
        for level_str, logits in level_logits.items():
            level = int(level_str)
            level_labels_list = self.level_labels[level]
            
            # Create targets for this level
            level_indices = [self.label_to_idx[label] for label in level_labels_list]
            level_mask = mx.zeros(labels.shape, dtype=mx.bool_)
            
            for idx in level_indices:
                level_mask = level_mask | (labels == idx)
            
            # Only compute loss for samples at this level
            if mx.any(level_mask):
                # Map global labels to level-specific labels
                level_targets = mx.zeros_like(labels)
                for i, global_idx in enumerate(level_indices):
                    mask = labels == global_idx
                    level_targets = mx.where(mask, i, level_targets)
                
                # Compute cross-entropy loss
                level_loss = nn.losses.cross_entropy(
                    logits[level_mask],
                    level_targets[level_mask],
                    reduction="mean"
                )
                
                total_loss = total_loss + level_loss
        
        # Hierarchical consistency loss
        if self.consistency_weight > 0:
            consistency_loss = self._compute_consistency_loss(level_logits)
            total_loss = total_loss + self.consistency_weight * consistency_loss
        
        return total_loss
    
    def _compute_consistency_loss(self, level_logits: Dict[str, mx.array]) -> mx.array:
        """Compute consistency loss between parent and child predictions."""
        consistency_loss = 0.0
        num_pairs = 0
        
        # For each parent-child pair, encourage consistency
        for parent_label, children in self.hierarchy.items():
            parent_level = self.label_to_level[parent_label]
            parent_idx_in_level = self.level_labels[parent_level].index(parent_label)
            
            # Get parent probabilities
            parent_logits = level_logits[str(parent_level)]
            parent_probs = mx.softmax(parent_logits, axis=-1)
            parent_prob = parent_probs[:, parent_idx_in_level]
            
            # Sum child probabilities
            child_prob_sum = 0.0
            for child_label in children:
                child_level = self.label_to_level[child_label]
                child_idx_in_level = self.level_labels[child_level].index(child_label)
                
                child_logits = level_logits[str(child_level)]
                child_probs = mx.softmax(child_logits, axis=-1)
                child_prob_sum = child_prob_sum + child_probs[:, child_idx_in_level]
            
            # Consistency: parent prob should approximate sum of child probs
            consistency_loss = consistency_loss + mx.mean((parent_prob - child_prob_sum) ** 2)
            num_pairs += 1
        
        if num_pairs > 0:
            consistency_loss = consistency_loss / num_pairs
        
        return consistency_loss


class EnsembleClassificationHead(nn.Module):
    """
    Ensemble classification head that combines multiple classification strategies.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_heads: int = 3,
        hidden_dims: Optional[List[int]] = None,
        activations: Optional[List[str]] = None,
        dropout_rates: Optional[List[float]] = None,
        ensemble_method: str = "average",  # "average", "weighted", "attention"
        temperature: float = 1.0,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.ensemble_method = ensemble_method
        self.temperature = temperature
        
        # Default configurations
        if hidden_dims is None:
            hidden_dims = [input_dim // 2] * num_heads
        if activations is None:
            activations = ["gelu", "relu", "silu"][:num_heads]
        if dropout_rates is None:
            dropout_rates = [0.1, 0.2, 0.3][:num_heads]
        
        # Create individual heads
        self.heads = []
        for i in range(num_heads):
            head = self._create_head(
                input_dim,
                num_classes,
                hidden_dims[i],
                activations[i],
                dropout_rates[i]
            )
            self.heads.append(head)
            setattr(self, f"head_{i}", head)
        
        # Ensemble combination layer
        if ensemble_method == "weighted":
            self.ensemble_weights = mx.ones((num_heads,)) / num_heads
        elif ensemble_method == "attention":
            self.attention_layer = nn.Linear(input_dim, num_heads)
    
    def _create_head(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int,
        activation: str,
        dropout_rate: float
    ) -> nn.Module:
        """Create individual classification head."""
        layers = [
            nn.Linear(input_dim, hidden_dim),
            self._get_activation(activation),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        ]
        return nn.Sequential(*layers)
    
    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "mish": nn.Mish(),
        }
        return activations.get(name, nn.GELU())
    
    def forward(self, x: mx.array) -> mx.array:
        """Forward pass through ensemble."""
        # Get predictions from all heads
        all_logits = []
        for head in self.heads:
            logits = head(x) / self.temperature
            all_logits.append(logits)
        
        # Stack logits: [batch_size, num_heads, num_classes]
        stacked_logits = mx.stack(all_logits, axis=1)
        
        # Combine predictions
        if self.ensemble_method == "average":
            combined_logits = mx.mean(stacked_logits, axis=1)
        elif self.ensemble_method == "weighted":
            weights = mx.softmax(self.ensemble_weights)
            weights = mx.reshape(weights, (1, -1, 1))
            combined_logits = mx.sum(stacked_logits * weights, axis=1)
        elif self.ensemble_method == "attention":
            # Compute attention weights
            attention_logits = self.attention_layer(x)
            attention_weights = mx.softmax(attention_logits, axis=-1)
            attention_weights = mx.expand_dims(attention_weights, axis=-1)
            
            # Weighted combination
            combined_logits = mx.sum(stacked_logits * attention_weights, axis=1)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        return combined_logits
    
    def compute_loss(
        self,
        logits: mx.array,
        labels: mx.array,
        class_weights: Optional[mx.array] = None,
        diversity_weight: float = 0.01,
    ) -> mx.array:
        """
        Compute ensemble loss with optional diversity term.
        
        Args:
            logits: Combined logits [batch_size, num_classes]
            labels: True labels [batch_size]
            class_weights: Weights for each class
            diversity_weight: Weight for diversity loss term
        """
        # Standard cross-entropy loss
        ce_loss = nn.losses.cross_entropy(logits, labels, reduction="none")
        
        # Apply class weights if provided
        if class_weights is not None:
            sample_weights = class_weights[labels]
            ce_loss = ce_loss * sample_weights
        
        loss = mx.mean(ce_loss)
        
        # Add diversity loss to encourage different predictions
        if diversity_weight > 0:
            diversity_loss = self._compute_diversity_loss()
            loss = loss + diversity_weight * diversity_loss
        
        return loss
    
    def _compute_diversity_loss(self) -> mx.array:
        """Compute diversity loss to encourage different predictions from heads."""
        # Use negative correlation between head predictions as diversity measure
        all_probs = []
        
        # Get softmax probabilities from each head
        for head in self.heads:
            # Use dummy input to get weights
            dummy_input = mx.zeros((1, head[0].weight.shape[1]))
            logits = head(dummy_input)
            probs = mx.softmax(logits, axis=-1)
            all_probs.append(probs)
        
        # Compute pairwise similarities
        diversity_loss = 0.0
        num_pairs = 0
        
        for i in range(len(all_probs)):
            for j in range(i + 1, len(all_probs)):
                # Cosine similarity between probability distributions
                similarity = mx.sum(all_probs[i] * all_probs[j]) / (
                    mx.sqrt(mx.sum(all_probs[i] ** 2)) * 
                    mx.sqrt(mx.sum(all_probs[j] ** 2))
                )
                diversity_loss = diversity_loss + similarity
                num_pairs += 1
        
        if num_pairs > 0:
            diversity_loss = diversity_loss / num_pairs
        
        return diversity_loss