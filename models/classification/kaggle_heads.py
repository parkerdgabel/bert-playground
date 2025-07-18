"""
Kaggle-specific classification heads for common competition types.

These heads target specific types of Kaggle competitions that are common
but may not be covered by standard classification heads.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from loguru import logger


class TimeSeriesClassificationHead(nn.Module):
    """
    Time series classification head for sequential data.
    Designed for Kaggle competitions involving temporal sequences.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        sequence_length: int,
        hidden_dim: int = 256,
        num_lstm_layers: int = 2,
        dropout_rate: float = 0.2,
        use_attention: bool = True,
        bidirectional: bool = True,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.use_attention = use_attention
        
        # LSTM layers for temporal modeling
        lstm_hidden = hidden_dim // 2 if bidirectional else hidden_dim
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            dropout=dropout_rate if num_lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        lstm_output_dim = hidden_dim
        
        # Attention mechanism for focusing on important time steps
        if use_attention:
            self.attention = nn.MultiHeadAttention(
                embed_dim=lstm_output_dim,
                num_heads=8,
                dropout=dropout_rate,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(lstm_output_dim)
        
        # Classification layers
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(lstm_output_dim)
        self.classifier = nn.Linear(lstm_output_dim, num_classes)
        
    def __call__(
        self, 
        embeddings: mx.array, 
        attention_mask: Optional[mx.array] = None
    ) -> Dict[str, mx.array]:
        """
        Forward pass for time series classification.
        
        Args:
            embeddings: [batch_size, sequence_length, input_dim]
            attention_mask: [batch_size, sequence_length]
            
        Returns:
            Dictionary containing logits and attention weights
        """
        batch_size, seq_len, input_dim = embeddings.shape
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(embeddings)
        
        # Apply attention if enabled
        if self.use_attention:
            # Self-attention over time steps
            attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
            lstm_out = self.attention_norm(lstm_out + attn_out)
        
        # Global pooling - use last time step or attention-weighted average
        if attention_mask is not None:
            # Mask padding tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(lstm_out.shape)
            lstm_out = lstm_out * mask_expanded
            
            # Weighted average based on mask
            lengths = attention_mask.sum(axis=1, keepdims=True).astype(mx.float32)
            pooled = lstm_out.sum(axis=1) / mx.maximum(lengths, mx.array(1.0))
        else:
            # Simple average pooling
            pooled = lstm_out.mean(axis=1)
        
        # Classification
        pooled = self.layer_norm(pooled)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        
        result = {"logits": logits}
        if self.use_attention and 'attn_weights' in locals():
            result["attention_weights"] = attn_weights
            
        return result


class RankingHead(nn.Module):
    """
    Learning-to-rank head for recommendation and ranking competitions.
    Implements listwise ranking using ListNet or similar approaches.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_hidden_layers: int = 2,
        dropout_rate: float = 0.1,
        ranking_loss: str = "listnet",  # listnet, ranknet, lambdarank
        temperature: float = 1.0,
    ):
        super().__init__()
        
        self.ranking_loss = ranking_loss
        self.temperature = temperature
        
        # Build ranking network
        layers = []
        current_dim = input_dim
        
        for i in range(num_hidden_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.LayerNorm(hidden_dim),
            ])
            current_dim = hidden_dim
        
        # Final scoring layer
        layers.append(nn.Linear(current_dim, 1))
        
        self.scoring_network = nn.Sequential(*layers)
        
    def __call__(self, embeddings: mx.array) -> Dict[str, mx.array]:
        """
        Forward pass for ranking.
        
        Args:
            embeddings: [batch_size, num_items, input_dim] or [batch_size, input_dim]
            
        Returns:
            Dictionary containing scores and rankings
        """
        if embeddings.ndim == 2:
            # Single item scoring
            scores = self.scoring_network(embeddings).squeeze(-1)
        else:
            # Multiple items - score each item
            batch_size, num_items, input_dim = embeddings.shape
            # Reshape for batch processing
            flat_embeddings = embeddings.reshape(-1, input_dim)
            flat_scores = self.scoring_network(flat_embeddings).squeeze(-1)
            scores = flat_scores.reshape(batch_size, num_items)
        
        # Apply temperature scaling
        scores = scores / self.temperature
        
        # Convert scores to rankings/probabilities
        if self.ranking_loss == "listnet":
            # ListNet uses softmax for probability distribution
            rankings = mx.softmax(scores, axis=-1)
        else:
            # For other methods, just return raw scores
            rankings = scores
        
        return {
            "scores": scores,
            "rankings": rankings,
            "logits": scores  # For compatibility
        }


class ContrastiveLearningHead(nn.Module):
    """
    Contrastive learning head for similarity and retrieval tasks.
    Useful for Kaggle competitions involving matching, similarity, or retrieval.
    """
    
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 128,
        temperature: float = 0.07,
        normalize_embeddings: bool = True,
        projection_hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.normalize_embeddings = normalize_embeddings
        
        # Projection network
        if projection_hidden_dim:
            self.projection = nn.Sequential(
                nn.Linear(input_dim, projection_hidden_dim),
                nn.ReLU(),
                nn.Linear(projection_hidden_dim, embedding_dim)
            )
        else:
            self.projection = nn.Linear(input_dim, embedding_dim)
    
    def __call__(self, embeddings: mx.array) -> Dict[str, mx.array]:
        """
        Forward pass for contrastive learning.
        
        Args:
            embeddings: [batch_size, input_dim] or [batch_size, 2, input_dim] for pairs
            
        Returns:
            Dictionary containing projected embeddings and similarities
        """
        if embeddings.ndim == 3:
            # Paired inputs - compute for both items
            batch_size, num_items, input_dim = embeddings.shape
            flat_embeddings = embeddings.reshape(-1, input_dim)
            projected = self.projection(flat_embeddings)
            projected = projected.reshape(batch_size, num_items, self.embedding_dim)
            
            if self.normalize_embeddings:
                projected = projected / mx.linalg.norm(projected, axis=-1, keepdims=True)
            
            # Compute similarity between pairs
            if num_items == 2:
                # Cosine similarity between paired items
                sim = (projected[:, 0] * projected[:, 1]).sum(axis=-1)
                logits = sim / self.temperature
            else:
                # All-pairs similarity
                # projected: [batch_size, num_items, embed_dim]
                sim_matrix = mx.matmul(projected, projected.transpose(0, 2, 1))
                logits = sim_matrix / self.temperature
        else:
            # Single embeddings
            projected = self.projection(embeddings)
            
            if self.normalize_embeddings:
                projected = projected / mx.linalg.norm(projected, axis=-1, keepdims=True)
            
            logits = projected  # For downstream similarity computation
        
        return {
            "embeddings": projected,
            "logits": logits,
            "similarities": logits if "sim" in locals() else None
        }


class MultiTaskHead(nn.Module):
    """
    Multi-task learning head for competitions with multiple objectives.
    Can handle different types of tasks simultaneously.
    """
    
    def __init__(
        self,
        input_dim: int,
        task_configs: Dict[str, Dict[str, Any]],
        shared_hidden_dim: int = 256,
        task_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize multi-task head.
        
        Args:
            input_dim: Input dimension
            task_configs: Dict of task_name -> {type: str, num_classes: int, ...}
            shared_hidden_dim: Shared representation dimension
            task_weights: Optional task weighting for loss combination
        """
        super().__init__()
        
        self.task_configs = task_configs
        self.task_weights = task_weights or {name: 1.0 for name in task_configs}
        
        # Shared representation layer
        self.shared_layer = nn.Sequential(
            nn.Linear(input_dim, shared_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(shared_hidden_dim)
        )
        
        # Task-specific heads
        self.task_heads = {}
        for task_name, config in task_configs.items():
            task_type = config.get("type", "classification")
            num_classes = config.get("num_classes", 2)
            hidden_dim = config.get("hidden_dim", 128)
            
            if task_type == "classification":
                head = nn.Sequential(
                    nn.Linear(shared_hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, num_classes)
                )
            elif task_type == "regression":
                head = nn.Sequential(
                    nn.Linear(shared_hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, 1)
                )
            else:
                raise ValueError(f"Unsupported task type: {task_type}")
            
            self.task_heads[task_name] = head
    
    def __call__(self, embeddings: mx.array) -> Dict[str, mx.array]:
        """
        Forward pass for multi-task learning.
        
        Args:
            embeddings: [batch_size, input_dim]
            
        Returns:
            Dictionary with task_name -> logits mappings
        """
        # Shared representation
        shared_repr = self.shared_layer(embeddings)
        
        # Task-specific predictions
        outputs = {}
        for task_name, head in self.task_heads.items():
            outputs[f"{task_name}_logits"] = head(shared_repr)
        
        # For compatibility, also include a main 'logits' output
        # Use the first task as the primary one
        primary_task = list(self.task_heads.keys())[0]
        outputs["logits"] = outputs[f"{primary_task}_logits"]
        
        return outputs


class MetricLearningHead(nn.Module):
    """
    Metric learning head for learning embeddings with specific distance properties.
    Useful for similarity-based Kaggle competitions.
    """
    
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 128,
        margin: float = 0.5,
        distance_metric: str = "cosine",  # cosine, euclidean, manhattan
        normalize_embeddings: bool = True,
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.distance_metric = distance_metric
        self.normalize_embeddings = normalize_embeddings
        
        # Embedding projection
        self.embedding_layer = nn.Sequential(
            nn.Linear(input_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
    
    def __call__(self, embeddings: mx.array) -> Dict[str, mx.array]:
        """
        Forward pass for metric learning.
        
        Args:
            embeddings: [batch_size, input_dim]
            
        Returns:
            Dictionary containing learned embeddings and distances
        """
        # Project to embedding space
        learned_embeddings = self.embedding_layer(embeddings)
        
        if self.normalize_embeddings:
            learned_embeddings = learned_embeddings / mx.linalg.norm(
                learned_embeddings, axis=-1, keepdims=True
            )
        
        return {
            "embeddings": learned_embeddings,
            "logits": learned_embeddings,  # For compatibility
        }
    
    def compute_distance(self, emb1: mx.array, emb2: mx.array) -> mx.array:
        """Compute distance between embeddings."""
        if self.distance_metric == "cosine":
            # Cosine similarity (higher = more similar)
            return 1 - (emb1 * emb2).sum(axis=-1)
        elif self.distance_metric == "euclidean":
            return mx.linalg.norm(emb1 - emb2, axis=-1)
        elif self.distance_metric == "manhattan":
            return mx.abs(emb1 - emb2).sum(axis=-1)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")


# Registry of Kaggle-specific heads
KAGGLE_HEAD_REGISTRY = {
    "time_series": TimeSeriesClassificationHead,
    "ranking": RankingHead,
    "contrastive": ContrastiveLearningHead,
    "multi_task": MultiTaskHead,
    "metric_learning": MetricLearningHead,
}


def create_kaggle_head(
    head_type: str,
    input_dim: int,
    **kwargs
) -> nn.Module:
    """
    Factory function for creating Kaggle-specific heads.
    
    Args:
        head_type: Type of head to create
        input_dim: Input dimension
        **kwargs: Head-specific arguments
        
    Returns:
        Initialized head module
    """
    if head_type not in KAGGLE_HEAD_REGISTRY:
        raise ValueError(
            f"Unknown Kaggle head type: {head_type}. "
            f"Available: {list(KAGGLE_HEAD_REGISTRY.keys())}"
        )
    
    head_class = KAGGLE_HEAD_REGISTRY[head_type]
    return head_class(input_dim=input_dim, **kwargs)