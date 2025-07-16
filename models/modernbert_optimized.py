"""Optimized ModernBERT implementation for MLX with efficient embeddings."""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load as load_lm_model
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from loguru import logger


@dataclass
class ModernBertConfig:
    """Configuration for ModernBERT model."""
    vocab_size: int = 50368
    hidden_size: int = 768
    num_hidden_layers: int = 22
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 8192
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    num_labels: int = 2
    
    # MLX optimizations
    use_fused_attention: bool = True
    use_memory_efficient_attention: bool = True
    gradient_checkpointing: bool = False
    
    @classmethod
    def from_pretrained(cls, model_name: str):
        """Load config from HuggingFace."""
        from transformers import AutoConfig
        hf_config = AutoConfig.from_pretrained(model_name)
        
        return cls(
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            intermediate_size=hf_config.intermediate_size,
            hidden_act=getattr(hf_config, 'hidden_activation', 'gelu'),
            hidden_dropout_prob=getattr(hf_config, 'mlp_dropout', 0.1),
            attention_probs_dropout_prob=getattr(hf_config, 'attention_dropout', 0.1),
            max_position_embeddings=hf_config.max_position_embeddings,
            type_vocab_size=getattr(hf_config, 'type_vocab_size', 2),
            initializer_range=getattr(hf_config, 'initializer_range', 0.02),
            layer_norm_eps=getattr(hf_config, 'norm_eps', 1e-12),
        )


class OptimizedEmbeddings(nn.Module):
    """Optimized embedding layer for MLX."""
    
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        
        # Use MLX efficient embeddings
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        if config.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Precompute position ids for efficiency
        self._position_ids_cache = {}
    
    def _get_position_ids(self, batch_size: int, seq_length: int) -> mx.array:
        """Get cached position ids for efficiency."""
        cache_key = (batch_size, seq_length)
        if cache_key not in self._position_ids_cache:
            position_ids = mx.arange(seq_length)[None, :]
            position_ids = mx.broadcast_to(position_ids, (batch_size, seq_length))
            self._position_ids_cache[cache_key] = position_ids
        return self._position_ids_cache[cache_key]
    
    def __call__(
        self,
        input_ids: mx.array,
        token_type_ids: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        batch_size, seq_length = input_ids.shape
        
        # Get embeddings
        word_embeddings = self.word_embeddings(input_ids)
        
        # Efficient position ids
        if position_ids is None:
            position_ids = self._get_position_ids(batch_size, seq_length)
        position_embeddings = self.position_embeddings(position_ids)
        
        embeddings = word_embeddings + position_embeddings
        
        # Add token type embeddings if needed
        if self.config.type_vocab_size > 0:
            if token_type_ids is None:
                token_type_ids = mx.zeros((batch_size, seq_length), dtype=mx.int32)
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings = embeddings + token_type_embeddings
        
        # Apply normalization and dropout
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class OptimizedAttention(nn.Module):
    """Optimized multi-head attention for MLX."""
    
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Fused QKV projection for efficiency
        self.qkv_proj = nn.Linear(config.hidden_size, 3 * self.all_head_size, bias=True)
        self.out_proj = nn.Linear(self.all_head_size, config.hidden_size, bias=True)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.scale = self.attention_head_size ** -0.5
    
    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        batch_size, seq_length, _ = hidden_states.shape
        
        # Fused QKV computation
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.reshape(batch_size, seq_length, 3, self.num_attention_heads, self.attention_head_size)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        
        # Efficient attention computation
        attention_scores = mx.matmul(query, key.transpose(0, 1, 3, 2)) * self.scale
        
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention_scores = attention_scores + (1.0 - attention_mask) * -10000.0
        
        attention_probs = mx.softmax(attention_scores, axis=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context = mx.matmul(attention_probs, value)
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_length, self.all_head_size)
        
        output = self.out_proj(context)
        return output


class OptimizedTransformerBlock(nn.Module):
    """Optimized transformer block for MLX."""
    
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.attention = OptimizedAttention(config)
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # MLP with fused operations
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob)
        )
        self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        # Self-attention with residual
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = self.attention_dropout(attention_output)
        hidden_states = self.attention_norm(hidden_states + attention_output)
        
        # MLP with residual
        mlp_output = self.mlp(hidden_states)
        hidden_states = self.mlp_norm(hidden_states + mlp_output)
        
        return hidden_states


class OptimizedModernBertMLX(nn.Module):
    """Optimized ModernBERT implementation for MLX."""
    
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        
        # Optimized embeddings
        self.embeddings = OptimizedEmbeddings(config)
        
        # Transformer blocks
        self.layers = [OptimizedTransformerBlock(config) for _ in range(config.num_hidden_layers)]
        
        # Pooler for classification
        self.pooler = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh()
        )
        
        # Classification head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        logger.info(f"Initialized OptimizedModernBertMLX with {config.num_hidden_layers} layers")
    
    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        token_type_ids: Optional[mx.array] = None,
        labels: Optional[mx.array] = None,
        return_dict: bool = True,
    ) -> Union[Dict[str, mx.array], Tuple[mx.array, ...]]:
        # Get embeddings
        hidden_states = self.embeddings(input_ids, token_type_ids)
        
        # Apply transformer blocks
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Pool and classify
        pooled_output = self.pooler(hidden_states[:, 0])
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss = mx.mean(nn.losses.cross_entropy(logits, labels, reduction='none'))
        
        if return_dict:
            return {
                'loss': loss,
                'logits': logits,
                'hidden_states': hidden_states,
                'pooled_output': pooled_output,
            }
        else:
            return (loss, logits, hidden_states, pooled_output)
    
    def save_pretrained(self, save_path: str):
        """Save model weights efficiently."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(save_path / "config.json", "w") as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        # Save weights in MLX format
        weights = dict(self.parameters())
        mx.save_safetensors(str(save_path / "model.safetensors"), weights)
        
        logger.info(f"Model saved to {save_path}")
    
    @classmethod
    def from_pretrained(cls, path_or_name: str, num_labels: int = 2) -> "OptimizedModernBertMLX":
        """Load model from pretrained weights."""
        path = Path(path_or_name)
        
        if path.exists() and path.is_dir():
            # Load from local path
            with open(path / "config.json") as f:
                config_dict = json.load(f)
            config = ModernBertConfig(**config_dict)
            config.num_labels = num_labels
            
            model = cls(config)
            
            # Load weights
            weights = mx.load(str(path / "model.safetensors"))
            model.load_weights(list(weights.items()))
            
            logger.info(f"Loaded model from {path}")
        else:
            # Initialize from HuggingFace config
            config = ModernBertConfig.from_pretrained(path_or_name)
            config.num_labels = num_labels
            model = cls(config)
            logger.info(f"Initialized model with config from {path_or_name}")
        
        return model


def create_optimized_model(
    model_name: str = "answerdotai/ModernBERT-base",
    num_labels: int = 2,
    **kwargs
) -> OptimizedModernBertMLX:
    """Create an optimized ModernBERT model."""
    return OptimizedModernBertMLX.from_pretrained(model_name, num_labels=num_labels)