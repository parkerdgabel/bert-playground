"""Unified ModernBERT implementation for MLX with all optimizations.

This module combines the best features from both the original and optimized
implementations, providing a single, high-performance ModernBERT model.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from loguru import logger
import warnings


@dataclass
class ModernBertConfig:
    """Configuration for ModernBERT model with MLX optimizations."""

    # Model architecture
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

    # MLX optimizations (always enabled in unified version)
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
            hidden_act=getattr(hf_config, "hidden_activation", "gelu"),
            hidden_dropout_prob=getattr(hf_config, "mlp_dropout", 0.1),
            attention_probs_dropout_prob=getattr(hf_config, "attention_dropout", 0.1),
            max_position_embeddings=hf_config.max_position_embeddings,
            type_vocab_size=getattr(hf_config, "type_vocab_size", 2),
            initializer_range=getattr(hf_config, "initializer_range", 0.02),
            layer_norm_eps=getattr(hf_config, "norm_eps", 1e-12),
        )

    def to_dict(self):
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, config_dict: Dict):
        """Create config from dictionary."""
        return cls(**config_dict)


class OptimizedEmbeddings(nn.Module):
    """Optimized embedding layer with position ID caching."""

    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Cache position IDs for efficiency
        self._position_ids_cache = {}

    def _get_position_ids(self, seq_length: int):
        """Get cached position IDs or create new ones."""
        if seq_length not in self._position_ids_cache:
            self._position_ids_cache[seq_length] = mx.arange(seq_length)
        return self._position_ids_cache[seq_length]

    def __call__(
        self,
        input_ids: mx.array,
        token_type_ids: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        batch_size, seq_length = input_ids.shape

        if position_ids is None:
            position_ids = self._get_position_ids(seq_length)
            position_ids = mx.broadcast_to(
                position_ids[None, :], (batch_size, seq_length)
            )

        if token_type_ids is None:
            token_type_ids = mx.zeros((batch_size, seq_length), dtype=mx.int32)

        inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class FusedMultiHeadAttention(nn.Module):
    """Optimized multi-head attention with fused QKV projections."""

    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.attention_dropout = config.attention_probs_dropout_prob

        # Fused QKV projection for efficiency
        self.qkv_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=True)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        training: bool = True,
    ) -> Tuple[mx.array, mx.array]:
        batch_size, seq_length, hidden_size = hidden_states.shape

        # Fused QKV computation
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.reshape(batch_size, seq_length, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scale = 1.0 / np.sqrt(self.head_dim)
        scores = mx.matmul(query, key.transpose(0, 1, 3, 2)) * scale

        if attention_mask is not None:
            # Ensure attention mask is properly shaped
            if attention_mask.ndim == 2:
                # Convert 2D mask to 4D mask for broadcasting
                # Shape: [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
                attention_mask = attention_mask[:, None, None, :]
                # Create a full attention mask by broadcasting
                attention_mask = (1.0 - attention_mask) * -10000.0
            scores = scores + attention_mask

        attention_probs = mx.softmax(scores, axis=-1)

        if training and self.attention_dropout > 0:
            attention_probs = self.dropout(attention_probs)

        context = mx.matmul(attention_probs, value)
        context = context.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_length, hidden_size
        )

        output = self.out_proj(context)

        return output, attention_probs


class TransformerBlock(nn.Module):
    """Transformer block with optimized attention and feed-forward."""

    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.attention = FusedMultiHeadAttention(config)
        self.attention_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU() if config.hidden_act == "gelu" else nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
        )
        self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        training: bool = True,
    ) -> Tuple[mx.array, mx.array]:
        # Apply layer norm first
        normed_hidden_states = self.attention_norm(hidden_states)
        
        # Self-attention with residual
        attn_output, attn_weights = self.attention(
            normed_hidden_states, attention_mask, training
        )
        hidden_states = hidden_states + attn_output

        # Feed-forward with residual
        mlp_output = self.mlp(self.mlp_norm(hidden_states))
        hidden_states = hidden_states + mlp_output

        return hidden_states, attn_weights


class ModernBertModel(nn.Module):
    """Unified ModernBERT model with all optimizations."""

    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        self.embeddings = OptimizedEmbeddings(config)
        self.encoder = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.pooler = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size), nn.Tanh()
        )

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        token_type_ids: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        output_attentions: bool = False,
        training: bool = True,
    ) -> Dict[str, mx.array]:
        batch_size, seq_length = input_ids.shape

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = mx.ones((batch_size, seq_length))

        # Convert to attention scores mask
        extended_attention_mask = (1.0 - attention_mask[:, None, None, :]) * -10000.0

        # Get embeddings
        hidden_states = self.embeddings(input_ids, token_type_ids, position_ids)

        # Pass through encoder
        all_attentions = []
        for layer in self.encoder.layers:
            hidden_states, attn_weights = layer(
                hidden_states, extended_attention_mask, training
            )
            if output_attentions:
                all_attentions.append(attn_weights)

        # Pool the first token (CLS token)
        pooled_output = self.pooler(hidden_states[:, 0])

        outputs = {"last_hidden_state": hidden_states, "pooler_output": pooled_output}

        if output_attentions:
            outputs["attentions"] = all_attentions

        return outputs

    @classmethod
    def from_pretrained(cls, model_path: Union[str, Path], **kwargs):
        """Load model from pretrained weights."""
        model_path = Path(model_path)

        if model_path.is_dir():
            # Load from local directory
            config_path = model_path / "config.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    config_dict = json.load(f)
                config = ModernBertConfig.from_dict(config_dict)
            else:
                config = ModernBertConfig()

            model = cls(config)

            # Load weights
            weights_path = model_path / "model.safetensors"
            if weights_path.exists():
                model.load_weights(str(weights_path))
            else:
                logger.warning(
                    f"No weights found at {weights_path}, using random initialization"
                )
        else:
            # Load from HuggingFace
            logger.info(f"Loading config from HuggingFace: {model_path}")
            config = ModernBertConfig.from_pretrained(str(model_path))
            model = cls(config)
            logger.warning(
                "HuggingFace weight conversion not implemented, using random initialization"
            )

        return model

    def save_pretrained(self, save_path: Union[str, Path]):
        """Save model to directory."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(save_path / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Save weights using safetensors
        from mlx.utils import tree_flatten
        
        flat_params = tree_flatten(self.parameters())
        weights = {k: v for k, v in flat_params}
        mx.save_safetensors(str(save_path / "model.safetensors"), weights)

        logger.info(f"Model saved to {save_path}")


# Backward compatibility imports
def create_model(
    config: Optional[ModernBertConfig] = None, **kwargs
) -> ModernBertModel:
    """Create a ModernBERT model with optional config."""
    if config is None:
        config = ModernBertConfig(**kwargs)
    return ModernBertModel(config)


# Legacy function names for compatibility
def MultiHeadAttention(config: ModernBertConfig):
    """Legacy function name - use FusedMultiHeadAttention instead."""
    warnings.warn(
        "MultiHeadAttention is deprecated. Use FusedMultiHeadAttention for better performance.",
        DeprecationWarning,
        stacklevel=2,
    )
    return FusedMultiHeadAttention(config)


def BertEmbeddings(config: ModernBertConfig):
    """Legacy function name - use OptimizedEmbeddings instead."""
    warnings.warn(
        "BertEmbeddings is deprecated. Use OptimizedEmbeddings for better performance.",
        DeprecationWarning,
        stacklevel=2,
    )
    return OptimizedEmbeddings(config)
