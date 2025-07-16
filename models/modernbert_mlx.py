import mlx.core as mx
import mlx.nn as nn
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import json
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
import numpy as np


@dataclass
class ModernBertConfig:
    vocab_size: int = 50368
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 8192
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    num_labels: int = 2  # Binary classification for Titanic
    
    @classmethod
    def from_pretrained(cls, model_name: str):
        # Load config from HuggingFace
        from transformers import AutoConfig
        hf_config = AutoConfig.from_pretrained(model_name)
        
        return cls(
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            intermediate_size=hf_config.intermediate_size,
            hidden_act=getattr(hf_config, 'hidden_activation', 'gelu'),  # ModernBERT uses 'hidden_activation'
            hidden_dropout_prob=getattr(hf_config, 'mlp_dropout', 0.1),  # ModernBERT uses 'mlp_dropout'
            attention_probs_dropout_prob=getattr(hf_config, 'attention_dropout', 0.1),  # ModernBERT uses 'attention_dropout'
            max_position_embeddings=hf_config.max_position_embeddings,
            type_vocab_size=getattr(hf_config, 'type_vocab_size', 2),
            initializer_range=getattr(hf_config, 'initializer_range', 0.02),
            layer_norm_eps=getattr(hf_config, 'norm_eps', 1e-12),  # ModernBERT uses 'norm_eps'
        )


class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None
    ) -> mx.array:
        batch_size, seq_length, _ = hidden_states.shape
        
        # Linear transformations
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)
        
        # Reshape for multi-head attention
        query_layer = query_layer.reshape(
            batch_size, seq_length, self.num_attention_heads, self.attention_head_size
        ).transpose(0, 2, 1, 3)
        
        key_layer = key_layer.reshape(
            batch_size, seq_length, self.num_attention_heads, self.attention_head_size
        ).transpose(0, 2, 1, 3)
        
        value_layer = value_layer.reshape(
            batch_size, seq_length, self.num_attention_heads, self.attention_head_size
        ).transpose(0, 2, 1, 3)
        
        # Attention scores
        attention_scores = mx.matmul(query_layer, key_layer.transpose(0, 1, 3, 2))
        attention_scores = attention_scores / mx.sqrt(mx.array(self.attention_head_size))
        
        # Apply attention mask
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention_scores = attention_scores + (1.0 - attention_mask) * -10000.0
        
        # Normalize attention scores
        attention_probs = mx.softmax(attention_scores, axis=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context_layer = mx.matmul(attention_probs, value_layer)
        
        # Reshape back
        context_layer = context_layer.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_length, self.all_head_size
        )
        
        return context_layer


class TransformerBlock(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.attention_output = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.activation = nn.gelu if config.hidden_act == "gelu" else nn.relu
        
    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None
    ) -> mx.array:
        # Self-attention
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = self.attention_output(attention_output)
        attention_output = self.attention_dropout(attention_output)
        hidden_states = self.attention_norm(hidden_states + attention_output)
        
        # Feed-forward
        intermediate_output = self.activation(self.intermediate(hidden_states))
        output = self.output(intermediate_output)
        output = self.output_dropout(output)
        hidden_states = self.output_norm(hidden_states + output)
        
        return hidden_states


class ModernBertMLX(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.embedding_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.embedding_dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Transformer blocks
        self.layers = [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        
        # Classification head
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        token_type_ids: Optional[mx.array] = None
    ) -> Dict[str, mx.array]:
        batch_size, seq_length = input_ids.shape
        
        # Create position ids
        position_ids = mx.arange(seq_length)[None, :].astype(mx.int32)
        position_ids = mx.broadcast_to(position_ids, (batch_size, seq_length))
        
        # Create token type ids if not provided
        if token_type_ids is None:
            token_type_ids = mx.zeros((batch_size, seq_length), dtype=mx.int32)
        
        # Embeddings
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.embedding_norm(embeddings)
        embeddings = self.embedding_dropout(embeddings)
        
        # Transformer blocks
        hidden_states = embeddings
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Pooling - take [CLS] token representation
        pooled_output = hidden_states[:, 0]
        pooled_output = mx.tanh(self.pooler(pooled_output))
        
        # Classification
        logits = self.classifier(pooled_output)
        
        return {
            'logits': logits,
            'hidden_states': hidden_states,
            'pooled_output': pooled_output
        }
    
    @classmethod
    def from_pretrained(cls, model_name: str, num_labels: int = 2):
        config = ModernBertConfig.from_pretrained(model_name)
        config.num_labels = num_labels
        
        # Initialize model
        model = cls(config)
        
        # Load pretrained weights (simplified - in practice, need proper weight conversion)
        print(f"Note: Weight loading from {model_name} not fully implemented. Using random initialization.")
        
        return model
    
    def save_pretrained(self, save_path: str):
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(save_path / "config.json", "w") as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        # Save weights using numpy format for compatibility
        import numpy as np
        weights = {}
        for name, param in self.parameters().items():
            weights[name] = np.array(param)
        
        # Save as numpy archive
        np.savez(str(save_path / "model_weights.npz"), **weights)


def create_model(
    model_name: str = "answerdotai/ModernBERT-base",
    num_labels: int = 2
) -> ModernBertMLX:
    return ModernBertMLX.from_pretrained(model_name, num_labels=num_labels)