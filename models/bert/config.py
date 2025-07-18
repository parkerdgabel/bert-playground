"""Unified BERT configuration for the modular BERT architecture.

This module provides configuration classes for BERT models, consolidating
the configuration requirements for both standard and CNN-hybrid variants.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import json
from pathlib import Path


@dataclass
class BertConfig:
    """Configuration class for BERT models.
    
    This configuration supports both standard BERT and CNN-enhanced variants.
    """
    # Model dimensions
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    
    # Vocabulary
    vocab_size: int = 50265
    type_vocab_size: int = 2
    max_position_embeddings: int = 512
    
    # Activation and normalization
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-12
    
    # Dropout
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    
    # Pooling
    pooler_hidden_size: Optional[int] = None
    pooler_dropout: float = 0.0
    pooler_activation: str = "tanh"
    
    # Initialization
    initializer_range: float = 0.02
    
    # Special tokens
    pad_token_id: int = 0
    bos_token_id: int = 101
    eos_token_id: int = 102
    
    
    # MLX optimizations
    use_fused_attention: bool = True
    use_memory_efficient_attention: bool = True
    gradient_checkpointing: bool = False
    
    # Additional pooling options
    compute_additional_pooling: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.pooler_hidden_size is None:
            self.pooler_hidden_size = self.hidden_size
        
        # Ensure head dimension is divisible by number of heads
        assert self.hidden_size % self.num_attention_heads == 0, (
            f"Hidden size ({self.hidden_size}) must be divisible by "
            f"number of attention heads ({self.num_attention_heads})"
        )
    
    def to_dict(self):
        """Convert config to dictionary."""
        return self.__dict__.copy()
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_hf_config(cls, hf_config_dict: Dict[str, Any]) -> "BertConfig":
        """Create BertConfig from HuggingFace config dictionary.
        
        Args:
            hf_config_dict: HuggingFace config dictionary
            
        Returns:
            BertConfig instance
        """
        # Map HuggingFace config keys to our config keys
        hf_to_mlx_mapping = {
            "vocab_size": "vocab_size",
            "hidden_size": "hidden_size", 
            "num_hidden_layers": "num_hidden_layers",
            "num_attention_heads": "num_attention_heads",
            "intermediate_size": "intermediate_size",
            "hidden_act": "hidden_act",
            "hidden_dropout_prob": "hidden_dropout_prob",
            "attention_probs_dropout_prob": "attention_probs_dropout_prob",
            "max_position_embeddings": "max_position_embeddings",
            "type_vocab_size": "type_vocab_size",
            "initializer_range": "initializer_range",
            "layer_norm_eps": "layer_norm_eps",
            "pad_token_id": "pad_token_id",
            "bos_token_id": "bos_token_id",
            "eos_token_id": "eos_token_id",
            # HuggingFace specific fields we can ignore or use defaults for
            "model_type": None,  # We don't need this
            "architectures": None,  # We don't need this
            "torch_dtype": None,  # MLX handles this differently
            "transformers_version": None,  # Not needed
            "use_cache": None,  # Not needed for our implementation
            "classifier_dropout": "hidden_dropout_prob",  # Map to our dropout
            "position_embedding_type": None,  # We use absolute by default
        }
        
        # Extract only the fields we care about
        mlx_config = {}
        for hf_key, mlx_key in hf_to_mlx_mapping.items():
            if hf_key in hf_config_dict and mlx_key is not None:
                mlx_config[mlx_key] = hf_config_dict[hf_key]
        
        return cls(**mlx_config)
    
    def to_hf_config(self) -> Dict[str, Any]:
        """Convert BertConfig to HuggingFace format.
        
        Returns:
            Dictionary in HuggingFace config format
        """
        hf_config = {
            "model_type": "bert",
            "architectures": ["BertModel"],
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "hidden_act": self.hidden_act,
            "hidden_dropout_prob": self.hidden_dropout_prob,
            "attention_probs_dropout_prob": self.attention_probs_dropout_prob,
            "max_position_embeddings": self.max_position_embeddings,
            "type_vocab_size": self.type_vocab_size,
            "initializer_range": self.initializer_range,
            "layer_norm_eps": self.layer_norm_eps,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "position_embedding_type": "absolute",
            "use_cache": True,
            "classifier_dropout": self.hidden_dropout_prob,
            "torch_dtype": "float32",
        }
        
        return hf_config


# Preset configurations
def get_base_config() -> BertConfig:
    """Get configuration for base BERT model."""
    return BertConfig(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
    )


def get_large_config() -> BertConfig:
    """Get configuration for large BERT model."""
    return BertConfig(
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
    )


def get_mini_config() -> BertConfig:
    """Get configuration for mini BERT model (for testing)."""
    return BertConfig(
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=1024,
        max_position_embeddings=128,
    )


