"""Unified BERT configuration for the modular BERT architecture.

This module provides configuration classes for BERT models, consolidating
the configuration requirements for both standard and CNN-hybrid variants.
"""

from dataclasses import dataclass, field
from typing import Optional, List


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
    
    # CNN hybrid settings (optional)
    use_cnn_layers: bool = False
    cnn_kernel_sizes: List[int] = field(default_factory=lambda: [3, 5, 7])
    cnn_num_filters: int = 128
    cnn_dropout: float = 0.1
    
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


def get_cnn_hybrid_config() -> BertConfig:
    """Get configuration for CNN-enhanced BERT model."""
    config = get_base_config()
    config.use_cnn_layers = True
    config.cnn_kernel_sizes = [3, 5, 7]
    config.cnn_num_filters = 128
    return config


# Backward compatibility aliases
ModernBertConfig = BertConfig
CNNHybridConfig = BertConfig  # CNN settings are part of BertConfig now