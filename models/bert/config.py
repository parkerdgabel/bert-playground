"""Unified BERT configuration for the modular BERT architecture.

This module provides configuration classes for BERT models, consolidating
the configuration requirements for both standard and CNN-hybrid variants.
"""

from dataclasses import dataclass
from typing import Any


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
    pooler_hidden_size: int | None = None
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

    # ModernBERT/neoBERT specific options
    use_rope: bool = False  # Use Rotary Position Embeddings
    use_geglu: bool = False  # Use GeGLU activation (ModernBERT)
    use_swiglu: bool = False  # Use SwiGLU activation (neoBERT)
    use_pre_norm: bool = False  # Use pre-normalization
    norm_type: str = "layer_norm"  # "layer_norm" or "rms_norm"
    use_bias: bool = True  # Whether to use bias in linear layers
    use_alternating_attention: bool = False  # Use alternating local/global attention
    local_attention_window_size: int = 128  # Window size for local attention
    global_attention_every_n_layers: int = 3  # Frequency of global attention layers
    gate_limit: float | None = None  # Optional gate limit for GLU activations

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
    def from_hf_config(cls, hf_config_dict: dict[str, Any]) -> "BertConfig":
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

    def to_hf_config(self) -> dict[str, Any]:
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

    def save_pretrained(self, save_path):
        """Save config to directory.
        
        Args:
            save_path: Directory to save config
        """
        import json
        from pathlib import Path
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        with open(save_path / "config.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2)


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


def get_modernbert_base_config() -> BertConfig:
    """Get configuration for ModernBERT base model."""
    return BertConfig(
        hidden_size=768,
        num_hidden_layers=22,  # Deeper than BERT base
        num_attention_heads=12,
        intermediate_size=2048,  # Smaller than BERT for efficiency
        max_position_embeddings=8192,  # Extended context
        hidden_act="gelu",
        layer_norm_eps=1e-6,
        # ModernBERT specific
        use_rope=True,
        use_geglu=True,
        use_pre_norm=True,
        norm_type="rms_norm",
        use_bias=False,
        use_alternating_attention=True,
        local_attention_window_size=128,
        global_attention_every_n_layers=3,
        # Optimizations
        use_fused_attention=True,
        use_memory_efficient_attention=True,
    )


def get_modernbert_large_config() -> BertConfig:
    """Get configuration for ModernBERT large model."""
    return BertConfig(
        hidden_size=1024,
        num_hidden_layers=28,
        num_attention_heads=16,
        intermediate_size=2730,  # 2.67x hidden_size
        max_position_embeddings=8192,
        hidden_act="gelu",
        layer_norm_eps=1e-6,
        # ModernBERT specific
        use_rope=True,
        use_geglu=True,
        use_pre_norm=True,
        norm_type="rms_norm",
        use_bias=False,
        use_alternating_attention=True,
        local_attention_window_size=128,
        global_attention_every_n_layers=3,
        # Optimizations
        use_fused_attention=True,
        use_memory_efficient_attention=True,
    )


def get_neobert_config() -> BertConfig:
    """Get configuration for neoBERT model (250M parameters)."""
    return BertConfig(
        hidden_size=768,
        num_hidden_layers=28,  # 28 layers (deeper than BERT base)
        num_attention_heads=12,
        intermediate_size=2048,  # Efficient intermediate size
        max_position_embeddings=4096,  # 4k context length
        vocab_size=50265,  # Standard BERT tokenizer vocab
        hidden_act="silu",  # For SwiGLU
        layer_norm_eps=1e-6,
        # neoBERT specific
        use_rope=True,
        use_swiglu=True,
        use_pre_norm=True,
        norm_type="rms_norm",
        use_bias=False,
        # No alternating attention in neoBERT
        use_alternating_attention=False,
        # Optimizations
        use_fused_attention=True,
        use_memory_efficient_attention=True,
        # Lower dropout for stable training
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )


def get_neobert_mini_config() -> BertConfig:
    """Get configuration for mini neoBERT model (for testing)."""
    return BertConfig(
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=512,
        hidden_act="silu",
        layer_norm_eps=1e-6,
        # neoBERT specific
        use_rope=True,
        use_swiglu=True,
        use_pre_norm=True,
        norm_type="rms_norm",
        use_bias=False,
        use_alternating_attention=False,
        # Optimizations
        use_fused_attention=True,
        use_memory_efficient_attention=True,
    )
