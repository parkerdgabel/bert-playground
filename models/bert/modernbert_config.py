"""
ModernBERT configuration class.

This module defines the configuration for ModernBERT models, incorporating
all the architectural improvements from Answer.AI's 2024 release.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import BertConfig


@dataclass
class ModernBertConfig(BertConfig):
    """
    Configuration class for ModernBERT models.

    ModernBERT introduces several architectural improvements over Classic BERT:
    - RoPE (Rotary Positional Embeddings) instead of learned positional embeddings
    - GeGLU activation functions instead of GELU
    - Alternating attention mechanism (global/local)
    - Streamlined architecture without bias terms
    - Additional normalization layers
    - Extended sequence length support (8192 tokens)
    """

    # Core architecture settings
    model_type: str = "modernbert"

    # Extended sequence length
    max_position_embeddings: int = 8192

    # RoPE configuration
    use_rope: bool = True
    rope_base: float = 10000.0
    rope_scaling: dict[str, Any] | None = None

    # GeGLU configuration
    use_geglu: bool = True
    geglu_limit: float | None = None  # Optional limit for GeGLU gate values

    # Alternating attention configuration
    use_alternating_attention: bool = True
    local_attention_window: int = 128
    global_attention_every_n_layers: int = 3

    # Streamlined architecture (no bias terms)
    use_bias: bool = False
    use_layer_norm_bias: bool = False

    # Additional normalization
    use_post_embedding_norm: bool = True
    post_embedding_norm_eps: float = 1e-12

    # Flash attention configuration
    use_flash_attention: bool = True
    flash_attention_dropout: float = 0.0

    # Model size presets
    model_size: str = "base"  # "base" or "large"

    # ModernBERT specific defaults
    hidden_act: str = "geglu"  # Use GeGLU by default

    def __post_init__(self):
        """Post-initialization setup for ModernBERT specific settings."""
        super().__post_init__()

        # Set model-specific defaults based on size if not explicitly set
        if (
            self.model_size == "base"
            and self.hidden_size == 768
            and self.num_hidden_layers == 12
        ):
            # Only apply defaults if we're using the base BERT defaults
            self.hidden_size = 768
            self.num_hidden_layers = 22
            self.num_attention_heads = 12
            self.intermediate_size = 3072
            self.vocab_size = 50368  # ModernBERT vocab size
        elif (
            self.model_size == "large"
            and self.hidden_size == 1024
            and self.num_hidden_layers == 24
        ):
            # Only apply defaults if we're using the large BERT defaults
            self.hidden_size = 1024
            self.num_hidden_layers = 28
            self.num_attention_heads = 16
            self.intermediate_size = 4096
            self.vocab_size = 50368  # ModernBERT vocab size

        # Validate alternating attention configuration
        if self.use_alternating_attention:
            if self.global_attention_every_n_layers <= 0:
                raise ValueError("global_attention_every_n_layers must be positive")
            if self.local_attention_window <= 0:
                raise ValueError("local_attention_window must be positive")

        # Validate RoPE configuration
        if self.use_rope and self.rope_base <= 0:
            raise ValueError("rope_base must be positive")

        # Set attention dropout for flash attention
        if self.use_flash_attention:
            self.flash_attention_dropout = self.attention_probs_dropout_prob

    def get_attention_type(self, layer_idx: int) -> str:
        """
        Determine attention type for a given layer.

        Args:
            layer_idx: Layer index (0-based)

        Returns:
            "global" or "local" attention type
        """
        if not self.use_alternating_attention:
            return "global"

        # Global attention every N layers (0-indexed)
        if (layer_idx + 1) % self.global_attention_every_n_layers == 0:
            return "global"
        else:
            return "local"

    def is_global_attention_layer(self, layer_idx: int) -> bool:
        """Check if a layer uses global attention."""
        return self.get_attention_type(layer_idx) == "global"

    def is_local_attention_layer(self, layer_idx: int) -> bool:
        """Check if a layer uses local attention."""
        return self.get_attention_type(layer_idx) == "local"

    @classmethod
    def from_bert_config(cls, bert_config: BertConfig) -> "ModernBertConfig":
        """
        Create ModernBertConfig from a standard BertConfig.

        Args:
            bert_config: Standard BERT configuration

        Returns:
            ModernBertConfig with BERT settings preserved
        """
        # Convert to dict and update with ModernBERT defaults
        config_dict = bert_config.to_dict()

        # Update with ModernBERT specific settings
        config_dict.update(
            {
                "model_type": "modernbert",
                "use_rope": True,
                "use_geglu": True,
                "use_alternating_attention": True,
                "use_bias": False,
                "use_layer_norm_bias": False,
                "use_post_embedding_norm": True,
                "use_flash_attention": True,
                "hidden_act": "geglu",
                "max_position_embeddings": 8192,
            }
        )

        return cls.from_dict(config_dict)

    @classmethod
    def get_base_config(cls) -> "ModernBertConfig":
        """Get ModernBERT-base configuration."""
        return cls(
            model_size="base",
            hidden_size=768,
            num_hidden_layers=22,
            num_attention_heads=12,
            intermediate_size=3072,
            vocab_size=50368,
        )

    @classmethod
    def get_large_config(cls) -> "ModernBertConfig":
        """Get ModernBERT-large configuration."""
        return cls(
            model_size="large",
            hidden_size=1024,
            num_hidden_layers=28,
            num_attention_heads=16,
            intermediate_size=4096,
            vocab_size=50368,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        config_dict = super().to_dict()

        # Add ModernBERT specific fields
        modernbert_fields = {
            "use_rope": self.use_rope,
            "rope_base": self.rope_base,
            "rope_scaling": self.rope_scaling,
            "use_geglu": self.use_geglu,
            "geglu_limit": self.geglu_limit,
            "use_alternating_attention": self.use_alternating_attention,
            "local_attention_window": self.local_attention_window,
            "global_attention_every_n_layers": self.global_attention_every_n_layers,
            "use_bias": self.use_bias,
            "use_layer_norm_bias": self.use_layer_norm_bias,
            "use_post_embedding_norm": self.use_post_embedding_norm,
            "post_embedding_norm_eps": self.post_embedding_norm_eps,
            "use_flash_attention": self.use_flash_attention,
            "flash_attention_dropout": self.flash_attention_dropout,
            "model_size": self.model_size,
        }

        config_dict.update(modernbert_fields)
        return config_dict

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "ModernBertConfig":
        """Create config from dictionary."""
        # Remove any extra fields not in the dataclass
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}

        return cls(**filtered_dict)

    def save_pretrained(self, save_directory: str | Path):
        """Save configuration to directory."""
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        config_file = save_directory / "config.json"
        with open(config_file, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_pretrained(cls, model_path: str | Path) -> "ModernBertConfig":
        """Load configuration from directory."""
        model_path = Path(model_path)
        config_file = model_path / "config.json"

        with open(config_file) as f:
            config_dict = json.load(f)

        return cls.from_dict(config_dict)

    def __repr__(self) -> str:
        """String representation of the config."""
        return (
            f"ModernBertConfig("
            f"model_size={self.model_size}, "
            f"hidden_size={self.hidden_size}, "
            f"num_layers={self.num_hidden_layers}, "
            f"num_heads={self.num_attention_heads}, "
            f"max_seq_len={self.max_position_embeddings}, "
            f"use_rope={self.use_rope}, "
            f"use_geglu={self.use_geglu}, "
            f"use_alternating_attention={self.use_alternating_attention}"
            f")"
        )


# Predefined configurations
MODERNBERT_BASE_CONFIG = ModernBertConfig.get_base_config()
MODERNBERT_LARGE_CONFIG = ModernBertConfig.get_large_config()


def get_modernbert_config(size: str = "base") -> ModernBertConfig:
    """
    Get a predefined ModernBERT configuration.

    Args:
        size: Model size ("base" or "large")

    Returns:
        ModernBertConfig instance
    """
    if size == "base":
        return MODERNBERT_BASE_CONFIG
    elif size == "large":
        return MODERNBERT_LARGE_CONFIG
    else:
        raise ValueError(f"Unknown model size: {size}. Choose 'base' or 'large'.")
