"""Domain model for BERT configuration.

This module contains the pure business logic for BERT model configuration,
free from any framework dependencies.
"""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class BertDomainConfig:
    """Configuration for BERT domain model.
    
    This represents the core business configuration for BERT models,
    independent of any ML framework implementation details.
    """
    
    # Model architecture parameters
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    
    # Vocabulary parameters
    vocab_size: int = 50265
    type_vocab_size: int = 2
    max_position_embeddings: int = 512
    
    # Activation and normalization
    hidden_activation: str = "gelu"
    layer_norm_epsilon: float = 1e-12
    
    # Regularization
    hidden_dropout_probability: float = 0.1
    attention_dropout_probability: float = 0.1
    
    # Pooling configuration
    pooler_hidden_size: Optional[int] = None
    pooler_dropout_probability: float = 0.0
    pooler_activation: str = "tanh"
    
    # Initialization
    initializer_range: float = 0.02
    
    # Special token IDs
    pad_token_id: int = 0
    cls_token_id: int = 101
    sep_token_id: int = 102
    
    # Modern BERT variants configuration
    use_rotary_embeddings: bool = False
    use_gated_linear_units: bool = False
    use_pre_layer_normalization: bool = False
    normalization_type: str = "layer_norm"  # "layer_norm" or "rms_norm"
    use_bias_in_linear: bool = True
    
    # Attention pattern configuration
    use_alternating_attention: bool = False
    local_attention_window_size: int = 128
    global_attention_frequency: int = 3
    
    # GLU activation configuration
    glu_activation_type: str = "geglu"  # "geglu" or "swiglu"
    glu_gate_limit: Optional[float] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.pooler_hidden_size is None:
            self.pooler_hidden_size = self.hidden_size
            
        # Business rule: hidden size must be divisible by number of heads
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"Hidden size ({self.hidden_size}) must be divisible by "
                f"number of attention heads ({self.num_attention_heads})"
            )
            
        # Business rule: intermediate size should be reasonable multiple of hidden size
        ratio = self.intermediate_size / self.hidden_size
        if ratio < 2.0 or ratio > 8.0:
            raise ValueError(
                f"Intermediate size ratio ({ratio:.2f}) should be between 2.0 and 8.0"
            )
    
    @property
    def head_dimension(self) -> int:
        """Calculate dimension per attention head."""
        return self.hidden_size // self.num_attention_heads
    
    @property
    def is_modern_variant(self) -> bool:
        """Check if this is a modern BERT variant (ModernBERT/neoBERT)."""
        return any([
            self.use_rotary_embeddings,
            self.use_gated_linear_units,
            self.use_pre_layer_normalization,
            self.normalization_type != "layer_norm",
            self.use_alternating_attention
        ])
    
    @property
    def parameter_count_estimate(self) -> int:
        """Estimate total parameter count for the model."""
        # Embedding parameters
        embedding_params = (
            self.vocab_size * self.hidden_size +  # Token embeddings
            self.type_vocab_size * self.hidden_size +  # Token type embeddings
            self.max_position_embeddings * self.hidden_size  # Position embeddings
        )
        
        # Attention parameters per layer
        attention_params_per_layer = (
            4 * self.hidden_size * self.hidden_size +  # Q, K, V, O projections
            (4 * self.hidden_size if self.use_bias_in_linear else 0)  # Biases
        )
        
        # FFN parameters per layer
        if self.use_gated_linear_units:
            # GLU variants have 3 projections
            ffn_params_per_layer = (
                3 * self.hidden_size * self.intermediate_size +
                self.intermediate_size * self.hidden_size
            )
        else:
            # Standard FFN has 2 projections
            ffn_params_per_layer = (
                2 * self.hidden_size * self.intermediate_size
            )
            
        if self.use_bias_in_linear:
            ffn_params_per_layer += self.intermediate_size + self.hidden_size
        
        # Layer norm parameters
        norm_params_per_layer = 2 * self.hidden_size  # gamma and beta
        if self.use_pre_layer_normalization:
            norm_params_per_layer *= 2  # Two norms per layer
        
        # Total for all layers
        total_layer_params = self.num_hidden_layers * (
            attention_params_per_layer + 
            ffn_params_per_layer + 
            norm_params_per_layer
        )
        
        # Pooler parameters
        pooler_params = (
            self.hidden_size * self.pooler_hidden_size +
            self.pooler_hidden_size
        )
        
        return embedding_params + total_layer_params + pooler_params
    
    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "vocab_size": self.vocab_size,
            "type_vocab_size": self.type_vocab_size,
            "max_position_embeddings": self.max_position_embeddings,
            "hidden_activation": self.hidden_activation,
            "layer_norm_epsilon": self.layer_norm_epsilon,
            "hidden_dropout_probability": self.hidden_dropout_probability,
            "attention_dropout_probability": self.attention_dropout_probability,
            "pooler_hidden_size": self.pooler_hidden_size,
            "pooler_dropout_probability": self.pooler_dropout_probability,
            "pooler_activation": self.pooler_activation,
            "initializer_range": self.initializer_range,
            "pad_token_id": self.pad_token_id,
            "cls_token_id": self.cls_token_id,
            "sep_token_id": self.sep_token_id,
            "use_rotary_embeddings": self.use_rotary_embeddings,
            "use_gated_linear_units": self.use_gated_linear_units,
            "use_pre_layer_normalization": self.use_pre_layer_normalization,
            "normalization_type": self.normalization_type,
            "use_bias_in_linear": self.use_bias_in_linear,
            "use_alternating_attention": self.use_alternating_attention,
            "local_attention_window_size": self.local_attention_window_size,
            "global_attention_frequency": self.global_attention_frequency,
            "glu_activation_type": self.glu_activation_type,
            "glu_gate_limit": self.glu_gate_limit,
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "BertDomainConfig":
        """Create configuration from dictionary."""
        return cls(**{
            k: v for k, v in config_dict.items()
            if k in cls.__annotations__
        })


# Preset configurations for common model sizes
class BertConfigPresets:
    """Factory for common BERT configuration presets."""
    
    @staticmethod
    def base() -> BertDomainConfig:
        """Standard BERT-base configuration."""
        return BertDomainConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
        )
    
    @staticmethod
    def large() -> BertDomainConfig:
        """Standard BERT-large configuration."""
        return BertDomainConfig(
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=4096,
        )
    
    @staticmethod
    def mini() -> BertDomainConfig:
        """Mini BERT configuration for testing."""
        return BertDomainConfig(
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=1024,
            max_position_embeddings=128,
        )
    
    @staticmethod
    def modern_bert_base() -> BertDomainConfig:
        """ModernBERT base configuration."""
        return BertDomainConfig(
            hidden_size=768,
            num_hidden_layers=22,
            num_attention_heads=12,
            intermediate_size=2048,
            max_position_embeddings=8192,
            hidden_activation="gelu",
            layer_norm_epsilon=1e-6,
            use_rotary_embeddings=True,
            use_gated_linear_units=True,
            use_pre_layer_normalization=True,
            normalization_type="rms_norm",
            use_bias_in_linear=False,
            use_alternating_attention=True,
            local_attention_window_size=128,
            global_attention_frequency=3,
            glu_activation_type="geglu",
        )
    
    @staticmethod
    def neo_bert() -> BertDomainConfig:
        """neoBERT configuration (250M parameters)."""
        return BertDomainConfig(
            hidden_size=768,
            num_hidden_layers=28,
            num_attention_heads=12,
            intermediate_size=2048,
            max_position_embeddings=4096,
            vocab_size=50265,
            hidden_activation="silu",
            layer_norm_epsilon=1e-6,
            use_rotary_embeddings=True,
            use_gated_linear_units=True,
            use_pre_layer_normalization=True,
            normalization_type="rms_norm",
            use_bias_in_linear=False,
            use_alternating_attention=False,
            glu_activation_type="swiglu",
            hidden_dropout_probability=0.1,
            attention_dropout_probability=0.1,
        )