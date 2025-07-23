"""Model entities representing BERT architecture and components."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class ActivationType(Enum):
    """Activation function types."""
    GELU = "gelu"
    RELU = "relu"
    SILU = "silu"
    GEGLU = "geglu"


class AttentionType(Enum):
    """Attention mechanism types."""
    STANDARD = "standard"
    FLASH = "flash"
    ALTERNATING = "alternating"
    SPARSE = "sparse"


@dataclass
class ModelArchitecture:
    """Defines the architecture of a BERT model."""
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    max_position_embeddings: int
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12
    activation: ActivationType = ActivationType.GELU
    attention_type: AttentionType = AttentionType.STANDARD
    use_rope: bool = False
    rope_theta: float = 10000.0
    use_bias: bool = True
    pad_token_id: int = 0
    
    def __post_init__(self):
        """Validate architecture parameters."""
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"Hidden size ({self.hidden_size}) must be divisible by "
                f"number of attention heads ({self.num_attention_heads})"
            )
        if self.intermediate_size < self.hidden_size:
            raise ValueError(
                f"Intermediate size ({self.intermediate_size}) should be "
                f"larger than hidden size ({self.hidden_size})"
            )


@dataclass
class ModelWeights:
    """Container for model weights/parameters."""
    embeddings: Dict[str, Any] = field(default_factory=dict)
    encoder_layers: List[Dict[str, Any]] = field(default_factory=list)
    pooler: Optional[Dict[str, Any]] = None
    task_head: Optional[Dict[str, Any]] = None
    
    @property
    def total_parameters(self) -> int:
        """Calculate total number of parameters."""
        # This is a simplified calculation - actual implementation
        # would recursively count all parameter shapes
        return 0
    
    def get_layer_weights(self, layer_idx: int) -> Dict[str, Any]:
        """Get weights for a specific layer."""
        if 0 <= layer_idx < len(self.encoder_layers):
            return self.encoder_layers[layer_idx]
        raise IndexError(f"Layer index {layer_idx} out of range")


@dataclass
class BertModel:
    """Core BERT model entity."""
    architecture: ModelArchitecture
    weights: Optional[ModelWeights] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_initialized(self) -> bool:
        """Check if model has weights."""
        return self.weights is not None
    
    @property
    def num_layers(self) -> int:
        """Get number of encoder layers."""
        return self.architecture.num_hidden_layers
    
    @property
    def hidden_dim(self) -> int:
        """Get hidden dimension size."""
        return self.architecture.hidden_size
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration as dictionary."""
        return {
            "vocab_size": self.architecture.vocab_size,
            "hidden_size": self.architecture.hidden_size,
            "num_hidden_layers": self.architecture.num_hidden_layers,
            "num_attention_heads": self.architecture.num_attention_heads,
            "intermediate_size": self.architecture.intermediate_size,
            "max_position_embeddings": self.architecture.max_position_embeddings,
            "hidden_dropout_prob": self.architecture.hidden_dropout_prob,
            "attention_probs_dropout_prob": self.architecture.attention_probs_dropout_prob,
            "layer_norm_eps": self.architecture.layer_norm_eps,
            "activation": self.architecture.activation.value,
            "attention_type": self.architecture.attention_type.value,
            "use_rope": self.architecture.use_rope,
            "rope_theta": self.architecture.rope_theta,
            "use_bias": self.architecture.use_bias,
            "pad_token_id": self.architecture.pad_token_id,
        }