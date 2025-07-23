"""Domain model for BERT architecture.

This module contains the pure business logic for BERT model architecture,
defining the structure and computation flow without framework dependencies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, TypeVar, Generic
from enum import Enum

from .bert_config import BertDomainConfig
from .bert_output import BertDomainOutput


# Generic type for framework-specific implementations
TArray = TypeVar('TArray')
TModule = TypeVar('TModule')


class AttentionType(Enum):
    """Types of attention mechanisms."""
    GLOBAL = "global"
    LOCAL = "local"
    SLIDING_WINDOW = "sliding_window"
    SPARSE = "sparse"


class ActivationType(Enum):
    """Types of activation functions."""
    GELU = "gelu"
    RELU = "relu"
    SILU = "silu"  # Swish
    TANH = "tanh"
    LINEAR = "linear"


class NormalizationType(Enum):
    """Types of normalization."""
    LAYER_NORM = "layer_norm"
    RMS_NORM = "rms_norm"
    BATCH_NORM = "batch_norm"


@dataclass
class LayerComponents:
    """Components that make up a transformer layer."""
    
    attention_type: AttentionType
    use_pre_normalization: bool
    normalization_type: NormalizationType
    feedforward_type: str  # "standard", "glu", "geglu", "swiglu"
    dropout_rate: float
    layer_index: int
    
    @property
    def is_global_attention(self) -> bool:
        """Check if this layer uses global attention."""
        return self.attention_type == AttentionType.GLOBAL
    
    @property
    def requires_position_encoding(self) -> bool:
        """Check if this layer type requires position encoding."""
        return self.attention_type in [AttentionType.GLOBAL, AttentionType.LOCAL]


class BertLayer(ABC, Generic[TModule]):
    """Abstract representation of a BERT transformer layer."""
    
    def __init__(self, config: BertDomainConfig, layer_index: int):
        self.config = config
        self.layer_index = layer_index
        self.components = self._create_components()
    
    @abstractmethod
    def _create_components(self) -> LayerComponents:
        """Create the components for this layer."""
        pass
    
    @property
    def attention_type(self) -> AttentionType:
        """Determine attention type for this layer."""
        if not self.config.use_alternating_attention:
            return AttentionType.GLOBAL
        
        # Alternating attention pattern
        if self.layer_index % self.config.global_attention_frequency == 0:
            return AttentionType.GLOBAL
        else:
            return AttentionType.LOCAL
    
    @property
    def is_final_layer(self) -> bool:
        """Check if this is the final layer."""
        return self.layer_index == self.config.num_hidden_layers - 1


class BertEmbeddings(ABC, Generic[TModule]):
    """Abstract representation of BERT embeddings."""
    
    def __init__(self, config: BertDomainConfig):
        self.config = config
        self.embedding_types = self._determine_embedding_types()
    
    def _determine_embedding_types(self) -> List[str]:
        """Determine which embedding types are needed."""
        types = ["token"]
        
        if self.config.type_vocab_size > 0:
            types.append("token_type")
            
        if not self.config.use_rotary_embeddings:
            types.append("position")
            
        return types
    
    @property
    def total_embedding_parameters(self) -> int:
        """Calculate total parameters in embeddings."""
        params = self.config.vocab_size * self.config.hidden_size
        
        if "token_type" in self.embedding_types:
            params += self.config.type_vocab_size * self.config.hidden_size
            
        if "position" in self.embedding_types:
            params += self.config.max_position_embeddings * self.config.hidden_size
            
        # Layer norm parameters
        params += 2 * self.config.hidden_size
        
        return params


class BertArchitecture(ABC, Generic[TModule, TArray]):
    """Abstract BERT architecture defining the model structure."""
    
    def __init__(self, config: BertDomainConfig):
        self.config = config
        self.validate_config()
        
    def validate_config(self):
        """Validate the configuration for architectural constraints."""
        # Already validated in config, but architecture can add more
        if self.config.use_alternating_attention:
            if self.config.num_hidden_layers < self.config.global_attention_frequency:
                raise ValueError(
                    "Number of layers must be >= global attention frequency "
                    "for alternating attention"
                )
    
    @property
    def num_layers(self) -> int:
        """Number of transformer layers."""
        return self.config.num_hidden_layers
    
    @property
    def hidden_size(self) -> int:
        """Hidden dimension size."""
        return self.config.hidden_size
    
    @property
    def supports_gradient_checkpointing(self) -> bool:
        """Whether architecture supports gradient checkpointing."""
        return True  # Most modern architectures support this
    
    @abstractmethod
    def compute_attention_mask(
        self, 
        attention_mask: Optional[TArray],
        input_shape: tuple[int, ...],
        device: Any = None
    ) -> TArray:
        """Compute extended attention mask for the architecture."""
        pass
    
    @abstractmethod
    def get_layer_types(self) -> List[LayerComponents]:
        """Get the types of all layers in the architecture."""
        pass
    
    def get_architectural_insights(self) -> Dict[str, Any]:
        """Get insights about the architecture."""
        layer_types = self.get_layer_types()
        
        global_layers = sum(1 for l in layer_types if l.is_global_attention)
        local_layers = len(layer_types) - global_layers
        
        return {
            "total_layers": len(layer_types),
            "global_attention_layers": global_layers,
            "local_attention_layers": local_layers,
            "uses_pre_normalization": self.config.use_pre_layer_normalization,
            "normalization_type": self.config.normalization_type,
            "activation_function": self.config.hidden_activation,
            "supports_long_context": self.config.max_position_embeddings > 512,
            "estimated_parameters": self._estimate_parameters(),
            "architectural_family": self._get_architectural_family(),
        }
    
    def _estimate_parameters(self) -> int:
        """Estimate total model parameters."""
        # Use config's parameter estimation
        return self.config.parameter_count_estimate
    
    def _get_architectural_family(self) -> str:
        """Determine the architectural family."""
        if self.config.use_rotary_embeddings and self.config.use_gated_linear_units:
            if self.config.glu_activation_type == "geglu":
                return "ModernBERT"
            elif self.config.glu_activation_type == "swiglu":
                return "neoBERT"
        elif not any([
            self.config.use_rotary_embeddings,
            self.config.use_gated_linear_units,
            self.config.use_pre_layer_normalization
        ]):
            return "Classic BERT"
        else:
            return "Custom BERT Variant"


class BertPooler(ABC, Generic[TModule]):
    """Abstract representation of BERT pooler."""
    
    def __init__(self, config: BertDomainConfig):
        self.config = config
        self.input_size = config.hidden_size
        self.output_size = config.pooler_hidden_size or config.hidden_size
        self.activation = config.pooler_activation
        self.dropout_prob = config.pooler_dropout_probability
    
    @property
    def num_parameters(self) -> int:
        """Calculate number of parameters in pooler."""
        # Dense layer: weight + bias
        return self.input_size * self.output_size + self.output_size


class AttentionPattern:
    """Defines attention patterns for different layer types."""
    
    @staticmethod
    def get_attention_pattern(
        layer_type: AttentionType,
        sequence_length: int,
        window_size: Optional[int] = None
    ) -> str:
        """Get a string representation of the attention pattern."""
        if layer_type == AttentionType.GLOBAL:
            return f"Global attention: all {sequence_length} positions attend to all"
        elif layer_type == AttentionType.LOCAL:
            window = window_size or 128
            return f"Local attention: each position attends to {window} neighbors"
        elif layer_type == AttentionType.SLIDING_WINDOW:
            window = window_size or 256
            return f"Sliding window: overlapping windows of size {window}"
        else:
            return f"Sparse attention: custom pattern"


class ModelCapabilities:
    """Defines what a BERT model architecture is capable of."""
    
    def __init__(self, config: BertDomainConfig):
        self.config = config
    
    @property
    def max_sequence_length(self) -> int:
        """Maximum sequence length the model can handle."""
        return self.config.max_position_embeddings
    
    @property
    def supports_token_types(self) -> bool:
        """Whether model supports token type embeddings."""
        return self.config.type_vocab_size > 0
    
    @property
    def supports_long_sequences(self) -> bool:
        """Whether model is designed for long sequences."""
        return self.config.max_position_embeddings > 512
    
    @property
    def memory_efficient(self) -> bool:
        """Whether model uses memory-efficient techniques."""
        return any([
            self.config.use_alternating_attention,
            self.config.use_rotary_embeddings,  # Saves position embedding memory
            not self.config.use_bias_in_linear,  # Saves bias parameters
        ])
    
    @property
    def inference_optimized(self) -> bool:
        """Whether model is optimized for inference."""
        return any([
            self.config.use_pre_layer_normalization,  # More stable
            self.config.normalization_type == "rms_norm",  # Faster
            self.config.use_gated_linear_units,  # Better quality/compute ratio
        ])
    
    def get_capabilities_summary(self) -> Dict[str, Any]:
        """Get a summary of model capabilities."""
        return {
            "max_sequence_length": self.max_sequence_length,
            "supports_token_types": self.supports_token_types,
            "supports_long_sequences": self.supports_long_sequences,
            "memory_efficient": self.memory_efficient,
            "inference_optimized": self.inference_optimized,
            "architectural_features": {
                "rotary_embeddings": self.config.use_rotary_embeddings,
                "gated_linear_units": self.config.use_gated_linear_units,
                "alternating_attention": self.config.use_alternating_attention,
                "pre_normalization": self.config.use_pre_layer_normalization,
            }
        }