"""Model entities representing BERT architecture and components.

Pure domain entities with no framework dependencies.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from enum import Enum


class ModelType(Enum):
    """Types of BERT models."""
    CLASSIC_BERT = "classic_bert"
    MODERN_BERT = "modern_bert"
    CUSTOM_BERT = "custom_bert"


class TaskType(Enum):
    """Task types for BERT models."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    TOKEN_CLASSIFICATION = "token_classification"
    MASKED_LM = "masked_lm"
    QUESTION_ANSWERING = "question_answering"


class ActivationType(Enum):
    """Activation function types."""
    GELU = "gelu"
    RELU = "relu"
    SILU = "silu"
    TANH = "tanh"
    GEGLU = "geglu"
    SWIGLU = "swiglu"


class AttentionType(Enum):
    """Attention mechanism types."""
    STANDARD = "standard"
    FLASH = "flash"
    ALTERNATING = "alternating"
    SPARSE = "sparse"
    LOCAL = "local"
    GLOBAL = "global"


@dataclass
class ModelArchitecture:
    """Pure specification of BERT model architecture.
    
    This defines the structure of a BERT model without any
    framework-specific implementation details.
    """
    # Core dimensions
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    max_position_embeddings: int
    
    # Model type and variants
    model_type: ModelType = ModelType.CLASSIC_BERT
    attention_type: AttentionType = AttentionType.STANDARD
    activation_type: ActivationType = ActivationType.GELU
    
    # Architectural features
    use_rope: bool = False
    rope_theta: float = 10000.0
    use_bias: bool = True
    use_gated_units: bool = False
    use_pre_layer_norm: bool = False
    use_alternating_attention: bool = False
    global_attention_frequency: int = 4
    
    # Regularization
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    
    # Normalization
    layer_norm_eps: float = 1e-12
    normalization_type: str = "layer_norm"
    
    # Special tokens
    pad_token_id: int = 0
    type_vocab_size: int = 2
    
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
        
    @property
    def head_size(self) -> int:
        """Size of each attention head."""
        return self.hidden_size // self.num_attention_heads
    
    @property
    def total_parameters(self) -> int:
        """Estimate total model parameters."""
        # Embeddings
        params = self.vocab_size * self.hidden_size  # Token embeddings
        if self.type_vocab_size > 0:
            params += self.type_vocab_size * self.hidden_size  # Token type
        if not self.use_rope:
            params += self.max_position_embeddings * self.hidden_size  # Position
        params += 2 * self.hidden_size  # LayerNorm
        
        # Transformer layers
        per_layer = 0
        # Attention
        per_layer += 4 * self.hidden_size * self.hidden_size  # Q,K,V,O projections
        if self.use_bias:
            per_layer += 4 * self.hidden_size  # Biases
        # FFN
        if self.use_gated_units:
            per_layer += 3 * self.hidden_size * self.intermediate_size  # Gate, up, down
        else:
            per_layer += 2 * self.hidden_size * self.intermediate_size  # Up, down
        if self.use_bias:
            per_layer += self.intermediate_size + self.hidden_size  # Biases
        # LayerNorms
        per_layer += 4 * self.hidden_size  # 2 LayerNorms x (weight + bias)
        
        params += per_layer * self.num_hidden_layers
        
        return params


@dataclass
class ModelWeights:
    """Container for model weights/parameters.
    
    This is a pure value object that holds weight data
    without framework-specific types.
    """
    embeddings: Dict[str, Any] = field(default_factory=dict)
    encoder_layers: list[Dict[str, Any]] = field(default_factory=list)
    pooler: Optional[Dict[str, Any]] = None
    task_head: Optional[Dict[str, Any]] = None
    
    def get_layer_weights(self, layer_idx: int) -> Dict[str, Any]:
        """Get weights for a specific layer."""
        if 0 <= layer_idx < len(self.encoder_layers):
            return self.encoder_layers[layer_idx]
        raise IndexError(f"Layer index {layer_idx} out of range")


@dataclass
class TaskHead:
    """Specification for a task-specific head."""
    task_type: TaskType
    input_size: int
    num_labels: Optional[int] = None
    dropout_prob: float = 0.1
    use_pooler: bool = True
    label_names: Optional[list[str]] = None
    
    def __post_init__(self):
        """Validate task head configuration."""
        if self.task_type in [TaskType.CLASSIFICATION, TaskType.TOKEN_CLASSIFICATION]:
            if self.num_labels is None or self.num_labels < 2:
                raise ValueError(f"{self.task_type.value} requires num_labels >= 2")


@dataclass
class BertModel:
    """Core BERT model entity.
    
    This represents a complete BERT model with its architecture,
    optional weights, and task configuration.
    """
    id: str
    name: str
    architecture: ModelArchitecture
    task_head: Optional[TaskHead] = None
    weights: Optional[ModelWeights] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_initialized(self) -> bool:
        """Check if model has weights."""
        return self.weights is not None
    
    @property
    def task_type(self) -> Optional[TaskType]:
        """Get task type if configured."""
        return self.task_head.task_type if self.task_head else None
    
    @property
    def num_labels(self) -> Optional[int]:
        """Get number of labels for classification tasks."""
        return self.task_head.num_labels if self.task_head else None
    
    def get_config(self) -> Dict[str, Any]:
        """Get complete model configuration."""
        config = {
            "id": self.id,
            "name": self.name,
            "architecture": {
                "vocab_size": self.architecture.vocab_size,
                "hidden_size": self.architecture.hidden_size,
                "num_hidden_layers": self.architecture.num_hidden_layers,
                "num_attention_heads": self.architecture.num_attention_heads,
                "intermediate_size": self.architecture.intermediate_size,
                "max_position_embeddings": self.architecture.max_position_embeddings,
                "model_type": self.architecture.model_type.value,
                "attention_type": self.architecture.attention_type.value,
                "activation_type": self.architecture.activation_type.value,
                "use_rope": self.architecture.use_rope,
                "use_bias": self.architecture.use_bias,
                "use_gated_units": self.architecture.use_gated_units,
                "hidden_dropout_prob": self.architecture.hidden_dropout_prob,
                "attention_probs_dropout_prob": self.architecture.attention_probs_dropout_prob,
            }
        }
        
        if self.task_head:
            config["task_head"] = {
                "task_type": self.task_head.task_type.value,
                "num_labels": self.task_head.num_labels,
                "use_pooler": self.task_head.use_pooler,
                "dropout_prob": self.task_head.dropout_prob,
            }
        
        return config


@dataclass
class ModelSpecification:
    """Specification for building a model.
    
    This is used by the model builder service to specify
    how a model should be constructed, independent of framework.
    """
    base_model: BertModel
    components: list[Dict[str, Any]] = field(default_factory=list)
    initialization_strategy: str = "default"
    pretrained_from: Optional[str] = None
    
    def add_component(self, component_type: str, config: Dict[str, Any]) -> None:
        """Add a component specification."""
        self.components.append({
            "type": component_type,
            "config": config
        })