"""Model builder service for framework-agnostic model construction.

This service builds model specifications that can be implemented
by any ML framework (MLX, PyTorch, JAX, etc.).
"""

from typing import Dict, List, Any, Optional
from domain.entities.model import (
    BertModel, ModelArchitecture, ModelType, TaskType, 
    TaskHead, ModelSpecification, ActivationType, AttentionType
)
from infrastructure.di import service


@service
class ModelBuilder:
    """Builds model specifications in a framework-agnostic way.
    
    This service contains the business logic for constructing
    BERT models without any framework-specific code.
    """
    
    def build_bert_model(
        self,
        model_id: str,
        model_name: str,
        vocab_size: int,
        num_labels: Optional[int] = None,
        task_type: Optional[TaskType] = None,
        model_variant: ModelType = ModelType.CLASSIC_BERT,
        **architecture_params
    ) -> BertModel:
        """Build a BERT model specification.
        
        Args:
            model_id: Unique identifier for the model
            model_name: Human-readable model name
            vocab_size: Size of the vocabulary
            num_labels: Number of labels for classification tasks
            task_type: Type of task the model will perform
            model_variant: Type of BERT variant to build
            **architecture_params: Additional architecture parameters
            
        Returns:
            BertModel specification
        """
        # Build architecture based on variant
        architecture = self._build_architecture(
            vocab_size=vocab_size,
            model_variant=model_variant,
            **architecture_params
        )
        
        # Build task head if specified
        task_head = None
        if task_type is not None:
            task_head = self._build_task_head(
                task_type=task_type,
                input_size=architecture.hidden_size,
                num_labels=num_labels
            )
        
        # Create model
        model = BertModel(
            id=model_id,
            name=model_name,
            architecture=architecture,
            task_head=task_head,
            metadata={
                "created_by": "ModelBuilder",
                "variant": model_variant.value
            }
        )
        
        return model
    
    def build_from_preset(
        self,
        preset_name: str,
        model_id: str,
        model_name: str,
        vocab_size: int,
        num_labels: Optional[int] = None,
        task_type: Optional[TaskType] = None
    ) -> BertModel:
        """Build a model from a preset configuration.
        
        Args:
            preset_name: Name of the preset (e.g., 'base', 'large', 'modern-base')
            model_id: Unique identifier for the model
            model_name: Human-readable model name
            vocab_size: Size of the vocabulary
            num_labels: Number of labels for classification tasks
            task_type: Type of task the model will perform
            
        Returns:
            BertModel specification
        """
        preset_config = self._get_preset_config(preset_name)
        
        return self.build_bert_model(
            model_id=model_id,
            model_name=model_name,
            vocab_size=vocab_size,
            num_labels=num_labels,
            task_type=task_type,
            **preset_config
        )
    
    def build_model_specification(
        self,
        model: BertModel,
        initialization_strategy: str = "default",
        pretrained_from: Optional[str] = None
    ) -> ModelSpecification:
        """Build a complete model specification for construction.
        
        Args:
            model: Base model to build from
            initialization_strategy: How to initialize weights
            pretrained_from: Optional pretrained model path
            
        Returns:
            ModelSpecification with all components
        """
        spec = ModelSpecification(
            base_model=model,
            initialization_strategy=initialization_strategy,
            pretrained_from=pretrained_from
        )
        
        # Add embedding components
        spec.add_component("embeddings", {
            "vocab_size": model.architecture.vocab_size,
            "hidden_size": model.architecture.hidden_size,
            "max_position_embeddings": model.architecture.max_position_embeddings,
            "type_vocab_size": model.architecture.type_vocab_size,
            "use_rope": model.architecture.use_rope,
            "rope_theta": model.architecture.rope_theta,
            "dropout_prob": model.architecture.hidden_dropout_prob,
        })
        
        # Add encoder layers
        for layer_idx in range(model.architecture.num_hidden_layers):
            layer_config = self._build_layer_specification(
                model.architecture, layer_idx
            )
            spec.add_component(f"encoder_layer_{layer_idx}", layer_config)
        
        # Add pooler if needed
        if model.task_head and model.task_head.use_pooler:
            spec.add_component("pooler", {
                "hidden_size": model.architecture.hidden_size,
                "activation": "tanh"
            })
        
        # Add task head if specified
        if model.task_head:
            head_config = self._build_head_specification(model.task_head)
            spec.add_component("task_head", head_config)
        
        return spec
    
    def _build_architecture(
        self,
        vocab_size: int,
        model_variant: ModelType,
        **params
    ) -> ModelArchitecture:
        """Build model architecture based on variant."""
        # Get default values for variant
        defaults = self._get_variant_defaults(model_variant)
        
        # Merge with provided params
        config = {**defaults, **params}
        
        # Create architecture
        return ModelArchitecture(
            vocab_size=vocab_size,
            hidden_size=config.get("hidden_size", 768),
            num_hidden_layers=config.get("num_hidden_layers", 12),
            num_attention_heads=config.get("num_attention_heads", 12),
            intermediate_size=config.get("intermediate_size", 3072),
            max_position_embeddings=config.get("max_position_embeddings", 512),
            model_type=model_variant,
            attention_type=AttentionType(config.get("attention_type", "standard")),
            activation_type=ActivationType(config.get("activation_type", "gelu")),
            use_rope=config.get("use_rope", False),
            rope_theta=config.get("rope_theta", 10000.0),
            use_bias=config.get("use_bias", True),
            use_gated_units=config.get("use_gated_units", False),
            use_pre_layer_norm=config.get("use_pre_layer_norm", False),
            use_alternating_attention=config.get("use_alternating_attention", False),
            global_attention_frequency=config.get("global_attention_frequency", 4),
            hidden_dropout_prob=config.get("hidden_dropout_prob", 0.1),
            attention_probs_dropout_prob=config.get("attention_probs_dropout_prob", 0.1),
            layer_norm_eps=config.get("layer_norm_eps", 1e-12),
            normalization_type=config.get("normalization_type", "layer_norm"),
            pad_token_id=config.get("pad_token_id", 0),
            type_vocab_size=config.get("type_vocab_size", 2),
        )
    
    def _build_task_head(
        self,
        task_type: TaskType,
        input_size: int,
        num_labels: Optional[int]
    ) -> TaskHead:
        """Build task-specific head."""
        # Validate num_labels for classification tasks
        if task_type in [TaskType.CLASSIFICATION, TaskType.TOKEN_CLASSIFICATION]:
            if num_labels is None or num_labels < 2:
                raise ValueError(f"{task_type.value} requires num_labels >= 2")
        
        # Set defaults based on task type
        use_pooler = task_type != TaskType.TOKEN_CLASSIFICATION
        
        return TaskHead(
            task_type=task_type,
            input_size=input_size,
            num_labels=num_labels,
            use_pooler=use_pooler,
            dropout_prob=0.1
        )
    
    def _build_layer_specification(
        self,
        architecture: ModelArchitecture,
        layer_idx: int
    ) -> Dict[str, Any]:
        """Build specification for a single encoder layer."""
        # Determine attention type for this layer
        if architecture.use_alternating_attention:
            is_global = layer_idx % architecture.global_attention_frequency == 0
            attention_type = "global" if is_global else "local"
        else:
            attention_type = architecture.attention_type.value
        
        return {
            "layer_index": layer_idx,
            "hidden_size": architecture.hidden_size,
            "num_attention_heads": architecture.num_attention_heads,
            "intermediate_size": architecture.intermediate_size,
            "attention_type": attention_type,
            "activation": architecture.activation_type.value,
            "use_gated_units": architecture.use_gated_units,
            "use_bias": architecture.use_bias,
            "use_pre_layer_norm": architecture.use_pre_layer_norm,
            "dropout_prob": architecture.hidden_dropout_prob,
            "attention_dropout_prob": architecture.attention_probs_dropout_prob,
            "layer_norm_eps": architecture.layer_norm_eps,
            "normalization_type": architecture.normalization_type,
        }
    
    def _build_head_specification(self, task_head: TaskHead) -> Dict[str, Any]:
        """Build specification for task head."""
        config = {
            "task_type": task_head.task_type.value,
            "input_size": task_head.input_size,
            "dropout_prob": task_head.dropout_prob,
            "use_pooler": task_head.use_pooler,
        }
        
        if task_head.num_labels is not None:
            config["num_labels"] = task_head.num_labels
            
        if task_head.label_names is not None:
            config["label_names"] = task_head.label_names
            
        return config
    
    def _get_variant_defaults(self, variant: ModelType) -> Dict[str, Any]:
        """Get default configuration for model variant."""
        if variant == ModelType.CLASSIC_BERT:
            return {
                "hidden_size": 768,
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
                "intermediate_size": 3072,
                "activation_type": "gelu",
                "attention_type": "standard",
                "use_rope": False,
                "use_bias": True,
                "use_gated_units": False,
                "use_pre_layer_norm": False,
            }
        elif variant == ModelType.MODERN_BERT:
            return {
                "hidden_size": 768,
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
                "intermediate_size": 2048,  # Smaller due to GLU
                "activation_type": "geglu",
                "attention_type": "alternating",
                "use_rope": True,
                "use_bias": False,
                "use_gated_units": True,
                "use_pre_layer_norm": True,
                "use_alternating_attention": True,
                "normalization_type": "rms_norm",
            }
        else:  # CUSTOM_BERT
            return {
                "hidden_size": 768,
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
                "intermediate_size": 3072,
                "activation_type": "gelu",
                "attention_type": "standard",
            }
    
    def _get_preset_config(self, preset_name: str) -> Dict[str, Any]:
        """Get preset configuration by name."""
        presets = {
            "base": {
                "model_variant": ModelType.CLASSIC_BERT,
                "hidden_size": 768,
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
                "intermediate_size": 3072,
            },
            "large": {
                "model_variant": ModelType.CLASSIC_BERT,
                "hidden_size": 1024,
                "num_hidden_layers": 24,
                "num_attention_heads": 16,
                "intermediate_size": 4096,
            },
            "modern-base": {
                "model_variant": ModelType.MODERN_BERT,
                "hidden_size": 768,
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
                "intermediate_size": 2048,
                "max_position_embeddings": 8192,
            },
            "modern-large": {
                "model_variant": ModelType.MODERN_BERT,
                "hidden_size": 1024,
                "num_hidden_layers": 24,
                "num_attention_heads": 16,
                "intermediate_size": 2730,
                "max_position_embeddings": 8192,
            },
            "tiny": {
                "model_variant": ModelType.CLASSIC_BERT,
                "hidden_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "intermediate_size": 512,
            },
        }
        
        if preset_name not in presets:
            raise ValueError(
                f"Unknown preset '{preset_name}'. "
                f"Available presets: {list(presets.keys())}"
            )
        
        return presets[preset_name]