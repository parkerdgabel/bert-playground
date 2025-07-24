"""MLX model adapter for BertModel entities."""

from typing import Any, Dict, Optional, List, Tuple
import mlx.core as mx
import mlx.nn as nn

from domain.entities.model import BertModel, ModelArchitecture, ActivationType, AttentionType


class MLXModelAdapter:
    """Adapts BertModel entities to MLX modules."""
    
    def __init__(self, bert_model: BertModel):
        """Initialize MLX model adapter.
        
        Args:
            bert_model: The domain BertModel entity
        """
        self.bert_model = bert_model
        self._mlx_model = self._create_mlx_model(bert_model.architecture)
        self._compiled_forward = None
        self._training = True
        
        # Initialize weights if provided
        if bert_model.weights is not None:
            self._load_weights_from_entity(bert_model.weights)
    
    def forward(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        token_type_ids: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        labels: Optional[mx.array] = None,
    ) -> Dict[str, Any]:
        """Forward pass through the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs
            position_ids: Position IDs
            labels: Labels for loss computation
            
        Returns:
            Dictionary with model outputs
        """
        # Use compiled forward if available
        forward_fn = self._compiled_forward if self._compiled_forward else self._mlx_model
        
        # Forward pass
        outputs = forward_fn(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        
        result = {
            "logits": outputs["logits"],
            "hidden_states": outputs.get("hidden_states"),
            "attentions": outputs.get("attentions"),
        }
        
        # Compute loss if labels provided
        if labels is not None:
            loss = self._compute_loss(outputs["logits"], labels)
            result["loss"] = loss
        
        return result
    
    def train(self, mode: bool = True) -> None:
        """Set training mode."""
        self._training = mode
        if hasattr(self._mlx_model, "train"):
            self._mlx_model.train(mode)
    
    def eval(self) -> None:
        """Set evaluation mode."""
        self.train(False)
    
    def compile(self) -> None:
        """Compile the model for optimized execution."""
        if self._compiled_forward is None:
            # Create a compiled version of forward pass
            def forward_fn(input_ids, attention_mask=None, token_type_ids=None, position_ids=None):
                return self._mlx_model(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                )
            
            self._compiled_forward = mx.compile(forward_fn)
    
    def named_parameters(self) -> List[Tuple[str, mx.array]]:
        """Get all parameters with names."""
        return list(self._mlx_model.parameters().items())
    
    def parameters(self) -> List[mx.array]:
        """Get all parameters."""
        return list(self._mlx_model.parameters().values())
    
    def count_parameters(self, trainable_only: bool = False) -> int:
        """Count model parameters."""
        total = 0
        for param in self.parameters():
            total += param.size
        return total
    
    def get_mlx_model(self) -> nn.Module:
        """Get the underlying MLX model."""
        return self._mlx_model
    
    # Private methods
    
    def _create_mlx_model(self, architecture: ModelArchitecture) -> nn.Module:
        """Create MLX model from architecture specification."""
        # Import MLX BERT implementation
        from models.bert.mlx_bert import BertModel as MLXBertModel
        
        # Convert architecture to MLX config
        config = {
            "vocab_size": architecture.vocab_size,
            "hidden_size": architecture.hidden_size,
            "num_hidden_layers": architecture.num_hidden_layers,
            "num_attention_heads": architecture.num_attention_heads,
            "intermediate_size": architecture.intermediate_size,
            "max_position_embeddings": architecture.max_position_embeddings,
            "hidden_dropout_prob": architecture.hidden_dropout_prob,
            "attention_probs_dropout_prob": architecture.attention_probs_dropout_prob,
            "layer_norm_eps": architecture.layer_norm_eps,
            "hidden_act": self._get_activation_name(architecture.activation),
            "use_rope": architecture.use_rope,
            "rope_theta": architecture.rope_theta,
            "use_bias": architecture.use_bias,
            "pad_token_id": architecture.pad_token_id,
        }
        
        # Handle attention type
        if architecture.attention_type == AttentionType.FLASH:
            config["use_flash_attention"] = True
        elif architecture.attention_type == AttentionType.ALTERNATING:
            config["use_alternating_attention"] = True
        
        # Create MLX model
        return MLXBertModel(**config)
    
    def _get_activation_name(self, activation: ActivationType) -> str:
        """Convert ActivationType enum to string."""
        activation_map = {
            ActivationType.GELU: "gelu",
            ActivationType.RELU: "relu",
            ActivationType.SILU: "silu",
            ActivationType.GEGLU: "geglu",
        }
        return activation_map.get(activation, "gelu")
    
    def _load_weights_from_entity(self, weights: Any) -> None:
        """Load weights from BertModel entity into MLX model."""
        # This would convert from the domain weight format to MLX arrays
        # For now, we'll skip the actual implementation
        pass
    
    def _compute_loss(self, logits: mx.array, labels: mx.array) -> mx.array:
        """Compute loss based on task type."""
        # Determine task type from logits shape
        if len(logits.shape) == 3:  # Token-level prediction
            # Reshape for cross-entropy loss
            batch_size, seq_length, vocab_size = logits.shape
            logits_flat = logits.reshape(-1, vocab_size)
            labels_flat = labels.reshape(-1)
            
            # Mask out padding tokens
            mask = labels_flat != -100
            active_logits = logits_flat[mask]
            active_labels = labels_flat[mask]
            
            if active_labels.size > 0:
                return nn.losses.cross_entropy(active_logits, active_labels)
            else:
                return mx.array(0.0)
        
        elif len(logits.shape) == 2:  # Sequence-level prediction
            num_classes = logits.shape[-1]
            
            if num_classes == 1:  # Regression
                return nn.losses.mse_loss(logits.squeeze(-1), labels)
            else:  # Classification
                return nn.losses.cross_entropy(logits, labels)
        
        else:
            raise ValueError(f"Unexpected logits shape: {logits.shape}")