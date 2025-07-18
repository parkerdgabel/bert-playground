"""BERT model with head - combines BertCore with any head from the heads directory.

This module provides the main interface for creating complete BERT models
with task-specific heads for Kaggle competitions.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Dict, Optional, Union, Any, Type, Tuple
from pathlib import Path
import json
from loguru import logger

from .core import BertCore, BertOutput
from .config import BertConfig
from ..heads.base_head import BaseKaggleHead, HeadConfig, HeadType, PoolingType
from ..heads.head_registry import HeadRegistry, CompetitionType, get_head_registry

# Import all head modules to trigger registration
from ..heads import classification_heads, regression_heads  # This triggers the decorators


class BertWithHead(nn.Module):
    """BERT model with attached task-specific head.
    
    This class combines a BertCore model with any head from the heads directory,
    providing a clean interface for end-to-end models.
    """
    
    def __init__(
        self,
        bert: BertCore,
        head: BaseKaggleHead,
        freeze_bert: bool = False,
        freeze_bert_layers: Optional[int] = None,
    ):
        """Initialize BERT with head.
        
        Args:
            bert: BertCore model
            head: Task-specific head
            freeze_bert: Whether to freeze all BERT parameters
            freeze_bert_layers: Number of BERT layers to freeze (from bottom)
        """
        super().__init__()
        
        self.bert = bert
        self.head = head
        
        # Validate compatibility
        self._validate_compatibility()
        
        # Apply freezing if requested
        if freeze_bert:
            self.bert.freeze_encoder()
        elif freeze_bert_layers is not None:
            self.bert.freeze_encoder(freeze_bert_layers)
        
        logger.info(f"Initialized BertWithHead: {bert.__class__.__name__} + {head.__class__.__name__}")
    
    def _validate_compatibility(self):
        """Validate that BERT and head are compatible."""
        bert_hidden_size = self.bert.get_hidden_size()
        head_input_size = self.head.config.input_size
        
        if bert_hidden_size != head_input_size:
            raise ValueError(
                f"BERT hidden size ({bert_hidden_size}) does not match "
                f"head input size ({head_input_size})"
            )
    
    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        token_type_ids: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        labels: Optional[mx.array] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs
    ) -> Union[Dict[str, mx.array], Tuple[mx.array, ...]]:
        """Make BertWithHead callable."""
        return self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
    
    def forward(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        token_type_ids: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        labels: Optional[mx.array] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs
    ) -> Union[Dict[str, mx.array], Tuple[mx.array, ...]]:
        """Forward pass through BERT and head.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len]
            labels: Optional labels for computing loss
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output all hidden states
            return_dict: Whether to return dict or tuple
            **kwargs: Additional arguments for the head
            
        Returns:
            Dictionary or tuple with model outputs
        """
        # Get BERT outputs
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            training=self.training,
        )
        
        # Pass to head
        # The head expects hidden_states and attention_mask
        head_outputs = self.head(
            hidden_states=bert_outputs.last_hidden_state,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = self.head.compute_loss(head_outputs, labels, **kwargs)
        
        if return_dict:
            # Combine outputs
            outputs = {
                "loss": loss,
                "logits": head_outputs.get("logits"),
                "predictions": head_outputs.get("predictions"),
                "probabilities": head_outputs.get("probabilities"),
                "bert_outputs": bert_outputs,
                "head_outputs": head_outputs,
            }
            
            # Add optional outputs
            if output_attentions:
                outputs["attentions"] = bert_outputs.attentions
            if output_hidden_states:
                outputs["hidden_states"] = bert_outputs.hidden_states
            
            return outputs
        else:
            # Return tuple for backward compatibility
            return (loss, head_outputs.get("logits", head_outputs.get("predictions")))
    
    def compute_loss(self, predictions: Dict[str, mx.array], labels: mx.array, **kwargs) -> mx.array:
        """Compute loss using the head's loss function.
        
        Args:
            predictions: Predictions from forward pass
            labels: Ground truth labels
            **kwargs: Additional arguments
            
        Returns:
            Loss value
        """
        return self.head.compute_loss(predictions, labels, **kwargs)
    
    def compute_metrics(self, predictions: Dict[str, mx.array], labels: mx.array, **kwargs) -> Dict[str, float]:
        """Compute metrics using the head's metric functions.
        
        Args:
            predictions: Predictions from forward pass
            labels: Ground truth labels
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of metrics
        """
        return self.head.compute_metrics(predictions, labels, **kwargs)
    
    def get_bert(self) -> BertCore:
        """Get the BERT model."""
        return self.bert
    
    def get_head(self) -> BaseKaggleHead:
        """Get the head model."""
        return self.head
    
    def freeze_bert(self, num_layers: Optional[int] = None):
        """Freeze BERT parameters.
        
        Args:
            num_layers: Number of layers to freeze from bottom. If None, freeze all.
        """
        self.bert.freeze_encoder(num_layers)
    
    def unfreeze_bert(self):
        """Unfreeze all BERT parameters."""
        self.bert.unfreeze_all()
    
    def save_pretrained(self, save_path: Union[str, Path]):
        """Save complete model to directory.
        
        Args:
            save_path: Directory to save model
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save BERT
        bert_path = save_path / "bert"
        self.bert.save_pretrained(bert_path)
        
        # Save head
        head_path = save_path / "head"
        head_path.mkdir(exist_ok=True)
        
        # Save head config
        head_config = self.head.get_config()
        # Convert enums to values for JSON serialization
        config_dict = head_config.__dict__.copy()
        config_dict["head_type"] = head_config.head_type.value
        config_dict["pooling_type"] = head_config.pooling_type.value
        config_dict["activation"] = head_config.activation.value
        
        with open(head_path / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        
        # Save head weights
        from mlx.utils import tree_flatten
        head_params = tree_flatten(self.head.parameters())
        head_weights = {k: v for k, v in head_params}
        mx.save_safetensors(str(head_path / "model.safetensors"), head_weights)
        
        # Save model metadata
        # Find the registered name for this head class
        registry = get_head_registry()
        head_name = None
        for name, spec in registry._heads.items():
            if spec.head_class == self.head.__class__:
                head_name = name
                break
        
        if head_name is None:
            # Fallback to class name
            head_name = self.head.__class__.__name__
        
        metadata = {
            "model_type": "BertWithHead",
            "bert_type": self.bert.__class__.__name__,
            "head_type": head_name,
            "head_class_name": self.head.__class__.__name__,
            "head_config_type": head_config.head_type.value,
        }
        
        with open(save_path / "model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {save_path}")
    
    @classmethod
    def from_pretrained(cls, model_path: Union[str, Path], **kwargs) -> "BertWithHead":
        """Load complete model from directory.
        
        Args:
            model_path: Path to model directory
            **kwargs: Additional arguments
            
        Returns:
            Loaded BertWithHead model
        """
        model_path = Path(model_path)
        
        # Load metadata
        with open(model_path / "model_metadata.json", "r") as f:
            metadata = json.load(f)
        
        # Load BERT
        bert = BertCore.from_pretrained(model_path / "bert")
        
        # Load head config
        with open(model_path / "head" / "config.json", "r") as f:
            head_config_dict = json.load(f)
        
        # Convert string head_type back to enum
        head_type = HeadType(head_config_dict["head_type"])
        head_config_dict["head_type"] = head_type
        
        # Convert other enums
        if "pooling_type" in head_config_dict:
            head_config_dict["pooling_type"] = PoolingType(head_config_dict["pooling_type"])
        if "activation" in head_config_dict:
            from ..heads.base_head import ActivationType
            head_config_dict["activation"] = ActivationType(head_config_dict["activation"])
        
        head_config = HeadConfig(**head_config_dict)
        
        # Get head class from registry
        registry = get_head_registry()  # Use global registry
        head_class = registry.get_head_class(metadata["head_type"])
        
        # Create head
        head = head_class(head_config)
        
        # Load head weights
        head_weights = mx.load(str(model_path / "head" / "model.safetensors"))
        head.load_weights(list(head_weights.items()))
        
        # Create model
        model = cls(bert, head, **kwargs)
        
        logger.info(f"Model loaded from {model_path}")
        return model


# Factory functions
def create_bert_with_head(
    bert_config: Optional[Union[BertConfig, Dict]] = None,
    head_config: Optional[Union[HeadConfig, Dict]] = None,
    head_type: Optional[Union[HeadType, str]] = None,
    bert_name: Optional[str] = None,
    freeze_bert: bool = False,
    freeze_bert_layers: Optional[int] = None,
    **kwargs
) -> BertWithHead:
    """Create a BERT model with head.
    
    Args:
        bert_config: BERT configuration
        head_config: Head configuration
        head_type: Type of head to create (if head_config not provided)
        bert_name: Pretrained BERT model name
        freeze_bert: Whether to freeze BERT parameters
        freeze_bert_layers: Number of BERT layers to freeze
        **kwargs: Additional arguments
        
    Returns:
        BertWithHead model
    """
    # Create BERT
    if bert_name:
        bert = BertCore.from_pretrained(bert_name)
    elif bert_config:
        bert = BertCore(bert_config)
    else:
        bert = BertCore(BertConfig())
    
    # Create head
    if head_config is None:
        if head_type is None:
            raise ValueError("Either head_config or head_type must be provided")
        
        # Convert string to enum if needed
        if isinstance(head_type, str):
            head_type = HeadType(head_type)
        
        # Get default config for head type
        from ..heads.base_head import get_default_config_for_head_type
        head_config = get_default_config_for_head_type(
            head_type,
            input_size=bert.get_hidden_size(),
            output_size=kwargs.get("num_labels", 2)
        )
    
    # Convert dict to HeadConfig if needed
    if isinstance(head_config, dict):
        head_config = HeadConfig(**head_config)
    
    # Get head class from registry
    registry = get_head_registry()  # Use global registry
    
    # We need to find a head that matches the head type
    # First, get all heads for this type
    head_names = []
    for name, spec in registry._heads.items():
        if spec.head_type == head_config.head_type:
            head_names.append(name)
    
    if not head_names:
        raise ValueError(f"No head found for type {head_config.head_type}")
    
    # Use the first one (highest priority)
    head_class = registry.get_head_class(head_names[0])
    
    # Create head
    head = head_class(head_config)
    
    # Create model
    return BertWithHead(
        bert=bert,
        head=head,
        freeze_bert=freeze_bert,
        freeze_bert_layers=freeze_bert_layers
    )


def create_bert_for_competition(
    competition_type: Union[CompetitionType, str],
    bert_config: Optional[Union[BertConfig, Dict]] = None,
    bert_name: Optional[str] = None,
    num_labels: int = 2,
    **kwargs
) -> BertWithHead:
    """Create a BERT model optimized for a specific competition type.
    
    Args:
        competition_type: Type of competition
        bert_config: BERT configuration
        bert_name: Pretrained BERT model name
        num_labels: Number of output labels
        **kwargs: Additional arguments
        
    Returns:
        BertWithHead model optimized for the competition
    """
    # Convert string to enum if needed
    if isinstance(competition_type, str):
        competition_type = CompetitionType(competition_type)
    
    # Create BERT
    if bert_name:
        bert = BertCore.from_pretrained(bert_name)
    elif bert_config:
        bert = BertCore(bert_config)
    else:
        bert = BertCore(BertConfig())
    
    # Get best head for competition
    registry = get_head_registry()  # Use global registry
    
    # Create head using the registry's method
    head = registry.create_head_from_competition(
        competition_type=competition_type,
        input_size=bert.get_hidden_size(),
        output_size=num_labels
    )
    
    # Create model
    return BertWithHead(bert=bert, head=head, **kwargs)