"""Core BERT model implementation with modular interface.

This module provides the consolidated BERT implementation that can be easily
attached to any head in the heads directory.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Dict, Optional, Tuple, Union, List
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from loguru import logger

from ..modernbert import ModernBertConfig, ModernBertModel


@dataclass
class BertOutput:
    """Standard output format for BERT models.
    
    This dataclass provides a consistent interface between BERT models
    and downstream heads, making it easy to swap models or heads.
    """
    # Primary outputs
    last_hidden_state: mx.array  # [batch_size, seq_len, hidden_size]
    pooler_output: mx.array      # [batch_size, hidden_size]
    
    # Optional outputs
    hidden_states: Optional[List[mx.array]] = None  # List of all hidden states
    attentions: Optional[List[mx.array]] = None     # List of attention weights
    
    # Additional pooled representations
    cls_output: Optional[mx.array] = None           # [batch_size, hidden_size] - CLS token
    mean_pooled: Optional[mx.array] = None          # [batch_size, hidden_size] - Mean pooling
    max_pooled: Optional[mx.array] = None           # [batch_size, hidden_size] - Max pooling
    
    # Metadata
    attention_mask: Optional[mx.array] = None       # [batch_size, seq_len] - For downstream use
    
    def get_pooled_output(self, pooling_type: str = "cls") -> mx.array:
        """Get pooled output by type.
        
        Args:
            pooling_type: Type of pooling - "cls", "mean", "max", or "pooler"
            
        Returns:
            Pooled representation
        """
        if pooling_type == "cls":
            return self.cls_output if self.cls_output is not None else self.last_hidden_state[:, 0, :]
        elif pooling_type == "mean":
            return self.mean_pooled if self.mean_pooled is not None else self._compute_mean_pooling()
        elif pooling_type == "max":
            return self.max_pooled if self.max_pooled is not None else self._compute_max_pooling()
        elif pooling_type == "pooler":
            return self.pooler_output
        else:
            raise ValueError(f"Unknown pooling type: {pooling_type}")
    
    def _compute_mean_pooling(self) -> mx.array:
        """Compute mean pooling if not already computed."""
        if self.attention_mask is not None:
            # Mask out padding tokens
            mask = self.attention_mask.astype(mx.float32)[..., None]
            masked_hidden = self.last_hidden_state * mask
            seq_lengths = mask.sum(axis=1)
            return masked_hidden.sum(axis=1) / seq_lengths
        else:
            return self.last_hidden_state.mean(axis=1)
    
    def _compute_max_pooling(self) -> mx.array:
        """Compute max pooling if not already computed."""
        if self.attention_mask is not None:
            # Mask out padding tokens with large negative values
            mask = self.attention_mask.astype(mx.float32)[..., None]
            masked_hidden = self.last_hidden_state + (1.0 - mask) * -1e9
            return masked_hidden.max(axis=1)
        else:
            return self.last_hidden_state.max(axis=1)


class BertCore(nn.Module):
    """Core BERT model with standardized interface.
    
    This class wraps the ModernBertModel and provides a clean interface
    for attaching any head from the heads directory.
    """
    
    def __init__(self, config: Union[ModernBertConfig, Dict]):
        """Initialize BERT core model.
        
        Args:
            config: Model configuration (ModernBertConfig or dict)
        """
        super().__init__()
        
        # Convert dict to config if needed
        if isinstance(config, dict):
            config = ModernBertConfig(**config)
        
        self.config = config
        
        # Initialize the underlying BERT model
        self.bert = ModernBertModel(config)
        
        # Optional: Additional pooling layers
        self.additional_pooling = config.__dict__.get("compute_additional_pooling", True)
        
        logger.info(f"Initialized BertCore with config: hidden_size={config.hidden_size}, "
                   f"num_layers={config.num_hidden_layers}, num_heads={config.num_attention_heads}")
    
    def forward(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        token_type_ids: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        compute_pooling: bool = True,
        training: bool = True,
    ) -> BertOutput:
        """Forward pass through BERT model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len]
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output all hidden states
            compute_pooling: Whether to compute additional pooling
            training: Whether in training mode
            
        Returns:
            BertOutput with all model outputs
        """
        # Get outputs from underlying BERT model
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            training=training,
        )
        
        # Extract outputs
        last_hidden_state = outputs["last_hidden_state"]
        pooler_output = outputs["pooler_output"]
        
        # Create BertOutput
        bert_output = BertOutput(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
            attention_mask=attention_mask,
        )
        
        # Add optional outputs
        if output_attentions and "attentions" in outputs:
            bert_output.attentions = outputs["attentions"]
        
        # Compute additional pooling if requested
        if compute_pooling and self.additional_pooling:
            # CLS output (already included in last_hidden_state[:, 0])
            bert_output.cls_output = last_hidden_state[:, 0, :]
            
            # Mean pooling
            if attention_mask is not None:
                mask = attention_mask.astype(mx.float32)[..., None]
                masked_hidden = last_hidden_state * mask
                seq_lengths = mask.sum(axis=1)
                bert_output.mean_pooled = masked_hidden.sum(axis=1) / seq_lengths
            else:
                bert_output.mean_pooled = last_hidden_state.mean(axis=1)
            
            # Max pooling
            if attention_mask is not None:
                mask = attention_mask.astype(mx.float32)[..., None]
                masked_hidden = last_hidden_state + (1.0 - mask) * -1e9
                bert_output.max_pooled = masked_hidden.max(axis=1)
            else:
                bert_output.max_pooled = last_hidden_state.max(axis=1)
        
        return bert_output
    
    def get_hidden_size(self) -> int:
        """Get the hidden size of the model."""
        return self.config.hidden_size
    
    def get_num_layers(self) -> int:
        """Get the number of layers in the model."""
        return self.config.num_hidden_layers
    
    def get_config(self) -> ModernBertConfig:
        """Get the model configuration."""
        return self.config
    
    @classmethod
    def from_pretrained(cls, model_path: Union[str, Path], **kwargs) -> "BertCore":
        """Load model from pretrained weights.
        
        Args:
            model_path: Path to model directory or HuggingFace model name
            **kwargs: Additional configuration parameters
            
        Returns:
            Loaded BertCore model
        """
        # Load underlying BERT model
        bert_model = ModernBertModel.from_pretrained(model_path, **kwargs)
        
        # Create BertCore wrapper
        core_model = cls(bert_model.config)
        core_model.bert = bert_model
        
        return core_model
    
    def save_pretrained(self, save_path: Union[str, Path]):
        """Save model to directory.
        
        Args:
            save_path: Directory to save model
        """
        self.bert.save_pretrained(save_path)
        
        # Also save any additional BertCore-specific config
        save_path = Path(save_path)
        bert_core_config = {
            "compute_additional_pooling": self.additional_pooling,
            "bert_core_version": "1.0.0",
        }
        
        with open(save_path / "bert_core_config.json", "w") as f:
            json.dump(bert_core_config, f, indent=2)
    
    def freeze_encoder(self, num_layers_to_freeze: Optional[int] = None):
        """Freeze encoder layers for fine-tuning.
        
        Args:
            num_layers_to_freeze: Number of layers to freeze from bottom.
                                If None, freeze all layers.
        """
        # Freeze embeddings
        for param in self.bert.embeddings.parameters():
            param.freeze()
        
        # Freeze encoder layers
        if num_layers_to_freeze is None:
            num_layers_to_freeze = len(self.bert.encoder.layers)
        
        for i, layer in enumerate(self.bert.encoder.layers):
            if i < num_layers_to_freeze:
                for param in layer.parameters():
                    param.freeze()
    
    def unfreeze_all(self):
        """Unfreeze all model parameters."""
        for param in self.parameters():
            param.unfreeze()


# Factory functions
def create_bert_core(
    model_name: Optional[str] = None,
    config: Optional[Union[ModernBertConfig, Dict]] = None,
    **kwargs
) -> BertCore:
    """Create a BertCore model.
    
    Args:
        model_name: Optional pretrained model name
        config: Optional configuration
        **kwargs: Additional configuration parameters
        
    Returns:
        BertCore model instance
    """
    if model_name:
        return BertCore.from_pretrained(model_name, **kwargs)
    elif config:
        return BertCore(config)
    else:
        # Use default config with kwargs
        config = ModernBertConfig(**kwargs)
        return BertCore(config)