"""
Base classes and components for BERT models.

This module contains the core building blocks that are shared across
different BERT architectures and provide the foundation for the model.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Dict, Optional, Tuple, Union, List
from dataclasses import dataclass

from .config import BertConfig
from .layers.attention import BertAttention
from .layers.feedforward import BertIntermediate, BertOutput
from .layers.embeddings import BertEmbeddings, BertPooler


@dataclass
class BertModelOutput:
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


class BertLayer(nn.Module):
    """Complete BERT transformer layer.
    
    Implementation following the original BERT paper architecture.
    """
    
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        
        # Attention layer
        self.attention = BertAttention(config)
        
        # Feed-forward network
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
    
    def forward(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        output_attentions: bool = False,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        """Forward pass through BERT layer.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            output_attentions: Whether to output attention weights
            
        Returns:
            Tuple of (layer_output, attention_probs)
        """
        # Self-attention
        attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            output_attentions,
        )
        attention_output = attention_outputs[0]
        
        # Feed-forward network
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        
        outputs = (layer_output,) + attention_outputs[1:]  # Add attentions if we output them
        return outputs
    
    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        output_attentions: bool = False,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        """Make the layer callable."""
        return self.forward(hidden_states, attention_mask, output_attentions)


class BaseBertModel(nn.Module):
    """Base class for BERT models.
    
    This class provides common functionality shared across different BERT variants.
    """
    
    def __init__(self, config: Union[BertConfig, Dict]):
        super().__init__()
        
        # Convert dict to config if needed
        if isinstance(config, dict):
            config = BertConfig(**config)
        
        self.config = config
        
        # Initialize embeddings
        self.embeddings = BertEmbeddings(config)
        
        # Initialize pooler
        self.pooler = BertPooler(config)
        
        # Initialize encoder layers (to be implemented by subclasses)
        self.encoder_layers = []
        
        # Optional: Additional pooling layers
        self.additional_pooling = config.__dict__.get("compute_additional_pooling", True)
    
    def _build_encoder(self):
        """Build encoder layers. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _build_encoder")
    
    def get_hidden_size(self) -> int:
        """Get the hidden size of the model."""
        return self.config.hidden_size
    
    def get_num_layers(self) -> int:
        """Get the number of layers in the model."""
        return self.config.num_hidden_layers
    
    def get_config(self) -> BertConfig:
        """Get the model configuration."""
        return self.config
    
    def freeze_encoder(self, num_layers_to_freeze: Optional[int] = None):
        """Freeze encoder layers for fine-tuning.
        
        Args:
            num_layers_to_freeze: Number of layers to freeze from bottom.
                                If None, freeze all layers.
        """
        # Freeze embeddings
        for param in self.embeddings.parameters():
            param.freeze()
        
        # Freeze encoder layers
        if num_layers_to_freeze is None:
            num_layers_to_freeze = self.config.num_hidden_layers
        
        # Freeze the specified number of encoder layers
        if num_layers_to_freeze > 0:
            layers_to_freeze = min(num_layers_to_freeze, len(self.encoder_layers))
            for i in range(layers_to_freeze):
                for param in self.encoder_layers[i].parameters():
                    param.freeze()
    
    def unfreeze_all(self):
        """Unfreeze all model parameters."""
        for param in self.parameters():
            param.unfreeze()
    
    def _compute_additional_pooling(self, last_hidden_state: mx.array, attention_mask: Optional[mx.array]) -> Dict[str, mx.array]:
        """Compute additional pooling representations.
        
        Args:
            last_hidden_state: Last hidden state [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Dictionary of additional pooling representations
        """
        pooling_outputs = {}
        
        # CLS output (first token)
        pooling_outputs["cls_output"] = last_hidden_state[:, 0, :]
        
        # Mean pooling
        if attention_mask is not None:
            mask = attention_mask.astype(mx.float32)[..., None]
            masked_hidden = last_hidden_state * mask
            seq_lengths = mask.sum(axis=1)
            pooling_outputs["mean_pooled"] = masked_hidden.sum(axis=1) / seq_lengths
        else:
            pooling_outputs["mean_pooled"] = last_hidden_state.mean(axis=1)
        
        # Max pooling
        if attention_mask is not None:
            mask = attention_mask.astype(mx.float32)[..., None]
            masked_hidden = last_hidden_state + (1.0 - mask) * -1e9
            pooling_outputs["max_pooled"] = masked_hidden.max(axis=1)
        else:
            pooling_outputs["max_pooled"] = last_hidden_state.max(axis=1)
        
        return pooling_outputs