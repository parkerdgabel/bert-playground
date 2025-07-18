"""
ModernBERT layer implementation.

This module implements the complete ModernBERT transformer layer
with all architectural improvements.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple

from .modernbert_config import ModernBertConfig
from .alternating_attention import create_attention_layer
from .activations import GeGLUMLP, get_normalization_function


class ModernBertAttentionOutput(nn.Module):
    """
    ModernBERT attention output processing.
    
    Streamlined version without bias terms.
    """
    
    def __init__(self, config: ModernBertConfig):
        """
        Initialize ModernBERT attention output.
        
        Args:
            config: ModernBERT configuration
        """
        super().__init__()
        self.config = config
        
        # Dense layer (no bias)
        self.dense = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=config.use_bias,
        )
        
        # Layer normalization (no bias)
        self.LayerNorm = get_normalization_function(
            norm_name="layer_norm",
            hidden_size=config.hidden_size,
            eps=config.layer_norm_eps,
        )
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, hidden_states: mx.array, input_tensor: mx.array) -> mx.array:
        """
        Forward pass through attention output.
        
        Args:
            hidden_states: Output from attention [batch_size, seq_len, hidden_size]
            input_tensor: Input tensor for residual connection [batch_size, seq_len, hidden_size]
            
        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    
    def __call__(self, hidden_states: mx.array, input_tensor: mx.array) -> mx.array:
        """Make the attention output callable."""
        return self.forward(hidden_states, input_tensor)


class ModernBertAttention(nn.Module):
    """
    ModernBERT attention layer.
    
    Combines alternating attention mechanism with output processing.
    """
    
    def __init__(self, config: ModernBertConfig, layer_idx: int):
        """
        Initialize ModernBERT attention.
        
        Args:
            config: ModernBERT configuration
            layer_idx: Index of the current layer
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Attention mechanism (alternating or global)
        self.self = create_attention_layer(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            layer_idx=layer_idx,
            config=config,
        )
        
        # Output processing
        self.output = ModernBertAttentionOutput(config)
    
    def forward(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        output_attentions: bool = False,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        """
        Forward pass through ModernBERT attention.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len]
            output_attentions: Whether to output attention weights
            
        Returns:
            Tuple of (attention_output, attention_weights)
        """
        # Apply attention mechanism
        self_outputs = self.self(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
        )
        
        # Process attention output
        attention_output = self.output(self_outputs[0], hidden_states)
        
        # Return outputs
        outputs = (attention_output,) + self_outputs[1:]  # Add attentions if we output them
        return outputs
    
    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        output_attentions: bool = False,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        """Make the attention layer callable."""
        return self.forward(hidden_states, attention_mask, position_ids, output_attentions)


class ModernBertFFNOutput(nn.Module):
    """
    ModernBERT feed-forward network output layer.
    
    Streamlined version without bias terms.
    """
    
    def __init__(self, config: ModernBertConfig):
        """
        Initialize ModernBERT FFN output.
        
        Args:
            config: ModernBERT configuration
        """
        super().__init__()
        self.config = config
        
        # Dense layer (no bias)
        self.dense = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=config.use_bias,
        )
        
        # Layer normalization (no bias)
        self.LayerNorm = get_normalization_function(
            norm_name="layer_norm",
            hidden_size=config.hidden_size,
            eps=config.layer_norm_eps,
        )
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, hidden_states: mx.array, input_tensor: mx.array) -> mx.array:
        """
        Forward pass through FFN output.
        
        Args:
            hidden_states: Output from FFN [batch_size, seq_len, intermediate_size]
            input_tensor: Input tensor for residual connection [batch_size, seq_len, hidden_size]
            
        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    
    def __call__(self, hidden_states: mx.array, input_tensor: mx.array) -> mx.array:
        """Make the FFN output callable."""
        return self.forward(hidden_states, input_tensor)


class ModernBertFeedForward(nn.Module):
    """
    ModernBERT feed-forward network.
    
    Uses GeGLU activation instead of standard GELU.
    """
    
    def __init__(self, config: ModernBertConfig):
        """
        Initialize ModernBERT feed-forward network.
        
        Args:
            config: ModernBERT configuration
        """
        super().__init__()
        self.config = config
        
        if config.use_geglu:
            # Use GeGLU MLP
            self.mlp = GeGLUMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                use_bias=config.use_bias,
                dropout_prob=config.hidden_dropout_prob,
                gate_limit=config.geglu_limit,
            )
        else:
            # Use standard MLP with GELU
            self.intermediate = nn.Linear(
                config.hidden_size,
                config.intermediate_size,
                bias=config.use_bias,
            )
            self.activation = nn.GELU()
            self.output = ModernBertFFNOutput(config)
    
    def forward(self, hidden_states: mx.array) -> mx.array:
        """
        Forward pass through feed-forward network.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            
        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        if self.config.use_geglu:
            # GeGLU MLP handles everything
            return self.mlp(hidden_states)
        else:
            # Standard MLP
            intermediate_states = self.intermediate(hidden_states)
            intermediate_states = self.activation(intermediate_states)
            return self.output(intermediate_states, hidden_states)
    
    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Make the feed-forward network callable."""
        return self.forward(hidden_states)


class ModernBertLayer(nn.Module):
    """
    Complete ModernBERT transformer layer.
    
    Combines all ModernBERT improvements:
    - Alternating attention mechanism
    - GeGLU activation
    - Streamlined architecture without bias terms
    - Enhanced normalization
    """
    
    def __init__(self, config: ModernBertConfig, layer_idx: int):
        """
        Initialize ModernBERT layer.
        
        Args:
            config: ModernBERT configuration
            layer_idx: Index of the current layer
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Attention layer
        self.attention = ModernBertAttention(config, layer_idx)
        
        # Feed-forward network
        self.feed_forward = ModernBertFeedForward(config)
    
    def forward(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        output_attentions: bool = False,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        """
        Forward pass through ModernBERT layer.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len]
            output_attentions: Whether to output attention weights
            
        Returns:
            Tuple of (layer_output, attention_weights)
        """
        # Self-attention
        attention_outputs = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
        )
        attention_output = attention_outputs[0]
        
        # Feed-forward network
        layer_output = self.feed_forward(attention_output)
        
        # Return outputs
        outputs = (layer_output,) + attention_outputs[1:]  # Add attentions if we output them
        return outputs
    
    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        output_attentions: bool = False,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        """Make the layer callable."""
        return self.forward(hidden_states, attention_mask, position_ids, output_attentions)
    
    def get_attention_type(self) -> str:
        """Get the attention type for this layer."""
        if hasattr(self.attention.self, 'get_attention_type'):
            return self.attention.self.get_attention_type()
        else:
            return "global"


def create_modernbert_layer(config: ModernBertConfig, layer_idx: int) -> ModernBertLayer:
    """
    Create a ModernBERT layer.
    
    Args:
        config: ModernBERT configuration
        layer_idx: Index of the current layer
        
    Returns:
        ModernBertLayer instance
    """
    return ModernBertLayer(config, layer_idx)


def create_modernbert_layers(config: ModernBertConfig) -> nn.Sequential:
    """
    Create all ModernBERT layers.
    
    Args:
        config: ModernBERT configuration
        
    Returns:
        Sequential container with all layers
    """
    layers = []
    for layer_idx in range(config.num_hidden_layers):
        layer = create_modernbert_layer(config, layer_idx)
        layers.append(layer)
    
    return nn.Sequential(*layers)


# Utility functions
def get_attention_pattern(config: ModernBertConfig) -> list:
    """
    Get the attention pattern for all layers.
    
    Args:
        config: ModernBERT configuration
        
    Returns:
        List of attention types for each layer
    """
    pattern = []
    for layer_idx in range(config.num_hidden_layers):
        if config.use_alternating_attention:
            if (layer_idx + 1) % config.global_attention_every_n_layers == 0:
                pattern.append("global")
            else:
                pattern.append("local")
        else:
            pattern.append("global")
    
    return pattern


def print_attention_pattern(config: ModernBertConfig):
    """
    Print the attention pattern for debugging.
    
    Args:
        config: ModernBERT configuration
    """
    pattern = get_attention_pattern(config)
    print("ModernBERT Attention Pattern:")
    for i, attention_type in enumerate(pattern):
        print(f"  Layer {i:2d}: {attention_type}")
    
    global_count = pattern.count("global")
    local_count = pattern.count("local")
    print(f"\nSummary: {global_count} global, {local_count} local layers")