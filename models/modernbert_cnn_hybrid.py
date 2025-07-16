"""CNN-enhanced ModernBERT hybrid model for sophisticated text classification."""

import mlx.core as mx
import mlx.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from loguru import logger
from .modernbert_optimized import OptimizedModernBertMLX, ModernBertConfig, OptimizedEmbeddings, OptimizedTransformerBlock


@dataclass
class CNNHybridConfig(ModernBertConfig):
    """Configuration for CNN-enhanced ModernBERT."""
    # CNN-specific parameters
    cnn_kernel_sizes: List[int] = (2, 3, 4, 5)
    cnn_num_filters: int = 128
    use_dilated_conv: bool = True
    dilation_rates: List[int] = (1, 2, 4)
    use_attention_fusion: bool = True
    use_highway: bool = True
    cnn_dropout: float = 0.5
    fusion_hidden_size: int = 512
    

class MultiScaleConv1D(nn.Module):
    """Multi-scale 1D convolution module with different kernel sizes."""
    
    def __init__(
        self,
        in_channels: int,
        num_filters: int,
        kernel_sizes: List[int],
        activation: str = "relu",
        dropout: float = 0.5,
    ):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.in_channels = in_channels
        
        # Create conv-like layers using Linear for each kernel size
        # This is more compatible with MLX and avoids transpose issues
        self.conv_layers = []
        for k in kernel_sizes:
            # Use a linear layer to simulate 1D convolution
            # For each position, we'll consider a window of size k
            conv_linear = nn.Linear(in_channels * k, num_filters)
            self.conv_layers.append((k, conv_linear))
        
        # Activation and dropout
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
    def __call__(self, x: mx.array) -> List[mx.array]:
        """Apply multi-scale convolutions.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            List of conv outputs, each of shape (batch_size, seq_len, num_filters)
        """
        batch_size, seq_len, hidden_size = x.shape
        outputs = []
        
        for kernel_size, conv_linear in self.conv_layers:
            # Pad the input
            padding = kernel_size // 2
            if padding > 0:
                # Add padding
                pad_left = mx.zeros((batch_size, padding, hidden_size))
                pad_right = mx.zeros((batch_size, padding, hidden_size))
                x_padded = mx.concatenate([pad_left, x, pad_right], axis=1)
            else:
                x_padded = x
            
            # Extract windows and apply linear transformation
            conv_outputs = []
            for i in range(seq_len):
                # Extract window
                window = x_padded[:, i:i+kernel_size, :]  # (batch, kernel_size, hidden_size)
                # Flatten window
                window_flat = window.reshape(batch_size, -1)  # (batch, kernel_size * hidden_size)
                # Apply linear transformation
                conv_out = conv_linear(window_flat)  # (batch, num_filters)
                conv_outputs.append(conv_out)
            
            # Stack outputs
            conv_output = mx.stack(conv_outputs, axis=1)  # (batch, seq_len, num_filters)
            
            # Apply activation and dropout
            conv_output = self.activation(conv_output)
            conv_output = self.dropout(conv_output)
            outputs.append(conv_output)
        
        return outputs


class DilatedConvBlock(nn.Module):
    """Dilated convolution block for capturing long-range dependencies."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation_rates: List[int] = (1, 2, 4),
        dropout: float = 0.5,
    ):
        super().__init__()
        self.dilation_rates = dilation_rates
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        
        # Create dilated conv layers using Linear
        self.dilated_convs = []
        for d in dilation_rates:
            # Effective kernel size with dilation
            effective_kernel = (kernel_size - 1) * d + 1
            conv_linear = nn.Linear(in_channels * kernel_size, out_channels)
            self.dilated_convs.append((d, conv_linear))
        
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_channels)
        
    def __call__(self, x: mx.array) -> mx.array:
        """Apply dilated convolutions and aggregate."""
        batch_size, seq_len, hidden_size = x.shape
        outputs = []
        
        for dilation, conv_linear in self.dilated_convs:
            # Apply dilated convolution
            conv_outputs = []
            for i in range(seq_len):
                # Collect dilated indices
                indices = []
                for k in range(self.kernel_size):
                    idx = i + k * dilation - (self.kernel_size // 2) * dilation
                    if 0 <= idx < seq_len:
                        indices.append(idx)
                
                if len(indices) == self.kernel_size:
                    # Extract dilated window
                    window_parts = [x[:, idx:idx+1, :] for idx in indices]
                    window = mx.concatenate(window_parts, axis=1)  # (batch, kernel_size, hidden_size)
                    # Flatten and apply linear
                    window_flat = window.reshape(batch_size, -1)
                    conv_out = conv_linear(window_flat)
                    conv_outputs.append(conv_out)
                else:
                    # Padding case - use zeros
                    conv_outputs.append(mx.zeros((batch_size, conv_linear.weight.shape[0])))
            
            # Stack outputs
            conv_output = mx.stack(conv_outputs, axis=1)  # (batch, seq_len, out_channels)
            conv_output = self.activation(conv_output)
            outputs.append(conv_output)
        
        # Aggregate by averaging
        aggregated = mx.mean(mx.stack(outputs), axis=0)
        aggregated = self.dropout(aggregated)
        aggregated = self.norm(aggregated)
        
        return aggregated


class AttentionFusion(nn.Module):
    """Attention-based fusion mechanism for combining multiple feature representations."""
    
    def __init__(self, input_dim: int, num_inputs: int):
        super().__init__()
        self.input_dim = input_dim
        self.num_inputs = num_inputs
        
        # Attention scoring network
        self.attention_scorer = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1),
        )
        
        # Feature transformation
        self.feature_transform = nn.Linear(input_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)
        
    def __call__(self, features: List[mx.array]) -> mx.array:
        """Fuse multiple feature representations using attention.
        
        Args:
            features: List of tensors, each of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Fused features of shape (batch_size, seq_len, input_dim)
        """
        # Stack features: (num_inputs, batch_size, seq_len, input_dim)
        stacked = mx.stack(features)
        
        # Compute attention scores
        scores = []
        for i in range(len(features)):
            # Global average pooling for each feature
            pooled = mx.mean(features[i], axis=1)  # (batch_size, input_dim)
            score = self.attention_scorer(pooled)  # (batch_size, 1)
            scores.append(score)
        
        # Normalize scores
        scores = mx.stack(scores, axis=1)  # (batch_size, num_inputs, 1)
        attention_weights = mx.softmax(scores, axis=1)
        
        # Apply attention weights
        weighted_features = []
        for i in range(len(features)):
            weight = attention_weights[:, i:i+1, :]  # (batch_size, 1, 1)
            weighted = features[i] * weight.reshape(-1, 1, 1)
            weighted_features.append(weighted)
        
        # Sum weighted features
        fused = mx.sum(mx.stack(weighted_features), axis=0)
        
        # Transform and normalize
        fused = self.feature_transform(fused)
        fused = self.norm(fused)
        
        return fused


class HighwayNetwork(nn.Module):
    """Highway network for better gradient flow."""
    
    def __init__(self, input_dim: int, num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers
        
        self.transforms = []
        self.gates = []
        
        for _ in range(num_layers):
            transform = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.ReLU(),
            )
            gate = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.Sigmoid(),
            )
            self.transforms.append(transform)
            self.gates.append(gate)
    
    def __call__(self, x: mx.array) -> mx.array:
        """Apply highway transformation."""
        for transform, gate in zip(self.transforms, self.gates):
            transform_out = transform(x)
            gate_out = gate(x)
            x = gate_out * transform_out + (1 - gate_out) * x
        return x


class CNNEnhancedModernBERT(nn.Module):
    """CNN-enhanced ModernBERT with sophisticated convolutional layers."""
    
    def __init__(self, config: CNNHybridConfig):
        super().__init__()
        self.config = config
        # Override hidden_size to match actual output dimension
        self.output_hidden_size = config.fusion_hidden_size
        
        # Base ModernBERT encoder
        self.embeddings = OptimizedEmbeddings(config)
        self.transformer_layers = [OptimizedTransformerBlock(config) for _ in range(config.num_hidden_layers)]
        
        # Multi-scale CNN layers
        self.multi_scale_cnn = MultiScaleConv1D(
            in_channels=config.hidden_size,
            num_filters=config.cnn_num_filters,
            kernel_sizes=config.cnn_kernel_sizes,
            dropout=config.cnn_dropout,
        )
        
        # Dilated convolutions
        if config.use_dilated_conv:
            self.dilated_conv = DilatedConvBlock(
                in_channels=config.hidden_size,
                out_channels=config.cnn_num_filters,
                dilation_rates=config.dilation_rates,
                dropout=config.cnn_dropout,
            )
        
        # Calculate total CNN output size
        num_conv_outputs = len(config.cnn_kernel_sizes)
        if config.use_dilated_conv:
            num_conv_outputs += 1
        
        # Feature fusion
        if config.use_attention_fusion:
            self.fusion = AttentionFusion(
                input_dim=config.cnn_num_filters,
                num_inputs=num_conv_outputs,
            )
            fusion_output_dim = config.cnn_num_filters
        else:
            # Simple concatenation
            fusion_output_dim = config.cnn_num_filters * num_conv_outputs
        
        # Highway network for better gradient flow
        if config.use_highway:
            self.highway = HighwayNetwork(fusion_output_dim)
        
        # Projection layer to combine BERT and CNN features
        self.feature_projection = nn.Sequential(
            nn.Linear(config.hidden_size + fusion_output_dim, config.fusion_hidden_size),
            nn.LayerNorm(config.fusion_hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.fusion_hidden_size, config.fusion_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.fusion_hidden_size // 2, config.num_labels),
        )
        
        logger.info(
            f"Initialized CNN-Enhanced ModernBERT with:\n"
            f"  - {config.num_hidden_layers} transformer layers\n"
            f"  - Multi-scale CNN with kernels: {config.cnn_kernel_sizes}\n"
            f"  - Dilated conv: {config.use_dilated_conv}\n"
            f"  - Attention fusion: {config.use_attention_fusion}\n"
            f"  - Highway network: {config.use_highway}"
        )
    
    def extract_bert_features(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        token_type_ids: Optional[mx.array] = None,
    ) -> mx.array:
        """Extract BERT features."""
        # Get embeddings
        hidden_states = self.embeddings(input_ids, token_type_ids)
        
        # Apply transformer blocks
        for layer in self.transformer_layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        return hidden_states
    
    def extract_cnn_features(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Extract CNN features from BERT embeddings."""
        # Apply multi-scale convolutions
        multi_scale_outputs = self.multi_scale_cnn(hidden_states)
        
        # Apply dilated convolutions if enabled
        all_conv_outputs = multi_scale_outputs
        if self.config.use_dilated_conv:
            dilated_output = self.dilated_conv(hidden_states)
            all_conv_outputs.append(dilated_output)
        
        # Mask padding positions if attention mask provided
        if attention_mask is not None:
            mask_expanded = attention_mask.reshape(-1, attention_mask.shape[1], 1)
            all_conv_outputs = [
                output * mask_expanded for output in all_conv_outputs
            ]
        
        # Fuse CNN features
        if self.config.use_attention_fusion:
            fused_cnn = self.fusion(all_conv_outputs)
        else:
            # Simple concatenation
            fused_cnn = mx.concatenate(all_conv_outputs, axis=-1)
        
        # Apply highway network if enabled
        if self.config.use_highway:
            fused_cnn = self.highway(fused_cnn)
        
        # Global max pooling over sequence
        pooled_cnn = mx.max(fused_cnn, axis=1)
        
        return pooled_cnn
    
    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        token_type_ids: Optional[mx.array] = None,
        labels: Optional[mx.array] = None,
        return_dict: bool = True,
    ) -> Union[Dict[str, mx.array], Tuple[mx.array, ...]]:
        """Forward pass combining BERT and CNN features."""
        
        # Extract BERT features
        bert_hidden_states = self.extract_bert_features(
            input_ids, attention_mask, token_type_ids
        )
        
        # Extract CNN features
        cnn_features = self.extract_cnn_features(bert_hidden_states, attention_mask)
        
        # Get BERT's CLS token representation
        bert_pooled = bert_hidden_states[:, 0, :]  # CLS token
        
        # Combine BERT and CNN features
        combined_features = mx.concatenate([bert_pooled, cnn_features], axis=-1)
        
        # Project to fusion hidden size
        fused_features = self.feature_projection(combined_features)
        
        # Classification
        logits = self.classifier(fused_features)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = mx.mean(nn.losses.cross_entropy(logits, labels, reduction='none'))
        
        if return_dict:
            return {
                'loss': loss,
                'logits': logits,
                'bert_hidden_states': bert_hidden_states,
                'cnn_features': cnn_features,
                'fused_features': fused_features,
                'pooled_output': fused_features,  # Add this for compatibility with TitanicClassifier
            }
        else:
            return (loss, logits, bert_hidden_states, cnn_features, fused_features)
    
    def save_pretrained(self, save_path: str):
        """Save model weights and config."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(save_path / "config.json", "w") as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        # Save weights
        weights = dict(self.parameters())
        mx.save_safetensors(str(save_path / "model.safetensors"), weights)
        
        logger.info(f"CNN-Enhanced ModernBERT saved to {save_path}")
    
    @classmethod
    def from_pretrained(
        cls,
        path_or_name: str,
        num_labels: int = 2,
        **kwargs
    ) -> "CNNEnhancedModernBERT":
        """Load model from pretrained weights or initialize new."""
        path = Path(path_or_name)
        
        if path.exists() and path.is_dir():
            # Load from local path
            with open(path / "config.json") as f:
                config_dict = json.load(f)
            config = CNNHybridConfig(**config_dict)
            config.num_labels = num_labels
            
            # Override config with kwargs
            for k, v in kwargs.items():
                if hasattr(config, k):
                    setattr(config, k, v)
            
            model = cls(config)
            
            # Load weights
            weights = mx.load(str(path / "model.safetensors"))
            model.load_weights(list(weights.items()))
            
            logger.info(f"Loaded CNN-Enhanced ModernBERT from {path}")
        else:
            # Initialize from HuggingFace config
            config = CNNHybridConfig.from_pretrained(path_or_name)
            config.num_labels = num_labels
            
            # Override with kwargs
            for k, v in kwargs.items():
                if hasattr(config, k):
                    setattr(config, k, v)
            
            model = cls(config)
            logger.info(f"Initialized CNN-Enhanced ModernBERT with config from {path_or_name}")
        
        return model


def create_cnn_hybrid_model(
    model_name: str = "answerdotai/ModernBERT-base",
    num_labels: int = 2,
    cnn_kernel_sizes: List[int] = (2, 3, 4, 5),
    cnn_num_filters: int = 128,
    use_dilated_conv: bool = True,
    use_attention_fusion: bool = True,
    use_highway: bool = True,
    **kwargs
) -> CNNEnhancedModernBERT:
    """Create a CNN-enhanced ModernBERT model."""
    return CNNEnhancedModernBERT.from_pretrained(
        model_name,
        num_labels=num_labels,
        cnn_kernel_sizes=cnn_kernel_sizes,
        cnn_num_filters=cnn_num_filters,
        use_dilated_conv=use_dilated_conv,
        use_attention_fusion=use_attention_fusion,
        use_highway=use_highway,
        **kwargs
    )