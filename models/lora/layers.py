"""LoRA and QLoRA layer implementations optimized for MLX.

This module provides efficient implementations of LoRA (Low-Rank Adaptation)
layers using MLX, designed for fast fine-tuning on Apple Silicon.
"""

import math
from typing import Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .config import LoRAConfig, QLoRAConfig
from ..quantization_utils import QuantizedLinear


class LoRALinear(nn.Module):
    """Linear layer with LoRA (Low-Rank Adaptation) for efficient fine-tuning.
    
    This layer adds trainable low-rank decomposition matrices (A and B) to a
    frozen pretrained linear layer, enabling efficient adaptation with minimal
    parameters.
    
    The forward pass computes: y = Wx + (BAx) * (alpha/r)
    where W is the frozen weight, B and A are the LoRA matrices.
    
    Args:
        in_features: Input dimension
        out_features: Output dimension  
        config: LoRA configuration
        base_layer: Optional existing linear layer to wrap
        device: MLX device
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: LoRAConfig,
        base_layer: Optional[nn.Linear] = None,
        device: Optional[mx.Device] = None,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        
        # Base layer (frozen during LoRA training)
        if base_layer is not None:
            self.base_layer = base_layer
            # Ensure dimensions match
            assert base_layer.weight.shape == (out_features, in_features)
        else:
            self.base_layer = nn.Linear(in_features, out_features, bias=True)
        
        # Freeze base layer
        self.base_layer.freeze()
        
        # LoRA parameters
        self.r = config.r
        self.alpha = config.alpha
        self.scaling = config.alpha / config.r
        
        # Initialize LoRA A and B matrices
        self._init_lora_layers()
        
        # Dropout for LoRA path
        self.lora_dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None
        
        # DoRA (Weight-Decomposed LoRA) components
        if config.use_dora:
            self.magnitude = mx.ones((out_features,))
            
        # RSLoRA (Rank-Stabilized LoRA) scaling
        if config.use_rslora:
            self.scaling = config.alpha / math.sqrt(config.r)
    
    def _init_lora_layers(self):
        """Initialize LoRA decomposition matrices."""
        init_method = self.config.init_lora_weights
        
        # LoRA A (down projection): [in_features, r]
        if init_method == "gaussian":
            # Kaiming uniform initialization scaled by alpha
            std = 1 / math.sqrt(self.in_features)
            self.lora_A = mx.random.normal(
                shape=(self.in_features, self.r),
                scale=std
            )
        elif init_method == "uniform":
            bound = 1 / math.sqrt(self.in_features)
            self.lora_A = mx.random.uniform(
                low=-bound,
                high=bound,
                shape=(self.in_features, self.r)
            )
        else:  # bert-style
            std = 0.02  # BERT's initialization std
            self.lora_A = mx.random.normal(
                shape=(self.in_features, self.r),
                scale=std
            )
        
        # LoRA B (up projection): [r, out_features] - initialized to zero
        self.lora_B = mx.zeros((self.r, self.out_features))
        
        # Optional LoRA bias
        if self.config.lora_bias_trainable:
            self.lora_bias = mx.zeros((self.out_features,))
        else:
            self.lora_bias = None
    
    def forward(self, x: mx.array) -> mx.array:
        """Forward pass with LoRA adaptation.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, in_features]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, out_features]
        """
        # Base layer forward (frozen)
        base_output = self.base_layer(x)
        
        # LoRA path
        if self.lora_dropout is not None:
            x_dropped = self.lora_dropout(x)
        else:
            x_dropped = x
        
        # Efficient LoRA computation: (x @ A) @ B
        lora_output = x_dropped @ self.lora_A  # [batch, seq, r]
        lora_output = lora_output @ self.lora_B  # [batch, seq, out]
        
        # Scale by alpha/r
        lora_output = lora_output * self.scaling
        
        # Add LoRA bias if enabled
        if self.lora_bias is not None:
            lora_output = lora_output + self.lora_bias
        
        # DoRA: apply magnitude vector
        if self.config.use_dora:
            # Normalize base weights and apply magnitude
            weight_norm = mx.sqrt(mx.sum(self.base_layer.weight ** 2, axis=1, keepdims=True))
            normalized_base = base_output / (weight_norm.T + 1e-8)
            base_output = normalized_base * self.magnitude
        
        return base_output + lora_output
    
    def merge_weights(self) -> nn.Linear:
        """Merge LoRA weights into base layer for inference.
        
        Returns:
            New Linear layer with merged weights
        """
        # Compute merged weight: W + (B @ A) * scaling
        merged_weight = self.base_layer.weight + (self.lora_B.T @ self.lora_A.T) * self.scaling
        
        # Create new linear layer with merged weights
        merged_layer = nn.Linear(self.in_features, self.out_features, bias=self.base_layer.bias is not None)
        merged_layer.weight = merged_weight
        
        if self.base_layer.bias is not None:
            merged_bias = self.base_layer.bias
            if self.lora_bias is not None:
                merged_bias = merged_bias + self.lora_bias
            merged_layer.bias = merged_bias
            
        return merged_layer
    
    def reset_lora_parameters(self):
        """Reset LoRA parameters to initial values."""
        self._init_lora_layers()
        if self.config.use_dora:
            self.magnitude = mx.ones((self.out_features,))
    
    @property
    def trainable_parameters(self) -> int:
        """Count of trainable parameters (LoRA only)."""
        count = self.r * (self.in_features + self.out_features)
        if self.lora_bias is not None:
            count += self.out_features
        if self.config.use_dora:
            count += self.out_features
        return count


class QLoRALinear(nn.Module):
    """Quantized Linear layer with LoRA for memory-efficient fine-tuning.
    
    Combines a 4-bit quantized base model with fp16 LoRA adapters, enabling
    fine-tuning of very large models on consumer hardware.
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        config: QLoRA configuration
        base_layer: Optional existing quantized linear layer
        device: MLX device
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: QLoRAConfig,
        base_layer: Optional[Union[nn.Linear, QuantizedLinear]] = None,
        device: Optional[mx.Device] = None,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        
        # Quantized base layer
        if base_layer is not None:
            if isinstance(base_layer, QuantizedLinear):
                self.base_layer = base_layer
            else:
                # Quantize the provided linear layer
                self.base_layer = self._quantize_linear(base_layer)
        else:
            # Create new quantized layer
            self.base_layer = self._create_quantized_layer()
        
        # Freeze quantized base
        self.base_layer.freeze()
        
        # LoRA components (in higher precision)
        self.r = config.r
        self.alpha = config.alpha
        self.scaling = config.alpha / config.r
        
        # Initialize LoRA matrices in specified compute dtype
        self.compute_dtype = self._get_compute_dtype()
        self._init_lora_layers()
        
        # Dropout
        self.lora_dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None
        
        # RSLoRA scaling
        if config.use_rslora:
            self.scaling = config.alpha / math.sqrt(config.r)
    
    def _get_compute_dtype(self) -> mx.Dtype:
        """Get MLX dtype from config string."""
        dtype_map = {
            "float16": mx.float16,
            "float32": mx.float32,
            "bfloat16": mx.bfloat16,
        }
        return dtype_map.get(self.config.bnb_4bit_compute_dtype, mx.float16)
    
    def _quantize_linear(self, linear: nn.Linear) -> QuantizedLinear:
        """Quantize a linear layer to 4-bit."""
        from ..quantization_utils import QuantizationConfig, quantize_module
        
        quant_config = QuantizationConfig(
            bits=4,
            group_size=128,
            quantization_type=self.config.bnb_4bit_quant_type,
        )
        
        return quantize_module(linear, quant_config)
    
    def _create_quantized_layer(self) -> QuantizedLinear:
        """Create a new quantized linear layer."""
        # Initialize with random weights then quantize
        temp_linear = nn.Linear(self.in_features, self.out_features)
        return self._quantize_linear(temp_linear)
    
    def _init_lora_layers(self):
        """Initialize LoRA matrices in compute dtype."""
        # LoRA A matrix
        std = 1 / math.sqrt(self.in_features)
        self.lora_A = mx.random.normal(
            shape=(self.in_features, self.r),
            scale=std,
            dtype=self.compute_dtype
        )
        
        # LoRA B matrix (initialized to zero)
        self.lora_B = mx.zeros(
            (self.r, self.out_features),
            dtype=self.compute_dtype
        )
        
        # Optional bias
        if self.config.lora_bias_trainable:
            self.lora_bias = mx.zeros(
                (self.out_features,),
                dtype=self.compute_dtype
            )
        else:
            self.lora_bias = None
    
    def forward(self, x: mx.array) -> mx.array:
        """Forward pass with quantized base and LoRA adaptation.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Cast input to compute dtype for LoRA path
        x_compute = x.astype(self.compute_dtype)
        
        # Quantized base forward
        base_output = self.base_layer(x)
        
        # LoRA path in higher precision
        if self.lora_dropout is not None:
            x_dropped = self.lora_dropout(x_compute)
        else:
            x_dropped = x_compute
        
        # Efficient LoRA computation
        lora_output = x_dropped @ self.lora_A
        lora_output = lora_output @ self.lora_B
        lora_output = lora_output * self.scaling
        
        if self.lora_bias is not None:
            lora_output = lora_output + self.lora_bias
        
        # Combine outputs (cast LoRA output to match base)
        output = base_output + lora_output.astype(base_output.dtype)
        
        return output
    
    @property
    def memory_footprint(self) -> dict:
        """Estimate memory usage of the layer."""
        base_params = self.in_features * self.out_features
        lora_params = self.r * (self.in_features + self.out_features)
        
        # 4-bit for base, compute_dtype for LoRA
        dtype_sizes = {
            mx.float16: 2,
            mx.float32: 4,
            mx.bfloat16: 2,
        }
        
        base_memory = base_params * 0.5  # 4-bit = 0.5 bytes per param
        lora_memory = lora_params * dtype_sizes.get(self.compute_dtype, 2)
        
        return {
            "base_params": base_params,
            "lora_params": lora_params,
            "base_memory_mb": base_memory / (1024 * 1024),
            "lora_memory_mb": lora_memory / (1024 * 1024),
            "total_memory_mb": (base_memory + lora_memory) / (1024 * 1024),
            "compression_ratio": base_params / (base_params * 0.5 + lora_params),
        }


class MultiLoRALinear(nn.Module):
    """Linear layer supporting multiple LoRA adapters.
    
    Useful for multi-task learning or ensemble approaches in Kaggle competitions.
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        base_layer: Base linear layer
        device: MLX device
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        base_layer: Optional[nn.Linear] = None,
        device: Optional[mx.Device] = None,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Base layer
        if base_layer is not None:
            self.base_layer = base_layer
        else:
            self.base_layer = nn.Linear(in_features, out_features)
        self.base_layer.freeze()
        
        # Dictionary of LoRA adapters
        self.lora_adapters = {}
        self.active_adapters = []
        self.combination_type = "average"
        self.adapter_weights = {}
    
    def add_adapter(self, name: str, config: LoRAConfig):
        """Add a new LoRA adapter.
        
        Args:
            name: Name of the adapter
            config: LoRA configuration
        """
        adapter = LoRALinear(
            self.in_features,
            self.out_features,
            config,
            base_layer=None,  # We handle base computation separately
        )
        self.lora_adapters[name] = adapter
        self.adapter_weights[name] = 1.0
    
    def activate_adapter(self, name: str):
        """Activate an adapter."""
        if name in self.lora_adapters and name not in self.active_adapters:
            self.active_adapters.append(name)
    
    def deactivate_adapter(self, name: str):
        """Deactivate an adapter.""" 
        if name in self.active_adapters:
            self.active_adapters.remove(name)
    
    def forward(self, x: mx.array) -> mx.array:
        """Forward pass with active adapters."""
        # Base output
        output = self.base_layer(x)
        
        if not self.active_adapters:
            return output
        
        # Collect LoRA outputs
        lora_outputs = []
        for adapter_name in self.active_adapters:
            adapter = self.lora_adapters[adapter_name]
            # Compute only LoRA path (not base)
            lora_out = x @ adapter.lora_A @ adapter.lora_B * adapter.scaling
            if adapter.lora_bias is not None:
                lora_out = lora_out + adapter.lora_bias
            lora_outputs.append(lora_out)
        
        # Combine LoRA outputs
        if self.combination_type == "average":
            combined_lora = mx.mean(mx.stack(lora_outputs), axis=0)
        elif self.combination_type == "weighted":
            weights = [self.adapter_weights.get(name, 1.0) for name in self.active_adapters]
            weights = mx.array(weights).reshape(-1, 1, 1)
            weighted = mx.stack(lora_outputs) * weights
            combined_lora = mx.sum(weighted, axis=0) / mx.sum(weights)
        else:  # concatenate
            combined_lora = mx.concatenate(lora_outputs, axis=-1)
            # Need projection back to output size
            if not hasattr(self, "projection"):
                self.projection = nn.Linear(
                    len(self.active_adapters) * self.out_features,
                    self.out_features
                )
            combined_lora = self.projection(combined_lora)
        
        return output + combined_lora