"""MLX model quantization utilities.

This module provides quantization support for MLX models:
- 4-bit and 8-bit quantization
- Layer-wise quantization configuration
- Quantization-aware training
- Model compression utilities
"""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from loguru import logger


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""

    bits: int = 4  # Number of bits (4 or 8)
    group_size: int = 64  # Group size for quantization
    quantize_embeddings: bool = False  # Whether to quantize embeddings
    quantize_output_layer: bool = False  # Whether to quantize output layer
    symmetric: bool = True  # Use symmetric quantization
    per_channel: bool = True  # Per-channel vs per-tensor quantization

    # Layer-specific configurations
    layer_configs: dict[str, dict] | None = None

    def get_layer_config(self, layer_name: str) -> dict | None:
        """Get configuration for specific layer."""
        if self.layer_configs and layer_name in self.layer_configs:
            return self.layer_configs[layer_name]
        return None


class QuantizedLinear(nn.Module):
    """Quantized linear layer for MLX."""

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bits: int = 4,
        group_size: int = 64,
        bias: bool = True,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.bits = bits
        self.group_size = group_size

        # Initialize quantized weight
        self.weight = mx.random.normal((output_dims, input_dims))
        if bias:
            self.bias = mx.zeros((output_dims,))
        else:
            self.bias = None

        # Quantization parameters
        self.scale = None
        self.zero_point = None
        self.quantized_weight = None

    def quantize_weights(self):
        """Quantize the weights."""
        # Calculate quantization range
        if self.bits == 4:
            qmin, qmax = -8, 7
        elif self.bits == 8:
            qmin, qmax = -128, 127
        else:
            raise ValueError(f"Unsupported bit width: {self.bits}")

        # Reshape weight for group quantization
        weight_reshaped = self.weight.reshape(-1, self.group_size)

        # Calculate scale and zero point per group
        w_min = mx.min(weight_reshaped, axis=1, keepdims=True)
        w_max = mx.max(weight_reshaped, axis=1, keepdims=True)

        # Symmetric quantization
        w_abs_max = mx.maximum(mx.abs(w_min), mx.abs(w_max))
        self.scale = w_abs_max / qmax
        self.zero_point = mx.zeros_like(self.scale)

        # Avoid division by zero
        self.scale = mx.maximum(self.scale, 1e-8)

        # Quantize
        quantized = mx.round(weight_reshaped / self.scale)
        quantized = mx.clip(quantized, qmin, qmax)

        # Store quantized weights
        self.quantized_weight = quantized.astype(mx.int8 if self.bits == 8 else mx.int8)

        logger.debug(
            f"Quantized linear layer: {self.input_dims}x{self.output_dims} "
            f"to {self.bits} bits with group_size={self.group_size}"
        )

    def dequantize_weights(self) -> mx.array:
        """Dequantize weights for computation."""
        if self.quantized_weight is None:
            return self.weight

        # Dequantize
        dequantized = self.quantized_weight.astype(mx.float32) * self.scale

        # Reshape back to original shape
        return dequantized.reshape(self.output_dims, self.input_dims)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with quantized weights."""
        # Get effective weights
        if self.quantized_weight is not None:
            weight = self.dequantize_weights()
        else:
            weight = self.weight

        # Linear transformation
        y = x @ weight.T

        if self.bias is not None:
            y = y + self.bias

        return y


class ModelQuantizer:
    """Utility class for quantizing MLX models."""

    def __init__(self, config: QuantizationConfig):
        self.config = config

    def quantize_model(self, model: nn.Module) -> nn.Module:
        """Quantize a model according to configuration."""
        logger.info(f"Starting model quantization with {self.config.bits} bits")

        # Track quantization statistics
        total_params = 0
        quantized_params = 0

        # Recursively quantize layers
        self._quantize_module(model, "", total_params, quantized_params)

        # Log statistics
        compression_ratio = (
            total_params / max(1, quantized_params) if quantized_params > 0 else 1.0
        )
        logger.info(
            f"Quantization complete: "
            f"Original params: {total_params:,}, "
            f"Quantized params: {quantized_params:,}, "
            f"Compression ratio: {compression_ratio:.2f}x"
        )

        return model

    def _quantize_module(
        self,
        module: nn.Module,
        name: str,
        total_params: int,
        quantized_params: int,
    ):
        """Recursively quantize modules."""
        for child_name, child in module.items():
            full_name = f"{name}.{child_name}" if name else child_name

            if isinstance(child, nn.Linear):
                # Check if we should quantize this layer
                if self._should_quantize_layer(full_name):
                    # Replace with quantized version
                    quantized = QuantizedLinear(
                        input_dims=child.weight.shape[1],
                        output_dims=child.weight.shape[0],
                        bits=self._get_layer_bits(full_name),
                        group_size=self._get_layer_group_size(full_name),
                        bias=child.bias is not None,
                    )

                    # Copy weights
                    quantized.weight = child.weight
                    if child.bias is not None:
                        quantized.bias = child.bias

                    # Quantize
                    quantized.quantize_weights()

                    # Replace module
                    setattr(module, child_name, quantized)

                    # Update statistics
                    param_count = child.weight.size
                    if child.bias is not None:
                        param_count += child.bias.size
                    total_params += param_count
                    quantized_params += (
                        param_count * self._get_layer_bits(full_name) // 32
                    )

            elif isinstance(child, nn.Module):
                # Recurse into submodules
                self._quantize_module(child, full_name, total_params, quantized_params)

    def _should_quantize_layer(self, layer_name: str) -> bool:
        """Determine if a layer should be quantized."""
        # Check layer-specific config
        layer_config = self.config.get_layer_config(layer_name)
        if layer_config and "quantize" in layer_config:
            return layer_config["quantize"]

        # Check global rules
        if "embed" in layer_name.lower() and not self.config.quantize_embeddings:
            return False

        return not (
            ("lm_head" in layer_name.lower() or "output" in layer_name.lower())
            and not self.config.quantize_output_layer
        )

    def _get_layer_bits(self, layer_name: str) -> int:
        """Get bit width for specific layer."""
        layer_config = self.config.get_layer_config(layer_name)
        if layer_config and "bits" in layer_config:
            return layer_config["bits"]
        return self.config.bits

    def _get_layer_group_size(self, layer_name: str) -> int:
        """Get group size for specific layer."""
        layer_config = self.config.get_layer_config(layer_name)
        if layer_config and "group_size" in layer_config:
            return layer_config["group_size"]
        return self.config.group_size


def create_default_quantization_config(
    model_type: str = "bert",
    bits: int = 4,
) -> QuantizationConfig:
    """Create default quantization configuration for model type."""
    if model_type == "bert":
        # BERT-specific configuration
        layer_configs = {
            # Higher precision for embeddings
            "embeddings": {"bits": 8, "group_size": 128},
            # Lower precision for intermediate layers
            "encoder.layers": {"bits": bits, "group_size": 64},
            # Higher precision for output
            "lm_head": {"bits": 8, "group_size": 64},
        }
    else:
        layer_configs = None

    return QuantizationConfig(
        bits=bits,
        group_size=64,
        quantize_embeddings=True,
        quantize_output_layer=True,
        symmetric=True,
        per_channel=True,
        layer_configs=layer_configs,
    )


def quantize_model_for_inference(
    model: nn.Module,
    config: QuantizationConfig | None = None,
) -> nn.Module:
    """Quantize model for inference with default settings."""
    if config is None:
        config = create_default_quantization_config()

    quantizer = ModelQuantizer(config)
    return quantizer.quantize_model(model)


def estimate_model_size(
    model: nn.Module,
    quantization_config: QuantizationConfig | None = None,
) -> dict[str, float]:
    """Estimate model size before and after quantization."""
    # Calculate original size
    original_params = sum(p.size for p in model.parameters())
    original_size_mb = original_params * 4 / (1024 * 1024)  # float32

    if quantization_config:
        # Estimate quantized size
        # This is a rough estimate
        avg_bits = quantization_config.bits
        quantized_size_mb = original_params * avg_bits / 8 / (1024 * 1024)
        compression_ratio = original_size_mb / quantized_size_mb
    else:
        quantized_size_mb = original_size_mb
        compression_ratio = 1.0

    return {
        "original_params": original_params,
        "original_size_mb": original_size_mb,
        "quantized_size_mb": quantized_size_mb,
        "compression_ratio": compression_ratio,
    }
