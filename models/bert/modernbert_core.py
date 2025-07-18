"""
ModernBERT core model implementation.

This module provides the complete ModernBERT model with all architectural
improvements from Answer.AI's 2024 release.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Dict, Optional, Tuple, Union, List
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from loguru import logger
import re
import os
from urllib.parse import urlparse

from .modernbert_config import ModernBertConfig
from .modernbert_embeddings import ModernBertEmbeddings, ModernBertPooler
from .modernbert_layer import ModernBertLayer
from .core import BertOutput  # Reuse the output dataclass


class ModernBertCore(nn.Module):
    """
    ModernBERT core model implementation.
    
    This class implements the complete ModernBERT architecture with all
    improvements from Answer.AI's 2024 release:
    - RoPE (Rotary Positional Embeddings)
    - GeGLU activation functions
    - Alternating attention mechanism
    - Streamlined architecture without bias terms
    - Enhanced normalization
    - 8192 sequence length support
    """
    
    def __init__(self, config: Union[ModernBertConfig, Dict]):
        """
        Initialize ModernBERT core model.
        
        Args:
            config: Model configuration (ModernBertConfig or dict)
        """
        super().__init__()
        
        # Convert dict to config if needed
        if isinstance(config, dict):
            config = ModernBertConfig(**config)
        
        self.config = config
        
        # Initialize the ModernBERT encoder
        self._build_encoder()
        
        # Additional pooling layers
        self.additional_pooling = config.compute_additional_pooling
        
        logger.info(
            f"Initialized ModernBertCore with config: "
            f"model_size={config.model_size}, "
            f"hidden_size={config.hidden_size}, "
            f"num_layers={config.num_hidden_layers}, "
            f"num_heads={config.num_attention_heads}, "
            f"max_seq_len={config.max_position_embeddings}, "
            f"use_rope={config.use_rope}, "
            f"use_geglu={config.use_geglu}, "
            f"use_alternating_attention={config.use_alternating_attention}"
        )
    
    def _build_encoder(self):
        """Build the ModernBERT encoder architecture."""
        # Create ModernBERT embedding layer
        self.embeddings = ModernBertEmbeddings(self.config)
        
        # Create encoder layers
        self.encoder_layers = []
        for layer_idx in range(self.config.num_hidden_layers):
            layer = ModernBertLayer(self.config, layer_idx)
            self.encoder_layers.append(layer)
        
        # Pooler for [CLS] token
        self.pooler = ModernBertPooler(self.config)
    
    def __call__(
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
        """Make ModernBertCore callable."""
        return self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            compute_pooling=compute_pooling,
            training=training,
        )
    
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
        """
        Forward pass through ModernBERT model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len] (not used in ModernBERT)
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output all hidden states
            compute_pooling: Whether to compute additional pooling
            training: Whether in training mode
            
        Returns:
            BertOutput with all model outputs
        """
        # Get embeddings (no position IDs in ModernBERT)
        embeddings = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=None,  # ModernBERT uses RoPE instead
        )
        
        # Pass through encoder layers
        hidden_states = embeddings
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        for layer in self.encoder_layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            # Pass through the layer with gradient checkpointing support
            if training and self.config.gradient_checkpointing:
                # Use gradient checkpointing for memory efficiency during training
                # Note: MLX gradient checkpointing implementation would go here
                # For now, fall back to regular forward pass
                layer_outputs = layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    output_attentions=output_attentions,
                )
            else:
                # Regular forward pass
                layer_outputs = layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    output_attentions=output_attentions,
                )
            
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        
        # Add last hidden state if requested
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        last_hidden_state = hidden_states
        
        # Get pooler output (from [CLS] token)
        pooler_output = self.pooler(last_hidden_state)
        
        # Create BertOutput
        bert_output = BertOutput(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
            attention_mask=attention_mask,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )
        
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
    
    def get_attention_pattern(self) -> List[str]:
        """Get the attention pattern for all layers."""
        pattern = []
        for layer_idx in range(self.config.num_hidden_layers):
            if self.config.use_alternating_attention:
                if (layer_idx + 1) % self.config.global_attention_every_n_layers == 0:
                    pattern.append("global")
                else:
                    pattern.append("local")
            else:
                pattern.append("global")
        return pattern
    
    def print_attention_pattern(self):
        """Print the attention pattern for debugging."""
        pattern = self.get_attention_pattern()
        print("ModernBERT Attention Pattern:")
        for i, attention_type in enumerate(pattern):
            print(f"  Layer {i:2d}: {attention_type}")
        
        global_count = pattern.count("global")
        local_count = pattern.count("local")
        print(f"\nSummary: {global_count} global, {local_count} local layers")
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, Path],
        cache_dir: Optional[str] = None,
        **kwargs
    ) -> "ModernBertCore":
        """
        Load ModernBERT model from pretrained weights.
        
        Args:
            model_path: Path to model directory or HuggingFace model ID
            cache_dir: Optional cache directory for downloaded models
            **kwargs: Additional configuration parameters
            
        Returns:
            Loaded ModernBertCore model
        """
        # Import hub utilities from the parent module
        from .core import _is_hub_model_id, _download_from_hub, _load_hf_config
        
        # Convert to string for hub model ID check
        model_path_str = str(model_path)
        
        # Check if this is a HuggingFace Hub model ID
        if _is_hub_model_id(model_path_str):
            logger.info(f"Detected HuggingFace Hub model ID: {model_path_str}")
            try:
                # Download from Hub
                downloaded_path = _download_from_hub(model_path_str, cache_dir)
                model_path = downloaded_path
            except Exception as e:
                logger.error(f"Failed to download from Hub: {e}")
                raise
        else:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model path does not exist: {model_path}")
        
        # Load config
        config_path = model_path / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            # Check if this is a ModernBERT config
            if config_dict.get("model_type") == "modernbert":
                config = ModernBertConfig.from_dict(config_dict)
            else:
                # Try to convert from standard BERT config
                logger.info("Converting standard BERT config to ModernBERT config")
                config = ModernBertConfig.from_dict(config_dict)
        else:
            logger.warning(f"No config.json found at {config_path}, using default config")
            config = ModernBertConfig()
        
        # Override config with any provided kwargs
        if kwargs:
            config_dict = config.to_dict()
            config_dict.update(kwargs)
            config = ModernBertConfig.from_dict(config_dict)
        
        # Create model
        model = cls(config)
        
        # Load weights if available
        weight_files = [
            "model.safetensors",
            "pytorch_model.safetensors",
            "model.bin",
        ]
        
        weights_loaded = False
        for weight_file in weight_files:
            weights_path = model_path / weight_file
            if weights_path.exists():
                try:
                    if weight_file.endswith('.safetensors'):
                        # Load safetensors format
                        weights = mx.load(str(weights_path))
                        model.load_weights(list(weights.items()))
                        logger.info(f"Loaded weights from {weights_path}")
                        weights_loaded = True
                        break
                    else:
                        logger.warning(f"Found {weight_file} but .safetensors format is preferred")
                except Exception as e:
                    logger.warning(f"Failed to load weights from {weights_path}: {e}")
                    continue
        
        if not weights_loaded:
            logger.warning(f"No compatible weight files found in {model_path}")
            logger.info("Model initialized with random weights")
        
        return model
    
    def save_pretrained(self, save_path: Union[str, Path]):
        """
        Save model to directory.
        
        Args:
            save_path: Directory to save model
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        self.config.save_pretrained(save_path)
        
        # Save weights
        from mlx.utils import tree_flatten
        weights = dict(tree_flatten(self.parameters()))
        mx.save_safetensors(str(save_path / "model.safetensors"), weights)
        
        # Save model metadata
        metadata = {
            "model_type": "ModernBertCore",
            "model_class": self.__class__.__name__,
            "model_size": self.config.model_size,
            "attention_pattern": self.get_attention_pattern(),
            "use_rope": self.config.use_rope,
            "use_geglu": self.config.use_geglu,
            "use_alternating_attention": self.config.use_alternating_attention,
            "version": "1.0.0",
        }
        
        with open(save_path / "model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"ModernBERT model saved to {save_path}")
    
    def freeze_encoder(self, num_layers_to_freeze: Optional[int] = None):
        """
        Freeze encoder layers for fine-tuning.
        
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
            
            logger.info(f"Frozen {layers_to_freeze} encoder layers")
    
    def unfreeze_all(self):
        """Unfreeze all model parameters."""
        for param in self.parameters():
            param.unfreeze()


# Factory functions
def create_modernbert_core(
    model_name: Optional[str] = None,
    config: Optional[Union[ModernBertConfig, Dict]] = None,
    model_size: str = "base",
    cache_dir: Optional[str] = None,
    **kwargs
) -> ModernBertCore:
    """
    Create a ModernBertCore model.
    
    Args:
        model_name: Optional pretrained model name (local path or HuggingFace model ID)
        config: Optional configuration
        model_size: Model size ("base" or "large")
        cache_dir: Optional cache directory for downloaded models
        **kwargs: Additional configuration parameters
        
    Returns:
        ModernBertCore model instance
    """
    if model_name:
        return ModernBertCore.from_pretrained(model_name, cache_dir=cache_dir, **kwargs)
    elif config:
        return ModernBertCore(config)
    else:
        # Use default config with kwargs
        if model_size == "base":
            config = ModernBertConfig.get_base_config()
        elif model_size == "large":
            config = ModernBertConfig.get_large_config()
        else:
            raise ValueError(f"Unknown model size: {model_size}")
        
        # Apply any additional kwargs
        if kwargs:
            config_dict = config.to_dict()
            config_dict.update(kwargs)
            config = ModernBertConfig.from_dict(config_dict)
        
        return ModernBertCore(config)


def create_modernbert_base(**kwargs) -> ModernBertCore:
    """Create ModernBERT-base model."""
    return create_modernbert_core(model_size="base", **kwargs)


def create_modernbert_large(**kwargs) -> ModernBertCore:
    """Create ModernBERT-large model."""
    return create_modernbert_core(model_size="large", **kwargs)