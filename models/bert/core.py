"""Core BERT model implementation with modular interface.

This module provides the consolidated BERT implementation that can be easily
attached to any head in the heads directory.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Dict, Optional, Tuple, Union, List
from pathlib import Path
import json
import numpy as np
from loguru import logger
import re
import os
from urllib.parse import urlparse

from .config import BertConfig
from .core_base import BaseBertModel, BertModelOutput, BertLayer
from .layers import BertEmbeddings, BertPooler


def _is_hub_model_id(model_path: str) -> bool:
    """Check if the model path is a HuggingFace Hub model ID.
    
    Args:
        model_path: Path or model ID to check
        
    Returns:
        True if it's a Hub model ID, False otherwise
    """
    # Check if it's a local path
    if os.path.exists(model_path):
        return False
    
    # Check if it's a URL
    if urlparse(model_path).scheme:
        return False
    
    # Check if it matches the pattern: organization/model-name
    # Allow for optional revision: organization/model-name@revision
    pattern = r'^[a-zA-Z0-9][\w\-\.]*\/[a-zA-Z0-9][\w\-\.]*(@[\w\-\.]+)?$'
    return bool(re.match(pattern, model_path))


def _download_from_hub(model_id: str, cache_dir: Optional[str] = None) -> Path:
    """Download model from HuggingFace Hub.
    
    Args:
        model_id: HuggingFace Hub model ID
        cache_dir: Optional cache directory
        
    Returns:
        Path to downloaded model directory
        
    Raises:
        ImportError: If huggingface_hub is not installed
        Exception: If download fails
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required to download models from the Hub. "
            "Install it with: pip install huggingface_hub"
        )
    
    logger.info(f"Downloading model '{model_id}' from HuggingFace Hub...")
    
    try:
        # Download the model files
        model_path = snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            local_files_only=False,
            # Only download the files we need for MLX
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.md"],
            ignore_patterns=["*.bin", "*.h5", "*.msgpack", "*.ot", "*.pt"],
        )
        
        logger.info(f"Model downloaded to: {model_path}")
        return Path(model_path)
        
    except Exception as e:
        logger.error(f"Failed to download model '{model_id}': {e}")
        raise


def _load_hf_config(config_path: Path) -> BertConfig:
    """Load HuggingFace config and convert to BertConfig.
    
    Args:
        config_path: Path to config.json file
        
    Returns:
        BertConfig instance
    """
    with open(config_path, 'r') as f:
        hf_config = json.load(f)
    
    # Check if this is a HuggingFace config (has model_type field)
    if "model_type" in hf_config:
        logger.info("Loading HuggingFace format config")
        return BertConfig.from_hf_config(hf_config)
    else:
        # Assume it's already in our format
        logger.info("Loading MLX format config")
        return BertConfig.from_dict(hf_config)


# For backward compatibility with existing code
BertOutput = BertModelOutput


class BertCore(BaseBertModel):
    """Core BERT model with standardized interface.
    
    This class provides the core BERT encoder with a clean interface
    for attaching any head from the heads directory.
    """
    
    def __init__(self, config: Union[BertConfig, Dict]):
        """Initialize BERT core model.
        
        Args:
            config: Model configuration (BertConfig or dict)
        """
        super().__init__(config)
        
        # Initialize the BERT encoder layers
        self._build_encoder()
        
        logger.info(f"Initialized BertCore with config: hidden_size={config.hidden_size}, "
                   f"num_layers={config.num_hidden_layers}, num_heads={config.num_attention_heads}")
    
    def _build_encoder(self):
        """Build the BERT encoder layers.
        
        This implementation creates the full BERT architecture with proper:
        - Token embeddings
        - Position embeddings (learned)
        - Token type embeddings (segment embeddings)
        - Layer normalization and dropout
        """
        # Create encoder layers using BERT-specific layers
        self.encoder_layers = []
        for _ in range(self.config.num_hidden_layers):
            layer = BertLayer(self.config)
            self.encoder_layers.append(layer)
        
        logger.info(f"Built BERT encoder with {len(self.encoder_layers)} layers")
    
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
    ) -> BertModelOutput:
        """Make BertCore callable."""
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
    ) -> BertModelOutput:
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
            BertModelOutput with all model outputs
        """
        # Get embeddings
        embeddings = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        
        # Pass through encoder layers
        hidden_states = embeddings
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        for layer in self.encoder_layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
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
        
        # Create BertModelOutput
        bert_output = BertModelOutput(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
            attention_mask=attention_mask,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )
        
        # Compute additional pooling if requested
        if compute_pooling and self.additional_pooling:
            pooling_outputs = self._compute_additional_pooling(last_hidden_state, attention_mask)
            bert_output.cls_output = pooling_outputs["cls_output"]
            bert_output.mean_pooled = pooling_outputs["mean_pooled"]
            bert_output.max_pooled = pooling_outputs["max_pooled"]
        
        return bert_output
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, Path],
        cache_dir: Optional[str] = None,
        **kwargs
    ) -> "BertCore":
        """Load BERT model from pretrained weights.
        
        Args:
            model_path: Path to model directory or HuggingFace model ID
            cache_dir: Optional cache directory for downloaded models
            **kwargs: Additional configuration parameters
            
        Returns:
            Loaded BertCore model
        """
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
            config = _load_hf_config(config_path)
        else:
            logger.warning(f"No config.json found at {config_path}, using default config")
            config = BertConfig()
        
        # Override config with any provided kwargs
        if kwargs:
            config_dict = config.to_dict()
            config_dict.update(kwargs)
            config = BertConfig.from_dict(config_dict)
        
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
        """Save model to directory.
        
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
            "model_type": "BertCore",
            "model_class": self.__class__.__name__,
            "version": "1.0.0",
        }
        
        with open(save_path / "model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"BERT model saved to {save_path}")


# Factory functions
def create_bert_core(
    model_name: Optional[str] = None,
    config: Optional[Union[BertConfig, Dict]] = None,
    cache_dir: Optional[str] = None,
    **kwargs
) -> BertCore:
    """Create a BertCore model.
    
    Args:
        model_name: Optional pretrained model name (local path or HuggingFace model ID)
        config: Optional configuration
        cache_dir: Optional cache directory for downloaded models
        **kwargs: Additional configuration parameters
        
    Returns:
        BertCore model instance
    """
    if model_name:
        return BertCore.from_pretrained(model_name, cache_dir=cache_dir, **kwargs)
    elif config:
        return BertCore(config)
    else:
        # Use default config with kwargs
        config = BertConfig(**kwargs)
        return BertCore(config)