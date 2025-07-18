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
import re
import os
from urllib.parse import urlparse

from .config import BertConfig


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


class BertLayer(nn.Module):
    """BERT transformer layer.
    
    This layer wraps MLX's TransformerEncoderLayer but handles BERT-specific
    attention patterns and layer normalization.
    """
    
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        
        # For now, use MLX's TransformerEncoderLayer as base
        # TODO: Replace with full BERT attention implementation
        self.transformer_layer = nn.TransformerEncoderLayer(
            dims=config.hidden_size,
            num_heads=config.num_attention_heads,
            mlp_dims=config.intermediate_size,
            dropout=config.hidden_dropout_prob,
            norm_first=False  # BERT uses post-norm (layer norm after attention/ffn)
        )
    
    def forward(self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None) -> mx.array:
        """Forward pass through BERT layer.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Hidden states [batch_size, seq_len, hidden_size]
        """
        # Convert attention mask to MLX format if needed
        if attention_mask is not None:
            # Convert from [batch, seq_len] to [batch, 1, 1, seq_len] for broadcasting
            mask = attention_mask[:, None, None, :]
        else:
            mask = None
        
        # Apply transformer layer
        output = self.transformer_layer(hidden_states, mask=mask)
        
        return output
    
    def __call__(self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None) -> mx.array:
        """Make the layer callable."""
        return self.forward(hidden_states, attention_mask)


class BertPooler(nn.Module):
    """BERT pooler layer.
    
    This layer pools the [CLS] token representation to create a fixed-size
    representation for classification tasks.
    """
    
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.pooler_hidden_size or config.hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(config.pooler_dropout)
    
    def forward(self, hidden_states: mx.array) -> mx.array:
        """Pool the [CLS] token representation.
        
        Args:
            hidden_states: Hidden states from encoder [batch_size, seq_len, hidden_size]
            
        Returns:
            Pooled representation [batch_size, pooler_hidden_size]
        """
        # Extract [CLS] token representation (first token)
        first_token_tensor = hidden_states[:, 0, :]  # [batch_size, hidden_size]
        
        # Apply linear transformation
        pooled_output = self.dense(first_token_tensor)
        
        # Apply activation
        pooled_output = self.activation(pooled_output)
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        return pooled_output
    
    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Make the pooler callable."""
        return self.forward(hidden_states)


class BertEmbeddings(nn.Module):
    """BERT embeddings layer.
    
    This layer combines token embeddings, position embeddings, and token type embeddings,
    applies layer normalization and dropout as per the original BERT architecture.
    """
    
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Position embeddings (learned, not sinusoidal like in Transformer)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Token type embeddings (for NSP task - sentence A vs sentence B)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        
        # Layer normalization and dropout
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Register position_ids as a buffer (similar to PyTorch implementation)
        self.register_buffer_persistent = False
        self.position_ids = mx.arange(config.max_position_embeddings)[None, :]  # [1, max_position_embeddings]
    
    def forward(
        self,
        input_ids: mx.array,
        token_type_ids: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        """Forward pass through embeddings.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len]
            
        Returns:
            Embeddings [batch_size, seq_len, hidden_size]
        """
        input_shape = input_ids.shape
        seq_length = input_shape[1]
        
        # Get position IDs
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        
        # Get token type IDs
        if token_type_ids is None:
            token_type_ids = mx.zeros(input_shape, dtype=mx.int32)
        
        # Get embeddings
        inputs_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)
        
        # Combine embeddings
        embeddings = inputs_embeds + position_embeds + token_type_embeds
        
        # Apply layer normalization and dropout
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings
    
    def __call__(self, input_ids: mx.array, token_type_ids: Optional[mx.array] = None, position_ids: Optional[mx.array] = None) -> mx.array:
        """Make the embeddings layer callable."""
        return self.forward(input_ids, token_type_ids, position_ids)


class BertCore(nn.Module):
    """Core BERT model with standardized interface.
    
    This class provides the core BERT encoder with a clean interface
    for attaching any head from the heads directory.
    """
    
    def __init__(self, config: Union[BertConfig, Dict]):
        """Initialize BERT core model.
        
        Args:
            config: Model configuration (BertConfig or dict)
        """
        super().__init__()
        
        # Convert dict to config if needed
        if isinstance(config, dict):
            config = BertConfig(**config)
        
        self.config = config
        
        # Initialize the BERT encoder layers
        self._build_encoder()
        
        # Optional: Additional pooling layers
        self.additional_pooling = config.__dict__.get("compute_additional_pooling", True)
        
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
        # Create BERT embedding layer
        self.embeddings = BertEmbeddings(self.config)
        
        # Create encoder layers using BERT-specific layers
        self.encoder_layers = []
        for _ in range(self.config.num_hidden_layers):
            layer = BertLayer(self.config)
            self.encoder_layers.append(layer)
        
        # Pooler for [CLS] token
        self.pooler = BertPooler(self.config)
    
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
        # Get embeddings
        embeddings = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids
        )
        
        # Pass through encoder layers
        hidden_states = embeddings
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
        last_hidden_state = hidden_states
        
        # Get pooler output (from [CLS] token)
        pooler_output = self.pooler(last_hidden_state)
        
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
    
    def get_config(self) -> BertConfig:
        """Get the model configuration."""
        return self.config
    
    @classmethod
    def from_pretrained(
        cls, 
        model_path: Union[str, Path], 
        cache_dir: Optional[str] = None,
        **kwargs
    ) -> "BertCore":
        """Load model from pretrained weights.
        
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
        # Try multiple weight file names
        weight_files = [
            "model.safetensors",  # Standard MLX format
            "pytorch_model.safetensors",  # HuggingFace format
            "model.bin",  # Fallback (though we prefer safetensors)
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
        with open(save_path / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        # Save weights
        from mlx.utils import tree_flatten
        weights = dict(tree_flatten(self.parameters()))
        mx.save_safetensors(str(save_path / "model.safetensors"), weights)
        
        # Also save any additional BertCore-specific config
        bert_core_config = {
            "compute_additional_pooling": self.additional_pooling,
            "bert_core_version": "1.0.0",
        }
        
        with open(save_path / "bert_core_config.json", "w") as f:
            json.dump(bert_core_config, f, indent=2)
        
        logger.info(f"Model saved to {save_path}")
    
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
                    
            logger.info(f"Frozen {layers_to_freeze} encoder layers")
    
    def unfreeze_all(self):
        """Unfreeze all model parameters."""
        for param in self.parameters():
            param.unfreeze()


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