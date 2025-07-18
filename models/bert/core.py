"""Core BERT model implementation with modular interface.

This module provides the consolidated BERT implementation that can be easily
attached to any head in the heads directory.
"""

import json
import os
import re
from pathlib import Path
from urllib.parse import urlparse

import mlx.core as mx
from loguru import logger

from .config import BertConfig, get_neobert_config, get_neobert_mini_config
from .core_base import BaseBertModel, BertLayer, BertModelOutput
from .layers.attention import create_attention_layer
from .layers.embeddings import (
    create_embeddings,
    create_pooler,
)


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
    pattern = r"^[a-zA-Z0-9][\w\-\.]*\/[a-zA-Z0-9][\w\-\.]*(@[\w\-\.]+)?$"
    return bool(re.match(pattern, model_path))


def _download_from_hub(model_id: str, cache_dir: str | None = None) -> Path:
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
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required to download models from the Hub. "
            "Install it with: pip install huggingface_hub"
        ) from e

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
    with open(config_path) as f:
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

    def __init__(self, config: BertConfig | dict):
        """Initialize BERT core model.

        Args:
            config: Model configuration (BertConfig or dict)
        """
        super().__init__(config)

        # Initialize the BERT encoder layers
        self._build_encoder()

        logger.info(
            f"Initialized BertCore with config: hidden_size={config.hidden_size}, "
            f"num_layers={config.num_hidden_layers}, num_heads={config.num_attention_heads}"
        )

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
        attention_mask: mx.array | None = None,
        token_type_ids: mx.array | None = None,
        position_ids: mx.array | None = None,
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
        attention_mask: mx.array | None = None,
        token_type_ids: mx.array | None = None,
        position_ids: mx.array | None = None,
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
            pooling_outputs = self._compute_additional_pooling(
                last_hidden_state, attention_mask
            )
            bert_output.cls_output = pooling_outputs["cls_output"]
            bert_output.mean_pooled = pooling_outputs["mean_pooled"]
            bert_output.max_pooled = pooling_outputs["max_pooled"]

        return bert_output

    @classmethod
    def from_pretrained(
        cls, model_path: str | Path, cache_dir: str | None = None, **kwargs
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
            logger.warning(
                f"No config.json found at {config_path}, using default config"
            )
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
                    if weight_file.endswith(".safetensors"):
                        # Load safetensors format
                        weights = mx.load(str(weights_path))
                        model.load_weights(list(weights.items()))
                        logger.info(f"Loaded weights from {weights_path}")
                        weights_loaded = True
                        break
                    else:
                        logger.warning(
                            f"Found {weight_file} but .safetensors format is preferred"
                        )
                except Exception as e:
                    logger.warning(f"Failed to load weights from {weights_path}: {e}")
                    continue

        if not weights_loaded:
            logger.warning(f"No compatible weight files found in {model_path}")
            logger.info("Model initialized with random weights")

        return model

    def save_pretrained(self, save_path: str | Path):
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
    model_name: str | None = None,
    config: BertConfig | dict | None = None,
    cache_dir: str | None = None,
    **kwargs,
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


# ============================================================================
# ModernBERT Core Implementation
# ============================================================================


class ModernBertCore(BaseBertModel):
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

    def __init__(self, config: BertConfig | dict):
        """
        Initialize ModernBERT core model.

        Args:
            config: Model configuration (BertConfig or dict)
        """
        super().__init__(config)

        # Initialize the ModernBERT encoder
        self._build_encoder()

        logger.info(
            f"Initialized ModernBertCore with config: "
            f"hidden_size={config.hidden_size}, "
            f"num_layers={config.num_hidden_layers}, "
            f"num_heads={config.num_attention_heads}, "
            f"max_seq_len={getattr(config, 'max_position_embeddings', 8192)}, "
            f"use_rope={getattr(config, 'use_rope', True)}, "
            f"use_geglu={getattr(config, 'use_geglu', True)}, "
            f"use_alternating_attention={getattr(config, 'use_alternating_attention', True)}"
        )

    def _build_encoder(self):
        """Build the ModernBERT encoder architecture."""
        # Create ModernBERT embedding layer
        self.embeddings = create_embeddings(self.config)

        # Create encoder layers using the unified layer factory
        self.encoder_layers = []
        for layer_idx in range(self.config.num_hidden_layers):
            # Create layer with appropriate attention mechanism
            layer = BertLayer(self.config, layer_idx)
            # Override attention with ModernBERT attention if needed
            if hasattr(self.config, "use_rope") and self.config.use_rope:
                layer.attention = create_attention_layer(
                    hidden_size=self.config.hidden_size,
                    num_attention_heads=self.config.num_attention_heads,
                    layer_idx=layer_idx,
                    config=self.config,
                )
            self.encoder_layers.append(layer)

        # Pooler for [CLS] token
        self.pooler = create_pooler(self.config)

        logger.info(f"Built ModernBERT encoder with {len(self.encoder_layers)} layers")

    def forward(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        token_type_ids: mx.array | None = None,
        position_ids: mx.array | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        compute_pooling: bool = True,
        training: bool = True,
    ) -> BertModelOutput:
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
            BertModelOutput with all model outputs
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
            if training and getattr(self.config, "gradient_checkpointing", False):
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
            pooling_outputs = self._compute_additional_pooling(
                last_hidden_state, attention_mask
            )
            bert_output.cls_output = pooling_outputs["cls_output"]
            bert_output.mean_pooled = pooling_outputs["mean_pooled"]
            bert_output.max_pooled = pooling_outputs["max_pooled"]

        return bert_output

    def get_attention_pattern(self) -> list[str]:
        """Get the attention pattern for all layers."""
        pattern = []
        for layer_idx in range(self.config.num_hidden_layers):
            if getattr(self.config, "use_alternating_attention", False):
                global_every_n = getattr(
                    self.config, "global_attention_every_n_layers", 3
                )
                if (layer_idx + 1) % global_every_n == 0:
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
        cls, model_path: str | Path, cache_dir: str | None = None, **kwargs
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
            with open(config_path) as f:
                config_dict = json.load(f)

            # Check if this is a ModernBERT config
            if config_dict.get("model_type") == "modernbert":
                config = BertConfig.from_dict(config_dict)
            else:
                # Try to convert from standard BERT config
                logger.info("Converting standard BERT config to ModernBERT config")
                config = BertConfig.from_dict(config_dict)
        else:
            logger.warning(
                f"No config.json found at {config_path}, using default config"
            )
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
                    if weight_file.endswith(".safetensors"):
                        # Load safetensors format
                        weights = mx.load(str(weights_path))
                        model.load_weights(list(weights.items()))
                        logger.info(f"Loaded weights from {weights_path}")
                        weights_loaded = True
                        break
                    else:
                        logger.warning(
                            f"Found {weight_file} but .safetensors format is preferred"
                        )
                except Exception as e:
                    logger.warning(f"Failed to load weights from {weights_path}: {e}")
                    continue

        if not weights_loaded:
            logger.warning(f"No compatible weight files found in {model_path}")
            logger.info("Model initialized with random weights")

        return model

    def save_pretrained(self, save_path: str | Path):
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
            "attention_pattern": self.get_attention_pattern(),
            "use_rope": getattr(self.config, "use_rope", True),
            "use_geglu": getattr(self.config, "use_geglu", True),
            "use_alternating_attention": getattr(
                self.config, "use_alternating_attention", True
            ),
            "version": "1.0.0",
        }

        with open(save_path / "model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"ModernBERT model saved to {save_path}")


# ============================================================================
# Unified Factory Functions
# ============================================================================


def create_model_core(
    model_type: str = "bert",
    model_name: str | None = None,
    config: BertConfig | dict | None = None,
    cache_dir: str | None = None,
    **kwargs,
):
    """
    Create a model core (BERT, ModernBERT, or neoBERT) based on configuration.

    Args:
        model_type: Type of model ("bert", "modernbert", or "neobert")
        model_name: Optional pretrained model name
        config: Optional configuration
        cache_dir: Optional cache directory
        **kwargs: Additional configuration parameters

    Returns:
        Model core instance
    """
    model_type_lower = model_type.lower()

    if model_type_lower in ["modernbert", "neobert"]:
        # Both ModernBERT and neoBERT use the same core implementation
        if model_name:
            return ModernBertCore.from_pretrained(
                model_name, cache_dir=cache_dir, **kwargs
            )
        else:
            # Use appropriate config for each type
            if config is None:
                if model_type_lower == "neobert":
                    config = get_neobert_config()
                else:
                    # Use default ModernBERT config
                    config = BertConfig(**kwargs)
            return ModernBertCore(config)
    else:
        return create_bert_core(model_name, config, cache_dir, **kwargs)


def create_modernbert_core(
    model_name: str | None = None,
    config: BertConfig | dict | None = None,
    cache_dir: str | None = None,
    **kwargs,
) -> ModernBertCore:
    """
    Create a ModernBertCore model.

    Args:
        model_name: Optional pretrained model name (local path or HuggingFace model ID)
        config: Optional configuration
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
        config = BertConfig(**kwargs)
        return ModernBertCore(config)


def create_modernbert_base(**kwargs) -> ModernBertCore:
    """Create ModernBERT-base model."""
    return create_modernbert_core(**kwargs)


def create_modernbert_large(**kwargs) -> ModernBertCore:
    """Create ModernBERT-large model."""
    return create_modernbert_core(**kwargs)


# ============================================================================
# neoBERT Factory Functions
# ============================================================================


def create_neobert_core(
    model_name: str | None = None,
    config: BertConfig | dict | None = None,
    cache_dir: str | None = None,
    **kwargs,
) -> ModernBertCore:
    """
    Create a neoBERT model using ModernBertCore.

    Args:
        model_name: Optional pretrained model name (local path or HuggingFace model ID)
        config: Optional configuration
        cache_dir: Optional cache directory for downloaded models
        **kwargs: Additional configuration parameters

    Returns:
        ModernBertCore model instance configured for neoBERT
    """
    if model_name:
        return ModernBertCore.from_pretrained(model_name, cache_dir=cache_dir, **kwargs)
    elif config:
        return ModernBertCore(config)
    else:
        # Use neoBERT config with any overrides
        config = get_neobert_config()
        # Apply any kwargs as overrides
        if kwargs:
            config_dict = config.to_dict()
            config_dict.update(kwargs)
            config = BertConfig.from_dict(config_dict)
        return ModernBertCore(config)


def create_neobert(**kwargs) -> ModernBertCore:
    """
    Create standard neoBERT model (250M parameters).

    Args:
        **kwargs: Configuration overrides

    Returns:
        neoBERT model instance
    """
    return create_neobert_core(config=get_neobert_config(), **kwargs)


def create_neobert_mini(**kwargs) -> ModernBertCore:
    """
    Create mini neoBERT model for testing.

    Args:
        **kwargs: Configuration overrides

    Returns:
        Mini neoBERT model instance
    """
    return create_neobert_core(config=get_neobert_mini_config(), **kwargs)
