"""
Base classes and components for BERT models.

This module contains the core building blocks that are shared across
different BERT architectures and provide the foundation for the model.
"""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .config import BertConfig
from .layers.attention import BertAttention
from .layers.embeddings import BertEmbeddings, BertPooler
from .layers.feedforward import BertIntermediate, BertOutput, create_feedforward_layer


@dataclass
class BertModelOutput:
    """Standard output format for BERT models.

    This dataclass provides a consistent interface between BERT models
    and downstream heads, making it easy to swap models or heads.
    """

    # Primary outputs
    last_hidden_state: mx.array  # [batch_size, seq_len, hidden_size]
    pooler_output: mx.array  # [batch_size, hidden_size]

    # Optional outputs
    hidden_states: list[mx.array] | None = None  # List of all hidden states
    attentions: list[mx.array] | None = None  # List of attention weights

    # Additional pooled representations
    cls_output: mx.array | None = None  # [batch_size, hidden_size] - CLS token
    mean_pooled: mx.array | None = None  # [batch_size, hidden_size] - Mean pooling
    max_pooled: mx.array | None = None  # [batch_size, hidden_size] - Max pooling

    # Metadata
    attention_mask: mx.array | None = None  # [batch_size, seq_len] - For downstream use

    def get_pooled_output(self, pooling_type: str = "cls") -> mx.array:
        """Get pooled output by type.

        Args:
            pooling_type: Type of pooling - "cls", "mean", "max", or "pooler"

        Returns:
            Pooled representation
        """
        if pooling_type == "cls":
            return (
                self.cls_output
                if self.cls_output is not None
                else self.last_hidden_state[:, 0, :]
            )
        elif pooling_type == "mean":
            return (
                self.mean_pooled
                if self.mean_pooled is not None
                else self._compute_mean_pooling()
            )
        elif pooling_type == "max":
            return (
                self.max_pooled
                if self.max_pooled is not None
                else self._compute_max_pooling()
            )
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

    Supports both post-normalization (classic BERT) and pre-normalization
    (ModernBERT/neoBERT) patterns.
    """

    def __init__(self, config: BertConfig, layer_idx: int | None = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Check if using pre-normalization
        self.use_pre_norm = getattr(config, "use_pre_norm", False)

        # Check normalization type
        self.norm_type = getattr(config, "norm_type", "layer_norm")

        # Attention layer
        self.attention = BertAttention(config)

        # Feed-forward network
        # Check if we should use a unified feedforward layer
        use_unified_ffn = any(
            [
                getattr(config, "use_geglu", False),
                getattr(config, "use_swiglu", False),
                self.use_pre_norm,
            ]
        )

        if use_unified_ffn:
            # Use factory to create appropriate feedforward layer
            self.feedforward = create_feedforward_layer(config)
            self.intermediate = None
            self.output = None
        else:
            # Classic BERT with separate intermediate and output
            self.intermediate = BertIntermediate(config)
            self.output = BertOutput(config)
            self.feedforward = None

        # For pre-normalization, we need additional norm layers
        if self.use_pre_norm:
            # Import norm layers
            if self.norm_type == "rms_norm":
                from .layers.activations import RMSNorm

                self.attention_norm = RMSNorm(
                    config.hidden_size, eps=config.layer_norm_eps, use_bias=False
                )
                self.ffn_norm = RMSNorm(
                    config.hidden_size, eps=config.layer_norm_eps, use_bias=False
                )
            else:
                self.attention_norm = nn.LayerNorm(
                    config.hidden_size, eps=config.layer_norm_eps
                )
                self.ffn_norm = nn.LayerNorm(
                    config.hidden_size, eps=config.layer_norm_eps
                )

            # Dropout for residual connections
            self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        position_ids: mx.array | None = None,
        output_attentions: bool = False,
    ) -> tuple[mx.array, mx.array | None]:
        """Forward pass through BERT layer.

        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            position_ids: Position IDs (for RoPE) [batch_size, seq_len]
            output_attentions: Whether to output attention weights

        Returns:
            Tuple of (layer_output, attention_probs)
        """
        if self.use_pre_norm:
            # Pre-normalization pattern (neoBERT/ModernBERT)
            # 1. Self-attention with pre-norm
            residual = hidden_states
            hidden_states = self.attention_norm(hidden_states)

            # Check if attention layer supports position_ids (for RoPE)
            if (
                hasattr(self.attention, "forward")
                and "position_ids" in self.attention.forward.__code__.co_varnames
            ):
                attention_outputs = self.attention(
                    hidden_states,
                    attention_mask,
                    position_ids,
                    output_attentions,
                )
            else:
                attention_outputs = self.attention(
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )

            attention_output = attention_outputs[0]
            attention_output = self.dropout(attention_output)
            hidden_states = residual + attention_output

            # 2. Feed-forward with pre-norm
            if self.feedforward is not None:
                # Using unified feedforward (already handles residual connection)
                hidden_states = self.feedforward(hidden_states)
            else:
                # Using classic BERT FFN
                residual = hidden_states
                hidden_states = self.ffn_norm(hidden_states)
                intermediate_output = self.intermediate(hidden_states)
                # For pre-norm, we need a different output handling
                ffn_output = self.output.dense(intermediate_output)
                ffn_output = self.output.dropout(ffn_output)
                hidden_states = residual + ffn_output

            outputs = (hidden_states,) + attention_outputs[1:]
            return outputs
        else:
            # Post-normalization pattern (classic BERT)
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

            outputs = (layer_output,) + attention_outputs[
                1:
            ]  # Add attentions if we output them
            return outputs

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        position_ids: mx.array | None = None,
        output_attentions: bool = False,
    ) -> tuple[mx.array, mx.array | None]:
        """Make the layer callable."""
        return self.forward(
            hidden_states, attention_mask, position_ids, output_attentions
        )


class BaseBertModel(nn.Module):
    """Base class for BERT models.

    This class provides common functionality shared across different BERT variants.
    """

    def __init__(self, config: BertConfig | dict):
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
        self.additional_pooling = config.__dict__.get(
            "compute_additional_pooling", True
        )

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

    def freeze_encoder(self, num_layers_to_freeze: int | None = None):
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

    def _compute_additional_pooling(
        self, last_hidden_state: mx.array, attention_mask: mx.array | None
    ) -> dict[str, mx.array]:
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
