"""BERT model with head for classification/regression tasks.

This module implements the BertWithHead wrapper that combines a BERT encoder
with task-specific heads, following clean separation of concerns.
"""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from loguru import logger

# Import all head modules
from ..heads.base import BaseHead
from ..heads.config import HeadConfig
from .config import BertConfig
from .core import BertCore


class BertWithHead(nn.Module):
    """BERT model with attached head for downstream tasks."""

    def __init__(
        self,
        bert: BertCore,
        head: BaseHead,
        freeze_bert: bool = False,
        freeze_bert_layers: int | None = None,
    ):
        """Initialize BERT with head.

        Args:
            bert: BERT core model
            head: Task-specific head
            freeze_bert: Whether to freeze all BERT parameters
            freeze_bert_layers: Number of BERT layers to freeze (from bottom)
        """
        super().__init__()
        self.bert = bert
        self.head = head
        self.num_labels = head.config.output_size

        # Freeze BERT if requested
        if freeze_bert:
            self.freeze_bert()
        elif freeze_bert_layers is not None:
            self.freeze_bert(freeze_bert_layers)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        token_type_ids: mx.array | None = None,
        position_ids: mx.array | None = None,
        labels: mx.array | None = None,
        **kwargs,
    ) -> dict[str, mx.array]:
        """Forward pass through BERT and head.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs for sentence pairs
            position_ids: Position IDs for custom positioning
            labels: Ground truth labels for computing loss
            **kwargs: Additional arguments passed to head

        Returns:
            Dictionary with at least 'logits' and optionally 'loss'
        """
        # BERT forward pass
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )

        # Head forward pass
        head_outputs = self.head(
            hidden_states=bert_outputs.last_hidden_state,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

        return head_outputs

    def save_pretrained(self, save_directory: str | Path) -> None:
        """Save model to directory with separate BERT and head.

        Args:
            save_directory: Directory to save model
        """
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save BERT
        bert_path = save_path / "bert"
        self.bert.save_pretrained(str(bert_path))

        # Save head
        head_path = save_path / "head"
        head_path.mkdir(exist_ok=True)

        # Save head config
        with open(head_path / "config.json", "w") as f:
            json.dump(self.head.config.__dict__, f, indent=2)

        # Save head weights
        import mlx.utils

        head_state = dict(mlx.utils.tree_flatten(self.head.parameters()))
        mx.save_safetensors(str(head_path / "model.safetensors"), head_state)

        # Save model metadata
        head_name = self.head.config.head_type

        metadata = {
            "model_type": "BertWithHead",
            "bert_type": self.bert.__class__.__name__,
            "head_type": head_name,
            "num_labels": self.num_labels,
        }

        with open(save_path / "model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved BertWithHead model to {save_path}")

    @classmethod
    def from_pretrained(cls, model_path: str | Path) -> "BertWithHead":
        """Load model from pretrained directory.

        Args:
            model_path: Path to saved model directory

        Returns:
            Loaded BertWithHead model
        """
        model_path = Path(model_path)

        # Load metadata
        with open(model_path / "model_metadata.json") as f:
            metadata = json.load(f)

        # Load BERT
        bert_path = model_path / "bert"
        if metadata["bert_type"] == "BertCore":
            bert = BertCore.from_pretrained(str(bert_path))
        else:
            # Handle other BERT types if needed
            from .core import ModernBertCore

            bert = ModernBertCore.from_pretrained(str(bert_path))

        # Load head config
        with open(model_path / "head" / "config.json") as f:
            head_config_dict = json.load(f)

        # Filter out parameters that HeadConfig doesn't accept
        valid_params = {
            "input_size",
            "output_size",
            "head_type",
            "hidden_sizes",
            "dropout_prob",
            "activation",
            "use_bias",
            "pooling_type",
            "use_layer_norm",
            "layer_norm_eps",
        }
        filtered_config = {
            k: v for k, v in head_config_dict.items() if k in valid_params
        }

        # No need to convert enums since we're using strings
        # head_config = HeadConfig(**filtered_config)  # Not needed, we use create_head directly

        # Create head based on type
        from ..heads import create_head

        # Remove head_type from config since we pass it explicitly
        create_head_config = {
            k: v for k, v in filtered_config.items() if k != "head_type"
        }

        head = create_head(
            head_type=metadata["head_type"],
            **create_head_config,
        )

        # Load head weights
        head_weights = mx.load(str(model_path / "head" / "model.safetensors"))
        head.load_weights(list(head_weights.items()))

        # Create model
        model = cls(bert=bert, head=head)

        logger.info(f"Loaded BertWithHead model from {model_path}")
        return model

    def get_num_labels(self) -> int:
        """Get number of labels."""
        return self.num_labels

    def get_bert(self) -> BertCore:
        """Get the BERT model."""
        return self.bert

    def get_head(self) -> BaseHead:
        """Get the head model."""
        return self.head

    def freeze_bert(self, num_layers: int | None = None):
        """Freeze BERT parameters.

        Args:
            num_layers: Number of layers to freeze (from bottom).
                       If None, freezes all layers.
        """
        if num_layers is None:
            # Freeze all BERT parameters
            # In MLX, we freeze modules, not individual parameters
            self.bert.freeze()
        else:
            # Freeze embeddings
            self.bert.embeddings.freeze()

            # Freeze specified number of layers
            for i in range(num_layers):
                if i < len(self.bert.encoder.layers):
                    self.bert.encoder.layers[i].freeze()

    def unfreeze_bert(self):
        """Unfreeze all BERT parameters."""
        # In MLX, we unfreeze modules
        self.bert.unfreeze()


# Factory functions
def create_bert_with_head(
    bert_config: BertConfig | dict | None = None,
    head_config: HeadConfig | dict | None = None,
    head_type: str | None = None,
    bert_name: str | None = None,
    freeze_bert: bool = False,
    freeze_bert_layers: int | None = None,
    **kwargs,
) -> BertWithHead:
    """Create BERT with attached head.

    Args:
        bert_config: BERT configuration
        head_config: Head configuration
        head_type: Type of head if head_config not provided
        bert_name: Name of pretrained BERT to load
        freeze_bert: Whether to freeze BERT parameters
        freeze_bert_layers: Number of BERT layers to freeze
        **kwargs: Additional arguments

    Returns:
        BertWithHead model
    """
    # Create BERT
    if bert_name:
        bert = BertCore.from_pretrained(bert_name)
    else:
        if bert_config is None:
            bert_config = BertConfig()
        elif isinstance(bert_config, dict):
            bert_config = BertConfig(**bert_config)
        bert = BertCore(bert_config)

    # Get num_labels for compatibility
    num_labels = kwargs.get("num_labels", 2)

    # Create head config if not provided
    if head_config is None:
        if head_type is None:
            raise ValueError("Either head_config or head_type must be provided")

        # Keep head_type as string
        # No conversion needed

        # Create head config based on type
        if head_type == "binary_classification":
            from ..heads.config import get_binary_classification_config

            head_config = get_binary_classification_config(bert.get_hidden_size())
        elif head_type == "multiclass_classification":
            from ..heads.config import get_multiclass_classification_config

            head_config = get_multiclass_classification_config(
                bert.get_hidden_size(), num_labels
            )
        elif head_type == "multilabel_classification":
            from ..heads.config import get_multilabel_classification_config

            head_config = get_multilabel_classification_config(
                bert.get_hidden_size(), num_labels
            )
        elif head_type == "regression":
            from ..heads.config import get_regression_preset_config

            head_config = get_regression_preset_config(bert.get_hidden_size())
        else:
            # Fallback to basic config
            head_config = HeadConfig(
                input_size=bert.get_hidden_size(),
                output_size=kwargs.get("num_labels", 2),
                head_type=head_type,
            )

    # Convert dict to HeadConfig if needed
    if isinstance(head_config, dict):
        head_config = HeadConfig(**head_config)

    # Create head based on type
    from ..heads import create_head

    # Extract config dict and remove duplicates
    head_config_dict = head_config.__dict__.copy()
    head_config_dict.pop("head_type", None)  # Already passed explicitly

    # Filter out any conflicting kwargs that might already be in head_config
    conflicting_params = {
        "num_labels",
        "num_classes",
        "input_size",
        "output_size",
        "head_type",
    }
    filtered_kwargs = {k: v for k, v in kwargs.items() if k not in conflicting_params}

    # Merge filtered kwargs with head config
    final_config = {**head_config_dict, **filtered_kwargs}

    head = create_head(
        head_type=head_config.head_type,
        **final_config,
    )

    # Create model
    model = BertWithHead(
        bert=bert,
        head=head,
        freeze_bert=freeze_bert,
        freeze_bert_layers=freeze_bert_layers,
    )

    return model


def create_bert_for_competition(
    competition_type: str,
    bert_config: BertConfig | dict | None = None,
    bert_name: str | None = None,
    num_labels: int = 2,
    **kwargs,
) -> BertWithHead:
    """Create a BERT model optimized for a specific competition type.

    Args:
        competition_type: Type of competition
        bert_config: BERT configuration
        bert_name: Name of pretrained BERT
        num_labels: Number of labels/classes
        **kwargs: Additional arguments

    Returns:
        BertWithHead model configured for the competition
    """
    # Create BERT
    if bert_name:
        bert = BertCore.from_pretrained(bert_name)
    else:
        if bert_config is None:
            bert_config = BertConfig()
        elif isinstance(bert_config, dict):
            bert_config = BertConfig(**bert_config)
        bert = BertCore(bert_config)

    # Map competition type to head type
    competition_to_head_map = {
        "binary_classification": "binary_classification",
        "multiclass_classification": "multiclass_classification",
        "multilabel_classification": "multilabel_classification",
        "regression": "regression",
        "ordinal_regression": "ordinal_regression",
        "time_series": "time_series",
        "ranking": "ranking",
    }

    head_type = competition_to_head_map.get(competition_type, "binary_classification")

    # Create head
    from ..heads import create_head

    head = create_head(
        head_type=head_type,
        input_size=bert.get_hidden_size(),
        output_size=num_labels,
        **kwargs,
    )

    # Create model
    model = BertWithHead(bert=bert, head=head)

    logger.info(
        f"Created BERT model for {competition_type} competition "
        f"with {num_labels} labels"
    )

    return model
