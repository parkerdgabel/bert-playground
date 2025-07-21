"""
Layer management utilities for BERT models.
"""

from typing import Any

from loguru import logger

from training.core.protocols import Model


class BERTLayerManager:
    """Manages BERT layers for training strategies."""

    def __init__(self, model: Model):
        """
        Initialize layer manager.

        Args:
            model: BERT model to manage
        """
        self.model = model
        self.bert_layers = self._identify_bert_layers()
        self.head_layers = self._identify_head_layers()
        logger.info(
            f"Identified {len(self.bert_layers)} BERT layers and {len(self.head_layers)} head layers"
        )

    def _identify_bert_layers(self) -> list[tuple[str, Any]]:
        """Identify BERT encoder layers."""
        bert_layers = []

        # Handle both classic BERT and ModernBERT
        # Look for the actual transformer blocks/layers (not sub-components)
        for name, module in self.model.named_modules():
            # Classic BERT: bert.encoder.layers.0, bert.encoder.layers.1, etc.
            # ModernBERT: modernbert.encoder.layers.0, etc.
            # Pattern: encoder.layers.{number} or encoder.blocks.{number}
            if "encoder.layers." in name or "encoder.blocks." in name:
                # Only add if this is the layer module itself, not sub-components
                parts = name.split(".")
                if parts[-1].isdigit():  # Ends with a digit (layer number)
                    bert_layers.append((name, module))
            # Alternative pattern for some implementations
            elif "transformer.layer." in name or "transformer.block." in name:
                parts = name.split(".")
                if parts[-1].isdigit():
                    bert_layers.append((name, module))

        # Sort by layer index
        def get_layer_index(name_module_tuple):
            name = name_module_tuple[0]
            parts = name.split(".")
            for part in parts:
                if part.isdigit():
                    return int(part)
            return 0

        bert_layers.sort(key=get_layer_index)
        return bert_layers

    def _identify_head_layers(self) -> list[tuple[str, Any]]:
        """Identify task-specific head layers."""
        head_layers = []

        for name, module in self.model.named_modules():
            if any(key in name for key in ["head", "classifier", "pooler"]):
                head_layers.append((name, module))

        return head_layers

    def freeze_bert(self, exclude_layers: list[int] | None = None):
        """
        Freeze BERT layers.

        Args:
            exclude_layers: Layer indices to exclude from freezing (keep trainable)
        """
        exclude_layers = exclude_layers or []

        # Freeze all BERT parameters
        for idx, (name, module) in enumerate(self.bert_layers):
            if idx not in exclude_layers:
                for param in module.parameters():
                    param.freeze()
                logger.debug(f"Froze layer {name}")
            else:
                logger.debug(f"Kept layer {name} trainable")

    def unfreeze_bert(self, layers_to_unfreeze: list[int] | None = None):
        """
        Unfreeze specific BERT layers.

        Args:
            layers_to_unfreeze: Layer indices to unfreeze (negative indices from end)
        """
        if layers_to_unfreeze is None:
            # Unfreeze all
            for name, module in self.bert_layers:
                for param in module.parameters():
                    param.unfreeze()
            logger.info("Unfroze all BERT layers")
        else:
            # Convert negative indices
            total_layers = len(self.bert_layers)
            indices = [
                idx if idx >= 0 else total_layers + idx for idx in layers_to_unfreeze
            ]

            for idx, (name, module) in enumerate(self.bert_layers):
                if idx in indices:
                    for param in module.parameters():
                        param.unfreeze()
                    logger.debug(f"Unfroze layer {name}")

    def get_parameter_groups(
        self,
        base_lr: float,
        layer_lr_decay: float = 0.95,
        head_lr_multiplier: float = 10.0,
    ) -> list[dict[str, Any]]:
        """
        Get parameter groups with layer-wise learning rates.

        Args:
            base_lr: Base learning rate for top BERT layer
            layer_lr_decay: Decay factor per layer depth
            head_lr_multiplier: Multiplier for head learning rate

        Returns:
            Parameter groups for optimizer
        """
        parameter_groups = []

        # BERT layers - different LR per layer
        num_layers = len(self.bert_layers)
        for idx, (name, module) in enumerate(self.bert_layers):
            # Deeper layers get smaller learning rates
            layer_lr = base_lr * (layer_lr_decay ** (num_layers - idx - 1))

            params = list(module.parameters())
            if params:  # Only add if there are parameters
                parameter_groups.append(
                    {"params": params, "lr": layer_lr, "name": f"bert_layer_{idx}"}
                )

        # Head layers - higher learning rate
        head_params = []
        for name, module in self.head_layers:
            head_params.extend(list(module.parameters()))

        if head_params:
            parameter_groups.append(
                {
                    "params": head_params,
                    "lr": base_lr * head_lr_multiplier,
                    "name": "head",
                }
            )

        # Embeddings - usually lowest learning rate
        embedding_params = []
        for name, param in self.model.named_parameters():
            if "embed" in name.lower() and param not in head_params:
                # Check if not already in BERT layers
                already_added = False
                for group in parameter_groups:
                    if param in group["params"]:
                        already_added = True
                        break

                if not already_added:
                    embedding_params.append(param)

        if embedding_params:
            parameter_groups.append(
                {
                    "params": embedding_params,
                    "lr": base_lr * (layer_lr_decay**num_layers),  # Smallest LR
                    "name": "embeddings",
                }
            )

        return parameter_groups
