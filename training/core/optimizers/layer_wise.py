"""
Custom optimizers for BERT training with layer-wise learning rates.

MLX doesn't natively support parameter groups, so we implement
custom optimizer wrappers that can handle different learning rates
for different parameter sets.
"""

from typing import Any

import mlx.core as mx
import mlx.nn as nn
from loguru import logger


class LayerWiseAdamW:
    """
    AdamW optimizer with layer-wise learning rates.

    This is a wrapper around MLX's AdamW that supports different
    learning rates for different parameter groups.
    """

    def __init__(
        self,
        param_groups: list[dict[str, Any]],
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        """
        Initialize layer-wise AdamW optimizer.

        Args:
            param_groups: List of parameter groups with 'params' and 'lr'
            betas: Adam beta parameters
            eps: Epsilon for numerical stability
            weight_decay: Weight decay coefficient
        """
        self.param_groups = param_groups
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # Create individual optimizers for each group
        self.optimizers = []
        for group in param_groups:
            optimizer = mx.optimizers.AdamW(
                learning_rate=group["lr"],
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
            )
            self.optimizers.append(
                {
                    "optimizer": optimizer,
                    "params": group["params"],
                    "name": group.get("name", "unnamed"),
                }
            )

        logger.info(
            f"Created LayerWiseAdamW with {len(self.optimizers)} parameter groups"
        )
        for opt_dict in self.optimizers:
            logger.debug(
                f"  Group '{opt_dict['name']}': {len(opt_dict['params'])} parameters, "
                f"lr={opt_dict['optimizer'].learning_rate}"
            )

    def update(self, model: nn.Module, gradients: dict[str, mx.array]):
        """
        Update model parameters with layer-wise learning rates.

        Args:
            model: Model to update
            gradients: Dictionary of gradients
        """
        # Get model parameters as a dict
        model_params = dict(model.named_parameters())

        # Update each parameter group separately
        for opt_dict in self.optimizers:
            # Find gradients for this group's parameters
            group_grads = {}
            group_params = {}

            for param in opt_dict["params"]:
                # Find the parameter name in model
                param_name = None
                for name, model_param in model_params.items():
                    if model_param is param:
                        param_name = name
                        break

                if param_name and param_name in gradients:
                    group_grads[param_name] = gradients[param_name]
                    group_params[param_name] = param

            # Update this group if we have gradients
            if group_grads:
                opt_dict["optimizer"].update(group_params, group_grads)

    @property
    def learning_rate(self):
        """Get average learning rate across groups."""
        if not self.optimizers:
            return 0.0
        return sum(opt["optimizer"].learning_rate for opt in self.optimizers) / len(
            self.optimizers
        )

    @property
    def state(self):
        """Get optimizer state (for checkpointing)."""
        return {
            "optimizers": [opt["optimizer"].state for opt in self.optimizers],
            "param_groups": self.param_groups,
            "betas": self.betas,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
        }

    def load_state(self, state: dict[str, Any]):
        """Load optimizer state."""
        for i, opt_state in enumerate(state["optimizers"]):
            if i < len(self.optimizers):
                self.optimizers[i]["optimizer"].state = opt_state


class BERTOptimizer:
    """
    Factory for creating BERT-optimized optimizers.
    """

    @staticmethod
    def create_optimizer(
        model: nn.Module,
        base_lr: float = 2e-5,
        weight_decay: float = 0.01,
        layer_lr_decay: float = 0.95,
        head_lr_multiplier: float = 10.0,
        use_layer_wise: bool = True,
    ):
        """
        Create optimizer for BERT model.

        Args:
            model: BERT model
            base_lr: Base learning rate
            weight_decay: Weight decay
            layer_lr_decay: LR decay per layer
            head_lr_multiplier: Multiplier for head LR
            use_layer_wise: Whether to use layer-wise LR

        Returns:
            Optimizer instance
        """
        if not use_layer_wise:
            # Standard AdamW
            return mx.optimizers.AdamW(learning_rate=base_lr, weight_decay=weight_decay)

        # Create layer-wise optimizer
        from models.bert.utils.layer_manager import BERTLayerManager

        layer_manager = BERTLayerManager(model)
        param_groups = layer_manager.get_parameter_groups(
            base_lr=base_lr,
            layer_lr_decay=layer_lr_decay,
            head_lr_multiplier=head_lr_multiplier,
        )

        return LayerWiseAdamW(param_groups=param_groups, weight_decay=weight_decay)
