"""
Multi-stage training strategies for BERT models.
"""

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from loguru import logger

# Import moved to avoid circular dependency
from ..core.config import OptimizerConfig
from ..core.optimization import create_optimizer
from ..core.optimizers import LayerWiseAdamW
from ..core.protocols import Model, Optimizer


class TrainingStage(Enum):
    """Training stages for BERT models."""

    FROZEN_BERT = "frozen_bert"  # Only train head
    PARTIAL_UNFREEZE = "partial_unfreeze"  # Unfreeze top layers
    FULL_FINETUNE = "full_finetune"  # Train entire model


@dataclass
class BERTTrainingStrategy:
    """Configuration for BERT training strategy."""

    # Stage configuration
    frozen_epochs: int = 1
    partial_unfreeze_epochs: int = 2
    full_finetune_epochs: int = 2

    # Layer configuration
    unfreeze_layers: list | None = None  # Which layers to unfreeze in partial stage
    num_layers_to_unfreeze: int = 4  # Number of top layers to unfreeze

    # Learning rate configuration
    head_lr_multiplier: float = 10.0  # Higher LR for task head
    layer_lr_decay: float = 0.95  # LR decay per layer (deeper = smaller LR)

    # Regularization
    freeze_embeddings: bool = True  # Keep embeddings frozen longer
    freeze_layer_norm: bool = False  # Whether to freeze layer norm

    def __post_init__(self):
        """Initialize unfreeze layers if not provided."""
        if self.unfreeze_layers is None:
            # By default, unfreeze top N layers
            self.unfreeze_layers = list(range(-self.num_layers_to_unfreeze, 0))


class MultiStageBERTTrainer:
    """Implements multi-stage training for BERT models."""

    def __init__(self, model: Model, strategy: BERTTrainingStrategy):
        """
        Initialize multi-stage trainer.

        Args:
            model: BERT model to train
            strategy: Training strategy configuration
        """
        # Import here to avoid circular dependency
        from models.bert.utils import BERTLayerManager
        
        self.model = model
        self.strategy = strategy
        self.layer_manager = BERTLayerManager(model)
        self.current_stage = TrainingStage.FROZEN_BERT
        self.stage_epoch = 0

    def get_current_stage(self, global_epoch: int) -> TrainingStage:
        """
        Determine current training stage based on epoch.

        Args:
            global_epoch: Current epoch number

        Returns:
            Current training stage
        """
        frozen_end = self.strategy.frozen_epochs
        partial_end = frozen_end + self.strategy.partial_unfreeze_epochs

        if global_epoch < frozen_end:
            return TrainingStage.FROZEN_BERT
        elif global_epoch < partial_end:
            return TrainingStage.PARTIAL_UNFREEZE
        else:
            return TrainingStage.FULL_FINETUNE

    def setup_stage(
        self, stage: TrainingStage, optimizer_config: OptimizerConfig
    ) -> Optimizer:
        """
        Setup model for a specific training stage.

        Args:
            stage: Training stage to setup
            optimizer_config: Base optimizer configuration

        Returns:
            Configured optimizer for the stage
        """
        logger.info(f"Setting up training stage: {stage.value}")

        if stage == TrainingStage.FROZEN_BERT:
            # Freeze all BERT layers
            self.layer_manager.freeze_bert()

            # Only train head parameters
            head_params = []
            for name, module in self.layer_manager.head_layers:
                head_params.extend(list(module.parameters()))

            # Create optimizer with higher LR for head
            config = replace(optimizer_config)
            config.learning_rate *= self.strategy.head_lr_multiplier
            optimizer = create_optimizer(head_params, config)

        elif stage == TrainingStage.PARTIAL_UNFREEZE:
            # Unfreeze top layers
            self.layer_manager.unfreeze_bert(self.strategy.unfreeze_layers)

            # Get layer-wise parameter groups
            param_groups = self.layer_manager.get_parameter_groups(
                base_lr=optimizer_config.learning_rate,
                layer_lr_decay=self.strategy.layer_lr_decay,
                head_lr_multiplier=self.strategy.head_lr_multiplier,
            )

            # Create optimizer with parameter groups
            optimizer = self._create_grouped_optimizer(param_groups, optimizer_config)

        else:  # FULL_FINETUNE
            # Unfreeze all layers
            self.layer_manager.unfreeze_bert()

            # Get all parameter groups with layer-wise LR
            param_groups = self.layer_manager.get_parameter_groups(
                base_lr=optimizer_config.learning_rate,
                layer_lr_decay=self.strategy.layer_lr_decay,
                head_lr_multiplier=self.strategy.head_lr_multiplier,
            )

            # Create optimizer
            optimizer = self._create_grouped_optimizer(param_groups, optimizer_config)

        self.current_stage = stage
        return optimizer

    def _create_grouped_optimizer(
        self, param_groups: list, config: OptimizerConfig
    ) -> Optimizer:
        """
        Create optimizer with parameter groups.
        """
        # Create layer-wise optimizer
        return LayerWiseAdamW(
            param_groups=param_groups,
            betas=(config.betas[0], config.betas[1])
            if hasattr(config, "betas")
            else (0.9, 0.999),
            eps=config.eps if hasattr(config, "eps") else 1e-8,
            weight_decay=config.weight_decay,
        )

    def should_transition_stage(self, global_epoch: int) -> bool:
        """
        Check if we should transition to next stage.

        Args:
            global_epoch: Current global epoch

        Returns:
            Whether to transition stages
        """
        new_stage = self.get_current_stage(global_epoch)
        return new_stage != self.current_stage

    def get_stage_info(self) -> dict[str, Any]:
        """Get information about current stage."""
        return {
            "stage": self.current_stage.value,
            "trainable_layers": len(
                [p for p in self.model.parameters() if not p.frozen]
            ),
            "total_layers": len(list(self.model.parameters())),
            "stage_epoch": self.stage_epoch,
        }
