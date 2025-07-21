"""
BERT-specific training strategies for Kaggle competitions.

This module implements advanced training techniques optimized for BERT models,
including multi-stage training, layer-wise learning rates, and gradual unfreezing.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import mlx.core as mx
import mlx.nn as nn
from loguru import logger

from ..core.protocols import Model, Optimizer
from ..core.optimization import create_optimizer
from ..core.config import OptimizerConfig


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
    unfreeze_layers: List[int] = None  # Which layers to unfreeze in partial stage
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
        logger.info(f"Identified {len(self.bert_layers)} BERT layers and {len(self.head_layers)} head layers")
    
    def _identify_bert_layers(self) -> List[Tuple[str, Any]]:
        """Identify BERT encoder layers."""
        bert_layers = []
        
        # Handle both classic BERT and ModernBERT
        for name, module in self.model.named_modules():
            if any(key in name for key in ["encoder", "bert", "transformer"]):
                if "layer" in name or "block" in name:
                    bert_layers.append((name, module))
        
        # Sort by layer depth
        bert_layers.sort(key=lambda x: x[0])
        return bert_layers
    
    def _identify_head_layers(self) -> List[Tuple[str, Any]]:
        """Identify task-specific head layers."""
        head_layers = []
        
        for name, module in self.model.named_modules():
            if any(key in name for key in ["head", "classifier", "pooler"]):
                head_layers.append((name, module))
        
        return head_layers
    
    def freeze_bert(self, exclude_layers: Optional[List[int]] = None):
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
    
    def unfreeze_bert(self, layers_to_unfreeze: Optional[List[int]] = None):
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
            indices = [idx if idx >= 0 else total_layers + idx 
                      for idx in layers_to_unfreeze]
            
            for idx, (name, module) in enumerate(self.bert_layers):
                if idx in indices:
                    for param in module.parameters():
                        param.unfreeze()
                    logger.debug(f"Unfroze layer {name}")
    
    def get_parameter_groups(self, base_lr: float, layer_lr_decay: float = 0.95,
                           head_lr_multiplier: float = 10.0) -> List[Dict[str, Any]]:
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
                parameter_groups.append({
                    "params": params,
                    "lr": layer_lr,
                    "name": f"bert_layer_{idx}"
                })
        
        # Head layers - higher learning rate
        head_params = []
        for name, module in self.head_layers:
            head_params.extend(list(module.parameters()))
        
        if head_params:
            parameter_groups.append({
                "params": head_params,
                "lr": base_lr * head_lr_multiplier,
                "name": "head"
            })
        
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
            parameter_groups.append({
                "params": embedding_params,
                "lr": base_lr * (layer_lr_decay ** num_layers),  # Smallest LR
                "name": "embeddings"
            })
        
        return parameter_groups


class MultiStageBERTTrainer:
    """Implements multi-stage training for BERT models."""
    
    def __init__(self, model: Model, strategy: BERTTrainingStrategy):
        """
        Initialize multi-stage trainer.
        
        Args:
            model: BERT model to train
            strategy: Training strategy configuration
        """
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
    
    def setup_stage(self, stage: TrainingStage, optimizer_config: OptimizerConfig) -> Optimizer:
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
            from dataclasses import replace
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
                head_lr_multiplier=self.strategy.head_lr_multiplier
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
                head_lr_multiplier=self.strategy.head_lr_multiplier
            )
            
            # Create optimizer
            optimizer = self._create_grouped_optimizer(param_groups, optimizer_config)
        
        self.current_stage = stage
        return optimizer
    
    def _create_grouped_optimizer(self, param_groups: List[Dict[str, Any]], 
                                config: OptimizerConfig) -> Optimizer:
        """
        Create optimizer with parameter groups.
        """
        from .optimizers import LayerWiseAdamW
        
        # Create layer-wise optimizer
        return LayerWiseAdamW(
            param_groups=param_groups,
            betas=(config.betas[0], config.betas[1]) if hasattr(config, 'betas') else (0.9, 0.999),
            eps=config.eps if hasattr(config, 'eps') else 1e-8,
            weight_decay=config.weight_decay
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
    
    def get_stage_info(self) -> Dict[str, Any]:
        """Get information about current stage."""
        return {
            "stage": self.current_stage.value,
            "trainable_layers": len([p for p in self.model.parameters() if not p.frozen]),
            "total_layers": len(list(self.model.parameters())),
            "stage_epoch": self.stage_epoch
        }


def create_bert_training_strategy(
    model_type: str = "bert",
    task_type: str = "classification",
    total_epochs: int = 5
) -> BERTTrainingStrategy:
    """
    Create a BERT training strategy based on model and task type.
    
    Args:
        model_type: Type of BERT model (bert, modernbert)
        task_type: Type of task (classification, regression, etc.)
        total_epochs: Total training epochs
        
    Returns:
        Configured training strategy
    """
    # Distribute epochs across stages
    if total_epochs <= 3:
        # Short training - minimal freezing
        frozen_epochs = 0
        partial_epochs = 1
        full_epochs = total_epochs - 1
    elif total_epochs <= 5:
        # Standard training
        frozen_epochs = 1
        partial_epochs = 2
        full_epochs = total_epochs - 3
    else:
        # Long training - more gradual
        frozen_epochs = 2
        partial_epochs = 3
        full_epochs = total_epochs - 5
    
    # Adjust for model type
    if model_type == "modernbert":
        # ModernBERT has more layers, unfreeze more gradually
        num_layers_to_unfreeze = 6
    else:
        num_layers_to_unfreeze = 4
    
    # Adjust for task type
    if task_type == "regression":
        # Regression often benefits from higher head LR
        head_lr_multiplier = 20.0
    else:
        head_lr_multiplier = 10.0
    
    return BERTTrainingStrategy(
        frozen_epochs=frozen_epochs,
        partial_unfreeze_epochs=partial_epochs,
        full_finetune_epochs=full_epochs,
        num_layers_to_unfreeze=num_layers_to_unfreeze,
        head_lr_multiplier=head_lr_multiplier
    )