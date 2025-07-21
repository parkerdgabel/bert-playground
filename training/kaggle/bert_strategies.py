"""
BERT-specific training strategies for Kaggle competitions.

This module contains Kaggle-specific utilities and the factory function
for creating BERT training strategies.
"""

from ..strategies.multi_stage import BERTTrainingStrategy


def create_bert_training_strategy(
    model_type: str = "bert", task_type: str = "classification", total_epochs: int = 5
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
        head_lr_multiplier=head_lr_multiplier,
    )
