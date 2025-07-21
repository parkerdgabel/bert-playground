"""
BERT ensemble training utilities for Kaggle competitions.

This module provides training utilities for BERT ensembles, leveraging
the core ensemble model functionality.
"""

from loguru import logger

from ...models.ensemble import BERTEnsembleConfig, BERTEnsembleModel
from ..core.protocols import DataLoader


class BERTEnsembleTrainer:
    """
    Trainer for BERT ensemble models.

    Handles training multiple models and combining them into an ensemble.
    """

    def __init__(self, ensemble_config: BERTEnsembleConfig):
        """
        Initialize ensemble trainer.

        Args:
            ensemble_config: Ensemble configuration
        """
        self.config = ensemble_config
        self.ensemble = BERTEnsembleModel(ensemble_config)

    def train_ensemble(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        trainer_class=None,
        trainer_config=None,
    ) -> BERTEnsembleModel:
        """
        Train ensemble of models.

        Args:
            train_loader: Training data
            val_loader: Validation data
            trainer_class: Trainer class to use
            trainer_config: Trainer configuration

        Returns:
            Trained ensemble
        """
        # Create diverse models
        models = self.ensemble.create_diverse_models(
            base_config={},
            head_type=trainer_config.kaggle.competition_type.value,
            num_labels=2,  # TODO: Get from config
        )

        cv_scores = []

        # Train each model
        for i, model in enumerate(models):
            logger.info(f"Training model {i + 1}/{len(models)}")

            # Create trainer for this model
            trainer = trainer_class(
                model=model, config=trainer_config, enable_bert_strategies=True
            )

            # Train
            result = trainer.train(train_loader, val_loader)

            # Track CV score
            best_score = result.best_val_metric or result.best_val_loss
            cv_scores.append(best_score)

            logger.info(f"Model {i + 1} best score: {best_score}")

        # Update ensemble weights based on CV performance
        self.ensemble.update_weights_from_cv(cv_scores)

        return self.ensemble
