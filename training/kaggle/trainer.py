"""
Kaggle-specific trainer implementation with competition optimizations.
"""

from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit

from data.augmentation.tta import BERTTestTimeAugmentation
from data.augmentation.text import BERTTextAugmenter
from data.augmentation.tabular import TabularToTextAugmenter
from models.ensemble import BERTEnsembleConfig, BERTEnsembleModel
from ..core.base import BaseTrainer
from ..core.protocols import DataLoader, Model, TrainingResult
from ..strategies.multi_stage import MultiStageBERTTrainer, TrainingStage
from .adversarial_validation import AdversarialValidationConfig, AdversarialValidator
from .bert_pseudo_labeling import (
    BERTPseudoLabeler,
    EnsemblePseudoLabeler,
    PseudoLabelingConfig,
)
from .bert_strategies import create_bert_training_strategy
from .config import CompetitionType, KaggleTrainerConfig


class KaggleTrainer(BaseTrainer):
    """
    Trainer optimized for Kaggle competitions.

    Features:
    - Cross-validation with OOF predictions
    - Ensemble support
    - Pseudo-labeling
    - Test-time augmentation (TTA)
    - Automatic submission generation
    - Competition metric tracking
    """

    def __init__(
        self,
        model: Model,
        config: KaggleTrainerConfig,
        test_dataloader: DataLoader | None = None,
        enable_bert_strategies: bool = True,
        tokenizer=None,
    ):
        """
        Initialize Kaggle trainer.

        Args:
            model: Model to train
            config: Kaggle trainer configuration
            test_dataloader: Test data loader for predictions
            enable_bert_strategies: Enable BERT-specific strategies
            tokenizer: Tokenizer for text augmentation
        """
        # Add Kaggle-specific callbacks
        callbacks = []

        # TODO: Re-enable when callbacks are properly implemented
        # callbacks = [
        #     CompetitionMetrics(
        #         metric_name=config.kaggle.competition_metric,
        #         maximize=config.kaggle.maximize_metric,
        #     ),
        # ]

        # if config.kaggle.enable_api and config.kaggle.auto_submit:
        #     callbacks.append(
        #         KaggleSubmissionCallback(
        #             competition_name=config.kaggle.competition_name,
        #             submission_message=config.kaggle.submission_message,
        #         )
        #     )

        super().__init__(model, config, callbacks)

        self.config: KaggleTrainerConfig = config
        self.test_dataloader = test_dataloader

        # Initialize CV splits
        self.cv_splits = []
        self.oof_predictions = None
        self.test_predictions = None

        # Ensemble models
        self.ensemble_models = []
        self.bert_ensemble = None

        # Create submission directory
        self.config.kaggle.submission_dir.mkdir(parents=True, exist_ok=True)

        # Initialize BERT strategies if enabled
        self.enable_bert_strategies = enable_bert_strategies
        self.bert_trainer = None

        if enable_bert_strategies and self._is_bert_model():
            # Multi-stage training
            self.bert_strategy = create_bert_training_strategy(
                model_type=self._get_model_type(),
                task_type=self._get_task_type(),
                total_epochs=config.training.num_epochs,
            )
            self.bert_trainer = MultiStageBERTTrainer(model, self.bert_strategy)
            logger.info("Initialized BERT multi-stage training strategies")

            # Ensemble configuration
            if config.kaggle.enable_ensemble:
                self.bert_ensemble_config = BERTEnsembleConfig(
                    model_types=[self._get_model_type()],
                    random_seeds=[42, 123, 456] if config.kaggle.cv_folds > 1 else [42],
                    ensemble_method=config.kaggle.ensemble_method or "weighted_average",
                )
                self.bert_ensemble = BERTEnsembleModel(self.bert_ensemble_config)
                logger.info("Initialized BERT ensemble framework")

            # Text augmentation (only initialize if tokenizer is provided)
            if tokenizer:
                self.text_augmenter = BERTTextAugmenter(tokenizer)
                self.tta_augmenter = BERTTestTimeAugmentation(self.text_augmenter)
                logger.info("Initialized BERT augmentation engines")
            else:
                self.text_augmenter = None
                self.tta_augmenter = None
                logger.warning("No tokenizer provided - text augmentation disabled")
            
            # Tabular augmenter doesn't need tokenizer but needs config
            from data.augmentation.config import AugmentationConfig
            augmentation_config = AugmentationConfig()
            self.tabular_augmenter = TabularToTextAugmenter(config=augmentation_config)

            # Pseudo-labeling configuration
            self.pseudo_labeling_config = PseudoLabelingConfig(
                confidence_threshold=0.95,
                num_iterations=config.kaggle.pseudo_labeling_iterations
                if hasattr(config.kaggle, "pseudo_labeling_iterations")
                else 0,
            )
            self.pseudo_labeler = None  # Will be initialized when needed

            # Adversarial validation
            self.adversarial_validator = None  # Will be initialized when needed

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None = None,
        resume_from: Path | None = None,
    ) -> TrainingResult:
        """
        Train with Kaggle optimizations.

        If CV is enabled, runs cross-validation training.
        Otherwise, runs standard training with Kaggle-specific features.

        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader (optional)
            resume_from: Path to resume from (optional)

        Returns:
            Training result with metrics
        """
        if self.config.kaggle.cv_folds > 1:
            logger.info(
                f"Starting {self.config.kaggle.cv_folds}-fold cross-validation training"
            )
            return self.train_with_cv(train_dataloader, val_dataloader)
        else:
            logger.info("Starting standard training with Kaggle optimizations")
            # Call parent train method with Kaggle callbacks already set
            if self.bert_trainer:
                # Use BERT-aware training
                result = self._train_with_bert_strategies(
                    train_dataloader, val_dataloader, resume_from
                )
            else:
                result = super().train(train_dataloader, val_dataloader, resume_from)

            # Generate predictions if test data is available
            if self.test_dataloader is not None:
                logger.info("Generating test predictions")
                self.test_predictions = self.predict(self.test_dataloader)

                # Create submission file
                submission_name = f"submission_{self.config.environment.run_name}"
                submission_path = self.create_submission(
                    submission_name=submission_name
                )
                logger.info(f"Submission saved to: {submission_path}")

            return result

    def train_with_cv(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None = None,
        groups: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """
        Train with cross-validation.

        Args:
            train_dataloader: Training data loader
            val_dataloader: Optional validation data loader (ignored for CV)
            groups: Optional group labels for GroupKFold

        Returns:
            Dictionary with CV results
        """
        logger.info(f"Starting {self.config.kaggle.cv_folds}-fold cross-validation")

        # Get data from loader (assuming it has a dataset attribute)
        if not hasattr(train_dataloader, "dataset"):
            raise ValueError("DataLoader must have a dataset attribute for CV")

        dataset = train_dataloader.dataset
        n_samples = len(dataset)

        # Initialize OOF predictions as MLX arrays
        if self.config.kaggle.competition_type == CompetitionType.BINARY_CLASSIFICATION:
            self.oof_predictions = mx.zeros([n_samples])
        elif (
            self.config.kaggle.competition_type
            == CompetitionType.MULTICLASS_CLASSIFICATION
        ):
            # Get number of classes from model
            num_classes = (
                self.model.config.num_labels
                if hasattr(self.model.config, "num_labels")
                else 10
            )
            self.oof_predictions = mx.zeros([n_samples, num_classes])
        else:
            self.oof_predictions = mx.zeros([n_samples])

        # Create CV splitter
        cv_splitter = self._create_cv_splitter()

        # Get labels for stratification if needed
        labels = None
        if hasattr(dataset, "labels"):
            labels = dataset.labels
        elif hasattr(dataset, "get_labels"):
            labels = dataset.get_labels()

        # Perform CV
        cv_scores = []
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(
            cv_splitter.split(range(n_samples), labels, groups)
        ):
            logger.info(f"\nTraining fold {fold + 1}/{self.config.kaggle.cv_folds}")

            # Create fold data loaders
            fold_train_loader = self._create_fold_dataloader(
                train_dataloader, train_idx
            )
            fold_val_loader = self._create_fold_dataloader(train_dataloader, val_idx)

            # Reset model for each fold
            self._reset_model_weights()

            # Train fold
            result = super().train(fold_train_loader, fold_val_loader)
            fold_results.append(result)

            # Get OOF predictions
            fold_predictions = self.predict(fold_val_loader)

            # Store OOF predictions
            if (
                self.config.kaggle.competition_type
                == CompetitionType.BINARY_CLASSIFICATION
            ):
                # Convert logits to probabilities
                fold_probs = mx.sigmoid(fold_predictions[:, 1])
                self.oof_predictions[val_idx] = fold_probs
            elif (
                self.config.kaggle.competition_type
                == CompetitionType.MULTICLASS_CLASSIFICATION
            ):
                # Convert logits to probabilities
                fold_probs = mx.softmax(fold_predictions, axis=-1)
                self.oof_predictions[val_idx] = fold_probs
            else:
                self.oof_predictions[val_idx] = fold_predictions.squeeze()

            # Calculate fold score
            fold_score = result.best_val_metric
            cv_scores.append(fold_score)

            logger.info(
                f"Fold {fold + 1} {self.config.kaggle.competition_metric}: {fold_score:.4f}"
            )

            # Save fold model if ensemble is enabled
            if self.config.kaggle.enable_ensemble:
                fold_model_path = (
                    self.config.environment.output_dir / f"fold_{fold}_model"
                )
                self.model.save_pretrained(fold_model_path)
                self.ensemble_models.append(fold_model_path)

        # Calculate CV score using MLX
        cv_scores_mx = mx.array(cv_scores)
        cv_mean = float(mx.mean(cv_scores_mx).item())
        cv_std = float(mx.std(cv_scores_mx).item())

        logger.info(
            f"\nCV {self.config.kaggle.competition_metric}: {cv_mean:.4f} ± {cv_std:.4f}"
        )

        # Save OOF predictions
        if self.config.kaggle.save_oof_predictions:
            self._save_oof_predictions()

        # Generate test predictions if test data is available
        if self.test_dataloader is not None:
            self.generate_test_predictions()

        return {
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "cv_scores": cv_scores,
            "fold_results": fold_results,
            "oof_predictions": self.oof_predictions,
            "test_predictions": self.test_predictions,
        }

    def generate_test_predictions(self) -> np.ndarray:
        """Generate predictions for test data."""
        if self.test_dataloader is None:
            raise ValueError("No test dataloader provided")

        logger.info("Generating test predictions")

        if self.config.kaggle.enable_ensemble and self.ensemble_models:
            # Ensemble prediction
            self.test_predictions = self._ensemble_predict(self.test_dataloader)
        else:
            # Single model prediction
            raw_predictions = self.predict(self.test_dataloader)

            # Convert to appropriate format
            if (
                self.config.kaggle.competition_type
                == CompetitionType.BINARY_CLASSIFICATION
            ):
                self.test_predictions = mx.sigmoid(raw_predictions[:, 1]).tolist()
            elif (
                self.config.kaggle.competition_type
                == CompetitionType.MULTICLASS_CLASSIFICATION
            ):
                self.test_predictions = mx.argmax(raw_predictions, axis=-1).tolist()
            else:
                self.test_predictions = raw_predictions.squeeze().tolist()

        # Apply test-time augmentation if enabled
        if self.config.kaggle.enable_tta:
            self.test_predictions = self._apply_tta(self.test_dataloader)

        # Save test predictions
        if self.config.kaggle.save_test_predictions:
            self._save_test_predictions()

        return self.test_predictions

    def create_submission(
        self,
        sample_submission_path: Path | None = None,
        submission_name: str | None = None,
    ) -> Path:
        """
        Create a submission file for Kaggle.

        Args:
            sample_submission_path: Path to sample submission file
            submission_name: Name for submission file

        Returns:
            Path to created submission file
        """
        if self.test_predictions is None:
            raise ValueError(
                "No test predictions available. Run generate_test_predictions first."
            )

        # Create submission DataFrame
        if sample_submission_path and sample_submission_path.exists():
            submission = pd.read_csv(sample_submission_path)

            # Update predictions
            if (
                self.config.kaggle.competition_type
                == CompetitionType.BINARY_CLASSIFICATION
            ):
                # Assuming target column is the second column
                target_col = submission.columns[1]
                submission[target_col] = self.test_predictions
            elif (
                self.config.kaggle.competition_type
                == CompetitionType.MULTICLASS_CLASSIFICATION
            ):
                target_col = submission.columns[1]
                submission[target_col] = self.test_predictions
            else:
                target_col = submission.columns[1]
                submission[target_col] = self.test_predictions
        else:
            # Create submission from scratch
            submission = pd.DataFrame(
                {
                    "id": range(len(self.test_predictions)),
                    "target": self.test_predictions,
                }
            )

        # Save submission
        if submission_name is None:
            submission_name = f"submission_{self.config.kaggle.competition_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"

        submission_path = self.config.kaggle.submission_dir / submission_name
        submission.to_csv(submission_path, index=False)

        logger.info(f"Created submission: {submission_path}")

        return submission_path

    def _create_cv_splitter(self):
        """Create appropriate CV splitter based on strategy."""
        if self.config.kaggle.cv_strategy == "stratified":
            return StratifiedKFold(
                n_splits=self.config.kaggle.cv_folds,
                shuffle=True,
                random_state=self.config.environment.seed,
            )
        elif self.config.kaggle.cv_strategy == "group":
            return GroupKFold(n_splits=self.config.kaggle.cv_folds)
        elif self.config.kaggle.cv_strategy == "time_series":
            return TimeSeriesSplit(n_splits=self.config.kaggle.cv_folds)
        else:
            return KFold(
                n_splits=self.config.kaggle.cv_folds,
                shuffle=True,
                random_state=self.config.environment.seed,
            )

    def _create_fold_dataloader(
        self, original_loader: DataLoader, indices: np.ndarray
    ) -> DataLoader:
        """Create a dataloader for a CV fold."""
        # This is a simplified version - actual implementation depends on the DataLoader
        # For now, we'll assume the DataLoader has a way to subset
        if hasattr(original_loader, "subset"):
            return original_loader.subset(indices)
        else:
            # Fallback: create new loader with subset of data
            logger.warning("DataLoader does not support subsetting, using full data")
            return original_loader

    def _reset_model_weights(self):
        """Reset model weights for new fold."""
        # Re-initialize model
        # This depends on the model architecture
        if hasattr(self.model, "reset_parameters"):
            self.model.reset_parameters()
        else:
            # Reinitialize from config
            logger.info("Reinitializing model weights")
            model_class = self.model.__class__
            if hasattr(self.model, "config"):
                self.model = model_class(self.model.config)
            else:
                logger.warning("Cannot reset model weights, using existing weights")

    def _ensemble_predict(self, dataloader: DataLoader) -> mx.array:
        """Generate ensemble predictions using MLX operations."""
        if self.bert_ensemble and len(self.bert_ensemble.models) > 0:
            # Use BERT ensemble model
            logger.info(
                f"Generating predictions with BERT ensemble ({len(self.bert_ensemble.models)} models)"
            )
            ensemble_preds = self.bert_ensemble.predict(dataloader)
            return ensemble_preds
        else:
            # Fallback to loading saved models
            logger.info(
                f"Generating ensemble predictions from {len(self.ensemble_models)} saved models"
            )

            all_predictions = []

            for model_path in self.ensemble_models:
                # Load model
                from ...models.factory import create_model_from_checkpoint

                model = create_model_from_checkpoint(model_path)

                # Get predictions (returns MLX array)
                predictions = self._predict_with_model(model, dataloader)
                all_predictions.append(predictions)

            # Stack predictions for ensemble
            stacked_preds = mx.stack(all_predictions, axis=0)

            # Combine predictions using MLX operations
            if self.config.kaggle.ensemble_method == "voting":
                # Simple averaging
                ensemble_preds = mx.mean(stacked_preds, axis=0)
            elif self.config.kaggle.ensemble_method == "blending":
                # Weighted averaging
                weights = self.config.kaggle.ensemble_weights or [1.0] * len(
                    all_predictions
                )
                weights = mx.array(weights)
                weights = weights / mx.sum(weights)

                # Reshape weights for broadcasting
                weights = (
                    weights[:, None, None]
                    if len(stacked_preds.shape) == 3
                    else weights[:, None]
                )
                ensemble_preds = mx.sum(stacked_preds * weights, axis=0)
            else:
                # Default to simple averaging
                ensemble_preds = mx.mean(stacked_preds, axis=0)

            return ensemble_preds

    def _predict_with_model(self, model: Model, dataloader: DataLoader) -> mx.array:
        """Generate predictions with a specific model."""
        predictions = []

        model.eval()
        for batch in dataloader:
            with mx.no_grad():
                outputs = model(batch)

                if "logits" in outputs:
                    batch_preds = outputs["logits"]
                else:
                    batch_preds = outputs["predictions"]

                predictions.append(batch_preds)

        return mx.concatenate(predictions, axis=0)

    def _apply_tta(self, dataloader: DataLoader) -> mx.array:
        """Apply test-time augmentation using BERT TTA augmenter."""
        if hasattr(self, "tta_augmenter") and self.tta_augmenter is not None:
            logger.info("Applying BERT-specific test-time augmentation")
            return self.tta_augmenter.predict_with_tta(
                self.model,
                dataloader,
                num_augmentations=self.config.kaggle.tta_iterations or 5,
            )
        else:
            # Fallback to simple TTA
            logger.info(
                f"Applying TTA with {self.config.kaggle.tta_iterations} iterations"
            )

            tta_predictions = []

            for i in range(self.config.kaggle.tta_iterations):
                # Get predictions with augmentation
                predictions = self.predict(dataloader)
                tta_predictions.append(predictions)

            # Average TTA predictions using MLX
            stacked = mx.stack(tta_predictions, axis=0)
            return mx.mean(stacked, axis=0)

    def _save_oof_predictions(self):
        """Save out-of-fold predictions."""
        oof_path = self.config.kaggle.submission_dir / "oof_predictions.npy"
        # Convert MLX array to numpy for saving
        oof_numpy = np.array(self.oof_predictions)
        np.save(oof_path, oof_numpy)
        logger.info(f"Saved OOF predictions to {oof_path}")

    def _save_test_predictions(self):
        """Save test predictions."""
        test_path = self.config.kaggle.submission_dir / "test_predictions.npy"
        # Convert MLX array to numpy for saving
        test_numpy = (
            np.array(self.test_predictions)
            if hasattr(self.test_predictions, "shape")
            else self.test_predictions
        )
        np.save(test_path, test_numpy)
        logger.info(f"Saved test predictions to {test_path}")

    def apply_pseudo_labeling(self, unlabeled_dataloader: DataLoader) -> DataLoader:
        """
        Apply pseudo-labeling to unlabeled data.

        Args:
            unlabeled_dataloader: DataLoader with unlabeled data

        Returns:
            Augmented dataloader with pseudo-labeled samples
        """
        if not self.enable_bert_strategies:
            logger.warning("Pseudo-labeling requires BERT strategies to be enabled")
            return unlabeled_dataloader

        # Initialize pseudo-labeler if not already done
        if self.pseudo_labeler is None:
            if self.bert_ensemble and len(self.bert_ensemble.models) > 1:
                # Use ensemble pseudo-labeler
                self.pseudo_labeler = EnsemblePseudoLabeler(
                    self.pseudo_labeling_config, self.bert_ensemble.models
                )
            else:
                # Use single model pseudo-labeler
                self.pseudo_labeler = BERTPseudoLabeler(self.pseudo_labeling_config)

        # Generate pseudo-labels
        pseudo_samples, confidences = self.pseudo_labeler.generate_pseudo_labels(
            self.model, unlabeled_dataloader
        )

        logger.info(f"Generated {len(pseudo_samples)} pseudo-labeled samples")

        # TODO: Create augmented dataloader combining original and pseudo-labeled data
        # This would require implementing dataset concatenation
        return unlabeled_dataloader

    def validate_with_adversarial(
        self, train_dataloader: DataLoader, test_dataloader: DataLoader
    ):
        """
        Perform adversarial validation to check train/test distribution.

        Args:
            train_dataloader: Training data
            test_dataloader: Test data (unlabeled)
        """
        if not self.enable_bert_strategies:
            logger.warning(
                "Adversarial validation requires BERT strategies to be enabled"
            )
            return

        # Initialize adversarial validator
        if self.adversarial_validator is None:
            adv_config = AdversarialValidationConfig()
            self.adversarial_validator = AdversarialValidator(self.model, adv_config)

        # Run adversarial validation
        results = self.adversarial_validator.validate(train_dataloader, test_dataloader)

        # Create adversarial split if AUC is high (indicating distribution shift)
        if results["auc"] > 0.7:
            logger.warning(
                f"High adversarial AUC ({results['auc']:.3f}) indicates train/test distribution shift"
            )
            logger.info("Creating adversarial validation split...")

            # Get adversarial split
            adv_train_idx, adv_val_idx = (
                self.adversarial_validator.create_adversarial_split(
                    train_dataloader, test_dataloader, val_size=0.2
                )
            )

            logger.info(
                f"Adversarial split created: {len(adv_train_idx)} train, {len(adv_val_idx)} val"
            )

            # Store for later use
            self.adversarial_split = (adv_train_idx, adv_val_idx)

    def _is_bert_model(self) -> bool:
        """Check if the model is a BERT variant."""
        model_class_name = self.model.__class__.__name__.lower()
        return any(
            bert_type in model_class_name
            for bert_type in ["bert", "roberta", "deberta"]
        )

    def _get_model_type(self) -> str:
        """Get the type of BERT model."""
        model_class_name = self.model.__class__.__name__.lower()
        if "modernbert" in model_class_name:
            return "modernbert"
        elif "roberta" in model_class_name:
            return "roberta"
        elif "deberta" in model_class_name:
            return "deberta"
        else:
            return "bert"

    def _get_task_type(self) -> str:
        """Get the task type from competition configuration."""
        comp_type = self.config.kaggle.competition_type
        if comp_type in [
            CompetitionType.BINARY_CLASSIFICATION,
            CompetitionType.MULTICLASS_CLASSIFICATION,
            CompetitionType.MULTILABEL_CLASSIFICATION,
        ]:
            return "classification"
        elif comp_type in [
            CompetitionType.REGRESSION,
            CompetitionType.ORDINAL_REGRESSION,
        ]:
            return "regression"
        else:
            return "other"

    def _train_with_bert_strategies(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None = None,
        resume_from: Path | None = None,
    ) -> TrainingResult:
        """
        Train with BERT-specific strategies including multi-stage training.

        This method hooks into the parent train method at key points to
        implement stage transitions.
        """
        logger.info("Starting BERT multi-stage training")

        # Store original train method
        original_train_epoch = self._train_epoch

        # Store current epoch for stage tracking
        self._bert_current_epoch = 0

        # Setup initial stage
        from dataclasses import replace

        original_optimizer_config = replace(self.config.optimizer)
        initial_optimizer = self.bert_trainer.setup_stage(
            TrainingStage.FROZEN_BERT, original_optimizer_config
        )

        # Store original optimizer
        original_optimizer = self.optimizer if hasattr(self, "optimizer") else None
        self.optimizer = initial_optimizer

        # Override _train_epoch to add stage transitions
        def bert_aware_train_epoch(dataloader, epoch):
            # Check for stage transition
            if self.bert_trainer.should_transition_stage(epoch):
                new_optimizer = self.bert_trainer.setup_stage(
                    self.bert_trainer.get_current_stage(epoch),
                    original_optimizer_config,
                )
                self.optimizer = new_optimizer

                stage_info = self.bert_trainer.get_stage_info()
                logger.info(f"Stage transition at epoch {epoch}: {stage_info['stage']}")
                logger.info(
                    f"Trainable params: {stage_info['trainable_layers']}/{stage_info['total_layers']}"
                )

            # Call original train epoch
            return original_train_epoch(dataloader, epoch)

        # Temporarily replace train_epoch
        self._train_epoch = bert_aware_train_epoch

        try:
            # Call parent train method
            result = super().train(train_dataloader, val_dataloader, resume_from)
        finally:
            # Restore original methods
            self._train_epoch = original_train_epoch
            if original_optimizer:
                self.optimizer = original_optimizer

        return result
