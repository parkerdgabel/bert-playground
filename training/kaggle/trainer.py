"""
Kaggle-specific trainer implementation with competition optimizations.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold, TimeSeriesSplit
from loguru import logger
import mlx.core as mx

from ..core.base import BaseTrainer
from ..core.protocols import DataLoader, Model, TrainingResult
from .config import KaggleTrainerConfig, CompetitionType
from .callbacks import KaggleSubmissionCallback, CompetitionMetrics


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
        test_dataloader: Optional[DataLoader] = None,
    ):
        """
        Initialize Kaggle trainer.
        
        Args:
            model: Model to train
            config: Kaggle trainer configuration
            test_dataloader: Test data loader for predictions
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
        
        # Create submission directory
        self.config.kaggle.submission_dir.mkdir(parents=True, exist_ok=True)
    
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        resume_from: Optional[Path] = None,
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
            logger.info(f"Starting {self.config.kaggle.cv_folds}-fold cross-validation training")
            return self.train_with_cv(train_dataloader, val_dataloader)
        else:
            logger.info("Starting standard training with Kaggle optimizations")
            # Call parent train method with Kaggle callbacks already set
            result = super().train(train_dataloader, val_dataloader, resume_from)
            
            # Generate predictions if test data is available
            if self.test_dataloader is not None:
                logger.info("Generating test predictions")
                self.test_predictions = self.predict(self.test_dataloader)
                
                # Create submission file
                submission_name = f"submission_{self.config.environment.run_name}"
                submission_path = self.create_submission(submission_name=submission_name)
                logger.info(f"Submission saved to: {submission_path}")
            
            return result
    
    def train_with_cv(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
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
        if not hasattr(train_dataloader, 'dataset'):
            raise ValueError("DataLoader must have a dataset attribute for CV")
        
        dataset = train_dataloader.dataset
        n_samples = len(dataset)
        
        # Initialize OOF predictions
        if self.config.kaggle.competition_type == CompetitionType.BINARY_CLASSIFICATION:
            self.oof_predictions = np.zeros(n_samples)
        elif self.config.kaggle.competition_type == CompetitionType.MULTICLASS_CLASSIFICATION:
            # Get number of classes from model
            num_classes = self.model.config.num_labels if hasattr(self.model.config, 'num_labels') else 10
            self.oof_predictions = np.zeros((n_samples, num_classes))
        else:
            self.oof_predictions = np.zeros(n_samples)
        
        # Create CV splitter
        cv_splitter = self._create_cv_splitter()
        
        # Get labels for stratification if needed
        labels = None
        if hasattr(dataset, 'labels'):
            labels = dataset.labels
        elif hasattr(dataset, 'get_labels'):
            labels = dataset.get_labels()
        
        # Perform CV
        cv_scores = []
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(cv_splitter.split(range(n_samples), labels, groups)):
            logger.info(f"\nTraining fold {fold + 1}/{self.config.kaggle.cv_folds}")
            
            # Create fold data loaders
            fold_train_loader = self._create_fold_dataloader(train_dataloader, train_idx)
            fold_val_loader = self._create_fold_dataloader(train_dataloader, val_idx)
            
            # Reset model for each fold
            self._reset_model_weights()
            
            # Train fold
            result = super().train(fold_train_loader, fold_val_loader)
            fold_results.append(result)
            
            # Get OOF predictions
            fold_predictions = self.predict(fold_val_loader)
            
            # Store OOF predictions
            if self.config.kaggle.competition_type == CompetitionType.BINARY_CLASSIFICATION:
                # Convert logits to probabilities
                self.oof_predictions[val_idx] = mx.sigmoid(fold_predictions[:, 1]).tolist()
            elif self.config.kaggle.competition_type == CompetitionType.MULTICLASS_CLASSIFICATION:
                # Convert logits to probabilities
                self.oof_predictions[val_idx] = mx.softmax(fold_predictions, axis=-1).tolist()
            else:
                self.oof_predictions[val_idx] = fold_predictions.squeeze().tolist()
            
            # Calculate fold score
            fold_score = result.best_val_metric
            cv_scores.append(fold_score)
            
            logger.info(f"Fold {fold + 1} {self.config.kaggle.competition_metric}: {fold_score:.4f}")
            
            # Save fold model if ensemble is enabled
            if self.config.kaggle.enable_ensemble:
                fold_model_path = self.config.environment.output_dir / f"fold_{fold}_model"
                self.model.save_pretrained(fold_model_path)
                self.ensemble_models.append(fold_model_path)
        
        # Calculate CV score
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        
        logger.info(f"\nCV {self.config.kaggle.competition_metric}: {cv_mean:.4f} ± {cv_std:.4f}")
        
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
            if self.config.kaggle.competition_type == CompetitionType.BINARY_CLASSIFICATION:
                self.test_predictions = mx.sigmoid(raw_predictions[:, 1]).tolist()
            elif self.config.kaggle.competition_type == CompetitionType.MULTICLASS_CLASSIFICATION:
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
        sample_submission_path: Optional[Path] = None,
        submission_name: Optional[str] = None,
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
            raise ValueError("No test predictions available. Run generate_test_predictions first.")
        
        # Create submission DataFrame
        if sample_submission_path and sample_submission_path.exists():
            submission = pd.read_csv(sample_submission_path)
            
            # Update predictions
            if self.config.kaggle.competition_type == CompetitionType.BINARY_CLASSIFICATION:
                # Assuming target column is the second column
                target_col = submission.columns[1]
                submission[target_col] = self.test_predictions
            elif self.config.kaggle.competition_type == CompetitionType.MULTICLASS_CLASSIFICATION:
                target_col = submission.columns[1]
                submission[target_col] = self.test_predictions
            else:
                target_col = submission.columns[1]
                submission[target_col] = self.test_predictions
        else:
            # Create submission from scratch
            submission = pd.DataFrame({
                'id': range(len(self.test_predictions)),
                'target': self.test_predictions,
            })
        
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
    
    def _create_fold_dataloader(self, original_loader: DataLoader, indices: np.ndarray) -> DataLoader:
        """Create a dataloader for a CV fold."""
        # This is a simplified version - actual implementation depends on the DataLoader
        # For now, we'll assume the DataLoader has a way to subset
        if hasattr(original_loader, 'subset'):
            return original_loader.subset(indices)
        else:
            # Fallback: create new loader with subset of data
            logger.warning("DataLoader does not support subsetting, using full data")
            return original_loader
    
    def _reset_model_weights(self):
        """Reset model weights for new fold."""
        # Re-initialize model
        # This depends on the model architecture
        if hasattr(self.model, 'reset_parameters'):
            self.model.reset_parameters()
        else:
            # Reinitialize from config
            logger.info("Reinitializing model weights")
            model_class = self.model.__class__
            if hasattr(self.model, 'config'):
                self.model = model_class(self.model.config)
            else:
                logger.warning("Cannot reset model weights, using existing weights")
    
    def _ensemble_predict(self, dataloader: DataLoader) -> np.ndarray:
        """Generate ensemble predictions."""
        logger.info(f"Generating ensemble predictions from {len(self.ensemble_models)} models")
        
        all_predictions = []
        
        for model_path in self.ensemble_models:
            # Load model
            model = self.model.__class__.load_pretrained(model_path)
            
            # Get predictions
            predictions = self._predict_with_model(model, dataloader)
            all_predictions.append(predictions)
        
        # Combine predictions
        if self.config.kaggle.ensemble_method == "voting":
            # Simple averaging
            ensemble_preds = np.mean(all_predictions, axis=0)
        elif self.config.kaggle.ensemble_method == "blending":
            # Weighted averaging
            weights = self.config.kaggle.ensemble_weights or [1.0] * len(all_predictions)
            weights = np.array(weights) / np.sum(weights)
            ensemble_preds = np.average(all_predictions, axis=0, weights=weights)
        else:
            # Default to simple averaging
            ensemble_preds = np.mean(all_predictions, axis=0)
        
        return ensemble_preds
    
    def _predict_with_model(self, model: Model, dataloader: DataLoader) -> np.ndarray:
        """Generate predictions with a specific model."""
        predictions = []
        
        for batch in dataloader:
            outputs = model(batch)
            
            if "logits" in outputs:
                batch_preds = outputs["logits"]
            else:
                batch_preds = outputs["predictions"]
            
            predictions.append(batch_preds)
        
        return mx.concatenate(predictions, axis=0).tolist()
    
    def _apply_tta(self, dataloader: DataLoader) -> np.ndarray:
        """Apply test-time augmentation."""
        logger.info(f"Applying TTA with {self.config.kaggle.tta_iterations} iterations")
        
        tta_predictions = []
        
        for i in range(self.config.kaggle.tta_iterations):
            # Get predictions with augmentation
            # This assumes the dataloader supports augmentation
            predictions = self.predict(dataloader)
            tta_predictions.append(predictions)
        
        # Average TTA predictions
        return np.mean(tta_predictions, axis=0)
    
    def _save_oof_predictions(self):
        """Save out-of-fold predictions."""
        oof_path = self.config.kaggle.submission_dir / "oof_predictions.npy"
        np.save(oof_path, self.oof_predictions)
        logger.info(f"Saved OOF predictions to {oof_path}")
    
    def _save_test_predictions(self):
        """Save test predictions."""
        test_path = self.config.kaggle.submission_dir / "test_predictions.npy"
        np.save(test_path, self.test_predictions)
        logger.info(f"Saved test predictions to {test_path}")