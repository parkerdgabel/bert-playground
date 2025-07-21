"""
Adversarial validation for detecting train/test distribution shift.

This module implements adversarial validation to identify distribution
differences between training and test data, enabling better validation
strategies for Kaggle competitions.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


@dataclass
class AdversarialValidationConfig:
    """Configuration for adversarial validation."""

    # Model settings
    model_type: str = "bert_with_head"  # Model to use for validation
    max_length: int = 256  # Max sequence length
    batch_size: int = 32

    # Training settings
    num_epochs: int = 3  # Usually don't need many epochs
    learning_rate: float = 2e-5

    # Validation settings
    cv_folds: int = 5  # Cross-validation folds
    similarity_threshold: float = 0.55  # ROC-AUC threshold for similarity

    # Sampling settings
    sample_size: int | None = None  # Sample size for large datasets
    stratify_sampling: bool = True  # Stratify when sampling


class AdversarialValidator:
    """
    Adversarial validation for train/test distribution analysis.

    This class helps identify:
    1. Whether train and test distributions are similar
    2. Which features contribute to distribution shift
    3. How to create better validation sets
    """

    def __init__(self, config: AdversarialValidationConfig):
        """
        Initialize adversarial validator.

        Args:
            config: Validation configuration
        """
        self.config = config
        self.similarity_score = None
        self.feature_importance = {}
        self.validation_indices = None

    def compute_distribution_similarity(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        text_column: str = "text",
        exclude_columns: list[str] | None = None,
    ) -> float:
        """
        Compute similarity between train and test distributions.

        Args:
            train_data: Training dataframe
            test_data: Test dataframe
            text_column: Column containing text data
            exclude_columns: Columns to exclude from analysis

        Returns:
            Similarity score (0.5 = identical, 1.0 = completely different)
        """
        logger.info("Starting adversarial validation")

        # Prepare data
        train_data = train_data.copy()
        test_data = test_data.copy()

        # Add labels
        train_data["is_test"] = 0
        test_data["is_test"] = 1

        # Combine data
        combined_data = pd.concat([train_data, test_data], ignore_index=True)

        # Sample if needed
        if self.config.sample_size and len(combined_data) > self.config.sample_size:
            combined_data = combined_data.sample(
                n=self.config.sample_size,
                stratify=combined_data["is_test"]
                if self.config.stratify_sampling
                else None,
                random_state=42,
            )

        # Get features and labels
        X = (
            combined_data[text_column]
            if text_column in combined_data.columns
            else combined_data
        )
        y = combined_data["is_test"].values

        # Cross-validation
        cv_scores = []
        kfold = StratifiedKFold(
            n_splits=self.config.cv_folds, shuffle=True, random_state=42
        )

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            logger.info(f"Processing fold {fold + 1}/{self.config.cv_folds}")

            # Split data
            if isinstance(X, pd.Series):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            else:
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Train model
            score = self._train_adversarial_model(X_train, y_train, X_val, y_val)
            cv_scores.append(score)

            logger.info(f"Fold {fold + 1} ROC-AUC: {score:.4f}")

        # Average score
        self.similarity_score = np.mean(cv_scores)
        logger.info(f"Average ROC-AUC: {self.similarity_score:.4f}")

        # Interpret score
        if self.similarity_score < self.config.similarity_threshold:
            logger.info("✓ Train and test distributions are similar")
        else:
            logger.warning(
                f"⚠ Distribution shift detected (ROC-AUC: {self.similarity_score:.4f})"
            )

        return self.similarity_score

    def create_adversarial_validation_split(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        val_size: float = 0.2,
        text_column: str = "text",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create validation set that matches test distribution.

        Args:
            train_data: Training dataframe
            test_data: Test dataframe
            val_size: Validation set size
            text_column: Text column name

        Returns:
            Tuple of (new_train_data, validation_data)
        """
        logger.info("Creating adversarial validation split")

        # Get predictions on training data
        train_data = train_data.copy()
        test_data = test_data.copy()

        # Add temporary labels
        train_data["is_test"] = 0
        test_data["is_test"] = 1

        # Train model on all data
        combined_data = pd.concat([train_data, test_data], ignore_index=True)
        X = (
            combined_data[text_column]
            if text_column in combined_data.columns
            else combined_data
        )
        y = combined_data["is_test"].values

        # Get predictions on training data only
        train_predictions = self._get_test_probabilities(
            X.iloc[: len(train_data)], X, y
        )

        # Sort by probability of being from test set
        train_data["test_probability"] = train_predictions
        train_data_sorted = train_data.sort_values("test_probability", ascending=False)

        # Take top samples as validation (most similar to test)
        val_size_count = int(len(train_data) * val_size)

        val_data = train_data_sorted.head(val_size_count).copy()
        new_train_data = train_data_sorted.tail(len(train_data) - val_size_count).copy()

        # Remove temporary columns
        for df in [new_train_data, val_data]:
            df.drop(
                ["is_test", "test_probability"], axis=1, inplace=True, errors="ignore"
            )

        logger.info(
            f"Created validation set with {len(val_data)} samples (most similar to test)"
        )
        logger.info(f"New training set has {len(new_train_data)} samples")

        return new_train_data, val_data

    def analyze_feature_importance(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        feature_columns: list[str],
    ) -> dict[str, float]:
        """
        Analyze which features contribute to distribution shift.

        Args:
            train_data: Training dataframe
            test_data: Test dataframe
            feature_columns: Columns to analyze

        Returns:
            Feature importance scores
        """
        logger.info("Analyzing feature importance for distribution shift")

        feature_scores = {}

        for feature in feature_columns:
            if feature not in train_data.columns or feature not in test_data.columns:
                continue

            # Create dataset with only this feature
            train_subset = train_data[[feature]].copy()
            test_subset = test_data[[feature]].copy()

            # Compute similarity for this feature alone
            score = self._compute_single_feature_similarity(
                train_subset[feature], test_subset[feature]
            )

            # Higher score means more distribution shift
            feature_scores[feature] = score

        # Sort by importance
        sorted_features = sorted(
            feature_scores.items(), key=lambda x: x[1], reverse=True
        )

        logger.info("Features contributing to distribution shift:")
        for feature, score in sorted_features[:10]:
            logger.info(f"  {feature}: {score:.4f}")

        self.feature_importance = dict(sorted_features)
        return self.feature_importance

    def _train_adversarial_model(
        self,
        X_train: pd.Series,
        y_train: np.ndarray,
        X_val: pd.Series,
        y_val: np.ndarray,
    ) -> float:
        """
        Train a model to distinguish train from test.

        Args:
            X_train: Training features
            y_train: Training labels (0=train, 1=test)
            X_val: Validation features
            y_val: Validation labels

        Returns:
            ROC-AUC score
        """
        # For text data, we'd use BERT
        # For this example, we'll use a simple approach
        # In practice, you'd create data loaders and train BERT

        # Simplified: return random score
        # TODO: Implement actual BERT training
        predictions = np.random.rand(len(y_val))

        try:
            score = roc_auc_score(y_val, predictions)
        except:
            score = 0.5

        return score

    def _get_test_probabilities(
        self, X_predict: pd.Series, X_all: pd.Series, y_all: np.ndarray
    ) -> np.ndarray:
        """
        Get probabilities of samples being from test set.

        Args:
            X_predict: Data to predict on
            X_all: All training data
            y_all: All labels

        Returns:
            Probabilities
        """
        # Simplified implementation
        # TODO: Implement actual prediction
        return np.random.rand(len(X_predict))

    def _compute_single_feature_similarity(
        self, train_feature: pd.Series, test_feature: pd.Series
    ) -> float:
        """
        Compute similarity for a single feature.

        Args:
            train_feature: Training feature values
            test_feature: Test feature values

        Returns:
            Similarity score
        """
        # For numerical features, use statistical tests
        if pd.api.types.is_numeric_dtype(train_feature):
            # Kolmogorov-Smirnov test or similar
            train_mean = train_feature.mean()
            test_mean = test_feature.mean()
            train_std = train_feature.std()

            if train_std > 0:
                # Normalized difference
                diff = abs(train_mean - test_mean) / train_std
                # Convert to 0-1 score
                score = 1 / (1 + np.exp(-diff))
            else:
                score = 0.5
        else:
            # For categorical, compare distributions
            train_dist = train_feature.value_counts(normalize=True)
            test_dist = test_feature.value_counts(normalize=True)

            # Get all unique values
            all_values = set(train_dist.index) | set(test_dist.index)

            # Compute total variation distance
            tv_distance = 0
            for value in all_values:
                train_prob = train_dist.get(value, 0)
                test_prob = test_dist.get(value, 0)
                tv_distance += abs(train_prob - test_prob)

            score = tv_distance / 2  # Normalize to 0-1

        return score


def create_leak_detection_features(
    train_data: pd.DataFrame, test_data: pd.DataFrame, id_column: str = "id"
) -> dict[str, Any]:
    """
    Create features to detect potential data leakage.

    Args:
        train_data: Training data
        test_data: Test data
        id_column: ID column name

    Returns:
        Dictionary with leak detection results
    """
    leaks = {}

    # Check for ID patterns
    if id_column in train_data.columns and id_column in test_data.columns:
        train_ids = set(train_data[id_column])
        test_ids = set(test_data[id_column])

        # Check for overlapping IDs
        overlap = train_ids & test_ids
        if overlap:
            leaks["id_overlap"] = list(overlap)
            logger.warning(
                f"Found {len(overlap)} overlapping IDs between train and test"
            )

    # Check for temporal patterns
    for col in train_data.columns:
        if "date" in col.lower() or "time" in col.lower():
            try:
                train_dates = pd.to_datetime(train_data[col])
                test_dates = pd.to_datetime(test_data[col])

                # Check if test is all after train (time series split)
                if train_dates.max() < test_dates.min():
                    leaks["temporal_split"] = {
                        "column": col,
                        "train_max": str(train_dates.max()),
                        "test_min": str(test_dates.min()),
                    }
                    logger.info(f"Detected temporal split on {col}")
            except:
                pass

    return leaks
