"""
BERT-specific pseudo-labeling with confidence scoring.

This module implements advanced pseudo-labeling techniques that leverage
BERT's attention patterns and ensemble uncertainty for confident predictions.
"""

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from loguru import logger

from models.bert.confidence import BERTConfidenceScorer
from ..core.protocols import DataLoader, Model


@dataclass
class PseudoLabelingConfig:
    """Configuration for pseudo-labeling."""

    # Confidence thresholds
    confidence_threshold: float = 0.95  # Min confidence for hard labels
    soft_label_threshold: float = 0.8  # Min confidence for soft labels

    # Attention-based confidence
    use_attention_confidence: bool = True
    attention_entropy_threshold: float = 0.3  # Low entropy = high confidence

    # Ensemble uncertainty
    use_ensemble_uncertainty: bool = True
    max_ensemble_std: float = 0.1  # Max std dev for confident prediction

    # Iterative pseudo-labeling
    num_iterations: int = 3
    confidence_increase_per_iter: float = 0.02  # Lower threshold each iteration

    # Label selection
    select_top_k: int | None = None  # Select top K most confident
    select_top_percent: float = 0.2  # Select top 20% most confident

    # Balancing
    balance_classes: bool = True  # Ensure balanced pseudo-labels
    min_samples_per_class: int = 10


class BERTPseudoLabeler:
    """
    BERT-specific pseudo-labeling with advanced confidence scoring.
    """

    def __init__(self, config: PseudoLabelingConfig):
        """
        Initialize pseudo-labeler.

        Args:
            config: Pseudo-labeling configuration
        """
        self.config = config
        self.confidence_scorer = BERTConfidenceScorer(
            use_attention_confidence=config.use_attention_confidence
        )
        self.iteration = 0

    def generate_pseudo_labels(
        self,
        model: Model,
        unlabeled_loader: DataLoader,
        return_soft_labels: bool = False,
    ) -> tuple[list[dict[str, Any]], mx.array]:
        """
        Generate pseudo-labels for unlabeled data.

        Args:
            model: Trained BERT model
            unlabeled_loader: DataLoader for unlabeled data
            return_soft_labels: Whether to return soft labels

        Returns:
            Tuple of (pseudo-labeled samples, confidence scores)
        """
        logger.info("Generating pseudo-labels")

        model.eval()

        all_samples = []
        all_predictions = []
        all_confidences = []
        all_logits = []

        for batch_idx, batch in enumerate(unlabeled_loader):
            with mx.no_grad():
                # Get model outputs
                outputs = model(batch)

                logits = outputs.get("logits")
                attention_weights = outputs.get("attention_weights")

                # Compute confidence
                confidence = self.confidence_scorer.compute_confidence(
                    logits=logits, attention_weights=attention_weights
                )

                # Get predictions
                if return_soft_labels:
                    predictions = mx.softmax(logits, axis=-1)
                else:
                    predictions = mx.argmax(logits, axis=-1)

                # Store results
                all_predictions.append(predictions)
                all_confidences.append(confidence)
                all_logits.append(logits)

                # Store samples (without moving to CPU yet)
                for i in range(len(batch["input_ids"])):
                    sample = {
                        "input_ids": batch["input_ids"][i],
                        "attention_mask": batch["attention_mask"][i],
                        "batch_idx": batch_idx,
                        "sample_idx": i,
                    }
                    all_samples.append(sample)

        # Concatenate results
        all_predictions = mx.concatenate(all_predictions, axis=0)
        all_confidences = mx.concatenate(all_confidences, axis=0)
        all_logits = mx.concatenate(all_logits, axis=0)

        # Select confident samples
        selected_indices = self._select_confident_samples(
            confidences=all_confidences, predictions=all_predictions, logits=all_logits
        )

        # Create pseudo-labeled dataset
        pseudo_labeled_samples = []

        for idx in selected_indices:
            idx = int(idx)
            sample = all_samples[idx].copy()

            if return_soft_labels:
                sample["labels"] = all_predictions[idx]  # Soft labels
            else:
                sample["labels"] = int(all_predictions[idx].item())  # Hard label

            sample["confidence"] = float(all_confidences[idx].item())
            sample["pseudo_label"] = True

            pseudo_labeled_samples.append(sample)

        logger.info(
            f"Selected {len(pseudo_labeled_samples)} confident samples "
            f"out of {len(all_samples)} total"
        )

        return pseudo_labeled_samples, all_confidences

    def iterative_pseudo_labeling(
        self,
        model: Model,
        labeled_loader: DataLoader,
        unlabeled_loader: DataLoader,
        trainer_fn: Any,
    ) -> list[Model]:
        """
        Perform iterative pseudo-labeling.

        Args:
            model: Initial trained model
            labeled_loader: Labeled training data
            unlabeled_loader: Unlabeled data
            trainer_fn: Function to train model

        Returns:
            List of models from each iteration
        """
        models = [model]
        current_threshold = self.config.confidence_threshold

        for iteration in range(self.config.num_iterations):
            logger.info(
                f"Pseudo-labeling iteration {iteration + 1}/{self.config.num_iterations}"
            )
            self.iteration = iteration

            # Generate pseudo-labels
            pseudo_samples, confidences = self.generate_pseudo_labels(
                model=models[-1], unlabeled_loader=unlabeled_loader
            )

            if not pseudo_samples:
                logger.warning("No confident samples found, stopping iterations")
                break

            # Create augmented dataset
            # This is simplified - in practice you'd merge with labeled data
            augmented_loader = self._create_augmented_loader(
                labeled_loader, pseudo_samples
            )

            # Train new model
            new_model = trainer_fn(augmented_loader)
            models.append(new_model)

            # Lower threshold for next iteration
            current_threshold = max(
                0.7,  # Minimum threshold
                current_threshold - self.config.confidence_increase_per_iter,
            )
            self.config.confidence_threshold = current_threshold

            logger.info(
                f"Iteration {iteration + 1} complete. "
                f"New threshold: {current_threshold:.3f}"
            )

        return models

    def _select_confident_samples(
        self, confidences: mx.array, predictions: mx.array, logits: mx.array
    ) -> mx.array:
        """
        Select samples based on confidence and other criteria.

        Args:
            confidences: Confidence scores
            predictions: Predicted labels
            logits: Raw logits

        Returns:
            Indices of selected samples
        """
        # Start with confidence threshold
        confident_mask = confidences >= self._get_current_threshold()
        confident_indices = mx.where(confident_mask)[0]

        if len(confident_indices) == 0:
            return mx.array([])

        # Apply top-K or top-percent selection
        if self.config.select_top_k:
            # Get top K most confident
            k = min(self.config.select_top_k, len(confident_indices))
            top_k_indices = mx.argsort(confidences)[-k:]
            confident_indices = mx.intersect1d(confident_indices, top_k_indices)

        elif self.config.select_top_percent:
            # Get top percent
            n_select = int(len(confidences) * self.config.select_top_percent)
            top_indices = mx.argsort(confidences)[-n_select:]
            confident_indices = mx.intersect1d(confident_indices, top_indices)

        # Balance classes if needed
        if self.config.balance_classes and len(confident_indices) > 0:
            confident_indices = self._balance_class_selection(
                confident_indices, predictions, confidences
            )

        return confident_indices

    def _balance_class_selection(
        self, indices: mx.array, predictions: mx.array, confidences: mx.array
    ) -> mx.array:
        """
        Balance selected samples across classes.

        Args:
            indices: Candidate indices
            predictions: Predicted labels
            confidences: Confidence scores

        Returns:
            Balanced indices
        """
        # Get predictions for selected indices
        selected_predictions = predictions[indices]

        # Count per class
        unique_classes, counts = mx.unique(selected_predictions, return_counts=True)

        # Find minimum count
        min_count = mx.min(counts)

        # If too imbalanced, subsample majority classes
        if min_count < self.config.min_samples_per_class and len(unique_classes) > 1:
            balanced_indices = []

            for class_id in unique_classes:
                # Get indices for this class
                class_mask = selected_predictions == class_id
                class_indices = indices[mx.where(class_mask)[0]]

                # Limit to reasonable multiple of minority class
                max_samples = max(
                    int(min_count * 2),  # At most 2x minority
                    self.config.min_samples_per_class,
                )

                if len(class_indices) > max_samples:
                    # Select most confident within class
                    class_confidences = confidences[class_indices]
                    top_in_class = mx.argsort(class_confidences)[-max_samples:]
                    class_indices = class_indices[top_in_class]

                balanced_indices.append(class_indices)

            # Concatenate balanced indices
            indices = mx.concatenate(balanced_indices)

        return indices

    def _get_current_threshold(self) -> float:
        """Get current confidence threshold based on iteration."""
        base_threshold = self.config.confidence_threshold
        reduction = self.iteration * self.config.confidence_increase_per_iter
        return max(0.7, base_threshold - reduction)

    def _create_augmented_loader(
        self, labeled_loader: DataLoader, pseudo_samples: list[dict[str, Any]]
    ) -> DataLoader:
        """
        Create data loader with labeled and pseudo-labeled data.

        This is a simplified version - in practice you'd properly
        merge datasets and create a new loader.
        """
        # TODO: Implement proper dataset merging
        logger.info(
            f"Created augmented dataset with {len(pseudo_samples)} pseudo-labels"
        )
        return labeled_loader


class EnsemblePseudoLabeler(BERTPseudoLabeler):
    """
    Pseudo-labeling using ensemble of BERT models.

    Uses disagreement between models as additional confidence signal.
    """

    def __init__(self, config: PseudoLabelingConfig, models: list[Model]):
        """
        Initialize ensemble pseudo-labeler.

        Args:
            config: Pseudo-labeling configuration
            models: List of trained models
        """
        super().__init__(config)
        self.models = models

    def generate_pseudo_labels(
        self,
        model: Model,  # Ignored, uses ensemble
        unlabeled_loader: DataLoader,
        return_soft_labels: bool = False,
    ) -> tuple[list[dict[str, Any]], mx.array]:
        """
        Generate pseudo-labels using ensemble.

        Overrides parent method to use ensemble predictions.
        """
        logger.info(f"Generating pseudo-labels with {len(self.models)}-model ensemble")

        # Get predictions from each model
        all_model_predictions = []
        all_model_logits = []
        all_model_confidences = []

        for model_idx, model in enumerate(self.models):
            logger.info(
                f"Getting predictions from model {model_idx + 1}/{len(self.models)}"
            )

            model.eval()
            model_predictions = []
            model_logits = []
            model_confidences = []

            for batch in unlabeled_loader:
                with mx.no_grad():
                    outputs = model(batch)
                    logits = outputs["logits"]

                    # Compute confidence for this model
                    confidence = self.confidence_scorer.compute_confidence(
                        logits=logits,
                        attention_weights=outputs.get("attention_weights"),
                    )

                    model_predictions.append(mx.argmax(logits, axis=-1))
                    model_logits.append(logits)
                    model_confidences.append(confidence)

            all_model_predictions.append(mx.concatenate(model_predictions))
            all_model_logits.append(mx.concatenate(model_logits))
            all_model_confidences.append(mx.concatenate(model_confidences))

        # Stack predictions from all models
        stacked_predictions = mx.stack(all_model_predictions)  # [n_models, n_samples]
        stacked_logits = mx.stack(all_model_logits)  # [n_models, n_samples, n_classes]
        stacked_confidences = mx.stack(all_model_confidences)  # [n_models, n_samples]

        # Compute ensemble statistics
        ensemble_predictions = mx.mode(stacked_predictions, axis=0)  # Majority vote
        ensemble_logits = mx.mean(stacked_logits, axis=0)  # Average logits

        # Compute ensemble confidence
        # 1. Average confidence across models
        avg_confidence = mx.mean(stacked_confidences, axis=0)

        # 2. Agreement score (how many models agree)
        agreement_scores = []
        for i in range(stacked_predictions.shape[1]):
            predictions_i = stacked_predictions[:, i]
            unique, counts = mx.unique(predictions_i, return_counts=True)
            max_agreement = mx.max(counts) / len(self.models)
            agreement_scores.append(max_agreement)
        agreement_score = mx.stack(agreement_scores)

        # 3. Prediction variance (low variance = high confidence)
        logit_variance = mx.var(stacked_logits, axis=0)
        avg_variance = mx.mean(logit_variance, axis=-1)
        variance_confidence = 1.0 / (1.0 + avg_variance)

        # Combine confidence signals
        ensemble_confidence = (
            0.4 * avg_confidence + 0.4 * agreement_score + 0.2 * variance_confidence
        )

        # Now continue with parent's selection logic
        # but using ensemble predictions and confidence
        return self._create_pseudo_labeled_samples(
            ensemble_predictions,
            ensemble_confidence,
            ensemble_logits,
            unlabeled_loader,
            return_soft_labels,
        )
