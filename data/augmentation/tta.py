"""
Test-time augmentation (TTA) for BERT models.
"""

import mlx.core as mx

from .base import BaseAugmenter
from .text import BERTTextAugmenter


class BERTTestTimeAugmentation(BaseAugmenter):
    """
    Test-time augmentation (TTA) for BERT models.

    Generates multiple augmented versions at inference time
    and combines predictions.
    """

    def __init__(self, augmenter: BERTTextAugmenter, num_augmentations: int = 5):
        """
        Initialize TTA.

        Args:
            augmenter: Text augmenter
            num_augmentations: Number of augmented versions
        """
        super().__init__()
        self.augmenter = augmenter
        self.num_augmentations = num_augmentations

    def augment(self, data: str, **kwargs) -> list[str]:
        """
        Augment a single text for TTA.

        Args:
            data: Input text
            **kwargs: Additional arguments

        Returns:
            List of augmented texts
        """
        return self.augmenter.augment_text(data, self.num_augmentations)

    def augment_batch(self, texts: list[str]) -> list[list[str]]:
        """
        Generate augmented versions for a batch of texts.

        Args:
            texts: Original texts

        Returns:
            List of augmented text lists
        """
        augmented_batch = []

        for text in texts:
            augmented = self.augmenter.augment_text(text, self.num_augmentations)
            augmented_batch.append(augmented)

        return augmented_batch

    def combine_predictions(
        self, predictions: list[mx.array], method: str = "mean"
    ) -> mx.array:
        """
        Combine predictions from augmented inputs.

        Args:
            predictions: List of prediction arrays
            method: Combination method (mean, max, vote)

        Returns:
            Combined predictions
        """
        stacked = mx.stack(predictions, axis=0)

        if method == "mean":
            return mx.mean(stacked, axis=0)
        elif method == "max":
            return mx.max(stacked, axis=0)
        elif method == "vote":
            # For classification - majority vote
            class_preds = mx.argmax(stacked, axis=-1)
            # Simple mode implementation using MLX
            # For each sample, find the most common prediction
            num_samples = class_preds.shape[1]
            combined = []
            for i in range(num_samples):
                votes = class_preds[:, i]
                # Count occurrences of each class
                num_classes = int(mx.max(votes).item()) + 1
                counts = mx.zeros([num_classes])
                for vote in votes:
                    idx = int(vote.item())
                    counts = mx.where(mx.arange(num_classes) == idx, counts + 1, counts)
                # Get most common
                mode = mx.argmax(counts)
                combined.append(mode)
            return mx.stack(combined)
        else:
            raise ValueError(f"Unknown combination method: {method}")

    def predict_with_tta(
        self, model, dataloader, num_augmentations: int = None
    ) -> mx.array:
        """
        Generate predictions using test-time augmentation.

        Args:
            model: Model to use for predictions
            dataloader: Data loader
            num_augmentations: Number of augmentations (uses default if None)

        Returns:
            Combined predictions
        """
        if num_augmentations is None:
            num_augmentations = self.num_augmentations

        all_predictions = []

        # Generate predictions for each augmented version
        for aug_idx in range(num_augmentations):
            predictions = []
            model.eval()

            for batch in dataloader:
                # Augment the batch (if text data is available)
                # This assumes the dataloader provides augmentable data
                # In practice, augmentation might happen at dataset level

                with mx.no_grad():
                    outputs = model(batch)
                    if "logits" in outputs:
                        batch_preds = outputs["logits"]
                    else:
                        batch_preds = outputs.get("predictions", outputs)
                    predictions.append(batch_preds)

            # Concatenate all batch predictions
            epoch_preds = mx.concatenate(predictions, axis=0)
            all_predictions.append(epoch_preds)

        # Combine predictions
        return self.combine_predictions(all_predictions, method="mean")
