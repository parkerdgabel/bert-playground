"""
Text augmentation strategies for BERT models.
"""

import mlx.core as mx

from .base import BaseAugmenter
from .config import BERTAugmentationConfig, TextAugmentationConfig


class BERTTextAugmenter(BaseAugmenter):
    """
    Text augmentation engine for BERT models.

    Implements various augmentation strategies to improve
    model robustness and performance.
    """

    def __init__(self, tokenizer, config=None):
        """
        Initialize augmenter.

        Args:
            tokenizer: BERT tokenizer
            config: Augmentation configuration (BERTAugmentationConfig or TextAugmentationConfig)
        """
        # Handle different config types
        if config is None:
            config = TextAugmentationConfig()
        elif isinstance(config, BERTAugmentationConfig):
            # Convert legacy config
            text_config = TextAugmentationConfig(
                mask_prob=config.mask_prob,
                random_token_prob=config.random_token_prob,
                delete_prob=config.delete_prob,
                synonym_prob=config.synonym_prob,
                sentence_shuffle_prob=config.sentence_order_prob,
            )
            config = text_config

        super().__init__(seed=42)  # Default seed
        self.tokenizer = tokenizer
        self.config = config

        # Get tokenizer properties safely
        self.vocab_size = getattr(tokenizer, "vocab_size", 30000)

        # Special token IDs
        self.mask_token_id = tokenizer.mask_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id

    def augment(self, data: str, **kwargs) -> str:
        """
        Augment a single text sample.

        Args:
            data: Input text
            **kwargs: Additional arguments

        Returns:
            Augmented text
        """
        return self.augment_text(data, num_augmentations=0)[0]

    def augment_text(self, text: str, num_augmentations: int = 1) -> list[str]:
        """
        Generate augmented versions of text.

        Args:
            text: Original text
            num_augmentations: Number of augmented versions

        Returns:
            List of augmented texts
        """
        augmented_texts = [text]  # Include original

        for _ in range(num_augmentations):
            # Apply random augmentation
            aug_text = text

            # Token-level augmentation
            if mx.random.uniform() < 0.5:
                aug_text = self.token_level_augmentation(aug_text)

            # Sentence-level augmentation
            if mx.random.uniform() < self.config.sentence_shuffle_prob:
                aug_text = self.shuffle_sentences(aug_text)

            # Add noise to make it different
            if aug_text == text:
                aug_text = self.add_minimal_noise(aug_text)

            augmented_texts.append(aug_text)

        return augmented_texts

    def token_level_augmentation(self, text: str) -> str:
        """
        Apply token-level augmentation (masking, replacement, deletion).

        Args:
            text: Input text

        Returns:
            Augmented text
        """
        # Tokenize
        tokens = self.tokenizer.tokenize(text)

        augmented_tokens = []
        for token in tokens:
            rand = float(mx.random.uniform().item())

            if rand < self.config.mask_prob:
                # Mask token
                augmented_tokens.append(self.tokenizer.mask_token)
            elif rand < self.config.mask_prob + self.config.random_token_prob:
                # Replace with random token
                random_token = self._get_random_token()
                augmented_tokens.append(random_token)
            elif (
                rand
                < self.config.mask_prob
                + self.config.random_token_prob
                + self.config.delete_prob
            ):
                # Delete token (skip)
                continue
            else:
                # Keep original
                augmented_tokens.append(token)

        # Convert back to text
        return self.tokenizer.convert_tokens_to_string(augmented_tokens)

    def augment_tokens(
        self, token_ids: list[int], attention_mask: list[int] | None = None
    ) -> tuple[list[int], list[int]]:
        """
        Augment tokenized input directly.

        Args:
            token_ids: Input token IDs
            attention_mask: Attention mask

        Returns:
            Augmented token IDs and attention mask
        """
        augmented_ids = []
        augmented_mask = [] if attention_mask else None

        for i, token_id in enumerate(token_ids):
            # Skip special tokens
            if token_id in [self.cls_token_id, self.sep_token_id, self.pad_token_id]:
                augmented_ids.append(token_id)
                if augmented_mask is not None:
                    augmented_mask.append(attention_mask[i])
                continue

            rand = float(mx.random.uniform().item())

            if rand < self.config.mask_prob:
                # Mask token
                augmented_ids.append(self.mask_token_id)
                if augmented_mask is not None:
                    augmented_mask.append(attention_mask[i])
            elif rand < self.config.mask_prob + self.config.random_token_prob:
                # Random token
                random_id = int(mx.random.uniform(1000, self.vocab_size).item())
                augmented_ids.append(random_id)
                if augmented_mask is not None:
                    augmented_mask.append(attention_mask[i])
            elif (
                rand
                < self.config.mask_prob
                + self.config.random_token_prob
                + self.config.delete_prob
            ):
                # Delete token
                continue
            else:
                # Keep original
                augmented_ids.append(token_id)
                if augmented_mask is not None:
                    augmented_mask.append(attention_mask[i])

        return augmented_ids, augmented_mask

    def shuffle_sentences(self, text: str) -> str:
        """
        Shuffle sentence order in text.

        Args:
            text: Input text

        Returns:
            Text with shuffled sentences
        """
        # Simple sentence splitting
        sentences = text.split(". ")

        if len(sentences) > 1:
            # MLX doesn't have shuffle, so we implement it
            indices = mx.random.permutation(len(sentences))
            shuffled = [sentences[int(i)] for i in indices]
            return ". ".join(shuffled)

        return text

    def add_minimal_noise(self, text: str) -> str:
        """
        Add minimal noise to ensure augmented text is different.

        Args:
            text: Input text

        Returns:
            Slightly modified text
        """
        # Add a space or punctuation variation
        modifications = [
            lambda t: t + " ",
            lambda t: " " + t,
            lambda t: t.replace(".", ". "),
            lambda t: t.replace(",", ", "),
        ]

        idx = int(mx.random.uniform(0, len(modifications)).item())
        mod_func = modifications[idx]
        return mod_func(text)

    def _get_random_token(self) -> str:
        """Get a random token from vocabulary."""
        # Get random token ID (avoid special tokens)
        random_id = int(mx.random.uniform(1000, self.vocab_size).item())
        return self.tokenizer.convert_ids_to_tokens([random_id])[0]
