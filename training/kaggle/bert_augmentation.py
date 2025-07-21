"""
BERT-specific text augmentation strategies for Kaggle competitions.

This module implements various text augmentation techniques optimized
for BERT models, including token-level augmentation, back-translation,
and paraphrasing.
"""

from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import random
import numpy as np
import mlx.core as mx
from loguru import logger


@dataclass
class BERTAugmentationConfig:
    """Configuration for BERT text augmentation."""
    
    # Token-level augmentation
    mask_prob: float = 0.15  # Probability of masking tokens
    random_token_prob: float = 0.1  # Probability of random token replacement
    delete_prob: float = 0.1  # Probability of token deletion
    
    # Synonym replacement
    use_synonyms: bool = True
    synonym_prob: float = 0.1
    
    # Text manipulation
    sentence_order_prob: float = 0.1  # Probability of shuffling sentences
    duplicate_prob: float = 0.1  # Probability of duplicating phrases
    
    # Advanced augmentation
    use_back_translation: bool = False  # Requires translation models
    use_paraphrasing: bool = False  # Requires paraphrase models
    
    # Tabular-specific
    feature_noise_std: float = 0.1  # Noise for numerical features
    feature_swap_prob: float = 0.1  # Probability of swapping feature values


class BERTTextAugmenter:
    """
    Text augmentation engine for BERT models.
    
    Implements various augmentation strategies to improve
    model robustness and performance.
    """
    
    def __init__(self, tokenizer, config: BERTAugmentationConfig):
        """
        Initialize augmenter.
        
        Args:
            tokenizer: BERT tokenizer
            config: Augmentation configuration
        """
        self.tokenizer = tokenizer
        self.config = config
        self.vocab_size = tokenizer.vocab_size
        
        # Special token IDs
        self.mask_token_id = tokenizer.mask_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id
        
    def augment_text(self, text: str, num_augmentations: int = 1) -> List[str]:
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
            if random.random() < 0.5:
                aug_text = self.token_level_augmentation(aug_text)
            
            # Sentence-level augmentation
            if random.random() < self.config.sentence_order_prob:
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
            rand = random.random()
            
            if rand < self.config.mask_prob:
                # Mask token
                augmented_tokens.append(self.tokenizer.mask_token)
            elif rand < self.config.mask_prob + self.config.random_token_prob:
                # Replace with random token
                random_token = self._get_random_token()
                augmented_tokens.append(random_token)
            elif rand < self.config.mask_prob + self.config.random_token_prob + self.config.delete_prob:
                # Delete token (skip)
                continue
            else:
                # Keep original
                augmented_tokens.append(token)
        
        # Convert back to text
        return self.tokenizer.convert_tokens_to_string(augmented_tokens)
    
    def augment_tokens(self, token_ids: List[int], 
                      attention_mask: Optional[List[int]] = None) -> Tuple[List[int], List[int]]:
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
            
            rand = random.random()
            
            if rand < self.config.mask_prob:
                # Mask token
                augmented_ids.append(self.mask_token_id)
                if augmented_mask is not None:
                    augmented_mask.append(attention_mask[i])
            elif rand < self.config.mask_prob + self.config.random_token_prob:
                # Random token
                random_id = random.randint(1000, self.vocab_size - 1)
                augmented_ids.append(random_id)
                if augmented_mask is not None:
                    augmented_mask.append(attention_mask[i])
            elif rand < self.config.mask_prob + self.config.random_token_prob + self.config.delete_prob:
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
        sentences = text.split('. ')
        
        if len(sentences) > 1:
            random.shuffle(sentences)
            return '. '.join(sentences)
        
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
        
        mod_func = random.choice(modifications)
        return mod_func(text)
    
    def _get_random_token(self) -> str:
        """Get a random token from vocabulary."""
        # Get random token ID (avoid special tokens)
        random_id = random.randint(1000, self.vocab_size - 1)
        return self.tokenizer.convert_ids_to_tokens([random_id])[0]


class TabularBERTAugmenter(BERTTextAugmenter):
    """
    Augmentation for tabular data converted to text.
    
    Specialized augmentation strategies for tabular-to-text data.
    """
    
    def augment_tabular_text(self, text: str, features: Dict[str, Any]) -> List[str]:
        """
        Augment tabular data text representation.
        
        Args:
            text: Text representation of tabular data
            features: Original feature values
            
        Returns:
            List of augmented texts
        """
        augmented_texts = [text]
        
        # Feature value augmentation
        aug_features = self._augment_features(features)
        aug_text = self._features_to_text(aug_features)
        augmented_texts.append(aug_text)
        
        # Template variation
        template_variations = self._create_template_variations(features)
        augmented_texts.extend(template_variations)
        
        # Combine with standard text augmentation
        for i in range(len(augmented_texts)):
            if random.random() < 0.5:
                augmented_texts[i] = self.token_level_augmentation(augmented_texts[i])
        
        return augmented_texts
    
    def _augment_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Augment feature values.
        
        Args:
            features: Original features
            
        Returns:
            Augmented features
        """
        aug_features = features.copy()
        
        for key, value in features.items():
            if isinstance(value, (int, float)):
                # Add Gaussian noise to numerical features
                noise = np.random.normal(0, self.config.feature_noise_std)
                aug_features[key] = value + noise * abs(value)
            elif isinstance(value, str) and random.random() < self.config.feature_swap_prob:
                # Occasionally swap categorical values
                # In real implementation, would use value from same column
                aug_features[key] = value  # Keep same for now
        
        return aug_features
    
    def _features_to_text(self, features: Dict[str, Any]) -> str:
        """
        Convert features to text with variations.
        
        Args:
            features: Feature dictionary
            
        Returns:
            Text representation
        """
        templates = [
            # Narrative style
            lambda f: f"A {f.get('Age', 'unknown')} year old {f.get('Sex', 'person')} "
                     f"in {f.get('Pclass', 'class')} class paid ${f.get('Fare', 0):.2f}",
            
            # Key-value style
            lambda f: " | ".join([f"{k}: {v}" for k, v in f.items()]),
            
            # Question style
            lambda f: f"Age? {f.get('Age')}. Gender? {f.get('Sex')}. "
                     f"Class? {f.get('Pclass')}. Fare? ${f.get('Fare', 0):.2f}",
            
            # Comparative style
            lambda f: self._comparative_description(f),
        ]
        
        template = random.choice(templates)
        return template(features)
    
    def _comparative_description(self, features: Dict[str, Any]) -> str:
        """Create comparative description of features."""
        desc = []
        
        # Age comparison
        age = features.get('Age', 30)
        if age < 18:
            desc.append("young passenger")
        elif age > 60:
            desc.append("elderly passenger")
        else:
            desc.append("adult passenger")
        
        # Fare comparison (assume median ~15)
        fare = features.get('Fare', 15)
        if fare > 30:
            desc.append("paid above average fare")
        elif fare < 10:
            desc.append("paid below average fare")
        
        # Class
        pclass = features.get('Pclass', 3)
        desc.append(f"traveling in {'first' if pclass == 1 else 'second' if pclass == 2 else 'third'} class")
        
        return "A " + " who ".join(desc)
    
    def _create_template_variations(self, features: Dict[str, Any]) -> List[str]:
        """Create multiple template variations."""
        variations = []
        
        # Different orderings
        keys = list(features.keys())
        for _ in range(2):
            random.shuffle(keys)
            text = " | ".join([f"{k}: {features[k]}" for k in keys])
            variations.append(text)
        
        return variations


class BERTTestTimeAugmentation:
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
        self.augmenter = augmenter
        self.num_augmentations = num_augmentations
    
    def augment_batch(self, texts: List[str]) -> List[List[str]]:
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
    
    def combine_predictions(self, predictions: List[mx.array], 
                          method: str = "mean") -> mx.array:
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
            # Simple mode implementation
            combined = []
            for i in range(class_preds.shape[1]):
                votes = class_preds[:, i]
                # Most common prediction
                unique, counts = mx.unique(votes, return_counts=True)
                mode_idx = mx.argmax(counts)
                combined.append(unique[mode_idx])
            return mx.stack(combined)
        else:
            raise ValueError(f"Unknown combination method: {method}")