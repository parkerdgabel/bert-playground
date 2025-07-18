"""
Data augmentation transforms for text and tabular data.
Provides various augmentation strategies to improve model robustness.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import random
import re
import numpy as np
from loguru import logger

from .base_transforms import Transform


class TextAugmentation(Transform):
    """Base class for text augmentation."""
    
    def __init__(
        self,
        text_field: str = "text",
        probability: float = 0.5,
        seed: Optional[int] = None
    ):
        """
        Initialize text augmentation.
        
        Args:
            text_field: Field containing text
            probability: Probability of applying augmentation
            seed: Random seed
        """
        self.text_field = text_field
        self.probability = probability
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def should_apply(self) -> bool:
        """Check if augmentation should be applied."""
        return random.random() < self.probability


class SynonymReplacement(TextAugmentation):
    """Replace words with synonyms."""
    
    def __init__(
        self,
        synonym_dict: Optional[Dict[str, List[str]]] = None,
        num_replacements: int = 1,
        **kwargs
    ):
        """
        Initialize synonym replacement.
        
        Args:
            synonym_dict: Dictionary of word -> synonyms
            num_replacements: Number of words to replace
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        self.synonym_dict = synonym_dict or self._get_default_synonyms()
        self.num_replacements = num_replacements
    
    def _get_default_synonyms(self) -> Dict[str, List[str]]:
        """Get default synonym dictionary."""
        return {
            "good": ["great", "excellent", "fine", "nice"],
            "bad": ["poor", "terrible", "awful", "horrible"],
            "big": ["large", "huge", "enormous", "massive"],
            "small": ["tiny", "little", "mini", "compact"],
            "fast": ["quick", "rapid", "swift", "speedy"],
            "slow": ["sluggish", "gradual", "leisurely", "unhurried"],
        }
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply synonym replacement."""
        if self.text_field not in data or not self.should_apply():
            return data
        
        result = data.copy()
        text = data[self.text_field]
        words = text.split()
        
        # Find replaceable words
        replaceable = []
        for i, word in enumerate(words):
            word_lower = word.lower().strip('.,!?";')
            if word_lower in self.synonym_dict:
                replaceable.append((i, word_lower))
        
        # Replace random words
        if replaceable:
            num_to_replace = min(self.num_replacements, len(replaceable))
            to_replace = random.sample(replaceable, num_to_replace)
            
            for idx, word in to_replace:
                synonym = random.choice(self.synonym_dict[word])
                # Preserve original case
                if words[idx][0].isupper():
                    synonym = synonym.capitalize()
                words[idx] = synonym
        
        result[self.text_field] = " ".join(words)
        return result
    
    def __repr__(self) -> str:
        return f"SynonymReplacement(n={self.num_replacements})"


class RandomInsertion(TextAugmentation):
    """Insert random words into text."""
    
    def __init__(
        self,
        word_list: Optional[List[str]] = None,
        num_insertions: int = 1,
        **kwargs
    ):
        """
        Initialize random insertion.
        
        Args:
            word_list: List of words to insert
            num_insertions: Number of insertions
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        self.word_list = word_list or ["however", "moreover", "therefore", "also", "furthermore"]
        self.num_insertions = num_insertions
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply random insertion."""
        if self.text_field not in data or not self.should_apply():
            return data
        
        result = data.copy()
        text = data[self.text_field]
        words = text.split()
        
        for _ in range(self.num_insertions):
            if len(words) > 0:
                insert_pos = random.randint(0, len(words))
                insert_word = random.choice(self.word_list)
                words.insert(insert_pos, insert_word)
        
        result[self.text_field] = " ".join(words)
        return result
    
    def __repr__(self) -> str:
        return f"RandomInsertion(n={self.num_insertions})"


class RandomSwap(TextAugmentation):
    """Randomly swap words in text."""
    
    def __init__(self, num_swaps: int = 1, **kwargs):
        """
        Initialize random swap.
        
        Args:
            num_swaps: Number of swaps
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        self.num_swaps = num_swaps
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply random swap."""
        if self.text_field not in data or not self.should_apply():
            return data
        
        result = data.copy()
        text = data[self.text_field]
        words = text.split()
        
        if len(words) < 2:
            return result
        
        for _ in range(self.num_swaps):
            idx1 = random.randint(0, len(words) - 1)
            idx2 = random.randint(0, len(words) - 1)
            if idx1 != idx2:
                words[idx1], words[idx2] = words[idx2], words[idx1]
        
        result[self.text_field] = " ".join(words)
        return result
    
    def __repr__(self) -> str:
        return f"RandomSwap(n={self.num_swaps})"


class RandomDeletion(TextAugmentation):
    """Randomly delete words from text."""
    
    def __init__(self, deletion_prob: float = 0.1, **kwargs):
        """
        Initialize random deletion.
        
        Args:
            deletion_prob: Probability of deleting each word
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        self.deletion_prob = deletion_prob
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply random deletion."""
        if self.text_field not in data or not self.should_apply():
            return data
        
        result = data.copy()
        text = data[self.text_field]
        words = text.split()
        
        # Keep at least one word
        if len(words) <= 1:
            return result
        
        new_words = []
        for word in words:
            if random.random() > self.deletion_prob:
                new_words.append(word)
        
        # Ensure at least one word remains
        if not new_words:
            new_words = [random.choice(words)]
        
        result[self.text_field] = " ".join(new_words)
        return result
    
    def __repr__(self) -> str:
        return f"RandomDeletion(p={self.deletion_prob})"


class BackTranslation(TextAugmentation):
    """Simulate back-translation augmentation."""
    
    def __init__(
        self,
        intermediate_languages: Optional[List[str]] = None,
        variation_templates: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize back-translation.
        
        Args:
            intermediate_languages: Languages to translate through
            variation_templates: Templates for variations
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        self.intermediate_languages = intermediate_languages or ["es", "fr", "de"]
        self.variation_templates = variation_templates or [
            "{text}",
            "In other words, {text}",
            "This means that {text}",
            "Put differently, {text}",
        ]
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply back-translation simulation."""
        if self.text_field not in data or not self.should_apply():
            return data
        
        result = data.copy()
        text = data[self.text_field]
        
        # Simulate back-translation with template variation
        template = random.choice(self.variation_templates)
        result[self.text_field] = template.format(text=text)
        
        return result
    
    def __repr__(self) -> str:
        return "BackTranslation()"


class MixUp(Transform):
    """MixUp augmentation for tabular data."""
    
    def __init__(
        self,
        alpha: float = 0.2,
        features_to_mix: Optional[List[str]] = None,
        label_field: str = "label",
    ):
        """
        Initialize MixUp.
        
        Args:
            alpha: Beta distribution parameter
            features_to_mix: Features to apply mixup to
            label_field: Label field name
        """
        self.alpha = alpha
        self.features_to_mix = features_to_mix
        self.label_field = label_field
    
    def __call__(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply MixUp to a pair of samples."""
        if len(data) < 2:
            return data[0] if data else {}
        
        # Sample mixing coefficient
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Mix samples
        sample1, sample2 = data[0], data[1]
        mixed = {}
        
        features = self.features_to_mix or list(sample1.keys())
        
        for key in sample1.keys():
            if key in features and isinstance(sample1[key], (int, float, np.ndarray)):
                # Mix numerical features
                mixed[key] = lam * sample1[key] + (1 - lam) * sample2[key]
            elif key == self.label_field:
                # Mix labels for soft targets
                if isinstance(sample1[key], (list, np.ndarray)):
                    mixed[key] = lam * np.array(sample1[key]) + (1 - lam) * np.array(sample2[key])
                else:
                    # For single labels, keep the dominant one
                    mixed[key] = sample1[key] if lam > 0.5 else sample2[key]
                    mixed[f"{key}_weight"] = max(lam, 1 - lam)
            else:
                # Keep from dominant sample
                mixed[key] = sample1[key] if lam > 0.5 else sample2[key]
        
        return mixed
    
    def __repr__(self) -> str:
        return f"MixUp(alpha={self.alpha})"


class CutMix(Transform):
    """CutMix augmentation for sequence data."""
    
    def __init__(
        self,
        alpha: float = 1.0,
        sequence_fields: Optional[List[str]] = None,
    ):
        """
        Initialize CutMix.
        
        Args:
            alpha: Beta distribution parameter
            sequence_fields: Sequence fields to apply cutmix to
        """
        self.alpha = alpha
        self.sequence_fields = sequence_fields or ["input_ids", "attention_mask"]
    
    def __call__(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply CutMix to a pair of samples."""
        if len(data) < 2:
            return data[0] if data else {}
        
        sample1, sample2 = data[0], data[1]
        mixed = sample1.copy()
        
        # Sample mixing coefficient
        lam = np.random.beta(self.alpha, self.alpha)
        
        for field in self.sequence_fields:
            if field in sample1 and field in sample2:
                seq1 = sample1[field]
                seq2 = sample2[field]
                
                if isinstance(seq1, (list, np.ndarray)):
                    seq_len = len(seq1)
                    cut_len = int(seq_len * (1 - lam))
                    
                    if cut_len > 0:
                        # Random position to cut
                        start_idx = random.randint(0, seq_len - cut_len)
                        end_idx = start_idx + cut_len
                        
                        # Mix sequences
                        mixed_seq = list(seq1)
                        mixed_seq[start_idx:end_idx] = seq2[start_idx:end_idx]
                        mixed[field] = mixed_seq
        
        return mixed
    
    def __repr__(self) -> str:
        return f"CutMix(alpha={self.alpha})"


class FeatureNoise(Transform):
    """Add noise to features."""
    
    def __init__(
        self,
        noise_type: str = "gaussian",
        noise_level: float = 0.1,
        features: Optional[List[str]] = None,
        probability: float = 0.5,
    ):
        """
        Initialize feature noise.
        
        Args:
            noise_type: Type of noise ('gaussian', 'uniform', 'dropout')
            noise_level: Noise level/std
            features: Features to add noise to
            probability: Probability of applying noise
        """
        self.noise_type = noise_type
        self.noise_level = noise_level
        self.features = features
        self.probability = probability
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Add noise to features."""
        if random.random() > self.probability:
            return data
        
        result = data.copy()
        features = self.features or [k for k, v in data.items() if isinstance(v, (int, float, np.ndarray))]
        
        for feature in features:
            if feature not in result:
                continue
            
            value = result[feature]
            
            if self.noise_type == "gaussian":
                if isinstance(value, (int, float)):
                    noise = np.random.normal(0, self.noise_level)
                    result[feature] = value + noise
                elif isinstance(value, (list, np.ndarray)):
                    noise = np.random.normal(0, self.noise_level, size=len(value))
                    result[feature] = np.array(value) + noise
                    
            elif self.noise_type == "uniform":
                if isinstance(value, (int, float)):
                    noise = np.random.uniform(-self.noise_level, self.noise_level)
                    result[feature] = value + noise
                elif isinstance(value, (list, np.ndarray)):
                    noise = np.random.uniform(-self.noise_level, self.noise_level, size=len(value))
                    result[feature] = np.array(value) + noise
                    
            elif self.noise_type == "dropout":
                if isinstance(value, (list, np.ndarray)):
                    mask = np.random.random(len(value)) > self.noise_level
                    result[feature] = np.array(value) * mask
        
        return result
    
    def __repr__(self) -> str:
        return f"FeatureNoise(type={self.noise_type}, level={self.noise_level})"


# Composite augmentation strategies
class EasyDataAugmentation(Transform):
    """Easy Data Augmentation (EDA) - combines multiple text augmentations."""
    
    def __init__(
        self,
        alpha_sr: float = 0.1,  # Synonym replacement
        alpha_ri: float = 0.1,  # Random insertion
        alpha_rs: float = 0.1,  # Random swap
        alpha_rd: float = 0.1,  # Random deletion
        num_aug: int = 1,
        **kwargs
    ):
        """Initialize EDA."""
        self.augmentations = []
        
        if alpha_sr > 0:
            self.augmentations.append(
                SynonymReplacement(probability=alpha_sr, **kwargs)
            )
        if alpha_ri > 0:
            self.augmentations.append(
                RandomInsertion(probability=alpha_ri, **kwargs)
            )
        if alpha_rs > 0:
            self.augmentations.append(
                RandomSwap(probability=alpha_rs, **kwargs)
            )
        if alpha_rd > 0:
            self.augmentations.append(
                RandomDeletion(deletion_prob=alpha_rd, **kwargs)
            )
        
        self.num_aug = num_aug
    
    def __call__(self, data: Dict[str, Any]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Apply EDA augmentations."""
        if not self.augmentations:
            return data
        
        if self.num_aug == 1:
            aug = random.choice(self.augmentations)
            return aug(data)
        else:
            results = [data]
            for _ in range(self.num_aug):
                aug = random.choice(self.augmentations)
                augmented = aug(data.copy())
                results.append(augmented)
            return results
    
    def __repr__(self) -> str:
        return f"EasyDataAugmentation(num_aug={self.num_aug})"