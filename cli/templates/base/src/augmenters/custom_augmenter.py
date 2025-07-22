"""Example custom data augmentation implementation.

This example shows how to create custom augmenters for domain-specific data augmentation.
"""

from typing import Any, Dict, List
import random

from k_bert.plugins import AugmenterPlugin, PluginMetadata, register_component


@register_component
class DomainSpecificAugmenter(AugmenterPlugin):
    """Domain-specific text augmentation example.
    
    This augmenter demonstrates techniques like:
    - Synonym replacement
    - Random insertion
    - Random swap
    - Domain-specific transformations
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the augmenter.
        
        Args:
            config: Configuration with keys:
                - augment_prob: Probability of augmentation (default: 0.5)
                - synonym_prob: Probability of synonym replacement (default: 0.3)
                - insert_prob: Probability of insertion (default: 0.2)
                - swap_prob: Probability of swap (default: 0.2)
                - max_changes: Maximum number of changes (default: 3)
        """
        super().__init__(config)
        
        self.augment_prob = self.config.get("augment_prob", 0.5)
        self.synonym_prob = self.config.get("synonym_prob", 0.3)
        self.insert_prob = self.config.get("insert_prob", 0.2)
        self.swap_prob = self.config.get("swap_prob", 0.2)
        self.max_changes = self.config.get("max_changes", 3)
        
        # Example synonym dictionary (expand for real use)
        self.synonyms = {
            "good": ["great", "excellent", "fine", "wonderful"],
            "bad": ["poor", "terrible", "awful", "horrible"],
            "big": ["large", "huge", "enormous", "massive"],
            "small": ["tiny", "little", "miniature", "compact"],
            # Add domain-specific synonyms
        }
    
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="DomainSpecificAugmenter",
            version="1.0.0",
            description="Domain-specific text augmentation with multiple techniques",
            author="Your Name",
            tags=["augmentation", "text", "nlp"],
        )
    
    def augment(
        self,
        data: Dict[str, Any],
        training: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Apply augmentation to a data sample.
        
        Args:
            data: Input data dictionary with 'text' field
            training: Whether in training mode
            
        Returns:
            Augmented data dictionary
        """
        # Only augment during training
        if not training:
            return data
        
        # Random decision to augment
        if random.random() > self.augment_prob:
            return data
        
        # Copy data to avoid modifying original
        augmented_data = data.copy()
        
        # Apply augmentation to text field
        if "text" in data:
            text = data["text"]
            augmented_text = self._augment_text(text)
            augmented_data["text"] = augmented_text
        
        # You can also augment other fields
        # For example, numerical features with noise
        for key, value in data.items():
            if key.startswith("feature_") and isinstance(value, (int, float)):
                # Add small noise to numerical features
                noise = random.gauss(0, 0.01)
                augmented_data[key] = value * (1 + noise)
        
        return augmented_data
    
    def _augment_text(self, text: str) -> str:
        """Apply text augmentation techniques."""
        words = text.split()
        
        if len(words) == 0:
            return text
        
        # Apply multiple augmentation techniques
        num_changes = random.randint(1, min(self.max_changes, len(words)))
        
        for _ in range(num_changes):
            technique = random.choices(
                ["synonym", "insert", "swap"],
                weights=[self.synonym_prob, self.insert_prob, self.swap_prob]
            )[0]
            
            if technique == "synonym":
                words = self._synonym_replacement(words)
            elif technique == "insert":
                words = self._random_insertion(words)
            elif technique == "swap":
                words = self._random_swap(words)
        
        return " ".join(words)
    
    def _synonym_replacement(self, words: List[str]) -> List[str]:
        """Replace random word with synonym."""
        if not words:
            return words
        
        # Find words that have synonyms
        replaceable = [
            (i, word) for i, word in enumerate(words)
            if word.lower() in self.synonyms
        ]
        
        if replaceable:
            idx, word = random.choice(replaceable)
            synonym = random.choice(self.synonyms[word.lower()])
            
            # Preserve capitalization
            if word[0].isupper():
                synonym = synonym.capitalize()
            
            words[idx] = synonym
        
        return words
    
    def _random_insertion(self, words: List[str]) -> List[str]:
        """Insert a random word."""
        if not words:
            return words
        
        # Simple insertion of common words
        insert_words = ["the", "a", "very", "quite", "really"]
        insert_word = random.choice(insert_words)
        insert_pos = random.randint(0, len(words))
        
        words.insert(insert_pos, insert_word)
        return words
    
    def _random_swap(self, words: List[str]) -> List[str]:
        """Swap two random words."""
        if len(words) < 2:
            return words
        
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return words
    
    def get_augmentation_params(self) -> Dict[str, Any]:
        """Get current augmentation parameters."""
        return {
            "augment_prob": self.augment_prob,
            "synonym_prob": self.synonym_prob,
            "insert_prob": self.insert_prob,
            "swap_prob": self.swap_prob,
            "max_changes": self.max_changes,
            "num_synonyms": len(self.synonyms),
        }


@register_component(name="tabular_augmenter")
class TabularDataAugmenter(AugmenterPlugin):
    """Augmenter specifically for tabular data.
    
    This demonstrates augmentation for structured/tabular data
    that will be converted to text for BERT.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize tabular augmenter."""
        super().__init__(config)
        
        self.numerical_noise = self.config.get("numerical_noise", 0.05)
        self.categorical_swap_prob = self.config.get("categorical_swap_prob", 0.1)
        self.missing_value_prob = self.config.get("missing_value_prob", 0.05)
    
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="TabularDataAugmenter",
            version="1.0.0",
            description="Augmentation for tabular data",
            tags=["augmentation", "tabular", "structured"],
        )
    
    def augment(
        self,
        data: Dict[str, Any],
        training: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Augment tabular data."""
        if not training:
            return data
        
        augmented = data.copy()
        
        # Get feature metadata if provided
        feature_types = kwargs.get("feature_types", {})
        
        for key, value in data.items():
            if key in ["text", "label", "labels", "target"]:
                continue
            
            feature_type = feature_types.get(key, "unknown")
            
            if feature_type == "numerical" or isinstance(value, (int, float)):
                # Add Gaussian noise to numerical features
                if random.random() < 0.8:  # 80% chance
                    noise = random.gauss(0, self.numerical_noise)
                    augmented[key] = value * (1 + noise)
                
            elif feature_type == "categorical" or isinstance(value, str):
                # Occasionally swap categorical values
                if random.random() < self.categorical_swap_prob:
                    # In real implementation, swap with another valid value
                    augmented[key] = f"AUG_{value}"
            
            # Occasionally introduce missing values
            if random.random() < self.missing_value_prob:
                augmented[key] = None
        
        return augmented
    
    def get_augmentation_params(self) -> Dict[str, Any]:
        """Get augmentation parameters."""
        return {
            "numerical_noise": self.numerical_noise,
            "categorical_swap_prob": self.categorical_swap_prob,
            "missing_value_prob": self.missing_value_prob,
        }