"""Example custom feature extraction implementations.

This shows how to create feature extractors that transform raw data
into features suitable for BERT models.
"""

from typing import Any, Dict, List
import re
from datetime import datetime

from k_bert.plugins import FeatureExtractorPlugin, PluginMetadata, register_component


@register_component
class TextStatisticsExtractor(FeatureExtractorPlugin):
    """Extract statistical features from text.
    
    This extractor computes various text statistics that can be
    useful as additional features alongside BERT embeddings.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the feature extractor."""
        super().__init__(config)
        
        self.compute_readability = self.config.get("compute_readability", True)
        self.compute_sentiment_markers = self.config.get("compute_sentiment_markers", True)
    
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="TextStatisticsExtractor",
            version="1.0.0",
            description="Extract statistical features from text",
            author="Your Name",
            tags=["features", "text", "statistics"],
        )
    
    def extract_features(
        self,
        data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Extract text statistics features.
        
        Args:
            data: Input data with 'text' field
            
        Returns:
            Dictionary with original data plus extracted features
        """
        features = data.copy()
        
        if "text" not in data:
            return features
        
        text = data["text"]
        
        # Basic statistics
        features["text_length"] = len(text)
        features["word_count"] = len(text.split())
        features["sentence_count"] = len(re.split(r'[.!?]+', text))
        features["avg_word_length"] = (
            sum(len(word) for word in text.split()) / max(len(text.split()), 1)
        )
        
        # Character type counts
        features["uppercase_count"] = sum(1 for c in text if c.isupper())
        features["digit_count"] = sum(1 for c in text if c.isdigit())
        features["punctuation_count"] = sum(1 for c in text if c in ".,!?;:'-\"")
        
        # Readability metrics
        if self.compute_readability:
            features.update(self._compute_readability(text))
        
        # Sentiment markers
        if self.compute_sentiment_markers:
            features.update(self._compute_sentiment_markers(text))
        
        return features
    
    def _compute_readability(self, text: str) -> Dict[str, float]:
        """Compute simple readability metrics."""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        # Remove empty sentences
        sentences = [s for s in sentences if s.strip()]
        
        if not words or not sentences:
            return {
                "avg_words_per_sentence": 0,
                "complex_word_ratio": 0,
            }
        
        # Average words per sentence
        avg_words_per_sentence = len(words) / len(sentences)
        
        # Complex words (3+ syllables, approximated by length)
        complex_words = sum(1 for word in words if len(word) > 6)
        complex_word_ratio = complex_words / len(words)
        
        return {
            "avg_words_per_sentence": avg_words_per_sentence,
            "complex_word_ratio": complex_word_ratio,
        }
    
    def _compute_sentiment_markers(self, text: str) -> Dict[str, int]:
        """Count sentiment marker words."""
        text_lower = text.lower()
        
        # Simple sentiment word lists (expand for real use)
        positive_words = [
            "good", "great", "excellent", "amazing", "wonderful",
            "fantastic", "love", "best", "happy", "positive"
        ]
        negative_words = [
            "bad", "terrible", "awful", "hate", "worst", "poor",
            "negative", "disappointing", "horrible", "wrong"
        ]
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        return {
            "positive_word_count": positive_count,
            "negative_word_count": negative_count,
            "sentiment_polarity": positive_count - negative_count,
        }
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names this extractor produces."""
        features = [
            "text_length", "word_count", "sentence_count", "avg_word_length",
            "uppercase_count", "digit_count", "punctuation_count"
        ]
        
        if self.compute_readability:
            features.extend(["avg_words_per_sentence", "complex_word_ratio"])
        
        if self.compute_sentiment_markers:
            features.extend([
                "positive_word_count", "negative_word_count", "sentiment_polarity"
            ])
        
        return features


@register_component
class TemporalFeatureExtractor(FeatureExtractorPlugin):
    """Extract temporal features from data.
    
    Useful for competitions where time-based patterns matter.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize temporal feature extractor."""
        super().__init__(config)
        
        self.date_columns = self.config.get("date_columns", [])
        self.reference_date = self.config.get("reference_date", None)
    
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="TemporalFeatureExtractor",
            version="1.0.0",
            description="Extract temporal and time-based features",
            tags=["features", "temporal", "datetime"],
        )
    
    def extract_features(
        self,
        data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Extract temporal features."""
        features = data.copy()
        
        # Process each date column
        for col in self.date_columns:
            if col in data:
                date_features = self._extract_date_features(data[col], col)
                features.update(date_features)
        
        # Auto-detect date columns if not specified
        if not self.date_columns:
            for key, value in data.items():
                if self._is_date(value):
                    date_features = self._extract_date_features(value, key)
                    features.update(date_features)
        
        return features
    
    def _is_date(self, value: Any) -> bool:
        """Check if value is a date."""
        if isinstance(value, (datetime, str)):
            if isinstance(value, str):
                # Try common date patterns
                patterns = [
                    r'\d{4}-\d{2}-\d{2}',
                    r'\d{2}/\d{2}/\d{4}',
                    r'\d{2}-\d{2}-\d{4}',
                ]
                return any(re.match(pattern, value) for pattern in patterns)
            return True
        return False
    
    def _extract_date_features(
        self,
        date_value: Any,
        column_name: str
    ) -> Dict[str, Any]:
        """Extract features from a date value."""
        features = {}
        
        # Parse date
        if isinstance(date_value, str):
            try:
                # Try common formats
                for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y']:
                    try:
                        date_obj = datetime.strptime(date_value, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    return features
            except:
                return features
        else:
            date_obj = date_value
        
        # Extract components
        features[f"{column_name}_year"] = date_obj.year
        features[f"{column_name}_month"] = date_obj.month
        features[f"{column_name}_day"] = date_obj.day
        features[f"{column_name}_dayofweek"] = date_obj.weekday()
        features[f"{column_name}_quarter"] = (date_obj.month - 1) // 3 + 1
        features[f"{column_name}_is_weekend"] = int(date_obj.weekday() >= 5)
        
        # Days since reference
        if self.reference_date:
            ref_date = datetime.strptime(self.reference_date, '%Y-%m-%d')
            features[f"{column_name}_days_since_ref"] = (date_obj - ref_date).days
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        if not self.date_columns:
            return []  # Dynamic based on data
        
        features = []
        for col in self.date_columns:
            features.extend([
                f"{col}_year", f"{col}_month", f"{col}_day",
                f"{col}_dayofweek", f"{col}_quarter", f"{col}_is_weekend"
            ])
            if self.reference_date:
                features.append(f"{col}_days_since_ref")
        
        return features