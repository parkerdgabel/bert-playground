"""
Enhanced BERT-specific text templates for Kaggle competitions.

This module extends the base text template engine with BERT-optimized
templates that leverage BERT's understanding of natural language and
improve performance on tabular data competitions.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from loguru import logger

from ...data.templates.engine import TextTemplateEngine, TemplateConfig
from ...data.core.base import CompetitionType


@dataclass
class BERTTemplateConfig(TemplateConfig):
    """Enhanced configuration for BERT-specific templates."""
    
    # BERT-specific features
    use_semantic_templates: bool = True  # Use natural language templates
    use_comparative_descriptions: bool = True  # Compare to statistics
    use_question_answering_format: bool = False  # Q&A style templates
    
    # Feature emphasis
    emphasize_important_features: bool = True  # Use attention markers
    importance_marker: str = "[IMPORTANT]"
    
    # Numerical handling
    use_binning_descriptions: bool = True  # Convert numbers to descriptions
    bin_descriptions: Dict[str, List[Tuple[float, str]]] = field(default_factory=dict)
    
    # Template diversity for ensemble
    num_template_variations: int = 3  # Number of different templates
    
    # Domain knowledge injection
    use_domain_knowledge: bool = True
    domain_descriptions: Dict[str, Dict[Any, str]] = field(default_factory=dict)


class BERTTextTemplateEngine(TextTemplateEngine):
    """
    Enhanced text template engine optimized for BERT models.
    
    Provides advanced templating strategies that leverage BERT's
    natural language understanding capabilities.
    """
    
    def __init__(self, config: Optional[BERTTemplateConfig] = None,
                 feature_statistics: Optional[Dict[str, Dict[str, float]]] = None):
        """
        Initialize BERT template engine.
        
        Args:
            config: BERT template configuration
            feature_statistics: Statistics for each feature (mean, std, etc.)
        """
        super().__init__(config or BERTTemplateConfig())
        self.config: BERTTemplateConfig = config or BERTTemplateConfig()
        self.feature_statistics = feature_statistics or {}
        
        # Template strategies
        self.template_strategies = [
            self._narrative_template,
            self._analytical_template,
            self._comparative_template,
        ]
        
        if self.config.use_question_answering_format:
            self.template_strategies.append(self._qa_template)
    
    def convert_row_with_variations(
        self,
        row: Union[pd.Series, Dict[str, Any]],
        competition_type: CompetitionType,
        **kwargs
    ) -> List[str]:
        """
        Convert row to multiple text variations for ensemble/augmentation.
        
        Args:
            row: Data row
            competition_type: Competition type
            **kwargs: Additional arguments
            
        Returns:
            List of text variations
        """
        variations = []
        
        # Use different template strategies
        for i in range(min(self.config.num_template_variations, len(self.template_strategies))):
            strategy = self.template_strategies[i]
            text = strategy(row, competition_type, **kwargs)
            variations.append(text)
        
        # Add base template variation
        base_text = super().convert_row(row, competition_type, **kwargs)
        if base_text not in variations:
            variations.append(base_text)
        
        return variations
    
    def _narrative_template(
        self,
        row: Union[pd.Series, Dict[str, Any]],
        competition_type: CompetitionType,
        **kwargs
    ) -> str:
        """
        Create a narrative-style description of the data.
        
        Example: "This is a 25-year-old male passenger traveling in first class..."
        """
        if isinstance(row, pd.Series):
            row_dict = row.to_dict()
        else:
            row_dict = dict(row)
        
        # Remove target if present
        target_column = kwargs.get('target_column')
        if target_column and target_column in row_dict:
            row_dict.pop(target_column)
        
        parts = []
        if self.config.use_special_tokens:
            parts.append(self.config.cls_token)
        
        # Build narrative based on competition type
        if competition_type == CompetitionType.BINARY_CLASSIFICATION:
            narrative = self._build_classification_narrative(row_dict, **kwargs)
        elif competition_type in [CompetitionType.REGRESSION, CompetitionType.ORDINAL_REGRESSION]:
            narrative = self._build_regression_narrative(row_dict, **kwargs)
        else:
            narrative = self._build_generic_narrative(row_dict, **kwargs)
        
        parts.append(narrative)
        
        if self.config.use_special_tokens:
            parts.append(self.config.sep_token)
        
        return " ".join(parts)
    
    def _analytical_template(
        self,
        row: Union[pd.Series, Dict[str, Any]],
        competition_type: CompetitionType,
        **kwargs
    ) -> str:
        """
        Create an analytical description focusing on key features.
        
        Example: "[IMPORTANT] High-value features: Age=25 (young), Fare=$50 (above average)..."
        """
        if isinstance(row, pd.Series):
            row_dict = row.to_dict()
        else:
            row_dict = dict(row)
        
        # Remove target
        target_column = kwargs.get('target_column')
        if target_column and target_column in row_dict:
            row_dict.pop(target_column)
        
        parts = []
        if self.config.use_special_tokens:
            parts.append(self.config.cls_token)
        
        # Identify important features (you could use feature importance here)
        important_features = self._identify_important_features(row_dict, **kwargs)
        other_features = {k: v for k, v in row_dict.items() if k not in important_features}
        
        # Add important features with emphasis
        if important_features and self.config.emphasize_important_features:
            parts.append(self.config.importance_marker)
            important_desc = self._describe_features(important_features, use_analysis=True)
            parts.append(f"Key factors: {important_desc}")
        
        # Add other features
        if other_features:
            other_desc = self._describe_features(other_features, use_analysis=False)
            parts.append(f"Additional details: {other_desc}")
        
        if self.config.use_special_tokens:
            parts.append(self.config.sep_token)
        
        return " ".join(parts)
    
    def _comparative_template(
        self,
        row: Union[pd.Series, Dict[str, Any]],
        competition_type: CompetitionType,
        **kwargs
    ) -> str:
        """
        Create a comparative description using statistics.
        
        Example: "Age of 25 is below average (mean: 30), Fare of $50 is 2x the median..."
        """
        if isinstance(row, pd.Series):
            row_dict = row.to_dict()
        else:
            row_dict = dict(row)
        
        # Remove target
        target_column = kwargs.get('target_column')
        if target_column and target_column in row_dict:
            row_dict.pop(target_column)
        
        parts = []
        if self.config.use_special_tokens:
            parts.append(self.config.cls_token)
        
        comparative_descriptions = []
        
        for feature, value in row_dict.items():
            if pd.notna(value) and feature in self.feature_statistics:
                stats = self.feature_statistics[feature]
                comp_desc = self._create_comparative_description(feature, value, stats)
                if comp_desc:
                    comparative_descriptions.append(comp_desc)
            else:
                # Fallback to simple description
                if pd.notna(value):
                    comparative_descriptions.append(f"{feature}: {value}")
        
        parts.append(". ".join(comparative_descriptions))
        
        if self.config.use_special_tokens:
            parts.append(self.config.sep_token)
        
        return " ".join(parts)
    
    def _qa_template(
        self,
        row: Union[pd.Series, Dict[str, Any]],
        competition_type: CompetitionType,
        **kwargs
    ) -> str:
        """
        Create a question-answering style template.
        
        Example: "Q: What is the passenger's age? A: 25. Q: What class? A: First class..."
        """
        if isinstance(row, pd.Series):
            row_dict = row.to_dict()
        else:
            row_dict = dict(row)
        
        # Remove target
        target_column = kwargs.get('target_column')
        if target_column and target_column in row_dict:
            row_dict.pop(target_column)
        
        parts = []
        if self.config.use_special_tokens:
            parts.append(self.config.cls_token)
        
        qa_pairs = []
        for feature, value in row_dict.items():
            if pd.notna(value):
                question = self._create_question_for_feature(feature)
                answer = self._format_answer(value, feature)
                qa_pairs.append(f"{question} {answer}")
        
        parts.append(" ".join(qa_pairs))
        
        if self.config.use_special_tokens:
            parts.append(self.config.sep_token)
        
        return " ".join(parts)
    
    def _build_classification_narrative(self, row_dict: Dict[str, Any], **kwargs) -> str:
        """Build narrative for classification tasks."""
        # Example for Titanic dataset
        narrative_parts = []
        
        # Start with subject
        if 'Sex' in row_dict and 'Age' in row_dict:
            age_desc = self._describe_age(row_dict.get('Age'))
            sex = row_dict.get('Sex', 'person').lower()
            narrative_parts.append(f"This is a {age_desc} {sex}")
        
        # Add class information
        if 'Pclass' in row_dict:
            class_desc = self._describe_class(row_dict.get('Pclass'))
            narrative_parts.append(f"traveling in {class_desc}")
        
        # Add fare information
        if 'Fare' in row_dict:
            fare_desc = self._describe_fare(row_dict.get('Fare'))
            narrative_parts.append(f"who paid {fare_desc}")
        
        # Add family information
        family_parts = []
        if 'SibSp' in row_dict and row_dict['SibSp'] > 0:
            family_parts.append(f"{row_dict['SibSp']} sibling(s)")
        if 'Parch' in row_dict and row_dict['Parch'] > 0:
            family_parts.append(f"{row_dict['Parch']} parent(s)/child(ren)")
        
        if family_parts:
            narrative_parts.append(f"traveling with {' and '.join(family_parts)}")
        
        # Combine with proper punctuation
        if narrative_parts:
            narrative = narrative_parts[0]
            for part in narrative_parts[1:]:
                narrative += f", {part}"
            narrative += "."
        else:
            # Fallback to generic
            narrative = self._build_generic_narrative(row_dict, **kwargs)
        
        return narrative
    
    def _build_regression_narrative(self, row_dict: Dict[str, Any], **kwargs) -> str:
        """Build narrative for regression tasks."""
        descriptions = []
        
        numerical_cols = kwargs.get('numerical_columns', [])
        categorical_cols = kwargs.get('categorical_columns', [])
        
        # Describe the subject/entity
        entity_desc = "This entity has"
        
        # Add numerical features with context
        num_descs = []
        for col in numerical_cols:
            if col in row_dict and pd.notna(row_dict[col]):
                value = row_dict[col]
                if col in self.feature_statistics:
                    stats = self.feature_statistics[col]
                    context = self._get_value_context(value, stats)
                    num_descs.append(f"{col} of {value} ({context})")
                else:
                    num_descs.append(f"{col} of {value}")
        
        if num_descs:
            descriptions.append(f"{entity_desc} {', '.join(num_descs)}")
        
        # Add categorical context
        cat_descs = []
        for col in categorical_cols:
            if col in row_dict and pd.notna(row_dict[col]):
                cat_descs.append(f"{col}: {row_dict[col]}")
        
        if cat_descs:
            descriptions.append(f"Categories include {', '.join(cat_descs)}")
        
        return ". ".join(descriptions)
    
    def _build_generic_narrative(self, row_dict: Dict[str, Any], **kwargs) -> str:
        """Build generic narrative for any task."""
        features = []
        
        for key, value in row_dict.items():
            if pd.notna(value):
                features.append(f"{key} is {value}")
        
        if features:
            return "Record with " + ", ".join(features[:5])  # Limit to 5 features
        else:
            return "Record with no specified features"
    
    def _identify_important_features(self, row_dict: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Identify important features based on domain knowledge or statistics."""
        important = {}
        
        # You could use actual feature importance here
        # For now, use heuristics
        
        # High-cardinality features
        for key, value in row_dict.items():
            if pd.notna(value):
                # Check if it's an outlier
                if key in self.feature_statistics:
                    stats = self.feature_statistics[key]
                    if isinstance(value, (int, float)):
                        mean = stats.get('mean', 0)
                        std = stats.get('std', 1)
                        if abs(value - mean) > 2 * std:
                            important[key] = value
                
                # Domain-specific importance (customize per competition)
                if key in ['Fare', 'Age', 'Pclass']:  # Titanic example
                    important[key] = value
        
        return important
    
    def _describe_features(self, features: Dict[str, Any], use_analysis: bool = True) -> str:
        """Describe features with optional analysis."""
        descriptions = []
        
        for key, value in features.items():
            if pd.notna(value):
                if use_analysis and key in self.feature_statistics:
                    stats = self.feature_statistics[key]
                    context = self._get_value_context(value, stats)
                    descriptions.append(f"{key}={value} ({context})")
                else:
                    descriptions.append(f"{key}={value}")
        
        return ", ".join(descriptions)
    
    def _create_comparative_description(self, feature: str, value: Any, 
                                      stats: Dict[str, float]) -> Optional[str]:
        """Create comparative description using statistics."""
        if not isinstance(value, (int, float)):
            return f"{feature} is {value}"
        
        mean = stats.get('mean', value)
        median = stats.get('median', value)
        std = stats.get('std', 1)
        
        comparisons = []
        
        # Compare to mean
        if value > mean + std:
            comparisons.append("significantly above average")
        elif value > mean:
            comparisons.append("above average")
        elif value < mean - std:
            comparisons.append("significantly below average")
        elif value < mean:
            comparisons.append("below average")
        else:
            comparisons.append("near average")
        
        # Compare to median
        if abs(value - median) / median > 0.5:
            ratio = value / median
            comparisons.append(f"{ratio:.1f}x the median")
        
        return f"{feature} of {value} is {', '.join(comparisons)}"
    
    def _get_value_context(self, value: float, stats: Dict[str, float]) -> str:
        """Get context description for a value."""
        mean = stats.get('mean', value)
        std = stats.get('std', 1)
        
        if value > mean + 2 * std:
            return "very high"
        elif value > mean + std:
            return "high"
        elif value > mean:
            return "above average"
        elif value < mean - 2 * std:
            return "very low"
        elif value < mean - std:
            return "low"
        elif value < mean:
            return "below average"
        else:
            return "average"
    
    def _describe_age(self, age: Any) -> str:
        """Create age description."""
        if pd.isna(age):
            return "person of unknown age"
        
        age = float(age)
        if age < 1:
            return "infant"
        elif age < 13:
            return "child"
        elif age < 20:
            return "teenager"
        elif age < 30:
            return "young adult"
        elif age < 50:
            return "middle-aged"
        elif age < 70:
            return "senior"
        else:
            return "elderly"
    
    def _describe_class(self, pclass: Any) -> str:
        """Create class description."""
        if pd.isna(pclass):
            return "unknown class"
        
        class_map = {
            1: "first class",
            2: "second class",
            3: "third class"
        }
        return class_map.get(int(pclass), f"class {pclass}")
    
    def _describe_fare(self, fare: Any) -> str:
        """Create fare description."""
        if pd.isna(fare):
            return "an unknown fare"
        
        fare = float(fare)
        if fare == 0:
            return "no fare"
        elif fare < 10:
            return "a very low fare"
        elif fare < 30:
            return "a modest fare"
        elif fare < 100:
            return "a substantial fare"
        else:
            return "a very high fare"
    
    def _create_question_for_feature(self, feature: str) -> str:
        """Create natural question for feature."""
        question_map = {
            'Age': "Q: What is the age?",
            'Sex': "Q: What is the gender?",
            'Pclass': "Q: What class?",
            'Fare': "Q: What was the fare?",
            'SibSp': "Q: How many siblings/spouses?",
            'Parch': "Q: How many parents/children?",
        }
        
        return question_map.get(feature, f"Q: What is the {feature}?")
    
    def _format_answer(self, value: Any, feature: str) -> str:
        """Format answer for Q&A template."""
        if pd.isna(value):
            return "A: Unknown."
        
        # Add context for some features
        if feature == 'Age':
            desc = self._describe_age(value)
            return f"A: {value} ({desc})."
        elif feature == 'Pclass':
            desc = self._describe_class(value)
            return f"A: {desc}."
        elif feature == 'Fare':
            return f"A: ${value:.2f}."
        else:
            return f"A: {value}."


def calculate_feature_statistics(df: pd.DataFrame, 
                               exclude_columns: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
    """
    Calculate statistics for each feature in the dataframe.
    
    Args:
        df: Input dataframe
        exclude_columns: Columns to exclude (e.g., target)
        
    Returns:
        Dictionary of statistics per feature
    """
    exclude_columns = exclude_columns or []
    statistics = {}
    
    for col in df.columns:
        if col not in exclude_columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                statistics[col] = {
                    'mean': df[col].mean(),
                    'median': df[col].median(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'q25': df[col].quantile(0.25),
                    'q75': df[col].quantile(0.75),
                }
            else:
                # For categorical, store value counts
                value_counts = df[col].value_counts()
                statistics[col] = {
                    'unique_count': len(value_counts),
                    'most_common': value_counts.index[0] if len(value_counts) > 0 else None,
                    'most_common_freq': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                }
    
    return statistics