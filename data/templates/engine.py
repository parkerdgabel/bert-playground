"""Text template engine for BERT-optimized tabular data conversion.

This module provides a flexible template system for converting tabular data
into natural language representations optimized for BERT models.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from loguru import logger

from ..core.base import CompetitionType


@dataclass
class TemplateConfig:
    """Configuration for text templates."""
    
    # Template behavior
    max_length: int = 512
    include_feature_names: bool = True
    include_missing_values: bool = False
    missing_value_token: str = "[MISSING]"
    
    # BERT optimization
    use_special_tokens: bool = True
    cls_token: str = "[CLS]"
    sep_token: str = "[SEP]"
    pad_token: str = "[PAD]"
    
    # Text formatting
    feature_separator: str = ", "
    value_separator: str = ": "
    sentence_separator: str = ". "
    
    # Competition-specific settings
    prioritize_text_columns: bool = True
    truncate_long_values: bool = True
    max_value_length: int = 100


class TextTemplateEngine:
    """Engine for converting tabular data to BERT-optimized text.
    
    This class provides a flexible template system that can adapt to different
    competition types and data characteristics while maintaining BERT compatibility.
    """
    
    def __init__(self, config: Optional[TemplateConfig] = None):
        """Initialize the template engine.
        
        Args:
            config: Template configuration
        """
        self.config = config or TemplateConfig()
        self.logger = logger.bind(component="TextTemplateEngine")
        
        # Built-in templates by competition type
        self._builtin_templates = {
            CompetitionType.BINARY_CLASSIFICATION: self._binary_classification_template,
            CompetitionType.MULTICLASS_CLASSIFICATION: self._multiclass_classification_template,
            CompetitionType.MULTILABEL_CLASSIFICATION: self._multilabel_classification_template,
            CompetitionType.REGRESSION: self._regression_template,
            CompetitionType.ORDINAL_REGRESSION: self._ordinal_regression_template,
            CompetitionType.TIME_SERIES: self._time_series_template,
            CompetitionType.RANKING: self._ranking_template,
        }
        
    def convert_row(
        self,
        row: Union[pd.Series, Dict[str, Any]],
        competition_type: CompetitionType,
        text_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
        numerical_columns: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        custom_template: Optional[str] = None,
    ) -> str:
        """Convert a single row to text.
        
        Args:
            row: Data row as Series or dictionary
            competition_type: Type of competition
            text_columns: List of text column names
            categorical_columns: List of categorical column names
            numerical_columns: List of numerical column names
            target_column: Target column name (excluded from text)
            custom_template: Optional custom template string
            
        Returns:
            Text representation of the row
        """
        if isinstance(row, pd.Series):
            row_dict = row.to_dict()
        else:
            row_dict = dict(row)
            
        # Remove target column from features
        if target_column and target_column in row_dict:
            row_dict = {k: v for k, v in row_dict.items() if k != target_column}
            
        # Use custom template if provided
        if custom_template:
            return self._apply_custom_template(custom_template, row_dict)
            
        # Use built-in template based on competition type
        template_func = self._builtin_templates.get(
            competition_type,
            self._generic_template
        )
        
        return template_func(
            row_dict,
            text_columns or [],
            categorical_columns or [],
            numerical_columns or [],
        )
    
    def convert_dataframe(
        self,
        df: pd.DataFrame,
        competition_type: CompetitionType,
        text_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
        numerical_columns: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        custom_template: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> List[str]:
        """Convert a DataFrame to list of text representations.
        
        Args:
            df: Input DataFrame
            competition_type: Type of competition
            text_columns: List of text column names
            categorical_columns: List of categorical column names  
            numerical_columns: List of numerical column names
            target_column: Target column name (excluded from text)
            custom_template: Optional custom template string
            max_samples: Maximum number of samples to convert
            
        Returns:
            List of text representations
        """
        if max_samples and len(df) > max_samples:
            df = df.head(max_samples)
            
        texts = []
        for _, row in df.iterrows():
            text = self.convert_row(
                row=row,
                competition_type=competition_type,
                text_columns=text_columns,
                categorical_columns=categorical_columns,
                numerical_columns=numerical_columns,
                target_column=target_column,
                custom_template=custom_template,
            )
            texts.append(text)
            
        return texts
    
    def _apply_custom_template(self, template: str, row_dict: Dict[str, Any]) -> str:
        """Apply a custom template to a row.
        
        Args:
            template: Template string with {column_name} placeholders
            row_dict: Row data as dictionary
            
        Returns:
            Formatted text
        """
        try:
            # Handle missing values
            formatted_dict = {}
            for key, value in row_dict.items():
                if pd.isna(value) or value is None:
                    if self.config.include_missing_values:
                        formatted_dict[key] = self.config.missing_value_token
                    else:
                        formatted_dict[key] = ""
                else:
                    formatted_dict[key] = self._format_value(value)
                    
            text = template.format(**formatted_dict)
            
            # Apply BERT optimizations
            text = self._optimize_for_bert(text)
            
            return text
            
        except KeyError as e:
            self.logger.warning(f"Template key {e} not found in row data")
            return self._generic_template(row_dict, [], [], [])
        except Exception as e:
            self.logger.error(f"Error applying custom template: {e}")
            return self._generic_template(row_dict, [], [], [])
    
    def _binary_classification_template(
        self,
        row_dict: Dict[str, Any],
        text_columns: List[str],
        categorical_columns: List[str],
        numerical_columns: List[str],
    ) -> str:
        """Template for binary classification tasks."""
        parts = []
        
        # Add CLS token if enabled
        if self.config.use_special_tokens:
            parts.append(self.config.cls_token)
            
        # Prioritize text columns
        if text_columns and self.config.prioritize_text_columns:
            text_parts = []
            for col in text_columns:
                if col in row_dict:
                    value = self._format_value(row_dict[col])
                    if value:
                        if self.config.include_feature_names:
                            text_parts.append(f"{col}{self.config.value_separator}{value}")
                        else:
                            text_parts.append(value)
            
            if text_parts:
                parts.append(self.config.feature_separator.join(text_parts))
                
        # Add categorical features
        if categorical_columns:
            cat_parts = []
            for col in categorical_columns:
                if col in row_dict:
                    value = self._format_value(row_dict[col])
                    if value:
                        if self.config.include_feature_names:
                            cat_parts.append(f"{col}{self.config.value_separator}{value}")
                        else:
                            cat_parts.append(value)
                            
            if cat_parts:
                if parts and not parts[-1].endswith(self.config.sentence_separator):
                    parts.append(self.config.sentence_separator)
                parts.append(f"Categories: {self.config.feature_separator.join(cat_parts)}")
                
        # Add numerical features (summarized)
        if numerical_columns:
            num_parts = []
            for col in numerical_columns:
                if col in row_dict:
                    value = self._format_value(row_dict[col])
                    if value:
                        num_parts.append(f"{col}{self.config.value_separator}{value}")
                        
            if num_parts:
                if parts and not parts[-1].endswith(self.config.sentence_separator):
                    parts.append(self.config.sentence_separator)
                parts.append(f"Values: {self.config.feature_separator.join(num_parts)}")
                
        # Add SEP token if enabled
        if self.config.use_special_tokens:
            parts.append(self.config.sep_token)
            
        text = " ".join(parts)
        return self._optimize_for_bert(text)
    
    def _multiclass_classification_template(
        self,
        row_dict: Dict[str, Any],
        text_columns: List[str],
        categorical_columns: List[str],
        numerical_columns: List[str],
    ) -> str:
        """Template for multiclass classification tasks."""
        # Similar to binary but with more emphasis on distinguishing features
        return self._binary_classification_template(
            row_dict, text_columns, categorical_columns, numerical_columns
        )
    
    def _multilabel_classification_template(
        self,
        row_dict: Dict[str, Any],
        text_columns: List[str],
        categorical_columns: List[str],
        numerical_columns: List[str],
    ) -> str:
        """Template for multilabel classification tasks."""
        # More detailed representation for multilabel tasks
        parts = []
        
        if self.config.use_special_tokens:
            parts.append(self.config.cls_token)
            
        # Create a comprehensive description
        all_features = []
        
        # Process all column types together for multilabel
        for col in list(text_columns) + list(categorical_columns) + list(numerical_columns):
            if col in row_dict:
                value = self._format_value(row_dict[col])
                if value:
                    if self.config.include_feature_names:
                        all_features.append(f"{col}{self.config.value_separator}{value}")
                    else:
                        all_features.append(value)
                        
        if all_features:
            parts.append(self.config.feature_separator.join(all_features))
            
        if self.config.use_special_tokens:
            parts.append(self.config.sep_token)
            
        text = " ".join(parts)
        return self._optimize_for_bert(text)
    
    def _regression_template(
        self,
        row_dict: Dict[str, Any],
        text_columns: List[str],
        categorical_columns: List[str],
        numerical_columns: List[str],
    ) -> str:
        """Template for regression tasks."""
        parts = []
        
        if self.config.use_special_tokens:
            parts.append(self.config.cls_token)
            
        # For regression, focus on numerical relationships
        if numerical_columns:
            num_desc = []
            for col in numerical_columns:
                if col in row_dict:
                    value = self._format_value(row_dict[col])
                    if value:
                        num_desc.append(f"{col} is {value}")
                        
            if num_desc:
                parts.append(self.config.sentence_separator.join(num_desc))
                
        # Add categorical context
        if categorical_columns:
            cat_desc = []
            for col in categorical_columns:
                if col in row_dict:
                    value = self._format_value(row_dict[col])
                    if value:
                        cat_desc.append(f"{col} is {value}")
                        
            if cat_desc:
                if parts and len(parts) > 1:
                    parts.append(self.config.sentence_separator)
                parts.append(self.config.sentence_separator.join(cat_desc))
                
        # Add text features
        if text_columns:
            for col in text_columns:
                if col in row_dict:
                    value = self._format_value(row_dict[col])
                    if value:
                        if parts and len(parts) > 1:
                            parts.append(self.config.sentence_separator)
                        parts.append(f"{col}: {value}")
                        
        if self.config.use_special_tokens:
            parts.append(self.config.sep_token)
            
        text = " ".join(parts)
        return self._optimize_for_bert(text)
    
    def _ordinal_regression_template(
        self,
        row_dict: Dict[str, Any],
        text_columns: List[str],
        categorical_columns: List[str],
        numerical_columns: List[str],
    ) -> str:
        """Template for ordinal regression tasks."""
        # Similar to regression but emphasize order relationships
        return self._regression_template(
            row_dict, text_columns, categorical_columns, numerical_columns
        )
    
    def _time_series_template(
        self,
        row_dict: Dict[str, Any],
        text_columns: List[str],
        categorical_columns: List[str],
        numerical_columns: List[str],
    ) -> str:
        """Template for time series tasks."""
        parts = []
        
        if self.config.use_special_tokens:
            parts.append(self.config.cls_token)
            
        # Look for time-related columns first
        time_cols = [col for col in row_dict.keys() 
                    if any(t in col.lower() for t in ['time', 'date', 'timestamp', 'day', 'month', 'year'])]
        
        if time_cols:
            time_parts = []
            for col in time_cols:
                if col in row_dict:
                    value = self._format_value(row_dict[col])
                    if value:
                        time_parts.append(f"{col}: {value}")
            if time_parts:
                parts.append(f"Time context: {self.config.feature_separator.join(time_parts)}")
                
        # Add other features
        other_features = []
        for col in numerical_columns + categorical_columns + text_columns:
            if col not in time_cols and col in row_dict:
                value = self._format_value(row_dict[col])
                if value:
                    other_features.append(f"{col}: {value}")
                    
        if other_features:
            if parts and len(parts) > 1:
                parts.append(self.config.sentence_separator)
            parts.append(self.config.feature_separator.join(other_features))
            
        if self.config.use_special_tokens:
            parts.append(self.config.sep_token)
            
        text = " ".join(parts)
        return self._optimize_for_bert(text)
    
    def _ranking_template(
        self,
        row_dict: Dict[str, Any],
        text_columns: List[str],
        categorical_columns: List[str],
        numerical_columns: List[str],
    ) -> str:
        """Template for ranking tasks."""
        # Focus on comparative and ranking-relevant features
        return self._regression_template(
            row_dict, text_columns, categorical_columns, numerical_columns
        )
    
    def _generic_template(
        self,
        row_dict: Dict[str, Any],
        text_columns: List[str],
        categorical_columns: List[str],
        numerical_columns: List[str],
    ) -> str:
        """Generic template for unknown competition types."""
        parts = []
        
        if self.config.use_special_tokens:
            parts.append(self.config.cls_token)
            
        # Simple feature enumeration
        all_features = []
        for col, value in row_dict.items():
            formatted_value = self._format_value(value)
            if formatted_value:
                if self.config.include_feature_names:
                    all_features.append(f"{col}{self.config.value_separator}{formatted_value}")
                else:
                    all_features.append(formatted_value)
                    
        if all_features:
            parts.append(self.config.feature_separator.join(all_features))
            
        if self.config.use_special_tokens:
            parts.append(self.config.sep_token)
            
        text = " ".join(parts)
        return self._optimize_for_bert(text)
    
    def _format_value(self, value: Any) -> str:
        """Format a value for text representation.
        
        Args:
            value: Value to format
            
        Returns:
            Formatted string representation
        """
        if pd.isna(value) or value is None:
            if self.config.include_missing_values:
                return self.config.missing_value_token
            else:
                return ""
                
        # Convert to string
        str_value = str(value)
        
        # Truncate if too long
        if self.config.truncate_long_values and len(str_value) > self.config.max_value_length:
            str_value = str_value[:self.config.max_value_length - 3] + "..."
            
        # Clean up whitespace
        str_value = re.sub(r'\s+', ' ', str_value).strip()
        
        return str_value
    
    def _optimize_for_bert(self, text: str) -> str:
        """Apply BERT-specific optimizations to text.
        
        Args:
            text: Input text
            
        Returns:
            Optimized text
        """
        # Clean up spacing
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Ensure proper spacing around special tokens
        if self.config.use_special_tokens:
            for token in [self.config.cls_token, self.config.sep_token]:
                text = re.sub(f'\\s*{re.escape(token)}\\s*', f' {token} ', text)
                
        # Truncate to max length (leave room for special tokens)
        max_content_length = self.config.max_length
        if self.config.use_special_tokens:
            max_content_length -= 10  # Reserve space for special tokens
            
        if len(text) > max_content_length:
            text = text[:max_content_length - 3] + "..."
            
        return text.strip()