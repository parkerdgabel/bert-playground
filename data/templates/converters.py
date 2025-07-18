"""Text converters for tabular data with BERT optimization.

This module provides high-level converters that combine templates with
BERT-specific optimizations for maximum model performance.
"""

from typing import Any, Dict, List, Optional, Union

import pandas as pd
from loguru import logger

from ..core.base import CompetitionType, DatasetSpec
from .engine import TextTemplateEngine, TemplateConfig
from .templates import (
    CompetitionTextTemplate,
    get_template_for_competition,
    get_template_for_type,
    suggest_template_from_columns,
)


class TabularTextConverter:
    """High-level converter for tabular data to BERT-optimized text.
    
    This class provides an easy-to-use interface for converting tabular
    datasets into text representations optimized for BERT models.
    """
    
    def __init__(
        self,
        competition_type: Optional[CompetitionType] = None,
        competition_name: Optional[str] = None,
        template_config: Optional[TemplateConfig] = None,
        custom_template: Optional[Union[str, CompetitionTextTemplate]] = None,
    ):
        """Initialize the tabular text converter.
        
        Args:
            competition_type: Type of competition
            competition_name: Name of competition (for template lookup)
            template_config: Template configuration
            custom_template: Custom template string or CompetitionTextTemplate
        """
        self.competition_type = competition_type
        self.competition_name = competition_name
        self.custom_template = custom_template
        
        # Initialize template engine
        self.config = template_config or TemplateConfig()
        self.engine = TextTemplateEngine(self.config)
        
        # Template selection
        self.selected_template: Optional[CompetitionTextTemplate] = None
        self._select_template()
        
        self.logger = logger.bind(component="TabularTextConverter")
        
    def _select_template(self) -> None:
        """Select the best template based on available information."""
        if isinstance(self.custom_template, CompetitionTextTemplate):
            self.selected_template = self.custom_template
            return
            
        # Try to get template by competition name
        if self.competition_name:
            template = get_template_for_competition(self.competition_name)
            if template:
                self.selected_template = template
                self.logger.info(f"Selected template for competition: {self.competition_name}")
                return
                
        # Try to get template by competition type
        if self.competition_type:
            template = get_template_for_type(self.competition_type)
            if template:
                self.selected_template = template
                self.logger.info(f"Selected template for type: {self.competition_type.value}")
                return
                
        self.logger.info("No specific template found, will use generic conversion")
    
    def fit(self, df: pd.DataFrame, target_column: Optional[str] = None) -> "TabularTextConverter":
        """Fit the converter to a dataset.
        
        Args:
            df: Training DataFrame
            target_column: Target column name
            
        Returns:
            Self for method chaining
        """
        # Analyze dataset structure
        self.target_column = target_column
        self.columns = list(df.columns)
        
        if target_column:
            self.feature_columns = [col for col in self.columns if col != target_column]
        else:
            self.feature_columns = self.columns
            
        # Analyze column types
        self.text_columns = []
        self.categorical_columns = []
        self.numerical_columns = []
        
        for col in self.feature_columns:
            dtype = df[col].dtype
            
            if dtype == 'object':
                # Distinguish between text and categorical
                unique_ratio = df[col].nunique() / len(df)
                avg_length = df[col].astype(str).str.len().mean()
                
                if unique_ratio > 0.8 or avg_length > 50:
                    self.text_columns.append(col)
                else:
                    self.categorical_columns.append(col)
            elif dtype in ['int64', 'int32', 'float64', 'float32']:
                # Check if it's categorical based on unique values
                unique_count = df[col].nunique()
                if unique_count <= 20 and dtype in ['int64', 'int32']:
                    self.categorical_columns.append(col)
                else:
                    self.numerical_columns.append(col)
                    
        # Try to suggest template from columns if not already selected
        if not self.selected_template:
            suggested = suggest_template_from_columns(self.columns)
            if suggested:
                self.selected_template = suggested
                self.logger.info(f"Suggested template: {suggested.description}")
                
        # Auto-detect competition type if not provided
        if not self.competition_type and self.target_column:
            self.competition_type = self._detect_competition_type(df, self.target_column)
            
        self.logger.info(
            f"Fitted converter: {len(self.text_columns)} text, "
            f"{len(self.categorical_columns)} categorical, "
            f"{len(self.numerical_columns)} numerical columns"
        )
        
        return self
    
    def transform(self, df: pd.DataFrame, max_samples: Optional[int] = None) -> List[str]:
        """Transform DataFrame to list of text representations.
        
        Args:
            df: DataFrame to transform
            max_samples: Maximum number of samples to transform
            
        Returns:
            List of text representations
        """
        if not hasattr(self, 'feature_columns'):
            raise ValueError("Converter must be fitted before transforming")
            
        # Use selected template if available
        if self.selected_template and isinstance(self.custom_template, str):
            template_string = self.custom_template
        elif self.selected_template:
            template_string = self.selected_template.template_string
        else:
            template_string = None
            
        # Convert using engine
        texts = self.engine.convert_dataframe(
            df=df,
            competition_type=self.competition_type or CompetitionType.UNKNOWN,
            text_columns=self.text_columns,
            categorical_columns=self.categorical_columns,
            numerical_columns=self.numerical_columns,
            target_column=self.target_column,
            custom_template=template_string,
            max_samples=max_samples,
        )
        
        return texts
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> List[str]:
        """Fit converter and transform DataFrame in one step.
        
        Args:
            df: DataFrame to fit and transform
            target_column: Target column name
            max_samples: Maximum number of samples to transform
            
        Returns:
            List of text representations
        """
        return self.fit(df, target_column).transform(df, max_samples)
    
    def _detect_competition_type(self, df: pd.DataFrame, target_column: str) -> CompetitionType:
        """Auto-detect competition type from target column.
        
        Args:
            df: DataFrame
            target_column: Target column name
            
        Returns:
            Detected CompetitionType
        """
        if target_column not in df.columns:
            return CompetitionType.UNKNOWN
            
        target_data = df[target_column]
        unique_values = target_data.nunique()
        
        if unique_values == 2:
            return CompetitionType.BINARY_CLASSIFICATION
        elif 3 <= unique_values <= 20 and target_data.dtype in ['int64', 'int32', 'object']:
            return CompetitionType.MULTICLASS_CLASSIFICATION
        elif target_data.dtype in ['float64', 'float32']:
            return CompetitionType.REGRESSION
        else:
            return CompetitionType.UNKNOWN
    
    def get_conversion_info(self) -> Dict[str, Any]:
        """Get information about the conversion setup.
        
        Returns:
            Dictionary with conversion information
        """
        info = {
            'competition_type': self.competition_type.value if self.competition_type else None,
            'competition_name': self.competition_name,
            'template_used': self.selected_template.description if self.selected_template else "Generic",
            'text_columns': getattr(self, 'text_columns', []),
            'categorical_columns': getattr(self, 'categorical_columns', []),
            'numerical_columns': getattr(self, 'numerical_columns', []),
            'target_column': getattr(self, 'target_column', None),
            'config': {
                'max_length': self.config.max_length,
                'include_feature_names': self.config.include_feature_names,
                'use_special_tokens': self.config.use_special_tokens,
            }
        }
        
        return info


class BERTTextConverter:
    """BERT-specific text converter with tokenization integration.
    
    This class extends TabularTextConverter with BERT tokenization
    and optimization specifically for the models in our BERT module.
    """
    
    def __init__(
        self,
        dataset_spec: DatasetSpec,
        tokenizer=None,
        template_config: Optional[TemplateConfig] = None,
        custom_template: Optional[Union[str, CompetitionTextTemplate]] = None,
    ):
        """Initialize BERT text converter.
        
        Args:
            dataset_spec: Dataset specification
            tokenizer: BERT tokenizer (optional)
            template_config: Template configuration
            custom_template: Custom template
        """
        self.dataset_spec = dataset_spec
        self.tokenizer = tokenizer
        
        # Configure template for BERT optimization
        bert_config = template_config or TemplateConfig()
        bert_config.max_length = dataset_spec.recommended_max_length
        bert_config.use_special_tokens = True
        
        # Initialize tabular converter
        self.tabular_converter = TabularTextConverter(
            competition_type=dataset_spec.competition_type,
            competition_name=dataset_spec.competition_name,
            template_config=bert_config,
            custom_template=custom_template,
        )
        
        self.logger = logger.bind(component="BERTTextConverter")
        
    def fit(self, df: pd.DataFrame) -> "BERTTextConverter":
        """Fit converter to dataset.
        
        Args:
            df: Training DataFrame
            
        Returns:
            Self for method chaining
        """
        self.tabular_converter.fit(df, self.dataset_spec.target_column)
        return self
    
    def transform_to_text(self, df: pd.DataFrame, max_samples: Optional[int] = None) -> List[str]:
        """Transform DataFrame to text representations.
        
        Args:
            df: DataFrame to transform
            max_samples: Maximum number of samples
            
        Returns:
            List of text representations
        """
        return self.tabular_converter.transform(df, max_samples)
    
    def transform_to_tokens(
        self,
        df: pd.DataFrame,
        max_samples: Optional[int] = None,
    ) -> Dict[str, List[Any]]:
        """Transform DataFrame to BERT tokens.
        
        Args:
            df: DataFrame to transform
            max_samples: Maximum number of samples
            
        Returns:
            Dictionary with tokenized data
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer required for token transformation")
            
        # Get text representations
        texts = self.transform_to_text(df, max_samples)
        
        # Tokenize texts
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.dataset_spec.recommended_max_length,
            return_tensors="np"
        )
        
        return {
            'input_ids': encodings['input_ids'].tolist(),
            'attention_mask': encodings['attention_mask'].tolist(),
            'texts': texts,
        }
    
    def fit_transform_to_text(
        self,
        df: pd.DataFrame,
        max_samples: Optional[int] = None,
    ) -> List[str]:
        """Fit and transform to text in one step.
        
        Args:
            df: DataFrame to fit and transform
            max_samples: Maximum number of samples
            
        Returns:
            List of text representations
        """
        return self.fit(df).transform_to_text(df, max_samples)
    
    def fit_transform_to_tokens(
        self,
        df: pd.DataFrame,
        max_samples: Optional[int] = None,
    ) -> Dict[str, List[Any]]:
        """Fit and transform to tokens in one step.
        
        Args:
            df: DataFrame to fit and transform
            max_samples: Maximum number of samples
            
        Returns:
            Dictionary with tokenized data
        """
        return self.fit(df).transform_to_tokens(df, max_samples)
    
    def get_bert_optimization_info(self) -> Dict[str, Any]:
        """Get BERT-specific optimization information.
        
        Returns:
            Dictionary with BERT optimization info
        """
        base_info = self.tabular_converter.get_conversion_info()
        
        bert_info = {
            **base_info,
            'dataset_spec': {
                'recommended_batch_size': self.dataset_spec.recommended_batch_size,
                'recommended_max_length': self.dataset_spec.recommended_max_length,
                'use_attention_mask': self.dataset_spec.use_attention_mask,
                'tokenizer_backend': self.dataset_spec.tokenizer_backend,
            },
            'tokenizer_available': self.tokenizer is not None,
        }
        
        return bert_info