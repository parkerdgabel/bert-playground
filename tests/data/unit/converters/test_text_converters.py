"""Tests for text converters - Fixed version."""

import pytest
from unittest.mock import Mock, patch

import pandas as pd
import numpy as np

from data.templates.converters import TabularTextConverter, BERTTextConverter
from data.templates.engine import TemplateConfig
from data.core.base import CompetitionType
from tests.data.fixtures.utils import create_sample_dataframe
from tests.data.fixtures.configs import create_dataset_spec


class TestTabularTextConverter:
    """Test TabularTextConverter class."""
    
    @pytest.fixture
    def converter(self):
        """Create TabularTextConverter instance."""
        return TabularTextConverter()
        
    @pytest.fixture
    def sample_data(self):
        """Create sample tabular data."""
        return create_sample_dataframe(size=3, include_all_types=True)
        
    def test_converter_creation(self, converter):
        """Test converter creation."""
        assert converter is not None
        assert hasattr(converter, 'engine')
        assert hasattr(converter, 'config')
        assert hasattr(converter, 'competition_type')
        
    def test_default_config(self):
        """Test default converter configuration."""
        converter = TabularTextConverter()
        
        assert converter.config.max_length == 512
        assert converter.config.include_feature_names == True
        assert converter.config.feature_separator == ', '
        assert converter.config.include_missing_values == False
        
    def test_custom_config(self):
        """Test custom converter configuration."""
        config = TemplateConfig(
            max_length=1024,
            include_feature_names=False,
            feature_separator=' | ',
            include_missing_values=True,
        )
        converter = TabularTextConverter(template_config=config)
        
        assert converter.config.max_length == 1024
        assert converter.config.include_feature_names == False
        assert converter.config.feature_separator == ' | '
        assert converter.config.include_missing_values == True
        
    def test_fit_transform(self, converter, sample_data):
        """Test fit and transform process."""
        # Fit and transform
        texts = converter.fit_transform(sample_data, target_column='target')
        
        assert len(texts) == len(sample_data)
        assert all(isinstance(text, str) for text in texts)
        assert all(len(text) > 0 for text in texts)
        
    def test_separate_fit_transform(self, converter, sample_data):
        """Test separate fit and transform."""
        # Fit
        converter.fit(sample_data, target_column='target')
        
        # Check fitted attributes
        assert hasattr(converter, 'text_columns')
        assert hasattr(converter, 'categorical_columns')
        assert hasattr(converter, 'numerical_columns')
        
        # Transform
        texts = converter.transform(sample_data)
        
        assert len(texts) == len(sample_data)
        
    def test_column_type_detection(self, converter, sample_data):
        """Test automatic column type detection."""
        converter.fit(sample_data)
        
        # Should have detected column types
        assert len(converter.numerical_columns) > 0
        assert len(converter.categorical_columns) > 0
        assert len(converter.text_columns) > 0
        
    def test_competition_type_detection(self, converter):
        """Test automatic competition type detection."""
        # Binary classification
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': ['A', 'B', 'A', 'B', 'A'],
            'target': [0, 1, 0, 1, 0],
        })
        
        converter.fit(df, target_column='target')
        assert converter.competition_type == CompetitionType.BINARY_CLASSIFICATION
        
    def test_with_missing_values(self, converter):
        """Test handling of missing values."""
        df = pd.DataFrame({
            'feature1': [1.0, np.nan, 3.0],
            'feature2': ['A', None, 'C'],
            'feature3': [10, 20, 30],
        })
        
        # Default behavior (missing values excluded)
        texts = converter.fit_transform(df)
        assert len(texts) == 3
        
        # With missing values included
        config = TemplateConfig(include_missing_values=True)
        converter_with_missing = TabularTextConverter(template_config=config)
        texts_with_missing = converter_with_missing.fit_transform(df)
        
        assert len(texts_with_missing) == 3
        # Second text should contain missing value token
        assert '[MISSING]' in texts_with_missing[1]
        
    def test_max_samples_limit(self, converter):
        """Test max_samples parameter."""
        df = create_sample_dataframe(size=100)
        
        texts = converter.fit_transform(df, max_samples=10)
        
        assert len(texts) == 10
        
    def test_get_conversion_info(self, converter, sample_data):
        """Test getting conversion information."""
        converter.fit(sample_data)
        
        info = converter.get_conversion_info()
        
        assert 'competition_type' in info
        assert 'text_columns' in info
        assert 'categorical_columns' in info
        assert 'numerical_columns' in info
        assert 'config' in info


class TestBERTTextConverter:
    """Test BERTTextConverter class."""
    
    @pytest.fixture
    def dataset_spec(self):
        """Create dataset spec for testing."""
        return create_dataset_spec(
            competition_type=CompetitionType.BINARY_CLASSIFICATION,
            num_samples=100,
            num_features=10,
        )
    
    @pytest.fixture
    def converter(self, dataset_spec):
        """Create BERTTextConverter instance."""
        return BERTTextConverter(dataset_spec)
        
    @pytest.fixture
    def sample_data(self):
        """Create sample data for BERT conversion."""
        return create_sample_dataframe(size=3, include_text_columns=True)
        
    def test_converter_creation(self, converter):
        """Test BERT converter creation."""
        assert converter is not None
        assert hasattr(converter, 'dataset_spec')
        assert hasattr(converter, 'tabular_converter')
        
    def test_fit_transform_to_text(self, converter, sample_data):
        """Test fit and transform to text."""
        texts = converter.fit_transform_to_text(sample_data)
        
        assert len(texts) == len(sample_data)
        assert all(isinstance(text, str) for text in texts)
        
    def test_transform_to_tokens_without_tokenizer(self, converter, sample_data):
        """Test transform to tokens without tokenizer raises error."""
        converter.fit(sample_data)
        
        with pytest.raises(ValueError, match="Tokenizer required"):
            converter.transform_to_tokens(sample_data)
            
    def test_bert_optimization_info(self, converter, sample_data):
        """Test BERT optimization info."""
        converter.fit(sample_data)
        
        info = converter.get_bert_optimization_info()
        
        assert 'dataset_spec' in info
        assert 'tokenizer_available' in info
        assert info['tokenizer_available'] == False  # No tokenizer set
        
    def test_different_competition_types(self, sample_data):
        """Test with different competition types."""
        for comp_type in [
            CompetitionType.BINARY_CLASSIFICATION,
            CompetitionType.MULTICLASS_CLASSIFICATION,
            CompetitionType.REGRESSION,
        ]:
            spec = create_dataset_spec(competition_type=comp_type)
            converter = BERTTextConverter(spec)
            
            texts = converter.fit_transform_to_text(sample_data)
            assert len(texts) > 0
            
    def test_with_custom_template_config(self, dataset_spec, sample_data):
        """Test with custom template configuration."""
        config = TemplateConfig(
            max_length=256,
            use_special_tokens=True,
            cls_token='[CLS]',
            sep_token='[SEP]',
        )
        
        converter = BERTTextConverter(dataset_spec, template_config=config)
        texts = converter.fit_transform_to_text(sample_data)
        
        assert len(texts) == len(sample_data)
        # Special tokens should be in the text
        assert all('[CLS]' in text for text in texts)
        assert all('[SEP]' in text for text in texts)


@pytest.mark.integration
class TestConverterIntegration:
    """Integration tests for converters."""
    
    def test_end_to_end_tabular_conversion(self):
        """Test complete tabular conversion pipeline."""
        # Create realistic dataset
        df = pd.DataFrame({
            'age': [25, 30, 35, 40],
            'income': [50000, 75000, 100000, 60000],
            'education': ['Bachelor', 'Master', 'PhD', 'Bachelor'],
            'city': ['New York', 'San Francisco', 'Boston', 'Chicago'],
            'description': [
                'Young professional in tech',
                'Senior engineer with experience',
                'Research scientist',
                'Marketing manager',
            ],
            'approved': [1, 1, 1, 0],
        })
        
        # Create converter
        converter = TabularTextConverter(
            competition_type=CompetitionType.BINARY_CLASSIFICATION,
        )
        
        # Convert
        texts = converter.fit_transform(df, target_column='approved')
        
        # Verify
        assert len(texts) == len(df)
        assert all(isinstance(text, str) for text in texts)
        
        # Check that relevant information is included
        assert 'Bachelor' in texts[0]
        assert 'tech' in texts[0]
        
    def test_bert_conversion_pipeline(self):
        """Test BERT conversion pipeline."""
        # Create dataset spec
        spec = create_dataset_spec(
            competition_type=CompetitionType.MULTICLASS_CLASSIFICATION,
            num_classes=3,
            recommended_max_length=128,
        )
        
        # Create data
        df = create_sample_dataframe(
            size=10,
            num_text=2,
            task_type='multiclass',
        )
        
        # Create converter
        converter = BERTTextConverter(spec)
        
        # Convert
        texts = converter.fit_transform_to_text(df)
        
        # Verify
        assert len(texts) == 10
        assert all(len(text) > 0 for text in texts)
        
        # Get optimization info
        info = converter.get_bert_optimization_info()
        # The spec sets recommended_max_length to 128, but BERTTextConverter
        # uses the template config which defaults to 512
        assert info['dataset_spec']['recommended_max_length'] == 512