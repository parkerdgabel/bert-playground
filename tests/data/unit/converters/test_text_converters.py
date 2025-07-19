"""Tests for text converters."""

import pytest
from unittest.mock import Mock, patch

import pandas as pd
import numpy as np

from data.templates.converters import TabularTextConverter, BERTTextConverter
from tests.data.fixtures.utils import (
    create_sample_dataframe,
    create_test_row,
)


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
        assert hasattr(converter, 'max_text_length')
        assert hasattr(converter, 'include_column_names')
        assert hasattr(converter, 'separator')
        
    def test_default_settings(self):
        """Test default converter settings."""
        converter = TabularTextConverter()
        
        assert converter.max_text_length == 512
        assert converter.include_column_names == True
        assert converter.separator == ', '
        assert converter.handle_missing == 'skip'
        
    def test_custom_settings(self):
        """Test custom converter settings."""
        converter = TabularTextConverter(
            max_text_length=1024,
            include_column_names=False,
            separator=' | ',
            handle_missing='placeholder',
        )
        
        assert converter.max_text_length == 1024
        assert converter.include_column_names == False
        assert converter.separator == ' | '
        assert converter.handle_missing == 'placeholder'
        
    def test_convert_simple_row(self, converter, sample_data):
        """Test converting simple row to text."""
        row = sample_data.iloc[0]
        text = converter.convert_row(row)
        
        assert isinstance(text, str)
        assert len(text) > 0
        assert len(text) <= converter.max_text_length
        
        # Check content is included
        for col in sample_data.columns:
            if pd.notna(row[col]):
                assert str(row[col]) in text
        
    def test_convert_row_with_column_names(self, sample_data):
        """Test converting row with column names included."""
        converter = TabularTextConverter(include_column_names=True)
        
        row = sample_data.iloc[0]
        text = converter.convert_row(row)
        
        # Column names should be in the text
        for col in sample_data.columns:
            if pd.notna(row[col]):
                assert f"{col}:" in text or f"{col} =" in text
        
    def test_convert_row_without_column_names(self, sample_data):
        """Test converting row without column names."""
        converter = TabularTextConverter(include_column_names=False)
        
        row = sample_data.iloc[0]
        text = converter.convert_row(row)
        
        # Column names should not be in the text
        for col in sample_data.columns:
            assert f"{col}:" not in text.lower()
            assert f"{col} =" not in text.lower()
        
    def test_convert_row_custom_separator(self, sample_data):
        """Test converting row with custom separator."""
        converter = TabularTextConverter(separator=' | ')
        
        row = sample_data.iloc[0]
        text = converter.convert_row(row)
        
        assert ' | ' in text
        
    def test_convert_batch(self, converter, sample_data):
        """Test converting batch of rows."""
        texts = converter.convert_batch(sample_data)
        
        assert len(texts) == len(sample_data)
        assert all(isinstance(text, str) for text in texts)
        assert all(len(text) > 0 for text in texts)
        
        # Check that different rows produce different texts
        assert len(set(texts)) == len(texts)  # All unique
        
    def test_handle_missing_skip(self):
        """Test handling missing values with skip strategy."""
        converter = TabularTextConverter(handle_missing='skip')
        
        row = create_test_row(include_missing=True)
        text = converter.convert_row(row)
        
        # Missing values should not appear
        assert 'None' not in text
        assert 'nan' not in text.lower()
        assert 'null' not in text.lower()
        
    def test_handle_missing_placeholder(self):
        """Test handling missing values with placeholder strategy."""
        converter = TabularTextConverter(handle_missing='placeholder')
        
        row = create_test_row(include_missing=True)
        text = converter.convert_row(row)
        
        # Should have placeholder for missing values
        assert '[missing]' in text or 'unknown' in text.lower()
        
    def test_handle_missing_empty(self):
        """Test handling missing values with empty strategy."""
        converter = TabularTextConverter(handle_missing='empty')
        
        row = create_test_row(include_missing=True)
        text = converter.convert_row(row)
        
        # Should handle empty values gracefully
        assert text is not None
        assert len(text) > 0
        
    def test_text_truncation(self, sample_data):
        """Test text truncation for long content."""
        converter = TabularTextConverter(max_text_length=50)
        
        # Create row with long content
        row = sample_data.iloc[0].copy()
        row['description'] = 'x' * 100  # Long text
        
        text = converter.convert_row(row)
        
        assert len(text) <= 50
        assert text.endswith('...') or len(text) == 50
        
    def test_special_characters_handling(self):
        """Test handling of special characters."""
        converter = TabularTextConverter()
        
        row = pd.Series({
            'text': 'Hello, "world"!',
            'value': 123.45,
            'special': 'Line1\nLine2\tTab',
        })
        
        text = converter.convert_row(row)
        
        # Should handle special characters
        assert 'Hello' in text
        assert 'world' in text
        assert '123.45' in text
        
    def test_numeric_formatting(self):
        """Test formatting of numeric values."""
        converter = TabularTextConverter()
        
        row = pd.Series({
            'integer': 42,
            'float': 3.14159,
            'scientific': 1.23e-4,
            'percentage': 0.95,
        })
        
        text = converter.convert_row(row)
        
        assert '42' in text
        assert '3.14' in text  # Should be formatted reasonably
        
    def test_categorical_handling(self, sample_data):
        """Test handling of categorical columns."""
        converter = TabularTextConverter()
        
        # Convert a column to categorical
        sample_data['category'] = pd.Categorical(['A', 'B', 'A'])
        
        texts = converter.convert_batch(sample_data)
        
        assert 'A' in texts[0]
        assert 'B' in texts[1]
        
    def test_datetime_handling(self):
        """Test handling of datetime values."""
        converter = TabularTextConverter()
        
        row = pd.Series({
            'date': pd.Timestamp('2023-01-15'),
            'time': pd.Timestamp('2023-01-15 14:30:00'),
            'name': 'Test',
        })
        
        text = converter.convert_row(row)
        
        assert '2023' in text
        assert 'Test' in text
        
    def test_list_column_handling(self):
        """Test handling of columns containing lists."""
        converter = TabularTextConverter()
        
        row = pd.Series({
            'tags': ['python', 'machine-learning', 'nlp'],
            'scores': [0.9, 0.8, 0.7],
            'name': 'Test',
        })
        
        text = converter.convert_row(row)
        
        # Lists should be converted to readable format
        assert 'python' in text
        assert 'machine-learning' in text
        
    def test_custom_formatter(self):
        """Test custom column formatter."""
        def currency_formatter(value):
            return f"${value:,.2f}"
            
        converter = TabularTextConverter(
            column_formatters={'price': currency_formatter}
        )
        
        row = pd.Series({
            'product': 'Widget',
            'price': 1234.56,
        })
        
        text = converter.convert_row(row)
        
        assert '$1,234.56' in text


class TestBERTTextConverter:
    """Test BERTTextConverter class."""
    
    @pytest.fixture
    def converter(self):
        """Create BERTTextConverter instance."""
        return BERTTextConverter()
        
    @pytest.fixture
    def sample_data(self):
        """Create sample data for BERT conversion."""
        return create_sample_dataframe(size=3, include_text_columns=True)
        
    def test_converter_creation(self, converter):
        """Test BERT converter creation."""
        assert converter is not None
        assert hasattr(converter, 'template_style')
        assert hasattr(converter, 'add_special_tokens')
        
    def test_bert_specific_formatting(self, converter, sample_data):
        """Test BERT-specific text formatting."""
        row = sample_data.iloc[0]
        text = converter.convert_row(row)
        
        # Should have BERT-friendly format
        assert isinstance(text, str)
        
        # Check for sentence structure
        assert '.' in text or '!' in text or '?' in text
        
    def test_template_styles(self, sample_data):
        """Test different template styles."""
        styles = ['descriptive', 'question_answer', 'factual', 'narrative']
        
        row = sample_data.iloc[0]
        texts = []
        
        for style in styles:
            converter = BERTTextConverter(template_style=style)
            text = converter.convert_row(row)
            texts.append(text)
            
        # Different styles should produce different texts
        assert len(set(texts)) > 1
        
    def test_special_token_handling(self, sample_data):
        """Test handling of special tokens."""
        converter = BERTTextConverter(add_special_tokens=True)
        
        row = sample_data.iloc[0]
        text = converter.convert_row(row)
        
        # Might include markers for downstream processing
        # (actual tokens added during tokenization)
        assert isinstance(text, str)
        
    def test_question_answer_format(self):
        """Test question-answer formatting."""
        converter = BERTTextConverter(template_style='question_answer')
        
        row = pd.Series({
            'product': 'Laptop',
            'price': 999.99,
            'rating': 4.5,
        })
        
        text = converter.convert_row(row)
        
        # Should be in Q&A format
        assert '?' in text  # Contains questions
        
    def test_classification_context(self):
        """Test formatting for classification tasks."""
        converter = BERTTextConverter(
            template_style='classification',
            task_type='sentiment'
        )
        
        row = pd.Series({
            'review': 'Great product!',
            'rating': 5,
        })
        
        text = converter.convert_row(row)
        
        # Should provide clear classification context
        assert 'review' in text.lower() or 'Great product!' in text
        
    def test_max_length_enforcement(self, sample_data):
        """Test max length enforcement for BERT."""
        converter = BERTTextConverter(max_text_length=128)
        
        # Create row with long content
        row = sample_data.iloc[0].copy()
        row['description'] = ' '.join(['word'] * 100)
        
        text = converter.convert_row(row)
        
        # Should respect BERT's typical max length
        assert len(text.split()) < 150  # Reasonable token count


@pytest.mark.integration
class TestConverterIntegration:
    """Integration tests for text converters."""
    
    def test_converter_pipeline(self):
        """Test complete conversion pipeline."""
        # Create realistic dataset
        df = create_sample_dataframe(size=100, include_all_types=True)
        
        # Test both converters
        converters = [
            TabularTextConverter(),
            BERTTextConverter(),
        ]
        
        for converter in converters:
            texts = converter.convert_batch(df)
            
            assert len(texts) == len(df)
            assert all(isinstance(t, str) for t in texts)
            assert all(len(t) > 0 for t in texts)
            
            # Check variety in outputs
            unique_texts = set(texts)
            assert len(unique_texts) > len(texts) * 0.9  # Most should be unique
            
    def test_converter_with_real_kaggle_data(self):
        """Test converter with Kaggle-like data."""
        # Create Kaggle-like dataset
        df = pd.DataFrame({
            'PassengerId': range(1, 101),
            'Survived': np.random.randint(0, 2, 100),
            'Pclass': np.random.randint(1, 4, 100),
            'Name': [f"Person {i}" for i in range(100)],
            'Sex': np.random.choice(['male', 'female'], 100),
            'Age': np.random.randint(1, 80, 100),
            'SibSp': np.random.randint(0, 5, 100),
            'Parch': np.random.randint(0, 5, 100),
            'Fare': np.random.uniform(10, 500, 100),
        })
        
        converter = BERTTextConverter(template_style='descriptive')
        texts = converter.convert_batch(df)
        
        # Check quality of conversion
        sample_text = texts[0]
        assert 'Person' in sample_text
        assert any(word in sample_text.lower() for word in ['passenger', 'class', 'age'])


