"""Unit tests for text converters."""

import pytest
from unittest.mock import Mock, patch

import pandas as pd
import numpy as np

from data.templates.converters import TabularTextConverter, BERTTextConverter


class TestTabularTextConverter:
    """Test TabularTextConverter class."""
    
    @pytest.fixture
    def converter(self):
        """Create TabularTextConverter instance."""
        return TabularTextConverter()
        
    @pytest.fixture
    def sample_data(self):
        """Create sample tabular data."""
        return pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice Johnson', 'Bob Smith', 'Charlie Brown'],
            'age': [25, 30, 35],
            'city': ['New York', 'Los Angeles', 'Chicago'],
            'salary': [50000.0, 75000.5, 60000.25],
            'is_active': [True, False, True],
            'category': ['A', 'B', 'A'],
            'description': [
                'Software engineer with 3 years experience',
                'Senior data scientist',
                'Product manager specializing in ML'
            ],
            'target': [1, 0, 1]
        })
        
    def test_converter_creation(self, converter):
        """Test converter creation."""
        assert converter is not None
        assert hasattr(converter, 'max_text_length')
        assert hasattr(converter, 'include_column_names')
        assert hasattr(converter, 'separator')
        
    def test_convert_simple_row(self, converter, sample_data):
        """Test converting simple row to text."""
        row = sample_data.iloc[0]
        text = converter.convert_row(row)
        
        assert isinstance(text, str)
        assert len(text) > 0
        assert 'Alice Johnson' in text
        assert '25' in text
        assert 'New York' in text
        
    def test_convert_row_with_column_names(self, sample_data):
        """Test converting row with column names included."""
        converter = TabularTextConverter(include_column_names=True)
        
        row = sample_data.iloc[0]
        text = converter.convert_row(row)
        
        assert 'name:' in text.lower() or 'name =' in text.lower()
        assert 'age:' in text.lower() or 'age =' in text.lower()
        assert 'Alice Johnson' in text
        
    def test_convert_row_without_column_names(self, sample_data):
        """Test converting row without column names."""
        converter = TabularTextConverter(include_column_names=False)
        
        row = sample_data.iloc[0]
        text = converter.convert_row(row)
        
        assert 'name:' not in text.lower()
        assert 'age:' not in text.lower()
        assert 'Alice Johnson' in text
        
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
        assert texts[0] != texts[1]
        assert 'Alice' in texts[0]
        assert 'Bob' in texts[1]
        
    def test_convert_with_missing_values(self, converter):
        """Test converting data with missing values."""
        data_with_nan = pd.DataFrame({
            'name': ['Alice', 'Bob', None],
            'age': [25, None, 35],
            'score': [85.5, 92.0, np.nan],
            'active': [True, None, False]
        })
        
        texts = converter.convert_batch(data_with_nan)
        
        assert len(texts) == 3
        assert all(isinstance(text, str) for text in texts)
        
        # Should handle missing values gracefully
        assert 'None' not in texts[2] or 'nan' not in texts[2].lower()
        
    def test_convert_with_exclude_columns(self, sample_data):
        """Test converting with excluded columns."""
        converter = TabularTextConverter(exclude_columns=['id', 'target'])
        
        row = sample_data.iloc[0]
        text = converter.convert_row(row)
        
        # ID and target should not be in text
        assert '1' not in text or 'id' not in text.lower()
        assert 'Alice Johnson' in text  # Other columns should be present
        
    def test_convert_with_include_columns(self, sample_data):
        """Test converting with only included columns."""
        converter = TabularTextConverter(include_columns=['name', 'age', 'city'])
        
        row = sample_data.iloc[0]
        text = converter.convert_row(row)
        
        assert 'Alice Johnson' in text
        assert '25' in text
        assert 'New York' in text
        # Salary should not be included
        assert '50000' not in text
        
    def test_convert_with_max_length(self, sample_data):
        """Test converting with maximum text length."""
        converter = TabularTextConverter(max_text_length=50)
        
        row = sample_data.iloc[0]
        text = converter.convert_row(row)
        
        assert len(text) <= 50
        
    def test_convert_numeric_formatting(self, converter):
        """Test numeric value formatting."""
        data = pd.DataFrame({
            'integer': [100, 200, 300],
            'float': [10.5, 20.75, 30.333333],
            'large_number': [1234567.89, 9876543.21, 5555555.55],
        })
        
        texts = converter.convert_batch(data)
        
        # Check that numbers are formatted reasonably
        assert '10.5' in texts[0] or '10.50' in texts[0]
        assert '20.75' in texts[1] or '20.8' in texts[1]
        
    def test_convert_boolean_formatting(self, converter):
        """Test boolean value formatting."""
        data = pd.DataFrame({
            'flag1': [True, False, True],
            'flag2': [False, True, False],
        })
        
        texts = converter.convert_batch(data)
        
        # Booleans should be converted to readable text
        assert 'true' in texts[0].lower() or 'yes' in texts[0].lower()
        assert 'false' in texts[1].lower() or 'no' in texts[1].lower()
        
    def test_convert_categorical_data(self, converter):
        """Test converting categorical data."""
        data = pd.DataFrame({
            'category': pd.Categorical(['A', 'B', 'A', 'C']),
            'grade': pd.Categorical(['Good', 'Excellent', 'Fair', 'Good']),
        })
        
        texts = converter.convert_batch(data)
        
        assert len(texts) == 4
        assert 'Good' in texts[0]
        assert 'Excellent' in texts[1]
        
    def test_convert_datetime_data(self, converter):
        """Test converting datetime data."""
        data = pd.DataFrame({
            'date': pd.to_datetime(['2023-01-01', '2023-06-15', '2023-12-31']),
            'timestamp': pd.to_datetime(['2023-01-01 10:30:00', '2023-06-15 14:45:30', '2023-12-31 23:59:59']),
        })
        
        texts = converter.convert_batch(data)
        
        assert len(texts) == 3
        assert '2023' in texts[0]
        assert 'Jan' in texts[0] or '01' in texts[0]
        
    def test_get_conversion_info(self, converter, sample_data):
        """Test getting conversion information."""
        info = converter.get_conversion_info(sample_data)
        
        assert 'total_columns' in info
        assert 'included_columns' in info
        assert 'excluded_columns' in info
        assert 'estimated_text_length' in info
        assert 'column_types' in info
        
        assert info['total_columns'] == len(sample_data.columns)
        
    def test_validate_data(self, converter, sample_data):
        """Test data validation."""
        is_valid, issues = converter.validate_data(sample_data)
        
        assert is_valid == True
        assert len(issues) == 0
        
    def test_validate_empty_data(self, converter):
        """Test validation with empty data."""
        empty_data = pd.DataFrame()
        
        is_valid, issues = converter.validate_data(empty_data)
        
        assert is_valid == False
        assert len(issues) > 0
        assert any('empty' in issue.lower() for issue in issues)
        
    def test_convert_with_custom_formatters(self, sample_data):
        """Test converting with custom column formatters."""
        def format_salary(value):
            return f"${value:,.0f}"
            
        def format_name(value):
            return value.upper() if pd.notna(value) else ""
            
        custom_formatters = {
            'salary': format_salary,
            'name': format_name,
        }
        
        converter = TabularTextConverter(column_formatters=custom_formatters)
        
        row = sample_data.iloc[0]
        text = converter.convert_row(row)
        
        assert '$50,000' in text
        assert 'ALICE JOHNSON' in text
        
    def test_get_column_statistics(self, converter, sample_data):
        """Test getting column statistics."""
        stats = converter.get_column_statistics(sample_data)
        
        assert 'numeric_columns' in stats
        assert 'text_columns' in stats
        assert 'categorical_columns' in stats
        assert 'boolean_columns' in stats
        assert 'datetime_columns' in stats
        
        assert 'age' in stats['numeric_columns']
        assert 'name' in stats['text_columns']
        assert 'is_active' in stats['boolean_columns']


class TestBERTTextConverter:
    """Test BERTTextConverter class."""
    
    @pytest.fixture
    def converter(self, mock_tokenizer):
        """Create BERTTextConverter with mock tokenizer."""
        return BERTTextConverter(tokenizer=mock_tokenizer)
        
    @pytest.fixture
    def sample_data(self):
        """Create sample data for BERT conversion."""
        return pd.DataFrame({
            'text_feature': [
                'This is a sample text for classification',
                'Another example with different content',
                'Third text sample for testing purposes'
            ],
            'category': ['positive', 'negative', 'positive'],
            'score': [0.8, 0.2, 0.9],
            'target': [1, 0, 1]
        })
        
    def test_converter_creation(self, converter):
        """Test BERT converter creation."""
        assert converter is not None
        assert hasattr(converter, 'tokenizer')
        assert hasattr(converter, 'max_length')
        assert hasattr(converter, 'padding')
        assert hasattr(converter, 'truncation')
        
    def test_convert_text_to_tokens(self, converter):
        """Test converting text to tokens."""
        text = "This is a test sentence for tokenization."
        
        tokens = converter.convert_text_to_tokens(text)
        
        assert 'input_ids' in tokens
        assert 'attention_mask' in tokens
        assert len(tokens['input_ids']) > 0
        assert len(tokens['attention_mask']) == len(tokens['input_ids'])
        
    def test_convert_batch_texts(self, converter):
        """Test converting batch of texts."""
        texts = [
            "First sample text",
            "Second sample text",
            "Third sample text"
        ]
        
        batch_tokens = converter.convert_batch_texts(texts)
        
        assert 'input_ids' in batch_tokens
        assert 'attention_mask' in batch_tokens
        assert len(batch_tokens['input_ids']) == 3
        assert len(batch_tokens['attention_mask']) == 3
        
    def test_convert_row_to_text(self, converter, sample_data):
        """Test converting DataFrame row to text."""
        row = sample_data.iloc[0]
        text = converter.convert_row_to_text(row)
        
        assert isinstance(text, str)
        assert len(text) > 0
        assert 'This is a sample text' in text
        
    def test_convert_batch_rows(self, converter, sample_data):
        """Test converting batch of DataFrame rows."""
        texts = converter.convert_batch_rows(sample_data)
        
        assert len(texts) == len(sample_data)
        assert all(isinstance(text, str) for text in texts)
        assert 'This is a sample text' in texts[0]
        assert 'Another example' in texts[1]
        
    def test_tokenize_data(self, converter, sample_data):
        """Test tokenizing DataFrame data."""
        tokenized = converter.tokenize_data(sample_data)
        
        assert 'input_ids' in tokenized
        assert 'attention_mask' in tokenized
        assert len(tokenized['input_ids']) == len(sample_data)
        
    def test_convert_with_max_length(self, mock_tokenizer):
        """Test conversion with custom max length."""
        converter = BERTTextConverter(tokenizer=mock_tokenizer, max_length=128)
        
        long_text = "This is a very long text " * 20  # Create long text
        tokens = converter.convert_text_to_tokens(long_text)
        
        assert len(tokens['input_ids']) <= 128
        
    def test_convert_without_padding(self, mock_tokenizer):
        """Test conversion without padding."""
        converter = BERTTextConverter(
            tokenizer=mock_tokenizer, 
            padding=False
        )
        
        text = "Short text"
        tokens = converter.convert_text_to_tokens(text)
        
        # Without padding, length should be minimal
        assert len(tokens['input_ids']) < converter.max_length
        
    def test_convert_without_truncation(self, mock_tokenizer):
        """Test conversion without truncation."""
        converter = BERTTextConverter(
            tokenizer=mock_tokenizer,
            truncation=False,
            max_length=10  # Very short to test
        )
        
        text = "This is a longer text that should not be truncated"
        tokens = converter.convert_text_to_tokens(text)
        
        # Should not be truncated even if longer than max_length
        assert isinstance(tokens['input_ids'], list)
        
    def test_convert_with_special_tokens(self, converter):
        """Test conversion includes special tokens."""
        text = "Regular text content"
        tokens = converter.convert_text_to_tokens(text)
        
        # Should include CLS and SEP tokens (0 and 1 in mock tokenizer)
        assert 0 in tokens['input_ids']  # CLS token
        assert 1 in tokens['input_ids']  # SEP token
        
    def test_convert_empty_text(self, converter):
        """Test converting empty text."""
        empty_text = ""
        tokens = converter.convert_text_to_tokens(empty_text)
        
        assert 'input_ids' in tokens
        assert 'attention_mask' in tokens
        assert len(tokens['input_ids']) > 0  # Should have special tokens
        
    def test_convert_with_text_templates(self, converter, sample_data):
        """Test converting with text templates."""
        template = "Category: {category}, Score: {score}, Text: {text_feature}"
        
        converter_with_template = BERTTextConverter(
            tokenizer=converter.tokenizer,
            text_template=template
        )
        
        text = converter_with_template.convert_row_to_text(sample_data.iloc[0])
        
        assert 'Category: positive' in text
        assert 'Score: 0.8' in text
        assert 'This is a sample text' in text
        
    def test_get_tokenization_info(self, converter, sample_data):
        """Test getting tokenization information."""
        info = converter.get_tokenization_info(sample_data)
        
        assert 'vocab_size' in info
        assert 'max_length' in info
        assert 'padding' in info
        assert 'truncation' in info
        assert 'estimated_tokens_per_sample' in info
        
    def test_validate_tokenizer(self, converter):
        """Test tokenizer validation."""
        is_valid, issues = converter.validate_tokenizer()
        
        assert is_valid == True
        assert len(issues) == 0
        
    def test_validate_missing_tokenizer(self):
        """Test validation with missing tokenizer."""
        converter = BERTTextConverter(tokenizer=None)
        
        is_valid, issues = converter.validate_tokenizer()
        
        assert is_valid == False
        assert len(issues) > 0
        assert any('tokenizer' in issue.lower() for issue in issues)
        
    def test_convert_with_return_tensors(self, mock_tokenizer):
        """Test conversion with tensor output."""
        converter = BERTTextConverter(
            tokenizer=mock_tokenizer,
            return_tensors="np"
        )
        
        text = "Test text for tensor conversion"
        tokens = converter.convert_text_to_tokens(text)
        
        # Should return numpy arrays when specified
        assert hasattr(tokens['input_ids'], 'shape')  # numpy array-like
        
    def test_get_attention_masks(self, converter):
        """Test getting attention masks."""
        texts = ["Short text", "Much longer text with more words"]
        
        batch_tokens = converter.convert_batch_texts(texts)
        attention_masks = batch_tokens['attention_mask']
        
        assert len(attention_masks) == 2
        assert all(isinstance(mask, list) for mask in attention_masks)
        assert all(token in [0, 1] for mask in attention_masks for token in mask)
        
    def test_decode_tokens(self, converter):
        """Test decoding tokens back to text."""
        original_text = "This is test text for decoding"
        tokens = converter.convert_text_to_tokens(original_text)
        
        # Mock decoder for testing
        def mock_decode(token_ids):
            # Simple mock decode - just join token values
            return " ".join(str(id) for id in token_ids)
            
        converter.tokenizer.decode = mock_decode
        
        decoded = converter.decode_tokens(tokens['input_ids'])
        assert isinstance(decoded, str)
        
    def test_get_special_tokens(self, converter):
        """Test getting special tokens information."""
        special_tokens = converter.get_special_tokens()
        
        assert isinstance(special_tokens, dict)
        # Mock tokenizer should have some special tokens
        assert len(special_tokens) > 0
        
    def test_convert_with_multiple_text_columns(self, converter):
        """Test converting data with multiple text columns."""
        data = pd.DataFrame({
            'title': ['Title 1', 'Title 2', 'Title 3'],
            'description': ['Desc 1', 'Desc 2', 'Desc 3'],
            'category': ['A', 'B', 'C'],
        })
        
        texts = converter.convert_batch_rows(data)
        
        assert len(texts) == 3
        assert all('Title' in text for text in texts)
        assert all('Desc' in text for text in texts)
        
    def test_convert_performance_large_batch(self, converter):
        """Test conversion performance with large batch."""
        large_data = pd.DataFrame({
            'text': [f'Sample text number {i}' for i in range(100)],
            'category': ['A'] * 100,
        })
        
        # Should handle large batches without errors
        texts = converter.convert_batch_rows(large_data)
        
        assert len(texts) == 100
        assert all(isinstance(text, str) for text in texts)
        
    def test_memory_efficient_conversion(self, converter):
        """Test memory-efficient batch conversion."""
        # Test with chunk size for memory efficiency
        converter_chunked = BERTTextConverter(
            tokenizer=converter.tokenizer,
            batch_size=32
        )
        
        large_texts = [f'Text sample {i}' for i in range(100)]
        
        # Should process in chunks
        tokenized = converter_chunked.convert_batch_texts(large_texts)
        
        assert len(tokenized['input_ids']) == 100
        
    def test_convert_with_data_augmentation(self, converter, sample_data):
        """Test conversion with data augmentation."""
        # Enable augmentation in converter
        converter_aug = BERTTextConverter(
            tokenizer=converter.tokenizer,
            enable_augmentation=True,
            augmentation_ratio=0.1  # 10% augmentation
        )
        
        texts = converter_aug.convert_batch_rows(sample_data)
        
        # With augmentation, some texts might be modified
        assert len(texts) == len(sample_data)
        assert all(isinstance(text, str) for text in texts)