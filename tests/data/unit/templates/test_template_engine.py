"""Tests for text template engine and templates."""

import pytest
from unittest.mock import Mock, patch

import pandas as pd
import numpy as np

from data.core.base import CompetitionType
from data.templates.engine import TextTemplateEngine, CompetitionTextTemplate
from data.templates.base_template import BaseTextTemplate
from tests.data.fixtures.utils import create_sample_dataframe
from tests.data.fixtures.configs import create_dataset_spec


class TestBaseTextTemplate:
    """Test BaseTextTemplate abstract class."""
    
    def test_base_template_creation(self):
        """Test base template cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseTextTemplate()
            
    def test_base_template_interface(self):
        """Test base template interface requirements."""
        class TestTemplate(BaseTextTemplate):
            def convert_row(self, row, **kwargs):
                return "test"
                
            def convert_batch(self, df, **kwargs):
                return ["test"] * len(df)
                
        template = TestTemplate()
        assert hasattr(template, 'convert_row')
        assert hasattr(template, 'convert_batch')


class TestCompetitionTextTemplate:
    """Test CompetitionTextTemplate class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return create_sample_dataframe(size=3)
        
    def test_template_creation(self):
        """Test template creation."""
        template = CompetitionTextTemplate(
            competition_type=CompetitionType.BINARY_CLASSIFICATION,
            template_name="test_template",
            description="Test template",
        )
        
        assert template.competition_type == CompetitionType.BINARY_CLASSIFICATION
        assert template.template_name == "test_template"
        assert template.description == "Test template"
        assert template.version == "1.0"
        
    def test_template_with_custom_patterns(self):
        """Test template with custom patterns."""
        custom_patterns = {
            'default': "Feature1: {feature1}, Feature2: {feature2}",
            'detailed': "Detailed: Feature1 is {feature1} and Feature2 is {feature2}",
        }
        
        template = CompetitionTextTemplate(
            competition_type=CompetitionType.REGRESSION,
            template_name="custom_template",
            template_patterns=custom_patterns,
        )
        
        assert template.template_patterns == custom_patterns
        assert 'default' in template.template_patterns
        assert 'detailed' in template.template_patterns
        
    def test_convert_row_default(self, sample_data):
        """Test converting single row with default template."""
        template = CompetitionTextTemplate(
            competition_type=CompetitionType.BINARY_CLASSIFICATION,
            template_name="test",
        )
        
        row = sample_data.iloc[0]
        text = template.convert_row(row)
        
        assert isinstance(text, str)
        assert len(text) > 0
        
    def test_convert_row_with_pattern(self, sample_data):
        """Test converting row with specific pattern."""
        custom_patterns = {
            'simple': "F1: {col_0}, F2: {col_1}",
        }
        
        template = CompetitionTextTemplate(
            competition_type=CompetitionType.BINARY_CLASSIFICATION,
            template_name="test",
            template_patterns=custom_patterns,
        )
        
        row = sample_data.iloc[0]
        text = template.convert_row(row, pattern_name="simple")
        
        assert "F1:" in text
        assert "F2:" in text
        
    def test_convert_row_missing_pattern(self, sample_data):
        """Test converting row with missing pattern falls back to default."""
        template = CompetitionTextTemplate(
            competition_type=CompetitionType.BINARY_CLASSIFICATION,
            template_name="test",
        )
        
        row = sample_data.iloc[0]
        # Should not raise error
        text = template.convert_row(row, pattern_name="nonexistent")
        assert isinstance(text, str)
        
    def test_convert_batch(self, sample_data):
        """Test converting batch of rows."""
        template = CompetitionTextTemplate(
            competition_type=CompetitionType.MULTICLASS_CLASSIFICATION,
            template_name="batch_test",
        )
        
        texts = template.convert_batch(sample_data)
        
        assert len(texts) == len(sample_data)
        assert all(isinstance(text, str) for text in texts)
        assert all(len(text) > 0 for text in texts)
        
    def test_template_variations(self, sample_data):
        """Test template with variations enabled."""
        patterns = {
            'v1': "Version 1: {col_0}",
            'v2': "Version 2: {col_0}",
            'v3': "Version 3: {col_0}",
        }
        
        template = CompetitionTextTemplate(
            competition_type=CompetitionType.BINARY_CLASSIFICATION,
            template_name="varied",
            template_patterns=patterns,
            enable_variations=True,
        )
        
        # Convert same row multiple times
        row = sample_data.iloc[0]
        texts = [template.convert_row(row) for _ in range(10)]
        
        # Should have variations
        unique_texts = set(texts)
        assert len(unique_texts) > 1
        
    def test_template_with_preprocessing(self):
        """Test template with preprocessing function."""
        def preprocess(row):
            # Convert numeric values to categories
            row = row.copy()
            for col in row.index:
                if pd.api.types.is_numeric_dtype(type(row[col])):
                    row[col] = f"NUM_{row[col]}"
            return row
            
        template = CompetitionTextTemplate(
            competition_type=CompetitionType.REGRESSION,
            template_name="preprocess_test",
            preprocessing_fn=preprocess,
        )
        
        row = pd.Series({'value': 42, 'name': 'test'})
        text = template.convert_row(row)
        
        assert "NUM_42" in text
        
    def test_template_metadata(self):
        """Test template metadata."""
        template = CompetitionTextTemplate(
            competition_type=CompetitionType.BINARY_CLASSIFICATION,
            template_name="metadata_test",
            description="Test template with metadata",
            author="Test Author",
            tags=["test", "classification"],
        )
        
        metadata = template.get_metadata()
        
        assert metadata['name'] == "metadata_test"
        assert metadata['description'] == "Test template with metadata"
        assert metadata['author'] == "Test Author"
        assert "test" in metadata['tags']
        assert metadata['version'] == "1.0"


class TestTextTemplateEngine:
    """Test TextTemplateEngine class."""
    
    @pytest.fixture
    def engine(self):
        """Create template engine instance."""
        return TextTemplateEngine()
        
    @pytest.fixture
    def sample_spec(self):
        """Create sample dataset spec."""
        return create_dataset_spec(
            competition_type=CompetitionType.BINARY_CLASSIFICATION,
            text_columns=['description'],
            categorical_columns=['category'],
            numerical_columns=['value'],
        )
        
    def test_engine_creation(self, engine):
        """Test engine creation."""
        assert engine is not None
        assert hasattr(engine, '_templates')
        assert hasattr(engine, '_converters')
        
    def test_register_template(self, engine):
        """Test registering custom template."""
        template = CompetitionTextTemplate(
            competition_type=CompetitionType.BINARY_CLASSIFICATION,
            template_name="custom",
        )
        
        engine.register_template("custom_binary", template)
        
        assert "custom_binary" in engine.list_templates()
        
    def test_get_template_for_spec(self, engine, sample_spec):
        """Test getting appropriate template for dataset spec."""
        template = engine.get_template_for_spec(sample_spec)
        
        assert template is not None
        assert isinstance(template, BaseTextTemplate)
        
    def test_create_converter(self, engine, sample_spec):
        """Test creating converter from spec."""
        converter = engine.create_converter(
            sample_spec,
            converter_type='tabular'
        )
        
        assert converter is not None
        assert hasattr(converter, 'convert_row')
        assert hasattr(converter, 'convert_batch')
        
    def test_convert_with_auto_selection(self, engine, sample_spec):
        """Test conversion with automatic template selection."""
        df = create_sample_dataframe(size=5)
        
        texts = engine.convert_dataset(df, sample_spec)
        
        assert len(texts) == len(df)
        assert all(isinstance(text, str) for text in texts)
        
    def test_convert_with_specific_template(self, engine, sample_spec):
        """Test conversion with specific template."""
        df = create_sample_dataframe(size=5)
        
        # Register specific template
        template = CompetitionTextTemplate(
            competition_type=CompetitionType.BINARY_CLASSIFICATION,
            template_name="specific",
            template_patterns={'default': "Row: {col_0}"},
        )
        engine.register_template("specific", template)
        
        texts = engine.convert_dataset(
            df,
            sample_spec,
            template_name="specific"
        )
        
        assert all("Row:" in text for text in texts)
        
    def test_template_caching(self, engine, sample_spec):
        """Test template caching for performance."""
        # Get template twice
        template1 = engine.get_template_for_spec(sample_spec)
        template2 = engine.get_template_for_spec(sample_spec)
        
        # Should return same instance (cached)
        assert template1 is template2
        
    def test_batch_conversion_performance(self, engine, sample_spec):
        """Test batch conversion is more efficient than row-by-row."""
        df = create_sample_dataframe(size=100)
        
        import time
        
        # Batch conversion
        start = time.time()
        batch_texts = engine.convert_dataset(df, sample_spec)
        batch_time = time.time() - start
        
        # Row-by-row conversion
        start = time.time()
        template = engine.get_template_for_spec(sample_spec)
        row_texts = [template.convert_row(row) for _, row in df.iterrows()]
        row_time = time.time() - start
        
        # Batch should be faster (or at least not much slower)
        assert batch_time <= row_time * 1.5
        
    def test_template_selection_by_competition_type(self, engine):
        """Test template selection based on competition type."""
        types_to_test = [
            CompetitionType.BINARY_CLASSIFICATION,
            CompetitionType.MULTICLASS_CLASSIFICATION,
            CompetitionType.REGRESSION,
            CompetitionType.RANKING,
        ]
        
        for comp_type in types_to_test:
            spec = create_dataset_spec(competition_type=comp_type)
            template = engine.get_template_for_spec(spec)
            
            assert template is not None
            # Template should be appropriate for competition type
            
    def test_custom_converter_registration(self, engine):
        """Test registering custom converter."""
        class CustomConverter:
            def convert_row(self, row):
                return "custom: " + str(row.values)
                
            def convert_batch(self, df):
                return [self.convert_row(row) for _, row in df.iterrows()]
                
        engine.register_converter("custom", CustomConverter)
        
        spec = create_dataset_spec()
        converter = engine.create_converter(spec, converter_type="custom")
        
        assert isinstance(converter, CustomConverter)
        
    def test_template_composition(self, engine):
        """Test composing multiple templates."""
        # Create templates with different focuses
        template1 = CompetitionTextTemplate(
            competition_type=CompetitionType.BINARY_CLASSIFICATION,
            template_name="features",
            template_patterns={'default': "Features: {col_0}, {col_1}"},
        )
        
        template2 = CompetitionTextTemplate(
            competition_type=CompetitionType.BINARY_CLASSIFICATION,
            template_name="context",
            template_patterns={'default': "Context: This is a binary classification task."},
        )
        
        engine.register_template("features", template1)
        engine.register_template("context", template2)
        
        # Compose templates
        df = create_sample_dataframe(size=2)
        spec = create_dataset_spec()
        
        texts = engine.convert_dataset(
            df,
            spec,
            template_names=["context", "features"],
            compose_method="concatenate"
        )
        
        assert all("Context:" in text and "Features:" in text for text in texts)


@pytest.mark.integration
class TestTemplateIntegration:
    """Integration tests for template system."""
    
    def test_end_to_end_conversion(self):
        """Test complete end-to-end conversion pipeline."""
        # Create dataset
        df = pd.DataFrame({
            'age': [25, 30, 35],
            'income': [50000, 75000, 100000],
            'education': ['Bachelor', 'Master', 'PhD'],
            'approved': [1, 1, 0],
        })
        
        # Create spec
        spec = create_dataset_spec(
            competition_type=CompetitionType.BINARY_CLASSIFICATION,
            target_column='approved',
            numerical_columns=['age', 'income'],
            categorical_columns=['education'],
        )
        
        # Create engine
        engine = TextTemplateEngine()
        
        # Convert
        texts = engine.convert_dataset(df, spec)
        
        # Verify
        assert len(texts) == len(df)
        
        # Check content
        assert '25' in texts[0] or 'twenty-five' in texts[0].lower()
        assert 'Bachelor' in texts[0]
        
    def test_multi_competition_support(self):
        """Test support for multiple competition types."""
        engine = TextTemplateEngine()
        
        # Test different competition types
        competition_configs = [
            (CompetitionType.BINARY_CLASSIFICATION, {'target': [0, 1, 0]}),
            (CompetitionType.MULTICLASS_CLASSIFICATION, {'target': [0, 1, 2]}),
            (CompetitionType.REGRESSION, {'target': [1.5, 2.3, 3.7]}),
            (CompetitionType.RANKING, {'rank': [1, 2, 3]}),
        ]
        
        for comp_type, target_data in competition_configs:
            df = create_sample_dataframe(size=3)
            for col, values in target_data.items():
                df[col] = values
                
            spec = create_dataset_spec(
                competition_type=comp_type,
                target_column=list(target_data.keys())[0],
            )
            
            texts = engine.convert_dataset(df, spec)
            
            assert len(texts) == len(df)
            assert all(isinstance(t, str) for t in texts)


@pytest.mark.slow
class TestTemplatePerformance:
    """Performance tests for template system."""
    
    def test_large_dataset_conversion(self):
        """Test conversion of large datasets."""
        engine = TextTemplateEngine()
        
        # Create large dataset
        df = create_sample_dataframe(size=10000)
        spec = create_dataset_spec(num_samples=10000)
        
        import time
        start = time.time()
        texts = engine.convert_dataset(df, spec)
        elapsed = time.time() - start
        
        assert len(texts) == 10000
        assert elapsed < 10.0  # Should convert 10k rows in less than 10 seconds
        
        # Calculate throughput
        throughput = len(texts) / elapsed
        assert throughput > 1000  # >1000 rows/second
        
    def test_template_caching_performance(self):
        """Test performance improvement from template caching."""
        engine = TextTemplateEngine()
        spec = create_dataset_spec()
        
        import time
        
        # First access (cache miss)
        start = time.time()
        template1 = engine.get_template_for_spec(spec)
        first_access = time.time() - start
        
        # Second access (cache hit)
        start = time.time()
        template2 = engine.get_template_for_spec(spec)
        second_access = time.time() - start
        
        # Cache hit should be much faster
        assert second_access < first_access * 0.1  # At least 10x faster