"""Tests for the template system."""

import pytest
import pandas as pd

from data.templates import (
    Template,
    TemplateConfig,
    KeyValueTemplate,
    NaturalLanguageTemplate,
    DescriptiveTemplate,
    CustomTemplate,
    MarkdownTableTemplate,
    JSONTemplate,
    get_template,
    get_template_registry,
)


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return {
        "name": "John Doe",
        "age": 30,
        "city": "New York",
        "salary": 75000.50,
        "is_active": True,
    }


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame for testing."""
    return pd.DataFrame([
        {"name": "John", "age": 30, "city": "NYC", "salary": 75000},
        {"name": "Jane", "age": 25, "city": "LA", "salary": 80000},
        {"name": "Bob", "age": 35, "city": "Chicago", "salary": 70000},
    ])


class TestTemplateConfig:
    """Test template configuration."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = TemplateConfig()
        assert not config.include_null_values
        assert config.decimal_places == 2
        assert config.null_representation == "missing"
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = TemplateConfig(
            include_null_values=True,
            decimal_places=4,
            max_text_length=100
        )
        assert config.include_null_values
        assert config.decimal_places == 4
        assert config.max_text_length == 100


class TestKeyValueTemplate:
    """Test key-value template."""
    
    def test_basic_conversion(self, sample_data):
        """Test basic key-value conversion."""
        template = KeyValueTemplate()
        result = template.convert(sample_data)
        
        assert "name: John Doe" in result
        assert "age: 30" in result
        assert "city: New York" in result
        assert " | " in result
    
    def test_with_column_types(self, sample_data):
        """Test with column type information."""
        template = KeyValueTemplate()
        template.set_column_types({
            "name": "text",
            "age": "numerical",
            "city": "categorical",
        })
        
        result = template.convert(sample_data)
        assert "name: John Doe" in result
        assert "age: 30" in result
    
    def test_custom_separator(self, sample_data):
        """Test custom separator."""
        config = TemplateConfig(custom_options={"separator": " || "})
        template = KeyValueTemplate(config)
        result = template.convert(sample_data)
        
        assert " || " in result
    
    def test_batch_conversion(self, sample_dataframe):
        """Test batch conversion."""
        template = KeyValueTemplate()
        results = template.convert_batch(sample_dataframe)
        
        assert len(results) == 3
        assert all("name:" in result for result in results)


class TestNaturalLanguageTemplate:
    """Test natural language template."""
    
    def test_basic_conversion(self, sample_data):
        """Test basic natural language conversion."""
        template = NaturalLanguageTemplate()
        template.set_column_types({
            "name": "text",
            "age": "numerical",
            "city": "categorical",
        })
        
        result = template.convert(sample_data)
        assert "The age is 30" in result
        assert "The city is New York" in result
    
    def test_without_column_types(self, sample_data):
        """Test without explicit column types."""
        template = NaturalLanguageTemplate()
        result = template.convert(sample_data)
        
        # Should still produce readable output
        assert "value" in result.lower()


class TestDescriptiveTemplate:
    """Test descriptive template."""
    
    def test_basic_conversion(self, sample_data):
        """Test basic descriptive conversion."""
        template = DescriptiveTemplate()
        result = template.convert(sample_data)
        
        assert "This record contains" in result
        assert "John Doe" in result
    
    def test_with_primary_columns(self, sample_data):
        """Test with primary columns specified."""
        config = TemplateConfig(custom_options={
            "primary_columns": ["name", "age"]
        })
        template = DescriptiveTemplate(config)
        result = template.convert(sample_data)
        
        assert "Key attributes" in result
        assert "name is John Doe" in result


class TestCustomTemplate:
    """Test custom template."""
    
    def test_basic_template_string(self, sample_data):
        """Test basic template string."""
        template = CustomTemplate(template_string="Person: {name}, Age: {age}")
        result = template.convert(sample_data)
        
        assert result == "Person: John Doe, Age: 30"
    
    def test_missing_placeholder(self):
        """Test with missing placeholder."""
        data = {"name": "John"}
        template = CustomTemplate(template_string="Name: {name}, Age: {age}")
        result = template.convert(data)
        
        assert "Name: John" in result
        assert "Age: missing" in result


class TestMarkdownTableTemplate:
    """Test markdown table template."""
    
    def test_basic_table(self, sample_data):
        """Test basic markdown table."""
        template = MarkdownTableTemplate()
        result = template.convert(sample_data)
        
        assert "|" in result
        assert "---" in result
        assert "John Doe" in result
    
    def test_column_order(self, sample_data):
        """Test custom column order."""
        config = TemplateConfig(custom_options={
            "column_order": ["name", "age", "city"]
        })
        template = MarkdownTableTemplate(config)
        result = template.convert(sample_data)
        
        lines = result.split("\n")
        header = lines[0]
        assert header.index("name") < header.index("age") < header.index("city")


class TestJSONTemplate:
    """Test JSON template."""
    
    def test_basic_json(self, sample_data):
        """Test basic JSON conversion."""
        template = JSONTemplate()
        result = template.convert(sample_data)
        
        import json
        parsed = json.loads(result)
        assert parsed["name"] == "John Doe"
        assert parsed["age"] == 30
    
    def test_compact_format(self, sample_data):
        """Test compact JSON format."""
        config = TemplateConfig(custom_options={"compact": True})
        template = JSONTemplate(config)
        result = template.convert(sample_data)
        
        assert "\n" not in result
        assert "  " not in result  # No indentation


class TestTemplateRegistry:
    """Test template registry."""
    
    def test_get_template(self):
        """Test getting template by name."""
        template = get_template("keyvalue")
        assert isinstance(template, KeyValueTemplate)
    
    def test_get_template_with_config(self):
        """Test getting template with configuration."""
        config = TemplateConfig(decimal_places=4)
        template = get_template("keyvalue", config)
        assert template.config.decimal_places == 4
    
    def test_list_templates(self):
        """Test listing available templates."""
        registry = get_template_registry()
        templates = registry.list_templates()
        
        assert "keyvalue" in templates
        assert "natural" in templates
        assert "json" in templates
    
    def test_template_info(self):
        """Test getting template information."""
        registry = get_template_registry()
        info = registry.get_template_info("keyvalue")
        
        assert info["name"] == "keyvalue"
        assert "class" in info
        assert "description" in info


class TestTemplateValidation:
    """Test template validation."""
    
    def test_validate_data(self, sample_data):
        """Test data validation."""
        template = KeyValueTemplate()
        assert template.validate(sample_data)
    
    def test_validate_invalid_data(self):
        """Test validation with invalid data."""
        template = KeyValueTemplate()
        assert not template.validate("not a dict")
    
    def test_filter_columns(self, sample_data):
        """Test column filtering."""
        config = TemplateConfig(
            include_columns=["name", "age"],
            exclude_columns=["city"]
        )
        template = KeyValueTemplate(config)
        
        filtered = template.filter_columns(sample_data)
        assert "name" in filtered
        assert "age" in filtered
        assert "city" not in filtered


class TestTemplateFormatting:
    """Test template value formatting."""
    
    def test_format_numerical(self):
        """Test numerical value formatting."""
        template = KeyValueTemplate()
        
        result = template.format_value(123.456, "test")
        assert result == "123.46"
    
    def test_format_with_scientific_notation(self):
        """Test scientific notation formatting."""
        config = TemplateConfig(
            use_scientific_notation=True,
            scientific_threshold=1000
        )
        template = KeyValueTemplate(config)
        
        result = template.format_value(1000000, "test")
        assert "e" in result.lower()
    
    def test_format_text_truncation(self):
        """Test text truncation."""
        config = TemplateConfig(
            max_text_length=10,
            truncate_indicator="..."
        )
        template = KeyValueTemplate(config)
        
        result = template.format_value("This is a very long text", "test")
        assert len(result) == 10
        assert result.endswith("...")
    
    def test_format_null_values(self):
        """Test null value formatting."""
        template = KeyValueTemplate()
        
        result = template.format_value(pd.NA, "test")
        assert result == "missing"
    
    def test_include_null_values(self, sample_data):
        """Test including null values."""
        data_with_null = sample_data.copy()
        data_with_null["missing_field"] = None
        
        config = TemplateConfig(include_null_values=True)
        template = KeyValueTemplate(config)
        result = template.convert(data_with_null)
        
        assert "missing_field: missing" in result