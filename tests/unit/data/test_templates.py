"""Unit tests for text template engine and templates."""

import pytest
from unittest.mock import Mock, patch

import pandas as pd

from data.core.base import CompetitionType
from data.templates.engine import TextTemplateEngine, CompetitionTextTemplate
from data.templates.base_template import BaseTextTemplate
from data.templates.converters import TabularTextConverter, BERTTextConverter


class TestBaseTextTemplate:
    """Test BaseTextTemplate abstract class."""
    
    def test_base_template_creation(self):
        """Test base template cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseTextTemplate()


class TestCompetitionTextTemplate:
    """Test CompetitionTextTemplate class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': ['A', 'B', 'C'],
            'target': [0, 1, 0]
        })
        
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
        assert "feature1" in text.lower()
        assert "1" in text  # Value should be included
        
    def test_convert_row_with_pattern(self, sample_data):
        """Test converting row with specific pattern."""
        custom_patterns = {
            'simple': "F1: {feature1}, F2: {feature2}",
        }
        
        template = CompetitionTextTemplate(
            competition_type=CompetitionType.BINARY_CLASSIFICATION,
            template_name="test",
            template_patterns=custom_patterns,
        )
        
        row = sample_data.iloc[0]
        text = template.convert_row(row, pattern_name="simple")
        
        assert text == "F1: 1, F2: A"
        
    def test_convert_row_missing_pattern(self, sample_data):
        """Test converting row with missing pattern falls back to default."""
        template = CompetitionTextTemplate(
            competition_type=CompetitionType.BINARY_CLASSIFICATION,
            template_name="test",
        )
        
        row = sample_data.iloc[0]
        text = template.convert_row(row, pattern_name="nonexistent")
        
        # Should fall back to default pattern
        assert isinstance(text, str)
        assert len(text) > 0
        
    def test_convert_batch(self, sample_data):
        """Test converting batch of rows."""
        template = CompetitionTextTemplate(
            competition_type=CompetitionType.BINARY_CLASSIFICATION,
            template_name="test",
        )
        
        texts = template.convert_batch(sample_data)
        
        assert len(texts) == len(sample_data)
        assert all(isinstance(text, str) for text in texts)
        assert all(len(text) > 0 for text in texts)
        
    def test_convert_batch_with_pattern(self, sample_data):
        """Test converting batch with specific pattern."""
        custom_patterns = {
            'batch_pattern': "Value: {feature1}",
        }
        
        template = CompetitionTextTemplate(
            competition_type=CompetitionType.REGRESSION,
            template_name="test",
            template_patterns=custom_patterns,
        )
        
        texts = template.convert_batch(sample_data, pattern_name="batch_pattern")
        
        expected_texts = ["Value: 1", "Value: 2", "Value: 3"]
        assert texts == expected_texts
        
    def test_get_available_patterns(self):
        """Test getting available patterns."""
        custom_patterns = {
            'pattern1': "Template 1",
            'pattern2': "Template 2",
        }
        
        template = CompetitionTextTemplate(
            competition_type=CompetitionType.MULTICLASS_CLASSIFICATION,
            template_name="test",
            template_patterns=custom_patterns,
        )
        
        patterns = template.get_available_patterns()
        
        assert 'pattern1' in patterns
        assert 'pattern2' in patterns
        assert len(patterns) >= 2
        
    def test_add_pattern(self):
        """Test adding new pattern."""
        template = CompetitionTextTemplate(
            competition_type=CompetitionType.BINARY_CLASSIFICATION,
            template_name="test",
        )
        
        template.add_pattern("new_pattern", "New: {feature1}")
        
        assert "new_pattern" in template.template_patterns
        assert template.template_patterns["new_pattern"] == "New: {feature1}"
        
    def test_remove_pattern(self):
        """Test removing pattern."""
        custom_patterns = {
            'removable': "Remove me",
            'keep': "Keep me",
        }
        
        template = CompetitionTextTemplate(
            competition_type=CompetitionType.REGRESSION,
            template_name="test",
            template_patterns=custom_patterns,
        )
        
        success = template.remove_pattern("removable")
        
        assert success == True
        assert "removable" not in template.template_patterns
        assert "keep" in template.template_patterns
        
    def test_remove_nonexistent_pattern(self):
        """Test removing non-existent pattern."""
        template = CompetitionTextTemplate(
            competition_type=CompetitionType.BINARY_CLASSIFICATION,
            template_name="test",
        )
        
        success = template.remove_pattern("nonexistent")
        assert success == False
        
    def test_validate_template_valid(self, sample_data):
        """Test template validation with valid template."""
        template = CompetitionTextTemplate(
            competition_type=CompetitionType.BINARY_CLASSIFICATION,
            template_name="test",
        )
        
        is_valid, issues = template.validate_template(sample_data)
        
        assert is_valid == True
        assert len(issues) == 0
        
    def test_validate_template_invalid_columns(self, sample_data):
        """Test template validation with invalid column references."""
        custom_patterns = {
            'invalid': "Missing: {nonexistent_column}",
        }
        
        template = CompetitionTextTemplate(
            competition_type=CompetitionType.BINARY_CLASSIFICATION,
            template_name="test",
            template_patterns=custom_patterns,
        )
        
        is_valid, issues = template.validate_template(sample_data)
        
        assert is_valid == False
        assert len(issues) > 0
        assert any("nonexistent_column" in issue for issue in issues)
        
    def test_get_template_info(self):
        """Test getting template information."""
        template = CompetitionTextTemplate(
            competition_type=CompetitionType.MULTICLASS_CLASSIFICATION,
            template_name="info_test",
            description="Test description",
        )
        
        info = template.get_template_info()
        
        assert info['template_name'] == "info_test"
        assert info['competition_type'] == "multiclass_classification"
        assert info['description'] == "Test description"
        assert 'num_patterns' in info
        assert 'version' in info


class TestTextTemplateEngine:
    """Test TextTemplateEngine class."""
    
    @pytest.fixture
    def engine(self):
        """Create TextTemplateEngine instance."""
        return TextTemplateEngine()
        
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35],
            'city': ['NYC', 'LA', 'Chicago'],
            'score': [85.5, 92.0, 78.5],
            'target': [1, 0, 1]
        })
        
    def test_engine_creation(self, engine):
        """Test engine creation."""
        assert engine is not None
        assert hasattr(engine, '_templates')
        assert hasattr(engine, '_converters')
        assert len(engine._templates) == 0
        
    def test_register_template(self, engine):
        """Test registering a template."""
        template = CompetitionTextTemplate(
            competition_type=CompetitionType.BINARY_CLASSIFICATION,
            template_name="test_template",
        )
        
        engine.register_template(template)
        
        assert "test_template" in engine._templates
        assert engine._templates["test_template"] == template
        
    def test_register_duplicate_template(self, engine):
        """Test registering duplicate template."""
        template1 = CompetitionTextTemplate(
            competition_type=CompetitionType.BINARY_CLASSIFICATION,
            template_name="duplicate",
        )
        template2 = CompetitionTextTemplate(
            competition_type=CompetitionType.REGRESSION,
            template_name="duplicate",
        )
        
        engine.register_template(template1)
        
        # Should not register duplicate
        with pytest.raises(ValueError, match="already registered"):
            engine.register_template(template2)
            
    def test_register_template_force_overwrite(self, engine):
        """Test force overwriting existing template."""
        template1 = CompetitionTextTemplate(
            competition_type=CompetitionType.BINARY_CLASSIFICATION,
            template_name="overwrite_test",
        )
        template2 = CompetitionTextTemplate(
            competition_type=CompetitionType.REGRESSION,
            template_name="overwrite_test",
        )
        
        engine.register_template(template1)
        engine.register_template(template2, force=True)
        
        # Should have been overwritten
        assert engine._templates["overwrite_test"].competition_type == CompetitionType.REGRESSION
        
    def test_get_template_exists(self, engine):
        """Test getting existing template."""
        template = CompetitionTextTemplate(
            competition_type=CompetitionType.BINARY_CLASSIFICATION,
            template_name="get_test",
        )
        
        engine.register_template(template)
        retrieved = engine.get_template("get_test")
        
        assert retrieved == template
        
    def test_get_template_missing(self, engine):
        """Test getting non-existent template."""
        template = engine.get_template("missing")
        assert template is None
        
    def test_list_templates_empty(self, engine):
        """Test listing templates when empty."""
        templates = engine.list_templates()
        assert templates == []
        
    def test_list_templates_multiple(self, engine):
        """Test listing multiple templates."""
        comp_types = [
            CompetitionType.BINARY_CLASSIFICATION,
            CompetitionType.REGRESSION,
            CompetitionType.MULTICLASS_CLASSIFICATION,
        ]
        
        for i, comp_type in enumerate(comp_types):
            template = CompetitionTextTemplate(
                competition_type=comp_type,
                template_name=f"template_{i}",
            )
            engine.register_template(template)
            
        templates = engine.list_templates()
        
        assert len(templates) == 3
        assert all(name.startswith('template_') for name in templates)
        
    def test_list_templates_filtered(self, engine):
        """Test listing templates with filter."""
        comp_types = [
            CompetitionType.BINARY_CLASSIFICATION,
            CompetitionType.REGRESSION,
            CompetitionType.BINARY_CLASSIFICATION,
        ]
        
        for i, comp_type in enumerate(comp_types):
            template = CompetitionTextTemplate(
                competition_type=comp_type,
                template_name=f"template_{i}",
            )
            engine.register_template(template)
            
        # Filter by binary classification
        binary_templates = engine.list_templates(
            competition_type=CompetitionType.BINARY_CLASSIFICATION
        )
        
        assert len(binary_templates) == 2
        assert 'template_0' in binary_templates
        assert 'template_2' in binary_templates
        
    def test_convert_data_with_template(self, engine, sample_data):
        """Test converting data with registered template."""
        template = CompetitionTextTemplate(
            competition_type=CompetitionType.BINARY_CLASSIFICATION,
            template_name="convert_test",
        )
        
        engine.register_template(template)
        
        texts = engine.convert_data(
            data=sample_data,
            template_name="convert_test",
        )
        
        assert len(texts) == len(sample_data)
        assert all(isinstance(text, str) for text in texts)
        
    def test_convert_data_missing_template(self, engine, sample_data):
        """Test converting data with missing template."""
        with pytest.raises(ValueError, match="Template .* not found"):
            engine.convert_data(
                data=sample_data,
                template_name="missing_template",
            )
            
    def test_convert_data_auto_detect(self, engine, sample_data):
        """Test converting data with auto-detected template."""
        # Register template for binary classification
        template = CompetitionTextTemplate(
            competition_type=CompetitionType.BINARY_CLASSIFICATION,
            template_name="auto_template",
        )
        engine.register_template(template)
        
        # Should auto-detect and use the template
        texts = engine.convert_data(
            data=sample_data,
            competition_type=CompetitionType.BINARY_CLASSIFICATION,
        )
        
        assert len(texts) == len(sample_data)
        
    def test_register_converter(self, engine):
        """Test registering a converter."""
        converter = TabularTextConverter()
        
        engine.register_converter("tabular", converter)
        
        assert "tabular" in engine._converters
        assert engine._converters["tabular"] == converter
        
    def test_get_converter(self, engine):
        """Test getting registered converter."""
        converter = BERTTextConverter()
        engine.register_converter("bert", converter)
        
        retrieved = engine.get_converter("bert")
        assert retrieved == converter
        
    def test_get_converter_missing(self, engine):
        """Test getting non-existent converter."""
        converter = engine.get_converter("missing")
        assert converter is None
        
    def test_remove_template(self, engine):
        """Test removing template."""
        template = CompetitionTextTemplate(
            competition_type=CompetitionType.REGRESSION,
            template_name="remove_test",
        )
        
        engine.register_template(template)
        assert "remove_test" in engine._templates
        
        success = engine.remove_template("remove_test")
        
        assert success == True
        assert "remove_test" not in engine._templates
        
    def test_remove_template_missing(self, engine):
        """Test removing non-existent template."""
        success = engine.remove_template("missing")
        assert success == False
        
    def test_clear_templates(self, engine):
        """Test clearing all templates."""
        # Register multiple templates
        for i in range(3):
            template = CompetitionTextTemplate(
                competition_type=CompetitionType.BINARY_CLASSIFICATION,
                template_name=f"clear_test_{i}",
            )
            engine.register_template(template)
            
        assert len(engine._templates) == 3
        
        engine.clear_templates()
        
        assert len(engine._templates) == 0
        
    def test_create_default_templates(self, engine):
        """Test creating default templates."""
        engine.create_default_templates()
        
        # Should have templates for common competition types
        templates = engine.list_templates()
        assert len(templates) > 0
        
        # Should have binary classification template
        binary_templates = engine.list_templates(
            competition_type=CompetitionType.BINARY_CLASSIFICATION
        )
        assert len(binary_templates) > 0
        
    def test_load_templates_from_config(self, engine, test_data_dir):
        """Test loading templates from configuration file."""
        # Create config file
        config = {
            "templates": [
                {
                    "template_name": "config_template",
                    "competition_type": "binary_classification",
                    "description": "From config",
                    "template_patterns": {
                        "default": "Config: {name} is {age} years old"
                    }
                }
            ]
        }
        
        config_file = test_data_dir / "template_config.json"
        import json
        with open(config_file, 'w') as f:
            json.dump(config, f)
            
        engine.load_templates_from_config(config_file)
        
        assert "config_template" in engine._templates
        template = engine._templates["config_template"]
        assert template.description == "From config"
        
    def test_save_templates_to_config(self, engine, test_data_dir):
        """Test saving templates to configuration file."""
        # Register a template
        template = CompetitionTextTemplate(
            competition_type=CompetitionType.REGRESSION,
            template_name="save_template",
            description="To be saved",
        )
        engine.register_template(template)
        
        # Save to config
        config_file = test_data_dir / "saved_config.json"
        engine.save_templates_to_config(config_file)
        
        # Verify file was created
        assert config_file.exists()
        
        # Load and verify content
        import json
        with open(config_file, 'r') as f:
            config = json.load(f)
            
        assert "templates" in config
        assert len(config["templates"]) == 1
        assert config["templates"][0]["template_name"] == "save_template"
        
    def test_validate_all_templates(self, engine, sample_data):
        """Test validating all templates."""
        # Register valid template
        valid_template = CompetitionTextTemplate(
            competition_type=CompetitionType.BINARY_CLASSIFICATION,
            template_name="valid",
        )
        engine.register_template(valid_template)
        
        # Register invalid template
        invalid_template = CompetitionTextTemplate(
            competition_type=CompetitionType.REGRESSION,
            template_name="invalid",
            template_patterns={"default": "Missing: {nonexistent}"}
        )
        engine.register_template(invalid_template)
        
        validation_results = engine.validate_all_templates(sample_data)
        
        assert "valid" in validation_results
        assert "invalid" in validation_results
        assert validation_results["valid"]["is_valid"] == True
        assert validation_results["invalid"]["is_valid"] == False
        
    def test_get_template_suggestions(self, engine, sample_data):
        """Test getting template suggestions for data."""
        # Register templates for different types
        comp_types = [
            CompetitionType.BINARY_CLASSIFICATION,
            CompetitionType.REGRESSION,
            CompetitionType.MULTICLASS_CLASSIFICATION,
        ]
        
        for i, comp_type in enumerate(comp_types):
            template = CompetitionTextTemplate(
                competition_type=comp_type,
                template_name=f"suggestion_{i}",
            )
            engine.register_template(template)
            
        suggestions = engine.get_template_suggestions(
            data=sample_data,
            competition_type=CompetitionType.BINARY_CLASSIFICATION,
        )
        
        assert len(suggestions) > 0
        assert all(suggestion['competition_type'] == "binary_classification" 
                  for suggestion in suggestions)
                  
    def test_engine_statistics(self, engine):
        """Test getting engine statistics."""
        # Register templates
        comp_types = [
            CompetitionType.BINARY_CLASSIFICATION,
            CompetitionType.REGRESSION,
            CompetitionType.BINARY_CLASSIFICATION,
        ]
        
        for i, comp_type in enumerate(comp_types):
            template = CompetitionTextTemplate(
                competition_type=comp_type,
                template_name=f"stats_{i}",
            )
            engine.register_template(template)
            
        stats = engine.get_statistics()
        
        assert stats['total_templates'] == 3
        assert stats['templates_by_type']['binary_classification'] == 2
        assert stats['templates_by_type']['regression'] == 1
        assert 'total_converters' in stats