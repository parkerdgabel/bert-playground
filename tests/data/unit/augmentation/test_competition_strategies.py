"""Tests for competition-specific augmentation strategies."""

import pytest

from data.augmentation.competition_strategies import (
    CompetitionTemplateAugmenter,
    TitanicAugmenter,
)
from data.augmentation.config import AugmentationConfig


class TestTitanicAugmenter:
    """Test TitanicAugmenter class."""

    def test_basic_conversion(self):
        """Test basic text conversion for Titanic data."""
        augmenter = TitanicAugmenter()
        
        sample = {
            "Name": "Smith, Mr. John",
            "Age": 30,
            "Sex": "male",
            "Pclass": 1,
            "Fare": 50.0,
            "Embarked": "S",
            "SibSp": 1,
            "Parch": 0,
        }
        
        text = augmenter.augment(sample)
        
        assert isinstance(text, str)
        assert "Smith, Mr. John" in text
        assert "30" in text
        assert "male" in text or "man" in text
        assert "first-class" in text
        assert "50" in text
        assert "Southampton" in text

    def test_missing_values(self):
        """Test handling missing values."""
        augmenter = TitanicAugmenter()
        
        sample = {
            "Name": "Unknown",
            "Age": None,
            "Sex": "female",
            "Pclass": 2,
        }
        
        text = augmenter.augment(sample)
        
        assert isinstance(text, str)
        assert "Unknown" in text
        assert "woman" in text
        assert "second-class" in text
        # Should handle missing age gracefully
        assert "unknown age" in text or "age unknown" in text

    def test_title_extraction(self):
        """Test passenger title extraction."""
        augmenter = TitanicAugmenter()
        
        test_cases = [
            ("Smith, Mr. John", "Mr"),
            ("Johnson, Mrs. Mary", "Mrs"),
            ("Williams, Miss. Jane", "Miss"),
            ("Brown, Master. Tom", "Master"),
            ("Davis, Dr. Robert", "Dr"),
        ]
        
        for name, expected_title in test_cases:
            sample = {"Name": name, "Sex": "male", "Pclass": 1}
            text = augmenter.augment(sample)
            assert expected_title in text or expected_title.lower() in text

    def test_embarkation_mapping(self):
        """Test embarkation port mapping."""
        augmenter = TitanicAugmenter()
        
        mappings = {
            "S": "Southampton",
            "C": "Cherbourg",
            "Q": "Queenstown",
        }
        
        for code, port in mappings.items():
            sample = {"Name": "Test", "Embarked": code, "Pclass": 1}
            text = augmenter.augment(sample)
            assert port in text

    def test_family_relationships(self):
        """Test family relationship descriptions."""
        augmenter = TitanicAugmenter()
        
        # Traveling alone
        sample = {"Name": "Solo", "SibSp": 0, "Parch": 0, "Pclass": 1}
        text = augmenter.augment(sample)
        assert "alone" in text
        
        # With family
        sample = {"Name": "Family", "SibSp": 2, "Parch": 1, "Pclass": 1}
        text = augmenter.augment(sample)
        assert "family" in text
        assert ("3" in text or "three" in text)  # Total family size

    def test_fare_formatting(self):
        """Test fare formatting."""
        augmenter = TitanicAugmenter()
        
        sample = {"Name": "Test", "Fare": 123.456, "Pclass": 1}
        text = augmenter.augment(sample)
        assert "$123.46" in text

    def test_long_name_truncation(self):
        """Test handling of very long names."""
        augmenter = TitanicAugmenter()
        
        long_name = "A" * 100  # Very long name
        sample = {"Name": long_name, "Pclass": 1}
        text = augmenter.augment(sample)
        
        # Should truncate with ellipsis
        assert "..." in text
        assert len(text) < 1000  # Reasonable length

    def test_augmentation_with_config(self):
        """Test augmentation with configuration."""
        config = AugmentationConfig()
        augmenter = TitanicAugmenter(config=config, seed=42)
        
        sample = {"Name": "Test", "Age": 25, "Sex": "male", "Pclass": 1}
        
        # Generate multiple versions
        texts = [augmenter.augment(sample) for _ in range(5)]
        
        # All should be valid strings
        assert all(isinstance(text, str) for text in texts)
        assert all(len(text) > 0 for text in texts)


class TestCompetitionTemplateAugmenter:
    """Test CompetitionTemplateAugmenter class."""

    def test_from_competition_name(self):
        """Test creating augmenter from competition name."""
        # Test with known competition
        augmenter = CompetitionTemplateAugmenter.from_competition_name("titanic")
        assert augmenter is not None
        assert augmenter.competition_name == "titanic"
        
        # Test with unknown competition
        augmenter = CompetitionTemplateAugmenter.from_competition_name("unknown")
        assert augmenter is not None
        assert augmenter.competition_name == "unknown"

    def test_generic_template(self):
        """Test generic template conversion."""
        augmenter = CompetitionTemplateAugmenter(
            competition_name="generic",
            feature_order=["feature1", "feature2", "feature3"],
        )
        
        sample = {
            "feature1": "value1",
            "feature2": 42,
            "feature3": 3.14,
        }
        
        text = augmenter.augment(sample)
        
        assert isinstance(text, str)
        assert "feature1: value1" in text
        assert "feature2: 42" in text
        assert "feature3: 3.14" in text

    def test_custom_template(self):
        """Test custom template function."""
        def custom_template(features):
            return f"Custom: {features.get('x', 0)} and {features.get('y', 0)}"
        
        augmenter = CompetitionTemplateAugmenter(
            competition_name="custom",
            template_fn=custom_template,
        )
        
        sample = {"x": 10, "y": 20}
        text = augmenter.augment(sample)
        
        assert text == "Custom: 10 and 20"

    def test_feature_descriptions(self):
        """Test with feature descriptions."""
        augmenter = CompetitionTemplateAugmenter(
            competition_name="test",
            feature_descriptions={
                "age": "Age of the person",
                "income": "Annual income in USD",
            },
        )
        
        sample = {"age": 30, "income": 50000}
        text = augmenter.augment(sample)
        
        assert isinstance(text, str)
        # Descriptions might be used in the text
        assert "30" in text
        assert "50000" in text

    def test_feature_ordering(self):
        """Test that features are ordered correctly."""
        augmenter = CompetitionTemplateAugmenter(
            competition_name="ordered",
            feature_order=["c", "a", "b"],
        )
        
        sample = {"a": 1, "b": 2, "c": 3}
        text = augmenter.augment(sample)
        
        # Check that 'c' appears before 'a' and 'b' in the text
        c_pos = text.find("c:")
        a_pos = text.find("a:")
        b_pos = text.find("b:")
        
        assert c_pos < a_pos < b_pos

    def test_numerical_formatting(self):
        """Test numerical value formatting."""
        augmenter = CompetitionTemplateAugmenter(
            competition_name="numbers",
            format_numbers=True,
        )
        
        sample = {
            "integer": 1000000,
            "float": 3.14159265,
            "small": 0.0001,
        }
        
        text = augmenter.augment(sample)
        
        assert isinstance(text, str)
        # Should format large numbers with commas or similar
        assert "1000000" in text or "1,000,000" in text
        # Should limit decimal places
        assert "3.14159265" in text or "3.14" in text

    def test_missing_features(self):
        """Test handling missing features."""
        augmenter = CompetitionTemplateAugmenter(
            competition_name="missing",
            feature_order=["a", "b", "c"],
        )
        
        sample = {"a": 1, "c": 3}  # 'b' is missing
        text = augmenter.augment(sample)
        
        assert isinstance(text, str)
        assert "a: 1" in text
        assert "c: 3" in text
        # Should handle missing 'b' gracefully
        assert "b: N/A" in text or "b: missing" in text or "b:" not in text

    def test_batch_augmentation(self):
        """Test batch augmentation."""
        augmenter = CompetitionTemplateAugmenter(
            competition_name="batch_test"
        )
        
        samples = [
            {"x": 1, "y": 2},
            {"x": 3, "y": 4},
            {"x": 5, "y": 6},
        ]
        
        texts = augmenter.augment_batch(samples)
        
        assert len(texts) == len(samples)
        assert all(isinstance(text, str) for text in texts)
        
        # Check that values are preserved
        assert "x: 1" in texts[0]
        assert "x: 3" in texts[1]
        assert "x: 5" in texts[2]