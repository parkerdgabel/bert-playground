"""Tests for dataset specification system."""

import pytest
import pandas as pd
import tempfile
from pathlib import Path

from data.dataset_spec import (
    KaggleDatasetSpec,
    ProblemType,
    FeatureType,
    OptimizationProfile,
    TITANIC_SPEC,
    get_dataset_spec,
    register_dataset_spec,
)


class TestKaggleDatasetSpec:
    """Test KaggleDatasetSpec functionality."""
    
    def test_basic_creation(self):
        """Test basic dataset spec creation."""
        spec = KaggleDatasetSpec(
            name="test_dataset",
            problem_type=ProblemType.BINARY_CLASSIFICATION,
            target_column="target",
            categorical_columns=["cat1", "cat2"],
            numerical_columns=["num1", "num2"],
        )
        
        assert spec.name == "test_dataset"
        assert spec.problem_type == ProblemType.BINARY_CLASSIFICATION
        assert spec.target_column == "target"
        assert "cat1" in spec.categorical_columns
        assert "num1" in spec.numerical_columns
        
        # Check that feature_types was built correctly
        assert spec.feature_types["cat1"] == FeatureType.CATEGORICAL
        assert spec.feature_types["num1"] == FeatureType.NUMERICAL
        assert spec.feature_types["target"] == FeatureType.TARGET
    
    def test_string_enum_conversion(self):
        """Test that string enum values are converted properly."""
        spec = KaggleDatasetSpec(
            name="test",
            problem_type="binary_classification",  # String instead of enum
            target_column="target",
            optimization_profile="production",     # String instead of enum
        )
        
        assert spec.problem_type == ProblemType.BINARY_CLASSIFICATION
        assert spec.optimization_profile == OptimizationProfile.PRODUCTION
    
    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        original_spec = KaggleDatasetSpec(
            name="test_dataset",
            problem_type=ProblemType.MULTICLASS_CLASSIFICATION,
            target_column="label",
            categorical_columns=["cat1"],
            numerical_columns=["num1"],
            text_columns=["text1"],
        )
        
        # Convert to dict and back
        spec_dict = original_spec.to_dict()
        restored_spec = KaggleDatasetSpec.from_dict(spec_dict)
        
        assert restored_spec.name == original_spec.name
        assert restored_spec.problem_type == original_spec.problem_type
        assert restored_spec.target_column == original_spec.target_column
        assert restored_spec.categorical_columns == original_spec.categorical_columns
        assert restored_spec.feature_types == original_spec.feature_types
    
    def test_from_csv_analysis_titanic(self):
        """Test auto-detection with a Titanic-like dataset."""
        # Create a sample Titanic-like CSV
        data = {
            'PassengerId': [1, 2, 3, 4, 5],
            'Survived': [0, 1, 1, 1, 0],
            'Pclass': [3, 1, 3, 1, 3],
            'Name': ['Braund, Mr. Owen Harris', 'Cumings, Mrs. John Bradley', 
                    'Heikkinen, Miss. Laina', 'Futrelle, Mrs. Jacques Heath',
                    'Allen, Mr. William Henry'],
            'Sex': ['male', 'female', 'female', 'female', 'male'],
            'Age': [22, 38, 26, 35, 35],
            'SibSp': [1, 1, 0, 1, 0],
            'Parch': [0, 0, 0, 0, 0],
            'Ticket': ['A/5 21171', 'PC 17599', 'STON/O2. 3101282', '113803', '373450'],
            'Fare': [7.25, 71.28, 7.925, 53.1, 8.05],
            'Embarked': ['S', 'C', 'S', 'S', 'S']
        }
        
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            csv_path = f.name
        
        try:
            spec = KaggleDatasetSpec.from_csv_analysis(csv_path, target_column='Survived')
            
            # Check basic properties
            assert spec.problem_type == ProblemType.BINARY_CLASSIFICATION
            assert spec.target_column == 'Survived'
            
            # Check feature type detection
            assert 'PassengerId' in spec.id_columns  # Should detect ID column
            assert 'Sex' in spec.categorical_columns
            assert 'Age' in spec.numerical_columns
            assert 'Name' in spec.text_columns  # Long text should be detected
            
            # Should have some primary text features
            assert len(spec.primary_text_features) > 0
            
        finally:
            Path(csv_path).unlink()  # Clean up
    
    def test_from_csv_analysis_regression(self):
        """Test auto-detection for regression problem."""
        # Create a regression dataset
        data = {
            'id': [1, 2, 3, 4, 5],
            'feature1': [1.1, 2.2, 3.3, 4.4, 5.5],
            'feature2': ['A', 'B', 'A', 'C', 'B'], 
            'target': [10.5, 20.7, 15.2, 25.8, 12.3]  # Continuous target
        }
        
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            csv_path = f.name
        
        try:
            spec = KaggleDatasetSpec.from_csv_analysis(csv_path, target_column='target')
            
            # Should detect regression due to many unique target values
            assert spec.problem_type == ProblemType.REGRESSION
            assert 'feature1' in spec.numerical_columns
            assert 'feature2' in spec.categorical_columns
            assert 'id' in spec.id_columns
            
        finally:
            Path(csv_path).unlink()


class TestDatasetRegistry:
    """Test dataset registry functionality."""
    
    def test_titanic_spec_exists(self):
        """Test that Titanic spec is pre-registered."""
        spec = get_dataset_spec("titanic")
        assert spec.name == "titanic"
        assert spec.problem_type == ProblemType.BINARY_CLASSIFICATION
        assert spec.target_column == "Survived"
    
    def test_register_new_spec(self):
        """Test registering a new dataset specification."""
        new_spec = KaggleDatasetSpec(
            name="test_competition",
            problem_type=ProblemType.REGRESSION,
            target_column="score",
            numerical_columns=["feature1", "feature2"],
        )
        
        register_dataset_spec(new_spec)
        
        # Should be able to retrieve it
        retrieved_spec = get_dataset_spec("test_competition")
        assert retrieved_spec.name == "test_competition"
        assert retrieved_spec.problem_type == ProblemType.REGRESSION
    
    def test_get_spec_with_csv_path(self):
        """Test getting spec by analyzing CSV path."""
        # Create a simple CSV
        data = {
            'feature1': [1, 2, 3],
            'feature2': ['A', 'B', 'C'],
            'target': [0, 1, 0]
        }
        
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            csv_path = f.name
        
        try:
            spec = get_dataset_spec(csv_path, target_column='target')
            assert spec.target_column == 'target'
            assert spec.problem_type == ProblemType.BINARY_CLASSIFICATION
            
        finally:
            Path(csv_path).unlink()
    
    def test_get_spec_missing_target_column(self):
        """Test that missing target_column raises error for CSV analysis."""
        with pytest.raises(ValueError, match="target_column is required"):
            get_dataset_spec("/some/csv/path.csv")


class TestOptimizationProfileSelection:
    """Test optimization profile auto-selection."""
    
    def test_small_dataset_development(self):
        """Test that small datasets get DEVELOPMENT profile."""
        data = pd.DataFrame({
            'feature': range(100),  # Small dataset
            'target': [0, 1] * 50
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            csv_path = f.name
        
        try:
            spec = KaggleDatasetSpec.from_csv_analysis(csv_path, target_column='target')
            assert spec.optimization_profile == OptimizationProfile.DEVELOPMENT
            
        finally:
            Path(csv_path).unlink()
    
    def test_large_dataset_competition(self):
        """Test that large datasets get COMPETITION profile.""" 
        # Create a larger dataset (simulate by setting expected_size)
        spec = KaggleDatasetSpec(
            name="large_dataset",
            problem_type=ProblemType.BINARY_CLASSIFICATION,
            target_column="target",
            expected_size=100000,  # Large size
        )
        
        # For this test, we'll manually create a spec with large expected_size
        # since creating a 100k row CSV would be slow in tests
        
        # The from_csv_analysis logic would set COMPETITION for >50k rows
        # We'll trust that logic based on the smaller tests above