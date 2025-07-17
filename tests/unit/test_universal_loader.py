"""Tests for universal Kaggle loader."""

import pytest
import pandas as pd
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

from data.universal_loader import (
    UniversalTextGenerator,
    UniversalKaggleLoader,
    TextGenerationStrategy,
    create_universal_loader,
    create_titanic_loader,
)
from data.dataset_spec import (
    KaggleDatasetSpec,
    ProblemType,
    FeatureType,
    OptimizationProfile,
)


class TestUniversalTextGenerator:
    """Test universal text generation."""
    
    @pytest.fixture
    def sample_dataset_spec(self):
        """Create a sample dataset specification."""
        return KaggleDatasetSpec(
            name="test_dataset",
            problem_type=ProblemType.BINARY_CLASSIFICATION,
            target_column="target",
            categorical_columns=["cat1", "cat2"],
            numerical_columns=["num1", "num2"],
            text_columns=["text1"],
            optimization_profile=OptimizationProfile.DEVELOPMENT,
        )
    
    def test_auto_strategy_selection(self, sample_dataset_spec):
        """Test automatic strategy selection."""
        generator = UniversalTextGenerator(
            dataset_spec=sample_dataset_spec,
            strategy=TextGenerationStrategy.AUTO,
        )
        
        # Should select structured for general datasets
        assert generator.strategy == TextGenerationStrategy.STRUCTURED
    
    def test_titanic_strategy_selection(self):
        """Test that Titanic dataset gets template-based strategy."""
        from data.dataset_spec import TITANIC_SPEC
        
        generator = UniversalTextGenerator(
            dataset_spec=TITANIC_SPEC,
            strategy=TextGenerationStrategy.AUTO,
        )
        
        assert generator.strategy == TextGenerationStrategy.TEMPLATE_BASED
    
    def test_feature_concatenation_generation(self, sample_dataset_spec):
        """Test feature concatenation text generation."""
        generator = UniversalTextGenerator(
            dataset_spec=sample_dataset_spec,
            strategy=TextGenerationStrategy.FEATURE_CONCATENATION,
        )
        
        row_dict = {
            "cat1": "A",
            "cat2": "B", 
            "num1": 1.5,
            "num2": 2,
            "text1": "Some text",
            "target": 1
        }
        
        text = generator.generate_text(row_dict)
        
        assert "cat1: A" in text
        assert "cat2: B" in text
        assert "num1: 1.50" in text
        assert "num2: 2" in text
        assert "text1: Some text" in text
    
    def test_structured_generation(self, sample_dataset_spec):
        """Test structured text generation."""
        generator = UniversalTextGenerator(
            dataset_spec=sample_dataset_spec,
            strategy=TextGenerationStrategy.STRUCTURED,
        )
        
        row_dict = {
            "cat1": "A",
            "num1": 1.5,
            "target": 1
        }
        
        text = generator.generate_text(row_dict)
        
        assert "Data record:" in text
        assert "cat1=A" in text
        assert "num1=1.50" in text
    
    def test_narrative_generation(self, sample_dataset_spec):
        """Test narrative text generation."""
        generator = UniversalTextGenerator(
            dataset_spec=sample_dataset_spec,
            strategy=TextGenerationStrategy.NARRATIVE,
        )
        
        row_dict = {
            "text1": "This is a sample text",
            "cat1": "A",
            "num1": 1.5,
            "target": 1
        }
        
        text = generator.generate_text(row_dict)
        
        assert "This is a sample text" in text
        assert "Additional context:" in text
    
    def test_missing_values_handling(self, sample_dataset_spec):
        """Test handling of missing/null values."""
        generator = UniversalTextGenerator(
            dataset_spec=sample_dataset_spec,
            strategy=TextGenerationStrategy.FEATURE_CONCATENATION,
        )
        
        row_dict = {
            "cat1": "A",
            "cat2": None,  # Missing value
            "num1": 1.5,
            "num2": float('nan'),  # NaN value
            "target": 1
        }
        
        text = generator.generate_text(row_dict)
        
        # Should include non-missing values
        assert "cat1: A" in text
        assert "num1: 1.50" in text
        
        # Should not include missing values
        assert "cat2:" not in text
        assert "num2:" not in text


class TestUniversalKaggleLoader:
    """Test universal Kaggle loader functionality."""
    
    @pytest.fixture
    def sample_train_csv(self):
        """Create a sample training CSV file."""
        data = {
            'id': [1, 2, 3, 4, 5],
            'feature1': [1.1, 2.2, 3.3, 4.4, 5.5],
            'feature2': ['A', 'B', 'A', 'C', 'B'],
            'text_feature': ['Hello', 'World', 'Test', 'Sample', 'Data'],
            'target': [0, 1, 0, 1, 0]
        }
        
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            csv_path = f.name
        
        yield csv_path
        
        Path(csv_path).unlink()
    
    @pytest.fixture
    def sample_test_csv(self):
        """Create a sample test CSV file (without target)."""
        data = {
            'id': [6, 7, 8],
            'feature1': [6.6, 7.7, 8.8],
            'feature2': ['D', 'E', 'F'],
            'text_feature': ['Test1', 'Test2', 'Test3'],
        }
        
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            csv_path = f.name
        
        yield csv_path
        
        Path(csv_path).unlink()
    
    @patch('data.universal_loader.create_mlx_pipeline')
    def test_initialization_with_auto_detection(self, mock_create_pipeline, sample_train_csv):
        """Test loader initialization with auto dataset detection."""
        # Mock the pipeline creation
        mock_pipeline = Mock()
        mock_create_pipeline.return_value = mock_pipeline
        
        loader = UniversalKaggleLoader(
            train_path=sample_train_csv,
            target_column="target",
            batch_size=16,
        )
        
        assert loader.train_path == sample_train_csv
        assert loader.dataset_spec.target_column == "target"
        assert loader.dataset_spec.problem_type == ProblemType.BINARY_CLASSIFICATION
        assert loader.batch_size == 16
        
        # Should have created training pipeline
        assert mock_create_pipeline.called
    
    def test_initialization_with_provided_spec(self, sample_train_csv):
        """Test loader initialization with provided dataset spec."""
        spec = KaggleDatasetSpec(
            name="custom_dataset",
            problem_type=ProblemType.REGRESSION,
            target_column="target",
            numerical_columns=["feature1"],
            categorical_columns=["feature2"],
        )
        
        with patch('data.universal_loader.create_mlx_pipeline') as mock_create_pipeline:
            mock_pipeline = Mock()
            mock_create_pipeline.return_value = mock_pipeline
            
            loader = UniversalKaggleLoader(
                train_path=sample_train_csv,
                dataset_spec=spec,
            )
            
            assert loader.dataset_spec.name == "custom_dataset"
            assert loader.dataset_spec.problem_type == ProblemType.REGRESSION
    
    def test_initialization_missing_requirements(self):
        """Test that initialization fails with missing requirements."""
        # Should fail with no train_path and no dataset_spec
        with pytest.raises(ValueError, match="Either train_path or dataset_spec must be provided"):
            UniversalKaggleLoader()
        
        # Should fail with train_path but no target_column or dataset_spec
        with pytest.raises(ValueError, match="target_column is required"):
            UniversalKaggleLoader(train_path="some_path.csv")
    
    @patch('data.universal_loader.create_mlx_pipeline')
    def test_multiple_data_sources(self, mock_create_pipeline, sample_train_csv, sample_test_csv):
        """Test initialization with multiple data sources."""
        mock_pipeline = Mock()
        mock_create_pipeline.return_value = mock_pipeline
        
        loader = UniversalKaggleLoader(
            train_path=sample_train_csv,
            test_path=sample_test_csv,
            target_column="target",
        )
        
        assert loader.train_path == sample_train_csv
        assert loader.test_path == sample_test_csv
        
        # Should have created both pipelines
        assert mock_create_pipeline.call_count >= 2
    
    @patch('data.universal_loader.create_mlx_pipeline')
    def test_text_augmentation(self, mock_create_pipeline, sample_train_csv):
        """Test text augmentation functionality."""
        mock_pipeline = Mock()
        mock_create_pipeline.return_value = mock_pipeline
        
        loader = UniversalKaggleLoader(
            train_path=sample_train_csv,
            target_column="target",
            augment=True,
        )
        
        # Test augmentation function
        original_text = "Sample text"
        loader._is_training_transform = True
        augmented = loader._augment_text(original_text)
        
        # Should be different from original or contain original
        assert augmented != original_text or original_text in augmented
    
    @patch('data.universal_loader.create_mlx_pipeline')
    def test_get_dataset_info(self, mock_create_pipeline, sample_train_csv):
        """Test dataset info retrieval."""
        mock_pipeline = Mock()
        mock_pipeline.get_dataset_info.return_value = {"mock": "info"}
        mock_create_pipeline.return_value = mock_pipeline
        
        loader = UniversalKaggleLoader(
            train_path=sample_train_csv,
            target_column="target",
        )
        
        info = loader.get_dataset_info()
        
        assert "dataset_spec" in info
        assert "paths" in info
        assert "configuration" in info
        assert "mlx_config" in info
        assert info["paths"]["train"] == sample_train_csv
    
    @patch('data.universal_loader.create_mlx_pipeline')
    def test_config_save_and_load(self, mock_create_pipeline, sample_train_csv):
        """Test configuration saving and loading."""
        mock_pipeline = Mock()
        mock_create_pipeline.return_value = mock_pipeline
        
        loader = UniversalKaggleLoader(
            train_path=sample_train_csv,
            target_column="target",
            batch_size=64,
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
        
        try:
            # Save config
            loader.save_config(config_path)
            
            # Check that file was created
            assert Path(config_path).exists()
            
            # Load config
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            assert config["batch_size"] == 64
            assert config["train_path"] == sample_train_csv
            
            # Test loading from config
            with patch('data.universal_loader.create_mlx_pipeline'):
                loaded_loader = UniversalKaggleLoader.from_config(
                    config_path,
                    batch_size=32  # Override
                )
                assert loaded_loader.batch_size == 32  # Should use override
                
        finally:
            Path(config_path).unlink()


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    @pytest.fixture
    def sample_csv(self):
        """Create a sample CSV file."""
        data = {
            'feature1': [1, 2, 3],
            'feature2': ['A', 'B', 'C'],
            'target': [0, 1, 0]
        }
        
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            csv_path = f.name
        
        yield csv_path
        
        Path(csv_path).unlink()
    
    @patch('data.universal_loader.create_mlx_pipeline')
    def test_create_universal_loader(self, mock_create_pipeline, sample_csv):
        """Test create_universal_loader convenience function."""
        mock_pipeline = Mock()
        mock_create_pipeline.return_value = mock_pipeline
        
        loader = create_universal_loader(
            train_path=sample_csv,
            target_column="target",
            batch_size=16,
        )
        
        assert isinstance(loader, UniversalKaggleLoader)
        assert loader.train_path == sample_csv
        assert loader.batch_size == 16
    
    @patch('data.universal_loader.create_mlx_pipeline')
    def test_create_titanic_loader(self, mock_create_pipeline):
        """Test create_titanic_loader convenience function."""
        mock_pipeline = Mock()
        mock_create_pipeline.return_value = mock_pipeline
        
        loader = create_titanic_loader(
            train_path="custom/train.csv",
            test_path="custom/test.csv",
        )
        
        assert isinstance(loader, UniversalKaggleLoader)
        assert loader.dataset_spec.name == "titanic"
        assert loader.train_path == "custom/train.csv"
        assert loader.test_path == "custom/test.csv"


class TestTextTransformFunction:
    """Test text transformation functionality."""
    
    @pytest.fixture
    def sample_dataset_spec(self):
        """Create a sample dataset specification."""
        return KaggleDatasetSpec(
            name="test_dataset",
            problem_type=ProblemType.BINARY_CLASSIFICATION,
            target_column="target",
            categorical_columns=["feature2"],
            numerical_columns=["feature1"],
        )
    
    @patch('data.universal_loader.create_mlx_pipeline')
    def test_text_transform_function_creation(self, mock_create_pipeline, sample_dataset_spec):
        """Test that text transform function is properly created."""
        mock_pipeline = Mock()
        mock_create_pipeline.return_value = mock_pipeline
        
        loader = UniversalKaggleLoader(
            train_path="dummy_path.csv",
            dataset_spec=sample_dataset_spec,
        )
        
        # Test the transform function
        sample = {
            "feature1": 1.5,
            "feature2": "A",
            "target": 1
        }
        
        transformed = loader.text_transform_fn(sample)
        
        assert "text" in transformed
        assert isinstance(transformed["text"], str)
        assert len(transformed["text"]) > 0
    
    @patch('data.universal_loader.create_mlx_pipeline')
    def test_text_transform_error_handling(self, mock_create_pipeline, sample_dataset_spec):
        """Test text transform error handling."""
        mock_pipeline = Mock()
        mock_create_pipeline.return_value = mock_pipeline
        
        loader = UniversalKaggleLoader(
            train_path="dummy_path.csv",
            dataset_spec=sample_dataset_spec,
        )
        
        # Patch text generator to raise exception
        with patch.object(loader.text_generator, 'generate_text', side_effect=Exception("Test error")):
            sample = {"feature1": 1.5}
            transformed = loader.text_transform_fn(sample)
            
            assert transformed["text"] == "Error generating text"