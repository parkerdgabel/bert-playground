"""Tests for MLX streaming functionality."""

import pytest
import pandas as pd
import tempfile
import mlx.core as mx
from pathlib import Path
from unittest.mock import Mock, patch

from data.mlx_streaming import (
    MLXStreamConfig,
    MLXCSVStreamer,
    MLXDataPipeline,
    create_mlx_pipeline,
)
from data.dataset_spec import (
    KaggleDatasetSpec,
    ProblemType,
    OptimizationProfile,
)


class TestMLXStreamConfig:
    """Test MLX stream configuration."""
    
    def test_development_profile(self):
        """Test development profile optimization settings."""
        config = MLXStreamConfig(
            optimization_profile=OptimizationProfile.DEVELOPMENT,
            batch_size=32,
        )
        
        # Should use minimal optimization
        assert config.prefetch_size == 2
        assert config.num_threads == 2
        assert config.buffer_size == 100
        assert config.enable_dynamic_batching is False
        assert config.enable_shape_optimization is False
    
    def test_production_profile(self):
        """Test production profile optimization settings."""
        config = MLXStreamConfig(
            optimization_profile=OptimizationProfile.PRODUCTION,
            batch_size=32,
        )
        
        # Should use balanced optimization
        assert config.prefetch_size >= 4
        assert config.num_threads >= 4
        assert config.buffer_size >= 500
    
    def test_competition_profile(self):
        """Test competition profile optimization settings."""
        config = MLXStreamConfig(
            optimization_profile=OptimizationProfile.COMPETITION,
            batch_size=32,
        )
        
        # Should use maximum optimization
        assert config.prefetch_size >= 8
        assert config.num_threads >= 8
        assert config.buffer_size >= 2000
        assert config.enable_dynamic_batching is True
        assert config.enable_shape_optimization is True


class TestMLXCSVStreamer:
    """Test MLX CSV streaming functionality."""
    
    @pytest.fixture
    def sample_csv(self):
        """Create a sample CSV file for testing."""
        data = {
            'id': [1, 2, 3, 4, 5],
            'feature1': [1.1, 2.2, 3.3, 4.4, 5.5],
            'feature2': ['A', 'B', 'A', 'C', 'B'],
            'text_feature': ['Hello world', 'Test text', 'Another example', 'Sample data', 'Final entry'],
            'target': [0, 1, 0, 1, 0]
        }
        
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            csv_path = f.name
        
        yield csv_path
        
        # Cleanup
        Path(csv_path).unlink()
    
    @pytest.fixture
    def sample_dataset_spec(self):
        """Create a sample dataset specification."""
        return KaggleDatasetSpec(
            name="test_dataset",
            problem_type=ProblemType.BINARY_CLASSIFICATION,
            target_column="target",
            categorical_columns=["feature2"],
            numerical_columns=["feature1"],
            text_columns=["text_feature"],
            id_columns=["id"],
            optimization_profile=OptimizationProfile.DEVELOPMENT,
        )
    
    def test_column_determination(self, sample_csv, sample_dataset_spec):
        """Test that correct columns are selected for loading."""
        streamer = MLXCSVStreamer(
            csv_path=sample_csv,
            dataset_spec=sample_dataset_spec,
        )
        
        # Should include target and feature columns, but not ID columns
        expected_columns = ["target", "feature1", "feature2", "text_feature"]
        assert set(streamer.columns_to_load) == set(expected_columns)
    
    @patch('data.mlx_streaming.AutoTokenizer')
    def test_streamer_initialization(self, mock_tokenizer, sample_csv, sample_dataset_spec):
        """Test MLX CSV streamer initialization."""
        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "[EOS]"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        streamer = MLXCSVStreamer(
            csv_path=sample_csv,
            dataset_spec=sample_dataset_spec,
        )
        
        assert streamer.csv_path == Path(sample_csv)
        assert streamer.dataset_spec == sample_dataset_spec
        assert mock_tokenizer.from_pretrained.called
        
        # Should set pad_token if None
        assert mock_tokenizer_instance.pad_token == "[EOS]"
    
    def test_tokenize_sample(self, sample_csv, sample_dataset_spec):
        """Test sample tokenization."""
        with patch('data.mlx_streaming.AutoTokenizer') as mock_tokenizer:
            # Mock tokenizer behavior
            mock_tokenizer_instance = Mock()
            mock_tokenizer_instance.pad_token = "[PAD]"
            
            # Mock tokenization result
            mock_tokens = {
                "input_ids": [[101, 102, 103, 0, 0]],  # Mock token IDs
                "attention_mask": [[1, 1, 1, 0, 0]]    # Mock attention mask
            }
            mock_tokenizer_instance.return_value = mock_tokens
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            streamer = MLXCSVStreamer(
                csv_path=sample_csv,
                dataset_spec=sample_dataset_spec,
            )
            
            # Test tokenization
            sample = {"text": "Hello world", "target": 1}
            result = streamer._tokenize_sample(sample)
            
            assert "input_ids" in result
            assert "attention_mask" in result
            assert "labels" in result
            assert result["labels"] == 1


class TestMLXDataPipeline:
    """Test complete MLX data pipeline."""
    
    @pytest.fixture
    def sample_csv(self):
        """Create a sample CSV file for testing."""
        data = {
            'feature1': [1.1, 2.2, 3.3],
            'feature2': ['A', 'B', 'C'],
            'target': [0, 1, 0]
        }
        
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            csv_path = f.name
        
        yield csv_path
        
        # Cleanup
        Path(csv_path).unlink()
    
    @pytest.fixture
    def sample_dataset_spec(self):
        """Create a sample dataset specification."""
        return KaggleDatasetSpec(
            name="test_pipeline",
            problem_type=ProblemType.BINARY_CLASSIFICATION,
            target_column="target",
            categorical_columns=["feature2"],
            numerical_columns=["feature1"],
            optimization_profile=OptimizationProfile.DEVELOPMENT,
        )
    
    @patch('data.mlx_streaming.AutoTokenizer')
    def test_pipeline_initialization(self, mock_tokenizer, sample_csv, sample_dataset_spec):
        """Test MLX data pipeline initialization."""
        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = "[PAD]"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        pipeline = MLXDataPipeline(
            csv_path=sample_csv,
            dataset_spec=sample_dataset_spec,
        )
        
        assert pipeline.csv_path == sample_csv
        assert pipeline.dataset_spec == sample_dataset_spec
        assert pipeline.streamer is not None
    
    @patch('data.mlx_streaming.AutoTokenizer')
    def test_get_dataset_info(self, mock_tokenizer, sample_csv, sample_dataset_spec):
        """Test dataset info retrieval."""
        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = "[PAD]"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        pipeline = MLXDataPipeline(
            csv_path=sample_csv,
            dataset_spec=sample_dataset_spec,
        )
        
        info = pipeline.get_dataset_info()
        
        assert "csv_path" in info
        assert "dataset_spec" in info
        assert "config" in info
        assert "tokenizer" in info
        
        assert info["csv_path"] == sample_csv
        assert info["dataset_spec"]["name"] == "test_pipeline"


class TestCreateMLXPipeline:
    """Test convenience function for creating MLX pipeline."""
    
    @pytest.fixture
    def sample_csv(self):
        """Create a sample CSV file for testing."""
        data = {
            'PassengerId': [1, 2, 3],
            'Survived': [0, 1, 0],
            'Pclass': [3, 1, 3],
            'Name': ['Braund, Mr. Owen Harris', 'Cumings, Mrs. John', 'Test Name'],
            'Sex': ['male', 'female', 'male'],
            'Age': [22, 38, 26],
        }
        
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            csv_path = f.name
        
        yield csv_path
        
        # Cleanup
        Path(csv_path).unlink()
    
    @patch('data.mlx_streaming.AutoTokenizer')
    def test_create_with_auto_detection(self, mock_tokenizer, sample_csv):
        """Test creating pipeline with auto dataset detection."""
        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = "[PAD]"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        pipeline = create_mlx_pipeline(
            csv_path=sample_csv,
            target_column="Survived"
        )
        
        assert pipeline is not None
        assert pipeline.dataset_spec.target_column == "Survived"
        assert pipeline.dataset_spec.problem_type == ProblemType.BINARY_CLASSIFICATION
    
    def test_create_missing_target_column(self, sample_csv):
        """Test that missing target column raises error."""
        with pytest.raises(ValueError, match="Either dataset_spec or target_column must be provided"):
            create_mlx_pipeline(csv_path=sample_csv)
    
    @patch('data.mlx_streaming.AutoTokenizer')
    def test_create_with_provided_spec(self, mock_tokenizer, sample_csv):
        """Test creating pipeline with provided dataset spec."""
        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = "[PAD]"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        spec = KaggleDatasetSpec(
            name="custom_spec",
            problem_type=ProblemType.BINARY_CLASSIFICATION,
            target_column="Survived",
            categorical_columns=["Sex"],
            numerical_columns=["Age"],
        )
        
        pipeline = create_mlx_pipeline(
            csv_path=sample_csv,
            dataset_spec=spec
        )
        
        assert pipeline.dataset_spec.name == "custom_spec"
        assert pipeline.dataset_spec == spec