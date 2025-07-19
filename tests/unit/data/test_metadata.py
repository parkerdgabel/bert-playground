"""Unit tests for competition metadata and dataset analyzer."""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from data.core.base import CompetitionType, DatasetSpec
from data.core.metadata import CompetitionMetadata, DatasetAnalyzer


class TestCompetitionMetadata:
    """Test CompetitionMetadata dataclass."""
    
    def test_basic_creation(self):
        """Test basic metadata creation."""
        metadata = CompetitionMetadata(
            competition_name="test_competition",
            competition_type=CompetitionType.BINARY_CLASSIFICATION,
            auto_detected=True,
            confidence_score=0.95,
        )
        
        assert metadata.competition_name == "test_competition"
        assert metadata.competition_type == CompetitionType.BINARY_CLASSIFICATION
        assert metadata.auto_detected == True
        assert metadata.confidence_score == 0.95
        
    def test_with_files(self):
        """Test metadata with file information."""
        metadata = CompetitionMetadata(
            competition_name="titanic",
            train_file="train.csv",
            test_file="test.csv",
            submission_file="sample_submission.csv",
            dataset_size_mb=23.5,
        )
        
        assert metadata.train_file == "train.csv"
        assert metadata.test_file == "test.csv"
        assert metadata.submission_file == "sample_submission.csv"
        assert metadata.dataset_size_mb == 23.5
        
    def test_optimization_recommendations(self):
        """Test optimization recommendation fields."""
        metadata = CompetitionMetadata(
            competition_name="test",
            recommended_batch_size=64,
            recommended_max_length=512,
            recommended_learning_rate=1e-5,
            use_unified_memory=True,
            optimal_prefetch_size=8,
        )
        
        assert metadata.recommended_batch_size == 64
        assert metadata.recommended_max_length == 512
        assert metadata.recommended_learning_rate == 1e-5
        assert metadata.use_unified_memory == True
        assert metadata.optimal_prefetch_size == 8
        
    def test_to_dict(self):
        """Test conversion to dictionary."""
        metadata = CompetitionMetadata(
            competition_name="test",
            competition_type=CompetitionType.REGRESSION,
            confidence_score=0.8,
        )
        
        data_dict = metadata.to_dict()
        
        assert isinstance(data_dict, dict)
        assert data_dict['competition_name'] == "test"
        assert data_dict['competition_type'] == "regression"
        assert data_dict['confidence_score'] == 0.8
        
    def test_from_dict(self):
        """Test creation from dictionary."""
        data_dict = {
            'competition_name': 'test',
            'competition_type': 'multiclass_classification',
            'auto_detected': False,
            'confidence_score': 0.7,
            'recommended_batch_size': 32,
        }
        
        metadata = CompetitionMetadata.from_dict(data_dict)
        
        assert metadata.competition_name == "test"
        assert metadata.competition_type == CompetitionType.MULTICLASS_CLASSIFICATION
        assert metadata.auto_detected == False
        assert metadata.confidence_score == 0.7
        assert metadata.recommended_batch_size == 32
        
    def test_save_load(self, test_data_dir):
        """Test saving and loading metadata."""
        metadata = CompetitionMetadata(
            competition_name="save_test",
            competition_type=CompetitionType.TIME_SERIES,
            confidence_score=0.9,
            train_file="train.csv",
        )
        
        save_path = test_data_dir / "metadata.json"
        metadata.save(save_path)
        
        # Verify file was created
        assert save_path.exists()
        
        # Load and verify
        loaded_metadata = CompetitionMetadata.load(save_path)
        
        assert loaded_metadata.competition_name == metadata.competition_name
        assert loaded_metadata.competition_type == metadata.competition_type
        assert loaded_metadata.confidence_score == metadata.confidence_score
        assert loaded_metadata.train_file == metadata.train_file


class TestDatasetAnalyzer:
    """Test DatasetAnalyzer class."""
    
    @pytest.fixture
    def analyzer(self):
        """Create DatasetAnalyzer instance."""
        return DatasetAnalyzer()
        
    def test_analyzer_creation(self, analyzer):
        """Test analyzer creation."""
        assert analyzer is not None
        assert hasattr(analyzer, '_find_data_files')
        assert hasattr(analyzer, '_detect_target_column')
        assert hasattr(analyzer, '_detect_competition_type')
        
    def test_find_data_files_single_file(self, test_data_dir, sample_titanic_data):
        """Test finding data files from single file."""
        analyzer = DatasetAnalyzer()
        
        # Create single CSV file
        csv_file = test_data_dir / "data.csv"
        sample_titanic_data.to_csv(csv_file, index=False)
        
        train_file, test_file, submission_file = analyzer._find_data_files(csv_file)
        
        assert train_file == csv_file
        assert test_file is None
        assert submission_file is None
        
    def test_find_data_files_directory(self, temp_csv_files):
        """Test finding data files from directory."""
        analyzer = DatasetAnalyzer()
        
        titanic_dir = temp_csv_files['titanic_train'].parent
        train_file, test_file, submission_file = analyzer._find_data_files(titanic_dir)
        
        assert train_file is not None
        assert train_file.name == "train.csv"
        assert test_file is not None
        assert test_file.name == "test.csv"
        
    def test_detect_target_column_common_names(self, sample_titanic_data):
        """Test target column detection with common names."""
        analyzer = DatasetAnalyzer()
        
        target_col = analyzer._detect_target_column(sample_titanic_data, None)
        assert target_col == "Survived"  # Should detect 'Survived' as target
        
    def test_detect_target_column_with_test_file(self, test_data_dir, sample_titanic_data):
        """Test target column detection using test file comparison."""
        analyzer = DatasetAnalyzer()
        
        # Create test file without target
        test_data = sample_titanic_data.drop('Survived', axis=1)
        test_file = test_data_dir / "test.csv"
        test_data.to_csv(test_file, index=False)
        
        target_col = analyzer._detect_target_column(sample_titanic_data, test_file)
        assert target_col == "Survived"
        
    def test_detect_competition_type_binary(self, sample_titanic_data):
        """Test binary classification detection."""
        analyzer = DatasetAnalyzer()
        
        comp_type, confidence = analyzer._detect_competition_type(
            sample_titanic_data, "Survived", "titanic"
        )
        
        assert comp_type == CompetitionType.BINARY_CLASSIFICATION
        assert confidence > 0.8
        
    def test_detect_competition_type_multiclass(self, sample_multiclass_data):
        """Test multiclass classification detection."""
        analyzer = DatasetAnalyzer()
        
        comp_type, confidence = analyzer._detect_competition_type(
            sample_multiclass_data, "target", "multiclass_test"
        )
        
        assert comp_type == CompetitionType.MULTICLASS_CLASSIFICATION
        assert confidence > 0.0
        
    def test_detect_competition_type_regression(self, sample_house_prices_data):
        """Test regression detection."""
        analyzer = DatasetAnalyzer()
        
        comp_type, confidence = analyzer._detect_competition_type(
            sample_house_prices_data, "SalePrice", "house_prices"
        )
        
        assert comp_type == CompetitionType.REGRESSION
        assert confidence > 0.0
        
    def test_detect_competition_type_by_name(self):
        """Test competition type detection by name patterns."""
        analyzer = DatasetAnalyzer()
        
        # Create dummy data
        data = pd.DataFrame({'feature': [1, 2, 3], 'target': [0, 1, 0]})
        
        # Test titanic pattern
        comp_type, confidence = analyzer._detect_competition_type(
            data, "target", "titanic_competition"
        )
        assert comp_type == CompetitionType.BINARY_CLASSIFICATION
        
        # Test house prices pattern
        comp_type, confidence = analyzer._detect_competition_type(
            data, "target", "house_price_prediction"
        )
        assert comp_type == CompetitionType.REGRESSION
        
    def test_generate_metadata_binary(self):
        """Test metadata generation for binary classification."""
        analyzer = DatasetAnalyzer()
        
        metadata = analyzer._generate_metadata(
            competition_name="test_binary",
            competition_type=CompetitionType.BINARY_CLASSIFICATION,
            confidence_score=0.9,
            train_file="train.csv",
            test_file="test.csv",
            submission_file="submission.csv",
            dataset_size_mb=50.0,
        )
        
        assert metadata.competition_name == "test_binary"
        assert metadata.competition_type == CompetitionType.BINARY_CLASSIFICATION
        assert metadata.recommended_head_type == "binary_classification"
        assert metadata.recommended_batch_size == 32
        assert metadata.recommended_pooling == "cls"
        
    def test_generate_metadata_regression(self):
        """Test metadata generation for regression."""
        analyzer = DatasetAnalyzer()
        
        metadata = analyzer._generate_metadata(
            competition_name="test_regression",
            competition_type=CompetitionType.REGRESSION,
            confidence_score=0.8,
            train_file="train.csv",
            test_file=None,
            submission_file=None,
            dataset_size_mb=20.0,
        )
        
        assert metadata.competition_type == CompetitionType.REGRESSION
        assert metadata.recommended_head_type == "regression"
        assert metadata.recommended_pooling == "mean"
        
    def test_generate_metadata_large_dataset(self):
        """Test metadata generation for large dataset."""
        analyzer = DatasetAnalyzer()
        
        metadata = analyzer._generate_metadata(
            competition_name="large_dataset",
            competition_type=CompetitionType.MULTICLASS_CLASSIFICATION,
            confidence_score=0.7,
            train_file="train.csv",
            test_file="test.csv",
            submission_file="submission.csv",
            dataset_size_mb=800.0,  # Large dataset
        )
        
        # Should have optimizations for large dataset
        assert metadata.recommended_batch_size >= 32
        assert metadata.optimal_prefetch_size >= 4
        assert metadata.enable_gradient_checkpointing == True
        
    def test_generate_dataset_spec(self, sample_titanic_data):
        """Test dataset spec generation."""
        analyzer = DatasetAnalyzer()
        
        metadata = CompetitionMetadata(
            competition_name="titanic",
            competition_type=CompetitionType.BINARY_CLASSIFICATION,
            recommended_batch_size=32,
            recommended_max_length=256,
        )
        
        spec = analyzer._generate_dataset_spec(
            competition_name="titanic",
            dataset_path=Path("/tmp/titanic"),
            competition_type=CompetitionType.BINARY_CLASSIFICATION,
            train_data=sample_titanic_data,
            target_column="Survived",
            metadata=metadata,
        )
        
        assert spec.competition_name == "titanic"
        assert spec.competition_type == CompetitionType.BINARY_CLASSIFICATION
        assert spec.target_column == "Survived"
        assert spec.num_samples == len(sample_titanic_data)
        assert spec.num_classes == 2
        
        # Check column categorization
        assert "Name" in spec.text_columns  # Long text
        assert "Sex" in spec.categorical_columns  # Object with low cardinality
        assert "Age" in spec.numerical_columns  # Numeric
        
    def test_calculate_dataset_size_file(self, test_data_dir, sample_titanic_data):
        """Test dataset size calculation for single file."""
        analyzer = DatasetAnalyzer()
        
        csv_file = test_data_dir / "test.csv"
        sample_titanic_data.to_csv(csv_file, index=False)
        
        size_mb = analyzer._calculate_dataset_size(csv_file)
        
        assert size_mb > 0
        assert isinstance(size_mb, float)
        
    def test_calculate_dataset_size_directory(self, temp_csv_files):
        """Test dataset size calculation for directory."""
        analyzer = DatasetAnalyzer()
        
        titanic_dir = temp_csv_files['titanic_train'].parent
        size_mb = analyzer._calculate_dataset_size(titanic_dir)
        
        assert size_mb > 0
        assert isinstance(size_mb, float)
        
    def test_analyze_competition_integration(self, temp_csv_files):
        """Test complete competition analysis integration."""
        analyzer = DatasetAnalyzer()
        
        titanic_dir = temp_csv_files['titanic_train'].parent
        
        metadata, spec = analyzer.analyze_competition(
            data_path=titanic_dir,
            competition_name="titanic_test",
            target_column="Survived",
        )
        
        # Check metadata
        assert metadata.competition_name == "titanic_test"
        assert metadata.auto_detected == True
        assert metadata.confidence_score > 0
        assert metadata.train_file is not None
        
        # Check spec
        assert spec.competition_name == "titanic_test"
        assert spec.target_column == "Survived"
        assert spec.num_samples > 0
        assert spec.num_features > 0
        
    def test_analyze_competition_auto_target_detection(self, temp_csv_files):
        """Test competition analysis with automatic target detection."""
        analyzer = DatasetAnalyzer()
        
        titanic_dir = temp_csv_files['titanic_train'].parent
        
        # Don't provide target column - should auto-detect
        metadata, spec = analyzer.analyze_competition(
            data_path=titanic_dir,
            competition_name="titanic_auto",
        )
        
        # Should detect 'Survived' as target
        assert spec.target_column == "Survived"
        assert metadata.competition_type == CompetitionType.BINARY_CLASSIFICATION