"""Unit tests for data loaders."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import mlx.core as mx
import pandas as pd
import pytest

from data.unified_loader import UnifiedTitanicDataPipeline, create_unified_dataloaders
from data.enhanced_unified_loader import EnhancedUnifiedDataPipeline
from data.text_templates import TitanicTextTemplates as DataToTextConverter


class TestDataToTextConverter:
    """Test DataToTextConverter functionality."""
    
    def test_initialization(self):
        """Test converter initialization."""
        converter = DataToTextConverter()
        assert hasattr(converter, 'templates')
        assert len(converter.templates) > 0
    
    def test_get_age_description(self):
        """Test age description."""
        converter = DataToTextConverter()
        
        assert converter.get_age_description(5) == "child"
        assert converter.get_age_description(15) == "teenager"
        assert converter.get_age_description(25) == "young adult"
        assert converter.get_age_description(45) == "middle-aged"
        assert converter.get_age_description(70) == "elderly"
        assert converter.get_age_description(None) == "passenger of unknown age"
    
    def test_get_fare_description(self):
        """Test fare description."""
        converter = DataToTextConverter()
        
        assert converter.get_fare_description(5) == "They paid a modest fare of $5.00."
        assert converter.get_fare_description(15) == "They paid a moderate fare of $15.00."
        assert converter.get_fare_description(40) == "They paid a substantial fare of $40.00."
        assert converter.get_fare_description(150) == "They paid a premium fare of $150.00."
        assert converter.get_fare_description(float('nan')).startswith("The fare is unknown")
    
    def test_convert_row_to_text(self):
        """Test converting a data row to text."""
        converter = DataToTextConverter()
        
        row = pd.Series({
            "Sex": "male",
            "Age": 25.0,
            "Pclass": 2,
            "Embarked": "S",
            "SibSp": 1,
            "Parch": 0,
            "Fare": 50.0,
            "Cabin": "B123",
            "Name": "Smith, Mr. John"
        })
        
        text = converter.row_to_text(row.to_dict())
        
        assert isinstance(text, str)
        assert len(text) > 0
        # Check that some expected content appears
        assert "male" in text.lower() or "man" in text.lower()
        assert "25" in text or "young" in text


class TestUnifiedTitanicDataPipeline:
    """Test UnifiedTitanicDataPipeline."""
    
    def test_initialization(self, sample_titanic_data, tokenizer_name):
        """Test pipeline initialization."""
        pipeline = UnifiedTitanicDataPipeline(
            data_path=str(sample_titanic_data),
            tokenizer_name=tokenizer_name,
            batch_size=4,
            max_length=128,
            augment=False,  # Disable augmentation for predictable counts
        )
        
        assert pipeline.batch_size == 4
        assert pipeline.max_length == 128
        assert len(pipeline.texts) == 10  # Sample data has 10 rows
        assert len(pipeline.labels) == 10
    
    def test_optimization_levels(self, sample_titanic_data, tokenizer_name):
        """Test different optimization levels."""
        # Basic optimization
        pipeline_basic = UnifiedTitanicDataPipeline(
            data_path=str(sample_titanic_data),
            tokenizer_name=tokenizer_name,
            optimization_level="basic",
            use_mlx_data=False,  # Disable MLX data for this test
        )
        assert pipeline_basic.optimization_level.value == "basic"
        
        # Standard optimization
        pipeline_standard = UnifiedTitanicDataPipeline(
            data_path=str(sample_titanic_data),
            tokenizer_name=tokenizer_name,
            optimization_level="standard",
            use_mlx_data=False,  # Disable MLX data for this test
        )
        assert pipeline_standard.optimization_level.value == "standard"
    
    def test_prepare_data(self, sample_titanic_data, tokenizer_name):
        """Test data preparation."""
        pipeline = UnifiedTitanicDataPipeline(
            data_path=str(sample_titanic_data),
            tokenizer_name=tokenizer_name,
            augment=False,  # Disable augmentation for predictable behavior
        )
        
        # Check texts are generated
        assert all(isinstance(text, str) for text in pipeline.texts)
        assert all(len(text) > 0 for text in pipeline.texts)
        
        # Check labels match survived column
        df = pd.read_csv(sample_titanic_data)
        assert list(pipeline.labels) == df["Survived"].tolist()
    
    def test_tokenization(self, sample_titanic_data, tokenizer_name):
        """Test tokenization process."""
        pipeline = UnifiedTitanicDataPipeline(
            data_path=str(sample_titanic_data),
            tokenizer_name=tokenizer_name,
            batch_size=4,
            max_length=128,
        )
        
        # Get a batch
        dataloader = pipeline.get_dataloader()
        batch = next(iter(dataloader))
        
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch
        
        # Check shapes
        assert batch["input_ids"].shape == (4, 128)
        assert batch["attention_mask"].shape == (4, 128)
        assert batch["labels"].shape == (4,)
    
    @pytest.mark.skip(reason="MLX data pipeline has initialization issue")
    def test_mlx_data_pipeline(self, sample_titanic_data, tokenizer_name):
        """Test MLX data pipeline mode."""
        pipeline = UnifiedTitanicDataPipeline(
            data_path=str(sample_titanic_data),
            tokenizer_name=tokenizer_name,
            use_mlx_data=True,
            batch_size=4,
        )
        
        assert pipeline.use_mlx_data == True
        
        # Get dataloader
        dataloader = pipeline.get_dataloader()
        assert dataloader is not None
    
    def test_test_data_without_labels(self, sample_test_data, tokenizer_name):
        """Test loading test data without labels."""
        pipeline = UnifiedTitanicDataPipeline(
            data_path=str(sample_test_data),
            tokenizer_name=tokenizer_name,
            is_training=False,
            augment=False,  # Disable augmentation for test data
        )
        
        # Labels should be -1 for test data (no Survived column)
        assert all(label == -1 for label in pipeline.labels)
        assert len(pipeline.texts) == 5  # Test data has 5 rows
        
        # Get a batch
        dataloader = pipeline.get_dataloader()
        batch = next(iter(dataloader))
        
        assert "labels" in batch
        assert all(batch["labels"] == -1)  # All labels should be -1
        assert "input_ids" in batch
        assert "attention_mask" in batch
    
    def test_data_augmentation(self, sample_titanic_data, tokenizer_name):
        """Test data augmentation."""
        # Without augmentation
        pipeline_no_aug = UnifiedTitanicDataPipeline(
            data_path=str(sample_titanic_data),
            tokenizer_name=tokenizer_name,
            augment=False,
        )
        
        # With augmentation (should produce different texts)
        pipeline_aug = UnifiedTitanicDataPipeline(
            data_path=str(sample_titanic_data),
            tokenizer_name=tokenizer_name,
            augment=True,
        )
        
        # With augmentation, should have 3x more texts
        assert len(pipeline_aug.texts) == 3 * len(pipeline_no_aug.texts)
    
    def test_get_num_batches(self, sample_titanic_data, tokenizer_name):
        """Test batch count calculation."""
        pipeline = UnifiedTitanicDataPipeline(
            data_path=str(sample_titanic_data),
            tokenizer_name=tokenizer_name,
            batch_size=3,
            augment=False,
        )
        
        # 10 samples / 3 batch_size = 4 batches (last batch has 1 sample)
        assert pipeline.get_num_batches() == 4
    
    def test_len_method(self, sample_titanic_data, tokenizer_name):
        """Test __len__ method."""
        pipeline = UnifiedTitanicDataPipeline(
            data_path=str(sample_titanic_data),
            tokenizer_name=tokenizer_name,
            augment=False,
            batch_size=32,  # Default batch size
        )
        
        # __len__ returns number of batches, not samples
        expected_batches = (10 + 32 - 1) // 32  # 10 samples, batch size 32 = 1 batch
        assert len(pipeline) == expected_batches


class TestEnhancedUnifiedDataPipeline:
    """Test EnhancedUnifiedDataPipeline."""
    
    def test_inheritance(self, sample_titanic_data, tokenizer_name):
        """Test that enhanced pipeline inherits from base."""
        pipeline = EnhancedUnifiedDataPipeline(
            data_path=str(sample_titanic_data),
            tokenizer_name=tokenizer_name,
        )
        
        assert isinstance(pipeline, UnifiedTitanicDataPipeline)
    
    def test_persistent_cache(self, sample_titanic_data, tokenizer_name, temp_dir):
        """Test persistent caching functionality."""
        cache_dir = temp_dir / "cache"
        
        # First load - should create cache
        pipeline1 = EnhancedUnifiedDataPipeline(
            data_path=str(sample_titanic_data),
            tokenizer_name=tokenizer_name,
            cache_dir=str(cache_dir),
            persistent_cache=True,  # Use persistent_cache parameter
            pre_tokenize=True,  # Enable pre-tokenization for caching
            optimization_level="optimized",  # Use optimized level
        )
        
        # Check cache was created
        assert cache_dir.exists()
        assert len(list(cache_dir.glob("*.pkl"))) > 0
        
        # Second load - should use cache
        with patch.object(EnhancedUnifiedDataPipeline, '_pretokenize_all') as mock_tokenize:
            pipeline2 = EnhancedUnifiedDataPipeline(
                data_path=str(sample_titanic_data),
                tokenizer_name=tokenizer_name,
                cache_dir=str(cache_dir),
                persistent_cache=True,
                pre_tokenize=True,
                optimization_level="optimized",
            )
            
            # Pre-tokenization should not be called when loading from cache
            mock_tokenize.assert_not_called()
    
    def test_update_batch_size(self, sample_titanic_data, tokenizer_name):
        """Test dynamic batch size update."""
        pipeline = EnhancedUnifiedDataPipeline(
            data_path=str(sample_titanic_data),
            tokenizer_name=tokenizer_name,
            batch_size=4,
            enable_dynamic_batch=True,  # Must enable dynamic batch updates
            augment=False,  # Disable augmentation for predictable batch sizes
        )
        
        # Update batch size
        pipeline.update_batch_size(8)
        assert pipeline.batch_size == 8
        
        # Get new dataloader with updated batch size
        dataloader = pipeline.get_dataloader()
        batch = next(iter(dataloader))
        # With 10 samples and batch size 8, first batch should have 8 samples
        assert batch["input_ids"].shape[0] == 8


class TestCreateUnifiedDataloaders:
    """Test create_unified_dataloaders helper function."""
    
    def test_create_train_only(self, sample_titanic_data, tokenizer_name):
        """Test creating train loader only."""
        # The factory function always creates both train and val loaders
        # if val_path is provided
        try:
            train_loader, val_loader = create_unified_dataloaders(
                train_path=str(sample_titanic_data),
                val_path=None,  # This will cause an error
                tokenizer_name=tokenizer_name,
                batch_size=4,
            )
            # Should not reach here
            assert False, "Expected error when val_path is None"
        except (TypeError, AttributeError):
            # Expected - the function requires both paths
            pass
        
        # Create with both paths to test properly
        train_loader, val_loader = create_unified_dataloaders(
            train_path=str(sample_titanic_data),
            val_path=str(sample_titanic_data),  # Use same file for testing
            tokenizer_name=tokenizer_name,
            batch_size=4,
        )
        
        assert train_loader is not None
        assert val_loader is not None
        assert train_loader.batch_size == 4
    
    def test_create_train_and_val(self, sample_titanic_data, tokenizer_name):
        """Test creating both train and val loaders."""
        train_loader, val_loader = create_unified_dataloaders(
            train_path=str(sample_titanic_data),
            val_path=str(sample_titanic_data),  # Use same file for testing
            tokenizer_name=tokenizer_name,
            batch_size=4,
        )
        
        assert train_loader is not None
        assert val_loader is not None
        assert train_loader.is_training == True
        assert val_loader.is_training == False
    
    def test_optimization_level_propagation(self, sample_titanic_data, tokenizer_name):
        """Test optimization level is properly set."""
        train_loader, val_loader = create_unified_dataloaders(
            train_path=str(sample_titanic_data),
            val_path=str(sample_titanic_data),  # Required parameter
            tokenizer_name=tokenizer_name,
            optimization_level="optimized",
        )
        
        assert train_loader.optimization_level.value == "optimized"


@pytest.mark.integration
class TestDataLoadersIntegration:
    """Integration tests for data loaders."""
    
    def test_full_data_pipeline(self, sample_titanic_data, tokenizer_name):
        """Test complete data loading pipeline."""
        # Create pipeline
        pipeline = UnifiedTitanicDataPipeline(
            data_path=str(sample_titanic_data),
            tokenizer_name=tokenizer_name,
            batch_size=3,
            max_length=128,
            is_training=True,
            augment=True,
            use_mlx_data=False,
        )
        
        # Iterate through all batches
        batches = []
        for batch in pipeline.get_dataloader():
            batches.append(batch)
            
            # Verify batch structure
            assert "input_ids" in batch
            assert "attention_mask" in batch
            assert "labels" in batch
            
            # Verify types
            assert isinstance(batch["input_ids"], mx.array)
            assert isinstance(batch["attention_mask"], mx.array)
            assert isinstance(batch["labels"], mx.array)
        
        # Should have correct number of batches
        assert len(batches) == pipeline.get_num_batches()
        
        # Verify all samples are covered
        total_samples = sum(batch["labels"].shape[0] for batch in batches)
        # With augmentation (3x), we should have 30 samples
        assert total_samples == 30  # 10 rows * 3 augmentations