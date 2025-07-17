"""Unit tests for V2 data loaders."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile

import mlx.core as mx
import mlx.data as dx
import pandas as pd
import pytest
import numpy as np

from data.dataloader import MLXTabularTextDataLoader, create_cached_dataloader, create_titanic_dataloader
from data.text_templates import TitanicTextTemplates


class TestTitanicTextTemplates:
    """Test TitanicTextTemplates functionality."""
    
    def test_initialization(self):
        """Test converter initialization."""
        converter = TitanicTextTemplates()
        assert hasattr(converter, 'templates')
        assert len(converter.templates) > 0
    
    def test_get_age_description(self):
        """Test age description."""
        converter = TitanicTextTemplates()
        
        assert converter.get_age_description(5) == "child"
        assert converter.get_age_description(15) == "teenager"
        assert converter.get_age_description(25) == "young adult"
        assert converter.get_age_description(45) == "middle-aged"
        assert converter.get_age_description(70) == "elderly"
        assert converter.get_age_description(None) == "passenger of unknown age"
    
    def test_get_fare_description(self):
        """Test fare description."""
        converter = TitanicTextTemplates()
        
        assert converter.get_fare_description(5) == "They paid a modest fare of $5.00."
        assert converter.get_fare_description(15) == "They paid a moderate fare of $15.00."
        assert converter.get_fare_description(40) == "They paid a substantial fare of $40.00."
        assert converter.get_fare_description(150) == "They paid a premium fare of $150.00."
        assert converter.get_fare_description(float('nan')).startswith("The fare is unknown")
    
    def test_convert_row_to_text(self):
        """Test converting a data row to text."""
        converter = TitanicTextTemplates()
        
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


class TestMLXTabularTextDataLoader:
    """Test MLXTabularTextDataLoader V2 implementation."""
    
    def test_initialization(self, sample_titanic_data, tokenizer_name):
        """Test dataloader initialization."""
        loader = MLXTabularTextDataLoader(
            csv_path=str(sample_titanic_data),
            tokenizer_name=tokenizer_name,
            batch_size=4,
            max_length=128,
            label_column="Survived",
        )
        
        assert loader.batch_size == 4
        assert loader.max_length == 128
        assert loader.label_column == "Survived"
        assert loader.tokenizer is not None
        assert loader.tokenizer_info is not None
    
    def test_analyze_csv(self, sample_titanic_data, tokenizer_name):
        """Test CSV analysis functionality."""
        loader = MLXTabularTextDataLoader(
            csv_path=str(sample_titanic_data),
            tokenizer_name=tokenizer_name,
            label_column="Survived",  # Specify label column
        )
        
        # Should have analyzed columns
        assert len(loader.all_columns) > 0
        assert "Survived" in loader.all_columns
        assert len(loader.text_columns) > 0
        assert "Survived" not in loader.text_columns  # Label column excluded
    
    def test_record_to_text(self, sample_titanic_data, tokenizer_name):
        """Test record to text transformation."""
        loader = MLXTabularTextDataLoader(
            csv_path=str(sample_titanic_data),
            tokenizer_name=tokenizer_name,
            label_column="Survived",
        )
        
        # Create sample record
        sample = {
            "Pclass": 2,
            "Name": "Smith, Mr. John",
            "Sex": "male",
            "Age": 25.0,
            "SibSp": 1,
            "Parch": 0,
            "Fare": 50.0,
            "Embarked": "S",
            "Survived": 1
        }
        
        result = loader._record_to_text(sample)
        
        assert "text" in result
        assert isinstance(result["text"], str)
        assert len(result["text"]) > 0
        assert "labels" in result
        assert result["labels"] == 1
    
    def test_padding_and_mask(self, sample_titanic_data, tokenizer_name):
        """Test padding and attention mask creation."""
        loader = MLXTabularTextDataLoader(
            csv_path=str(sample_titanic_data),
            tokenizer_name=tokenizer_name,
            max_length=128,
        )
        
        # Create sample with tokenized input
        sample = {
            "input_ids": [101, 2023, 2003, 1037, 3231, 102],  # Example token IDs
            "labels": 1
        }
        
        result = loader._add_padding_and_mask(sample)
        
        assert "input_ids" in result
        assert "attention_mask" in result
        assert "labels" in result
        
        # Check shapes
        assert len(result["input_ids"]) == 128
        assert len(result["attention_mask"]) == 128
        
        # Check padding - we have 6 original tokens + 2 special tokens (CLS/SEP) = 8 total
        # So positions 0-7 should have attention mask 1, rest should be 0
        assert result["attention_mask"][7] == 1  # Last non-padded position
        assert result["attention_mask"][8] == 0  # First padded position
        assert result["attention_mask"][-1] == 0  # Last position (padded)
    
    
    def test_create_stream(self, sample_titanic_data, tokenizer_name):
        """Test stream creation."""
        loader = MLXTabularTextDataLoader(
            csv_path=str(sample_titanic_data),
            tokenizer_name=tokenizer_name,
            batch_size=4,
            max_length=128,
        )
        
        # Create training stream
        stream = loader.create_stream(is_training=True)
        assert isinstance(stream, dx.Stream)
        
        # Create evaluation stream
        stream_eval = loader.create_stream(is_training=False)
        assert isinstance(stream_eval, dx.Stream)
    
    def test_stream_iteration(self, sample_titanic_data, tokenizer_name):
        """Test iterating through the stream."""
        loader = MLXTabularTextDataLoader(
            csv_path=str(sample_titanic_data),
            tokenizer_name=tokenizer_name,
            batch_size=3,
            max_length=128,
        )
        
        # Get stream
        stream = loader.create_stream(is_training=False)  # No shuffle for predictable results
        
        # Iterate and collect batches
        batches = []
        for i, batch in enumerate(stream):
            if i >= 5:  # Limit iterations for testing
                break
            batches.append(batch)
            
            # Verify batch structure
            assert "input_ids" in batch
            assert "attention_mask" in batch
            assert "labels" in batch
            
            # Verify types - MLX-data returns numpy arrays
            assert isinstance(batch["input_ids"], np.ndarray)
            assert isinstance(batch["attention_mask"], np.ndarray)
            assert isinstance(batch["labels"], np.ndarray)
            
            # Verify shapes
            assert batch["input_ids"].shape[1] == 128  # max_length
            assert batch["attention_mask"].shape[1] == 128
            assert len(batch["labels"].shape) == 1
        
        assert len(batches) > 0
    
    def test_create_buffer(self, sample_titanic_data, tokenizer_name):
        """Test buffer creation for small datasets."""
        loader = MLXTabularTextDataLoader(
            csv_path=str(sample_titanic_data),
            tokenizer_name=tokenizer_name,
            batch_size=4,
            max_length=128,
        )
        
        # Create buffer
        buffer = loader.create_buffer()
        assert isinstance(buffer, dx.Buffer)
    
    def test_dataloader_len(self, sample_titanic_data, tokenizer_name):
        """Test __len__ method."""
        loader = MLXTabularTextDataLoader(
            csv_path=str(sample_titanic_data),
            tokenizer_name=tokenizer_name,
            batch_size=3,
        )
        
        # Should return approximate number of batches
        # 10 samples / 3 batch_size = 4 batches
        assert len(loader) == 4
    
    def test_dataloader_iter(self, sample_titanic_data, tokenizer_name):
        """Test __iter__ method."""
        loader = MLXTabularTextDataLoader(
            csv_path=str(sample_titanic_data),
            tokenizer_name=tokenizer_name,
            batch_size=4,
        )
        
        # Should be iterable
        assert hasattr(loader, '__iter__')
        iterator = iter(loader)
        assert iterator is not None


class TestCreateTitanicDataloader:
    """Test the create_titanic_dataloader helper function."""
    
    def test_create_titanic_dataloader(self, sample_titanic_data):
        """Test creating Titanic-specific dataloader."""
        stream = create_titanic_dataloader(
            data_path=str(sample_titanic_data),
            batch_size=4,
            max_length=128,
            is_training=True
        )
        
        assert isinstance(stream, dx.Stream)
        
        # Get first batch to verify
        batch = next(iter(stream))
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch


class TestCachedDataloader:
    """Test cached dataloader functionality."""
    
    def test_create_cached_dataloader(self, sample_titanic_data, tokenizer_name, temp_dir):
        """Test cached dataloader creation."""
        cache_path = temp_dir / "cache"
        
        # Create cached dataloader
        loader = create_cached_dataloader(
            csv_path=str(sample_titanic_data),
            cache_path=str(cache_path),
            tokenizer_name=tokenizer_name,
            batch_size=4,
            max_length=128,
        )
        
        assert isinstance(loader, MLXTabularTextDataLoader)
        assert cache_path.exists()


class TestDataLoaderWithTestData:
    """Test dataloader with test data (no labels)."""
    
    def test_test_data_without_labels(self, sample_test_data, tokenizer_name):
        """Test loading test data without labels."""
        loader = MLXTabularTextDataLoader(
            csv_path=str(sample_test_data),
            tokenizer_name=tokenizer_name,
            batch_size=4,
            max_length=128,
            label_column="Survived",  # Column doesn't exist in test data
        )
        
        # Get stream
        stream = loader.create_stream(is_training=False)
        
        # Get a batch
        batch = next(iter(stream))
        
        assert "labels" in batch
        assert np.all(batch["labels"] == 0)  # Default label when column missing
        assert "input_ids" in batch
        assert "attention_mask" in batch


@pytest.mark.integration
class TestDataLoadersIntegration:
    """Integration tests for V2 data loaders."""
    
    def test_full_data_pipeline(self, sample_titanic_data, tokenizer_name):
        """Test complete data loading pipeline."""
        # Create dataloader
        loader = MLXTabularTextDataLoader(
            csv_path=str(sample_titanic_data),
            tokenizer_name=tokenizer_name,
            batch_size=3,
            max_length=128,
            label_column="Survived",
        )
        
        # Create stream
        stream = loader.create_stream(is_training=True)
        
        # Iterate through batches
        batches = []
        total_samples = 0
        
        for i, batch in enumerate(stream):
            if i >= 10:  # Limit iterations
                break
                
            batches.append(batch)
            total_samples += batch["labels"].shape[0]
            
            # Verify batch structure
            assert "input_ids" in batch
            assert "attention_mask" in batch
            assert "labels" in batch
            
            # Verify types - MLX-data returns numpy arrays
            assert isinstance(batch["input_ids"], np.ndarray)
            assert isinstance(batch["attention_mask"], np.ndarray)
            assert isinstance(batch["labels"], np.ndarray)
            
            # Verify shapes
            assert batch["input_ids"].shape[1] == 128
            assert batch["attention_mask"].shape[1] == 128
        
        assert len(batches) > 0
        assert total_samples > 0
    
    def test_performance_comparison(self, sample_titanic_data, tokenizer_name):
        """Test that V2 dataloader works efficiently."""
        import time
        
        loader = MLXTabularTextDataLoader(
            csv_path=str(sample_titanic_data),
            tokenizer_name=tokenizer_name,
            batch_size=4,
            max_length=128,
            prefetch_size=2,
        )
        
        stream = loader.create_stream(is_training=False)
        
        # Time the iteration
        start_time = time.time()
        batch_count = 0
        
        for i, batch in enumerate(stream):
            if i >= 5:
                break
            batch_count += 1
            # Force evaluation
            _ = batch["input_ids"].shape
        
        elapsed_time = time.time() - start_time
        
        assert batch_count > 0
        assert elapsed_time < 10.0  # Should be fast for small dataset