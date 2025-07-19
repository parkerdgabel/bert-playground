"""Integration tests for complete data pipeline."""

import pytest
from pathlib import Path
import tempfile
import time

import pandas as pd
import mlx.core as mx

from data.core.base import CompetitionType, DatasetSpec
from data.loaders.mlx_loader import MLXDataLoader, MLXLoaderConfig
from data.loaders.streaming import StreamingPipeline, StreamingConfig
from data.templates.engine import TextTemplateEngine
from tests.data.fixtures.datasets import create_kaggle_like_dataset
from tests.data.fixtures.configs import create_dataset_spec


class TestDataPipelineIntegration:
    """Test complete data pipeline integration."""
    
    @pytest.fixture
    def titanic_data(self):
        """Create Titanic-like dataset."""
        return create_kaggle_like_dataset('titanic', size=100)
        
    @pytest.fixture
    def pipeline_config(self):
        """Create pipeline configuration."""
        return {
            'loader': MLXLoaderConfig(
                batch_size=32,
                shuffle=True,
                num_workers=2,
            ),
            'streaming': StreamingConfig(
                buffer_size=512,
                chunk_size=32,
            ),
        }
        
    def test_dataset_to_loader_pipeline(self, titanic_data):
        """Test dataset creation to data loader pipeline."""
        # Create dataset spec
        spec = DatasetSpec(
            competition_name="titanic",
            dataset_path=Path("/tmp/titanic"),
            competition_type=CompetitionType.BINARY_CLASSIFICATION,
            num_samples=len(titanic_data['train']),
            num_features=len(titanic_data['train'].columns) - 1,
            target_column="Survived",
            num_classes=2,
        )
        
        # Create mock dataset
        from tests.data.fixtures.datasets import MockKaggleDataset
        dataset = MockKaggleDataset(spec)
        dataset._data = titanic_data['train']
        
        # Create loader
        loader_config = MLXLoaderConfig(batch_size=16)
        loader = MLXDataLoader(dataset, loader_config)
        
        # Load batches
        batches = []
        for i, batch in enumerate(loader):
            batches.append(batch)
            if i >= 2:  # Just test a few batches
                break
                
        assert len(batches) == 3
        assert all('input_ids' in b for b in batches)
        assert all('labels' in b for b in batches)
        
    def test_text_conversion_pipeline(self, titanic_data):
        """Test text conversion in data pipeline."""
        # Create dataset spec
        spec = create_dataset_spec(
            competition_name="titanic",
            competition_type=CompetitionType.BINARY_CLASSIFICATION,
            target_column="Survived",
            categorical_columns=['Sex', 'Embarked'],
            numerical_columns=['Age', 'Fare', 'SibSp', 'Parch'],
            text_columns=['Name'],
        )
        
        # Create template engine
        engine = TextTemplateEngine()
        
        # Convert data
        texts = engine.convert_dataset(titanic_data['train'], spec)
        
        assert len(texts) == len(titanic_data['train'])
        assert all(isinstance(t, str) for t in texts)
        
        # Check content
        sample_text = texts[0]
        row = titanic_data['train'].iloc[0]
        
        # Should contain passenger information
        if pd.notna(row['Name']):
            assert str(row['Name']) in sample_text or 'passenger' in sample_text.lower()
            
    def test_streaming_pipeline_integration(self, titanic_data):
        """Test streaming pipeline with real data."""
        # Create dataset
        spec = create_dataset_spec(
            competition_name="titanic_streaming",
            num_samples=len(titanic_data['train']),
        )
        
        from tests.data.fixtures.datasets import StreamingDataset
        dataset = StreamingDataset(spec, data=titanic_data['train'])
        
        # Create streaming pipeline
        config = StreamingConfig(
            buffer_size=256,
            chunk_size=16,
            num_workers=2,
        )
        
        pipeline = StreamingPipeline(dataset, config)
        
        pipeline.start()
        
        try:
            # Stream samples
            samples = []
            for i, sample in enumerate(pipeline):
                samples.append(sample)
                if i >= 50:
                    break
                    
            assert len(samples) == 51
            
            # Check throughput
            stats = pipeline.get_throughput_stats()
            assert stats['samples_per_second'] > 0
            
        finally:
            pipeline.stop()
            
    def test_caching_integration(self):
        """Test caching across pipeline components."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            cache_dir.mkdir()
            
            # Create dataset with caching
            spec = create_dataset_spec(
                competition_name="cached_test",
                enable_caching=True,
            )
            
            from tests.data.fixtures.datasets import MockKaggleDataset
            dataset = MockKaggleDataset(spec, cache_dir=cache_dir)
            
            # First access - cache miss
            start = time.time()
            _ = dataset[0]
            first_access = time.time() - start
            
            # Second access - cache hit
            start = time.time()
            _ = dataset[0]
            second_access = time.time() - start
            
            # Cache hit should be faster
            assert second_access < first_access
            
    def test_memory_management_integration(self, titanic_data):
        """Test memory management across components."""
        from data.loaders.memory import UnifiedMemoryManager, MemoryConfig
        
        # Create memory manager
        memory_config = MemoryConfig(
            pool_size_mb=128,
            enable_unified_memory=True,
        )
        memory_manager = UnifiedMemoryManager(memory_config)
        
        # Create dataset and loader with memory manager
        spec = create_dataset_spec(num_samples=len(titanic_data['train']))
        
        from tests.data.fixtures.datasets import MockKaggleDataset
        dataset = MockKaggleDataset(spec)
        dataset._data = titanic_data['train']
        
        loader_config = MLXLoaderConfig(
            batch_size=32,
            memory_manager=memory_manager,
        )
        loader = MLXDataLoader(dataset, loader_config)
        
        # Track memory usage
        initial_memory = memory_manager.get_memory_info()
        
        # Load several batches
        for i, batch in enumerate(loader):
            if i >= 5:
                break
                
        final_memory = memory_manager.get_memory_info()
        
        # Memory should be efficiently managed
        memory_growth = final_memory['allocated_mb'] - initial_memory['allocated_mb']
        assert memory_growth < 50  # Less than 50MB growth
        
    def test_multi_dataset_pipeline(self):
        """Test pipeline with multiple datasets."""
        # Create multiple datasets
        datasets_info = [
            ('titanic', CompetitionType.BINARY_CLASSIFICATION, 100),
            ('iris', CompetitionType.MULTICLASS_CLASSIFICATION, 150),
            ('housing', CompetitionType.REGRESSION, 200),
        ]
        
        loaders = []
        
        for name, comp_type, size in datasets_info:
            # Create dataset
            spec = create_dataset_spec(
                competition_name=name,
                competition_type=comp_type,
                num_samples=size,
            )
            
            from tests.data.fixtures.datasets import MockKaggleDataset
            dataset = MockKaggleDataset(spec, size=size)
            
            # Create loader
            loader = MLXDataLoader(dataset, MLXLoaderConfig(batch_size=32))
            loaders.append((name, loader))
            
        # Load from all datasets
        for name, loader in loaders:
            batch = next(iter(loader))
            assert batch is not None
            assert 'input_ids' in batch
            
    def test_error_handling_integration(self):
        """Test error handling across pipeline."""
        # Create dataset that may have errors
        spec = create_dataset_spec(num_samples=100)
        
        from tests.data.fixtures.datasets import FaultyDataset
        dataset = FaultyDataset(spec, error_rate=0.1)
        
        # Create loader with error handling
        loader_config = MLXLoaderConfig(
            batch_size=16,
            error_handling='skip',
            max_retries=3,
        )
        
        loader = MLXDataLoader(dataset, loader_config)
        
        # Should handle errors gracefully
        successful_batches = 0
        errors = 0
        
        for i, batch in enumerate(loader):
            if batch is not None:
                successful_batches += 1
            else:
                errors += 1
                
            if i >= 10:
                break
                
        # Should have some successful batches despite errors
        assert successful_batches > 0
        
    def test_data_augmentation_pipeline(self, titanic_data):
        """Test data augmentation in pipeline."""
        # Define augmentation function
        def augment_text(sample):
            # Simple augmentation - add prefix/suffix
            if 'text' in sample:
                prefixes = ['Passenger: ', 'Record: ', 'Entry: ']
                import random
                prefix = random.choice(prefixes)
                sample['text'] = prefix + sample['text']
            return sample
            
        # Create dataset with augmentation
        spec = create_dataset_spec(num_samples=len(titanic_data['train']))
        
        from tests.data.fixtures.datasets import MockKaggleDataset
        dataset = MockKaggleDataset(spec, transform=augment_text)
        dataset._data = titanic_data['train']
        
        # Create loader
        loader = MLXDataLoader(dataset)
        
        # Check augmentation applied
        batch = next(iter(loader))
        texts = batch.get('text', [])
        
        if texts:
            assert any(t.startswith(('Passenger:', 'Record:', 'Entry:')) for t in texts)


@pytest.mark.slow
class TestDataPipelinePerformance:
    """Performance tests for data pipeline."""
    
    def test_end_to_end_throughput(self):
        """Test end-to-end pipeline throughput."""
        # Create large dataset
        large_data = create_kaggle_like_dataset('large_dataset', size=10000)
        
        spec = create_dataset_spec(
            competition_name="performance_test",
            num_samples=10000,
        )
        
        # Create components
        from tests.data.fixtures.datasets import MockKaggleDataset
        dataset = MockKaggleDataset(spec, size=10000)
        
        engine = TextTemplateEngine()
        
        loader_config = MLXLoaderConfig(
            batch_size=64,
            num_workers=4,
            prefetch_size=4,
        )
        loader = MLXDataLoader(dataset, loader_config)
        
        # Measure throughput
        start_time = time.time()
        samples_processed = 0
        
        for i, batch in enumerate(loader):
            samples_processed += batch['input_ids'].shape[0]
            if i >= 50:  # Process 50 batches
                break
                
        elapsed = time.time() - start_time
        throughput = samples_processed / elapsed
        
        # Should achieve good throughput
        assert throughput > 1000  # >1000 samples/second
        
    def test_memory_efficiency_large_dataset(self):
        """Test memory efficiency with large dataset."""
        # Create large dataset
        spec = create_dataset_spec(num_samples=100000)
        
        from tests.data.fixtures.datasets import MockKaggleDataset
        dataset = MockKaggleDataset(spec, size=100000)
        
        # Configure for memory efficiency
        from data.loaders.memory import UnifiedMemoryManager, MemoryConfig
        
        memory_config = MemoryConfig(
            pool_size_mb=256,
            enable_unified_memory=True,
            auto_cleanup=True,
        )
        memory_manager = UnifiedMemoryManager(memory_config)
        
        loader_config = MLXLoaderConfig(
            batch_size=128,
            memory_manager=memory_manager,
            drop_last=True,
        )
        
        loader = MLXDataLoader(dataset, loader_config)
        
        # Process batches and monitor memory
        max_memory = 0
        
        for i, batch in enumerate(loader):
            if i % 10 == 0:
                memory_info = memory_manager.get_memory_info()
                max_memory = max(max_memory, memory_info['allocated_mb'])
                
            if i >= 100:  # Process 100 batches
                break
                
        # Memory usage should stay bounded
        assert max_memory < 512  # Less than 512MB
        
    def test_streaming_vs_batch_performance(self):
        """Compare streaming vs batch loading performance."""
        spec = create_dataset_spec(num_samples=5000)
        
        from tests.data.fixtures.datasets import MockKaggleDataset, StreamingDataset
        
        # Batch loading
        batch_dataset = MockKaggleDataset(spec, size=5000)
        batch_loader = MLXDataLoader(
            batch_dataset,
            MLXLoaderConfig(batch_size=64)
        )
        
        start = time.time()
        batch_count = sum(1 for _ in batch_loader)
        batch_time = time.time() - start
        
        # Streaming loading
        stream_dataset = StreamingDataset(spec, size=5000)
        stream_config = StreamingConfig(
            buffer_size=1024,
            chunk_size=64,
        )
        stream_pipeline = StreamingPipeline(stream_dataset, stream_config)
        
        stream_pipeline.start()
        
        try:
            start = time.time()
            stream_count = 0
            for i, _ in enumerate(stream_pipeline):
                if i >= 5000:
                    break
                stream_count += 1
            stream_time = time.time() - start
            
        finally:
            stream_pipeline.stop()
            
        # Both should be reasonably fast
        assert batch_time < 5.0
        assert stream_time < 5.0
        
        # Streaming might be slightly slower but more memory efficient
        assert stream_time < batch_time * 2.0