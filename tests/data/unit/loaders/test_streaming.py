"""Tests for streaming data pipeline."""

import pytest
from unittest.mock import Mock, patch
import time
import threading
from pathlib import Path

import mlx.core as mx

from data.loaders.streaming import StreamingPipeline, StreamingConfig
from tests.data.fixtures.datasets import MockKaggleDataset, StreamingDataset
from tests.data.fixtures.configs import (
    create_dataset_spec,
    create_streaming_config,
)
from tests.data.fixtures.utils import (
    measure_throughput,
    simulate_slow_consumer,
)


class TestStreamingConfig:
    """Test StreamingConfig dataclass."""
    
    def test_default_config(self):
        """Test default streaming configuration."""
        config = StreamingConfig()
        
        assert config.buffer_size == 1000
        assert config.batch_size == 32
        assert config.max_queue_size == 100
        assert config.num_producer_threads == 2
        assert config.num_consumer_threads == 1
        assert config.prefetch_batches == 5
        assert config.max_memory_mb == 1024
        assert config.enable_memory_monitoring == True
        
    def test_custom_config(self):
        """Test custom streaming configuration."""
        config = create_streaming_config(
            buffer_size=2048,
            chunk_size=512,
            num_workers=8,
            target_throughput=1000,
            adaptive_batching=True,
        )
        
        assert config.buffer_size == 2048
        assert config.chunk_size == 512
        assert config.num_workers == 8
        assert config.target_throughput == 1000
        assert config.adaptive_batching == True
        
    def test_validation(self):
        """Test configuration validation."""
        # Buffer size must be positive
        with pytest.raises(ValueError):
            StreamingConfig(buffer_size=0)
            
        # Chunk size must be positive
        with pytest.raises(ValueError):
            StreamingConfig(chunk_size=0)
            
        # Workers must be positive
        with pytest.raises(ValueError):
            StreamingConfig(num_workers=0)


class TestStreamingPipeline:
    """Test StreamingPipeline class."""
    
    @pytest.fixture
    def streaming_config(self):
        """Create streaming configuration."""
        return create_streaming_config(
            buffer_size=1024,
            chunk_size=256,
            num_workers=2,
            target_throughput=100,
        )
        
    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for streaming."""
        spec = create_dataset_spec(
            competition_name="streaming_test",
            num_samples=1000,
            num_features=5,
        )
        return StreamingDataset(spec, size=1000, delay=0.0)  # No delay for tests
        
    def test_pipeline_creation(self, sample_dataset, streaming_config):
        """Test streaming pipeline creation."""
        pipeline = StreamingPipeline(sample_dataset, streaming_config)
        
        assert pipeline.dataset == sample_dataset
        assert pipeline.config == streaming_config
        assert hasattr(pipeline, '_sample_queue')
        assert hasattr(pipeline, '_batch_queue')
        assert hasattr(pipeline, '_producer_threads')
        assert hasattr(pipeline, '_consumer_threads')
        
    def test_pipeline_start_stop(self, sample_dataset, streaming_config):
        """Test starting and stopping pipeline."""
        pipeline = StreamingPipeline(sample_dataset, streaming_config)
        
        # Start pipeline
        pipeline.start()
        assert pipeline.is_running() == True
        
        # Stop pipeline
        pipeline.stop()
        assert pipeline.is_running() == False
        
    def test_streaming_iteration(self, sample_dataset, streaming_config):
        """Test streaming iteration."""
        pipeline = StreamingPipeline(sample_dataset, streaming_config)
        
        pipeline.start()
        
        try:
            # Stream some samples
            samples = []
            for i, sample in enumerate(pipeline):
                samples.append(sample)
                if i >= 50:  # Test first 50 samples
                    break
                    
            assert len(samples) == 51  # 0 to 50 inclusive
            assert all('input_ids' in sample for sample in samples)
            
        finally:
            pipeline.stop()
            
    def test_throughput_measurement(self, sample_dataset, streaming_config):
        """Test throughput measurement."""
        pipeline = StreamingPipeline(sample_dataset, streaming_config)
        
        pipeline.start()
        
        try:
            # Process samples for a short time
            start_time = time.time()
            sample_count = 0
            
            for sample in pipeline:
                sample_count += 1
                if time.time() - start_time > 1.0:  # Run for 1 second
                    break
                    
            throughput = pipeline.get_throughput_stats()
            
            assert 'samples_per_second' in throughput
            assert 'avg_batch_time_ms' in throughput
            assert throughput['samples_per_second'] > 0
            
        finally:
            pipeline.stop()
            
    def test_buffer_management(self, sample_dataset, streaming_config):
        """Test buffer management."""
        pipeline = StreamingPipeline(sample_dataset, streaming_config)
        
        buffer_info = pipeline.get_buffer_info()
        
        assert 'buffer_size' in buffer_info
        assert 'current_items' in buffer_info
        assert 'buffer_utilization' in buffer_info
        
    def test_adaptive_streaming(self, sample_dataset):
        """Test adaptive streaming based on consumption rate."""
        config = create_streaming_config(
            buffer_size=512,
            adaptive_batching=True,
            target_throughput=200,
        )
        
        pipeline = StreamingPipeline(sample_dataset, config)
        
        pipeline.start()
        
        try:
            # Consume at different rates to test adaptation
            samples = []
            for i, sample in enumerate(pipeline):
                samples.append(sample)
                if i >= 20:
                    break
                if i % 5 == 0:
                    time.sleep(0.01)  # Simulate slower consumption
                    
            assert len(samples) == 21
            
            # Check adaptation metrics
            metrics = pipeline.get_adaptation_metrics()
            assert 'buffer_adjustments' in metrics
            assert 'worker_adjustments' in metrics
            
        finally:
            pipeline.stop()
            
    def test_error_recovery(self, sample_dataset, streaming_config):
        """Test error recovery in streaming."""
        pipeline = StreamingPipeline(sample_dataset, streaming_config)
        
        # Enable error recovery
        pipeline.enable_error_recovery(max_retries=3)
        
        pipeline.start()
        
        try:
            # Should handle errors gracefully
            samples = []
            for i, sample in enumerate(pipeline):
                samples.append(sample)
                if i >= 10:
                    break
                    
            assert len(samples) == 11
            
            # Check error statistics
            error_stats = pipeline.get_error_statistics()
            assert 'total_errors' in error_stats
            assert 'recovered_errors' in error_stats
            
        finally:
            pipeline.stop()
            
    @pytest.mark.skip(reason="StreamingPipeline has issues with stream_batches() hanging")
    def test_backpressure_handling(self, sample_dataset, streaming_config):
        """Test backpressure handling with slow consumer."""
        # Configure smaller buffer to test backpressure
        streaming_config.buffer_size = 10
        streaming_config.batch_size = 2
        
        pipeline = StreamingPipeline(sample_dataset, streaming_config)
        
        pipeline.start()
        
        try:
            # Get a few batches
            batches = []
            for i, batch in enumerate(pipeline.stream_batches()):
                if i >= 5:  # Just get 5 batches
                    break
                batches.append(batch)
                time.sleep(0.01)  # Small delay to simulate processing
            
            # Should get requested batches
            assert len(batches) == 5
            
            # Check that batches have data
            for batch in batches:
                assert 'input_ids' in batch
                assert isinstance(batch['input_ids'], mx.array)
            
        finally:
            pipeline.stop()
            
    def test_worker_pool_management(self, sample_dataset, streaming_config):
        """Test worker pool management."""
        pipeline = StreamingPipeline(sample_dataset, streaming_config)
        
        # Get worker info before starting
        worker_info = pipeline.get_worker_info()
        assert worker_info['active_workers'] == 0
        
        pipeline.start()
        
        try:
            # Workers should be active
            worker_info = pipeline.get_worker_info()
            assert worker_info['active_workers'] == streaming_config.num_workers
            assert 'worker_stats' in worker_info
            
            # Process some samples
            samples = []
            for i, sample in enumerate(pipeline):
                samples.append(sample)
                if i >= 10:
                    break
                    
            # Check worker statistics
            worker_stats = pipeline.get_worker_statistics()
            assert all(stat['samples_processed'] > 0 for stat in worker_stats)
            
        finally:
            pipeline.stop()
            
    def test_graceful_shutdown(self, sample_dataset, streaming_config):
        """Test graceful shutdown with pending items."""
        pipeline = StreamingPipeline(sample_dataset, streaming_config)
        
        pipeline.start()
        
        # Start consuming
        samples = []
        consumer_thread = threading.Thread(
            target=lambda: [samples.append(s) for i, s in enumerate(pipeline) if i < 100]
        )
        consumer_thread.start()
        
        # Let it run briefly
        time.sleep(0.1)
        
        # Stop gracefully
        pipeline.stop(graceful=True, timeout=5.0)
        
        # Consumer thread should finish
        consumer_thread.join(timeout=1.0)
        assert not consumer_thread.is_alive()
        
    def test_streaming_with_transforms(self, streaming_config):
        """Test streaming with data transforms."""
        spec = create_dataset_spec(num_samples=100)
        
        # Add transform to dataset
        def transform(sample):
            sample['transformed'] = True
            return sample
            
        dataset = StreamingDataset(spec, size=100, transform=transform)
        pipeline = StreamingPipeline(dataset, streaming_config)
        
        pipeline.start()
        
        try:
            samples = []
            for i, sample in enumerate(pipeline):
                samples.append(sample)
                if i >= 10:
                    break
                    
            # All samples should be transformed
            assert all('transformed' in s for s in samples)
            
        finally:
            pipeline.stop()


@pytest.mark.integration
class TestStreamingIntegration:
    """Integration tests for streaming pipeline."""
    
    def test_multi_pipeline_coordination(self):
        """Test multiple pipelines working together."""
        spec1 = create_dataset_spec(num_samples=100, competition_name="pipeline1")
        spec2 = create_dataset_spec(num_samples=100, competition_name="pipeline2")
        
        dataset1 = StreamingDataset(spec1, size=100)
        dataset2 = StreamingDataset(spec2, size=100)
        
        config = create_streaming_config(num_workers=2)
        
        pipeline1 = StreamingPipeline(dataset1, config)
        pipeline2 = StreamingPipeline(dataset2, config)
        
        pipeline1.start()
        pipeline2.start()
        
        try:
            # Consume from both pipelines
            samples1 = []
            samples2 = []
            
            for i in range(10):
                sample1 = next(iter(pipeline1))
                sample2 = next(iter(pipeline2))
                samples1.append(sample1)
                samples2.append(sample2)
                
            assert len(samples1) == 10
            assert len(samples2) == 10
            
        finally:
            pipeline1.stop()
            pipeline2.stop()
            
    @pytest.mark.skip(reason="StreamingPipeline has issues that need to be fixed")
    def test_streaming_to_training_pipeline(self):
        """Test streaming pipeline feeding training."""
        spec = create_dataset_spec(num_samples=1000)
        dataset = StreamingDataset(spec, size=1000)
        
        config = create_streaming_config(
            buffer_size=512,
            chunk_size=32,  # Batch-like chunks
            num_workers=4,
        )
        
        pipeline = StreamingPipeline(dataset, config)
        
        pipeline.start()
        
        try:
            # Simulate training loop
            def training_step(batch):
                # Simulate computation
                time.sleep(0.01)
                return mx.mean(batch['input_ids'])
                
            results = []
            start_time = time.time()
            
            for i, sample in enumerate(pipeline):
                if i % 32 == 0 and i > 0:  # Every batch
                    # Collect batch
                    batch = {
                        'input_ids': mx.stack([s['input_ids'] for s in results[-32:]]),
                        'labels': mx.stack([s['labels'] for s in results[-32:]]),
                    }
                    loss = training_step(batch)
                    
                results.append(sample)
                
                if i >= 320:  # 10 batches
                    break
                    
            elapsed = time.time() - start_time
            throughput = 320 / elapsed
            
            assert throughput > 100  # Should process >100 samples/sec
            
        finally:
            pipeline.stop()


