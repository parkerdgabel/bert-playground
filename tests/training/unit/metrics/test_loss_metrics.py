"""Unit tests for loss metrics."""

import pytest
import mlx.core as mx
import numpy as np

from training.metrics.loss import MeanLoss, SmoothLoss, LossCollection


class TestMeanLoss:
    """Test mean loss metric."""
    
    def test_single_update(self):
        """Test single loss update."""
        metric = MeanLoss()
        
        metric.update(0.5)
        assert metric.compute() == 0.5
        
    def test_multiple_updates(self):
        """Test multiple loss updates."""
        metric = MeanLoss()
        
        metric.update(1.0)
        metric.update(2.0)
        metric.update(3.0)
        
        # Mean of [1, 2, 3] = 2
        assert metric.compute() == 2.0
        
    def test_batch_updates(self):
        """Test batch loss updates."""
        metric = MeanLoss()
        
        # Batch 1: 10 samples, loss 1.0
        metric.update(1.0, batch_size=10)
        
        # Batch 2: 20 samples, loss 2.0
        metric.update(2.0, batch_size=20)
        
        # Weighted mean: (10*1 + 20*2) / 30 = 50/30 = 1.667
        assert abs(metric.compute() - 5/3) < 1e-6
        
    def test_reset(self):
        """Test resetting mean loss."""
        metric = MeanLoss()
        
        metric.update(5.0)
        assert metric.compute() == 5.0
        
        metric.reset()
        assert metric.compute() == 0.0
        
        metric.update(2.0)
        assert metric.compute() == 2.0


class TestSmoothLoss:
    """Test exponentially smoothed loss."""
    
    def test_smoothing_factor(self):
        """Test smoothing with different factors."""
        # High smoothing (factor=0.99)
        metric = SmoothLoss(smoothing=0.99)
        metric.update(1.0)
        metric.update(2.0)
        # Should be close to 1.0 due to high smoothing
        assert abs(metric.compute() - 1.01) < 0.01
        
    def test_exponential_smoothing(self):
        """Test exponential smoothing."""
        metric = SmoothLoss(smoothing=0.9)
        
        metric.update(1.0)
        assert metric.compute() == 1.0
        
        metric.update(2.0)
        # 0.9 * 1.0 + 0.1 * 2.0 = 0.9 + 0.2 = 1.1
        assert abs(metric.compute() - 1.1) < 1e-6
        
    def test_smoothing_convergence(self):
        """Test smoothing convergence."""
        metric = SmoothLoss(smoothing=0.5)
        
        # Update many times with same value
        for _ in range(10):
            metric.update(5.0)
            
        # Should converge to 5.0
        assert abs(metric.compute() - 5.0) < 0.1


class TestLossCollection:
    """Test loss metrics collection."""
    
    def test_collection_update(self):
        """Test updating loss collection."""
        collection = LossCollection()
        
        collection.update(1.0)
        
        results = collection.compute()
        assert results["loss"] == 1.0
        assert results["smooth_loss"] == 1.0
        
    def test_collection_multiple_updates(self):
        """Test multiple updates to same metric."""
        collection = LossCollection()
        
        collection.update(1.0)
        collection.update(2.0)
        collection.update(3.0)
        
        results = collection.compute()
        assert results["loss"] == 2.0  # Mean of [1, 2, 3]
        
    def test_collection_reset(self):
        """Test resetting collection."""
        collection = LossCollection()
        
        collection.update(5.0)
        collection.reset()
        
        # After reset, metrics should return 0
        results = collection.compute()
        assert results["loss"] == 0.0
        assert results["smooth_loss"] == 0.0
        
    def test_collection_with_batch_sizes(self):
        """Test collection with batch sizes."""
        collection = LossCollection()
        
        # Different batch sizes
        collection.update(1.0, batch_size=10)
        collection.update(2.0, batch_size=20)
        
        results = collection.compute()
        # Weighted mean: (10*1 + 20*2) / 30 = 1.667
        assert abs(results["loss"] - 5/3) < 1e-6