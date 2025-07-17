"""Tests for the advanced memory management system."""

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from training.memory_manager import (
    AppleSiliconMemoryManager,
    MemoryMetrics,
    MemoryOptimizer,
    MemoryThresholds,
)


class TestMemoryThresholds:
    """Test memory thresholds configuration."""
    
    def test_default_thresholds(self):
        """Test default threshold values."""
        thresholds = MemoryThresholds()
        
        assert thresholds.critical_memory == 0.95
        assert thresholds.high_memory == 0.85
        assert thresholds.optimal_memory == 0.75
        assert thresholds.low_memory == 0.50
        assert thresholds.min_batch_size == 4
        assert thresholds.max_batch_size == 256
    
    def test_custom_thresholds(self):
        """Test custom threshold configuration."""
        thresholds = MemoryThresholds(
            critical_memory=0.9,
            high_memory=0.8,
            optimal_memory=0.7,
            low_memory=0.4,
            min_batch_size=8,
            max_batch_size=128,
        )
        
        assert thresholds.critical_memory == 0.9
        assert thresholds.high_memory == 0.8
        assert thresholds.optimal_memory == 0.7
        assert thresholds.low_memory == 0.4
        assert thresholds.min_batch_size == 8
        assert thresholds.max_batch_size == 128


class TestMemoryMetrics:
    """Test memory metrics data structure."""
    
    def test_memory_metrics_creation(self):
        """Test creating memory metrics."""
        timestamp = time.time()
        metrics = MemoryMetrics(
            timestamp=timestamp,
            total_memory_gb=16.0,
            used_memory_gb=8.0,
            available_memory_gb=8.0,
            memory_percentage=0.5,
        )
        
        assert metrics.timestamp == timestamp
        assert metrics.total_memory_gb == 16.0
        assert metrics.used_memory_gb == 8.0
        assert metrics.available_memory_gb == 8.0
        assert metrics.memory_percentage == 0.5
        assert metrics.mlx_cache_size_mb == 0.0  # Default value


class TestAppleSiliconMemoryManager:
    """Test the Apple Silicon memory manager."""
    
    @pytest.fixture
    def memory_manager(self):
        """Create a memory manager for testing."""
        thresholds = MemoryThresholds(
            min_batch_size=4,
            max_batch_size=64,
        )
        return AppleSiliconMemoryManager(thresholds)
    
    def test_initialization(self, memory_manager):
        """Test memory manager initialization."""
        assert memory_manager.thresholds is not None
        assert isinstance(memory_manager.metrics_history, list)
        assert len(memory_manager.metrics_history) == 0
        assert memory_manager.gc_stats["collections"] == 0
    
    @patch("psutil.virtual_memory")
    def test_get_current_metrics(self, mock_memory, memory_manager):
        """Test getting current memory metrics."""
        # Mock memory info
        mock_memory.return_value = MagicMock(
            total=16 * 1024**3,  # 16 GB
            available=8 * 1024**3,  # 8 GB available
        )
        
        with patch("psutil.Process") as mock_process:
            mock_process.return_value.memory_info.return_value = MagicMock(
                rss=1024**3  # 1 GB process memory
            )
            
            metrics = memory_manager.get_current_metrics()
            
            assert metrics.total_memory_gb == 16.0
            assert metrics.available_memory_gb == 8.0
            assert metrics.used_memory_gb == 8.0
            assert metrics.memory_percentage == 0.5
            assert metrics.process_memory_gb == 1.0
    
    def test_should_adjust_batch_size_critical(self, memory_manager):
        """Test batch size adjustment for critical memory usage."""
        with patch.object(memory_manager, 'get_current_metrics') as mock_metrics:
            mock_metrics.return_value = MagicMock(memory_percentage=0.96)  # Critical
            
            should_adjust, new_size, reason = memory_manager.should_adjust_batch_size(32)
            
            assert should_adjust is True
            assert new_size == 8  # 32 // 4
            assert "Critical memory usage" in reason
    
    def test_should_adjust_batch_size_high(self, memory_manager):
        """Test batch size adjustment for high memory usage."""
        with patch.object(memory_manager, 'get_current_metrics') as mock_metrics:
            mock_metrics.return_value = MagicMock(memory_percentage=0.86)  # High
            
            should_adjust, new_size, reason = memory_manager.should_adjust_batch_size(32)
            
            assert should_adjust is True
            assert new_size == 16  # 32 // 2
            assert "High memory usage" in reason
    
    def test_should_adjust_batch_size_low(self, memory_manager):
        """Test batch size adjustment for low memory usage."""
        with patch.object(memory_manager, 'get_current_metrics') as mock_metrics:
            mock_metrics.return_value = MagicMock(memory_percentage=0.4)  # Low
            
            should_adjust, new_size, reason = memory_manager.should_adjust_batch_size(16)
            
            assert should_adjust is True
            assert new_size == 32  # 16 * 2
            assert "Low memory usage" in reason
    
    def test_should_adjust_batch_size_no_change(self, memory_manager):
        """Test no batch size adjustment needed."""
        with patch.object(memory_manager, 'get_current_metrics') as mock_metrics:
            mock_metrics.return_value = MagicMock(memory_percentage=0.7)  # Optimal
            
            should_adjust, new_size, reason = memory_manager.should_adjust_batch_size(32)
            
            assert should_adjust is False
            assert new_size == 32
            assert "No adjustment needed" in reason
    
    def test_force_garbage_collection(self, memory_manager):
        """Test garbage collection functionality."""
        with patch.object(memory_manager, 'get_current_metrics') as mock_metrics:
            # Mock before and after memory states
            mock_metrics.side_effect = [
                MagicMock(memory_percentage=0.9, used_memory_gb=14.4),  # Before
                MagicMock(memory_percentage=0.8, used_memory_gb=12.8),  # After
            ]
            
            with patch("gc.collect", return_value=10) as mock_gc:
                stats = memory_manager.force_garbage_collection()
                
                assert stats["objects_collected"] == 10
                assert abs(stats["memory_freed_gb"] - 1.6) < 1e-10  # 14.4 - 12.8
                assert abs(stats["improvement"] - 0.1) < 1e-10  # 0.9 - 0.8
                assert "gc_time_seconds" in stats
                mock_gc.assert_called_once()
    
    def test_get_memory_recommendations_critical(self, memory_manager):
        """Test memory recommendations for critical usage."""
        with patch.object(memory_manager, 'get_current_metrics') as mock_metrics:
            mock_metrics.return_value = MagicMock(memory_percentage=0.96)
            
            recommendations = memory_manager.get_memory_recommendations()
            
            assert recommendations["status"] == "critical"
            assert "Immediately reduce batch size" in recommendations["actions"]
            assert "Force aggressive garbage collection" in recommendations["actions"]
    
    def test_get_memory_recommendations_optimal(self, memory_manager):
        """Test memory recommendations for optimal usage."""
        with patch.object(memory_manager, 'get_current_metrics') as mock_metrics:
            mock_metrics.return_value = MagicMock(memory_percentage=0.7)
            
            recommendations = memory_manager.get_memory_recommendations()
            
            assert recommendations["status"] == "optimal"
            assert len(recommendations["actions"]) == 0
    
    def test_save_memory_report(self, memory_manager):
        """Test saving memory report."""
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "memory_report.json"
            
            # Add some fake metrics directly
            from training.memory_manager import MemoryMetrics
            fake_metrics = MemoryMetrics(
                timestamp=time.time(),
                total_memory_gb=16.0,
                used_memory_gb=8.0,
                available_memory_gb=8.0,
                memory_percentage=0.5,
            )
            memory_manager.metrics_history.append(fake_metrics)
            
            with patch.object(memory_manager, 'get_current_metrics') as mock_metrics:
                mock_metrics.return_value = fake_metrics
                
                memory_manager.save_memory_report(report_path)
                
                assert report_path.exists()
                
                # Load and verify report structure
                import json
                with open(report_path) as f:
                    report = json.load(f)
                
                assert "system_info" in report
                assert "current_metrics" in report
                assert "gc_stats" in report
                assert "recommendations" in report


class TestMemoryOptimizer:
    """Test the memory optimizer."""
    
    @pytest.fixture
    def memory_optimizer(self):
        """Create a memory optimizer for testing."""
        memory_manager = AppleSiliconMemoryManager()
        return MemoryOptimizer(memory_manager)
    
    def test_initialization(self, memory_optimizer):
        """Test memory optimizer initialization."""
        assert memory_optimizer.memory_manager is not None
        assert isinstance(memory_optimizer.optimization_history, list)
    
    def test_optimize_training_memory_no_change(self, memory_optimizer):
        """Test memory optimization when no changes are needed."""
        with patch.object(memory_optimizer.memory_manager, 'should_adjust_batch_size') as mock_adjust:
            mock_adjust.return_value = (False, 32, "No adjustment needed")
            
            with patch.object(memory_optimizer.memory_manager, 'get_current_metrics') as mock_metrics:
                mock_metrics.return_value = MagicMock(memory_percentage=0.7)  # Optimal
                
                new_batch_size, optimization_info = memory_optimizer.optimize_training_memory(32)
                
                assert new_batch_size == 32
                assert len(optimization_info["optimizations_applied"]) == 0
    
    def test_optimize_training_memory_with_adjustment(self, memory_optimizer):
        """Test memory optimization with batch size adjustment."""
        with patch.object(memory_optimizer.memory_manager, 'should_adjust_batch_size') as mock_adjust:
            mock_adjust.return_value = (True, 16, "High memory usage")
            
            with patch.object(memory_optimizer.memory_manager, 'get_current_metrics') as mock_metrics:
                mock_metrics.return_value = MagicMock(memory_percentage=0.9)  # High
                
                with patch.object(memory_optimizer.memory_manager, 'force_garbage_collection') as mock_gc:
                    mock_gc.return_value = {"objects_collected": 5}
                    
                    new_batch_size, optimization_info = memory_optimizer.optimize_training_memory(32)
                    
                    assert new_batch_size == 16
                    assert "batch_size_adjustment" in optimization_info["optimizations_applied"]
                    assert "garbage_collection" in optimization_info["optimizations_applied"]
    
    def test_optimize_training_memory_forced(self, memory_optimizer):
        """Test forced memory optimization."""
        with patch.object(memory_optimizer.memory_manager, 'should_adjust_batch_size') as mock_adjust:
            mock_adjust.return_value = (False, 32, "No adjustment needed")
            
            with patch.object(memory_optimizer.memory_manager, 'get_current_metrics') as mock_metrics:
                mock_metrics.return_value = MagicMock(memory_percentage=0.7)
                
                with patch.object(memory_optimizer.memory_manager, 'force_garbage_collection') as mock_gc:
                    mock_gc.return_value = {"objects_collected": 3}
                    
                    with patch.object(memory_optimizer.memory_manager, 'optimize_for_apple_silicon') as mock_apple:
                        mock_apple.return_value = {"unified_memory": True}
                        
                        new_batch_size, optimization_info = memory_optimizer.optimize_training_memory(
                            32, force_optimization=True
                        )
                        
                        assert new_batch_size == 32
                        assert "garbage_collection" in optimization_info["optimizations_applied"]
                        # Apple Silicon optimizations only applied if is_apple_silicon is True