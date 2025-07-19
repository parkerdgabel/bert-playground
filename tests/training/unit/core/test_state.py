"""Unit tests for training state management."""

import pytest
from pathlib import Path
import tempfile
import shutil

from training.core.protocols import TrainingState, TrainingResult


class TestTrainingState:
    """Test TrainingState functionality."""
    
    def test_default_initialization(self):
        """Test default state initialization."""
        state = TrainingState()
        
        assert state.epoch == 0
        assert state.global_step == 0
        assert state.samples_seen == 0
        assert state.train_loss == 0.0
        assert state.val_loss == 0.0
        assert state.best_val_loss == float('inf')
        assert state.best_val_metric == 0.0
        assert state.metrics == {}
        assert state.train_history == []
        assert state.val_history == []
        assert state.should_stop == False
        assert state.improvement_streak == 0
        assert state.no_improvement_count == 0
        
    def test_state_to_dict(self):
        """Test converting state to dictionary."""
        state = TrainingState(
            epoch=5,
            global_step=100,
            train_loss=0.5,
            val_loss=0.4,
            metrics={"accuracy": 0.9}
        )
        
        state_dict = state.to_dict()
        
        assert state_dict["epoch"] == 5
        assert state_dict["global_step"] == 100
        assert state_dict["train_loss"] == 0.5
        assert state_dict["val_loss"] == 0.4
        assert state_dict["metrics"]["accuracy"] == 0.9
        
    def test_state_from_dict(self):
        """Test creating state from dictionary."""
        state_dict = {
            "epoch": 3,
            "global_step": 50,
            "samples_seen": 1000,
            "train_loss": 0.3,
            "val_loss": 0.2,
            "best_val_loss": 0.15,
            "best_val_metric": 0.95,
            "metrics": {"f1": 0.85},
            "train_history": [{"loss": 0.5}, {"loss": 0.4}],
            "val_history": [{"loss": 0.3}, {"loss": 0.2}],
            "should_stop": False,
            "improvement_streak": 2,
            "no_improvement_count": 0,
        }
        
        state = TrainingState.from_dict(state_dict)
        
        assert state.epoch == 3
        assert state.global_step == 50
        assert state.train_loss == 0.3
        assert state.val_loss == 0.2
        assert state.best_val_loss == 0.15
        assert state.metrics["f1"] == 0.85
        assert len(state.train_history) == 2
        
    def test_state_update_best_metrics(self):
        """Test updating best metrics."""
        state = TrainingState()
        
        # Initial best values
        assert state.best_val_loss == float('inf')
        assert state.best_val_metric == 0.0
        
        # Update with better loss
        state.val_loss = 0.5
        if state.val_loss < state.best_val_loss:
            state.best_val_loss = state.val_loss
            state.no_improvement_count = 0
            state.improvement_streak += 1
        
        assert state.best_val_loss == 0.5
        assert state.improvement_streak == 1
        assert state.no_improvement_count == 0
        
        # Update with worse loss
        state.val_loss = 0.6
        if state.val_loss >= state.best_val_loss:
            state.no_improvement_count += 1
            state.improvement_streak = 0
            
        assert state.best_val_loss == 0.5  # Unchanged
        assert state.improvement_streak == 0
        assert state.no_improvement_count == 1


class TestTrainingResult:
    """Test TrainingResult functionality."""
    
    def test_result_initialization(self):
        """Test result initialization."""
        result = TrainingResult(
            final_train_loss=0.2,
            final_val_loss=0.15,
            best_val_loss=0.14,
            best_val_metric=0.92,
            final_metrics={"accuracy": 0.91, "f1": 0.89},
            train_history=[{"loss": 0.5}, {"loss": 0.3}, {"loss": 0.2}],
            val_history=[{"loss": 0.4}, {"loss": 0.2}, {"loss": 0.15}],
        )
        
        assert result.final_train_loss == 0.2
        assert result.final_val_loss == 0.15
        assert result.best_val_loss == 0.14
        assert result.best_val_metric == 0.92
        assert result.final_metrics["accuracy"] == 0.91
        assert len(result.train_history) == 3
        assert len(result.val_history) == 3
        
    def test_result_with_paths(self):
        """Test result with model paths."""
        result = TrainingResult(
            final_train_loss=0.1,
            final_val_loss=0.08,
            best_val_loss=0.08,
            best_val_metric=0.95,
            final_metrics={},
            train_history=[],
            val_history=[],
            final_model_path=Path("/tmp/final_model"),
            best_model_path=Path("/tmp/best_model"),
            total_epochs=10,
            total_steps=1000,
            total_time=3600.0,
            early_stopped=False,
        )
        
        assert result.final_model_path == Path("/tmp/final_model")
        assert result.best_model_path == Path("/tmp/best_model")
        assert result.total_epochs == 10
        assert result.total_steps == 1000
        assert result.total_time == 3600.0
        assert result.early_stopped == False
        
    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        result = TrainingResult(
            final_train_loss=0.1,
            final_val_loss=0.08,
            best_val_loss=0.07,
            best_val_metric=0.96,
            final_metrics={"auc": 0.98},
            train_history=[],
            val_history=[],
            final_model_path=Path("/models/final"),
            best_model_path=Path("/models/best"),
            mlflow_run_id="abc123",
            early_stopped=True,
            stop_reason="No improvement for 3 epochs",
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["final_train_loss"] == 0.1
        assert result_dict["final_val_loss"] == 0.08
        assert result_dict["best_val_loss"] == 0.07
        assert result_dict["final_metrics"]["auc"] == 0.98
        assert result_dict["final_model_path"] == "/models/final"
        assert result_dict["best_model_path"] == "/models/best"
        assert result_dict["mlflow_run_id"] == "abc123"
        assert result_dict["early_stopped"] == True
        assert result_dict["stop_reason"] == "No improvement for 3 epochs"