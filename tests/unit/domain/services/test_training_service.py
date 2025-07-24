"""Unit tests for training service domain logic.

These tests verify the pure business logic of the training service
without any external dependencies or framework-specific code.
"""

import pytest
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from datetime import datetime

from domain.services.training_service import (
    TrainingConfig,
    TrainingState,
    TrainingPhase,
    OptimizerType,
    SchedulerType,
    LinearSchedule,
    CosineSchedule,
    EpochStrategy,
    StepStrategy,
    TrainingMetrics,
    TrainingService
)


class TestTrainingConfig:
    """Test training configuration logic."""
    
    def test_default_configuration(self):
        """Test default configuration values."""
        config = TrainingConfig()
        assert config.num_epochs == 3
        assert config.batch_size == 32
        assert config.learning_rate == 5e-5
        assert config.optimizer_type == OptimizerType.ADAMW
        assert config.scheduler_type == SchedulerType.WARMUP_LINEAR
    
    def test_effective_batch_size(self):
        """Test effective batch size calculation."""
        config = TrainingConfig(batch_size=16, gradient_accumulation_steps=4)
        assert config.effective_batch_size == 64
    
    def test_compute_total_steps(self):
        """Test total steps computation."""
        config = TrainingConfig(batch_size=10, num_epochs=2)
        total_steps = config.compute_total_steps(num_training_samples=100)
        assert total_steps == 20  # 100 samples / 10 batch_size * 2 epochs
    
    def test_compute_warmup_steps_from_ratio(self):
        """Test warmup steps computation from ratio."""
        config = TrainingConfig(warmup_ratio=0.1)
        warmup_steps = config.compute_warmup_steps(total_steps=1000)
        assert warmup_steps == 100
    
    def test_compute_warmup_steps_explicit(self):
        """Test explicit warmup steps."""
        config = TrainingConfig(warmup_steps=50, warmup_ratio=0.1)
        warmup_steps = config.compute_warmup_steps(total_steps=1000)
        assert warmup_steps == 50  # Explicit value takes precedence
    
    def test_validation_eval_strategy(self):
        """Test validation of eval strategy."""
        with pytest.raises(ValueError, match="eval_steps must be specified"):
            TrainingConfig(eval_strategy="steps", eval_steps=None)
    
    def test_validation_save_strategy(self):
        """Test validation of save strategy."""
        with pytest.raises(ValueError, match="save_steps must be specified"):
            TrainingConfig(save_strategy="steps", save_steps=None)


class TestTrainingState:
    """Test training state management."""
    
    def test_initial_state(self):
        """Test initial training state."""
        state = TrainingState()
        assert state.epoch == 0
        assert state.global_step == 0
        assert state.should_stop is False
        assert state.current_phase == TrainingPhase.WARMUP
    
    def test_current_phase_transitions(self):
        """Test phase transitions based on state."""
        state = TrainingState()
        
        # Initial phase is warmup
        assert state.current_phase == TrainingPhase.WARMUP
        
        # After first step, move to training
        state.global_step = 1
        assert state.current_phase == TrainingPhase.TRAINING
        
        # When stopping, move to cooldown
        state.should_stop = True
        assert state.current_phase == TrainingPhase.COOLDOWN
    
    def test_update_metrics_train(self):
        """Test updating training metrics."""
        state = TrainingState()
        metrics = {"loss": 0.5, "accuracy": 0.9}
        
        state.update_metrics(metrics, is_eval=False)
        
        assert len(state.train_history) == 1
        assert state.train_history[0]["loss"] == 0.5
        assert state.train_history[0]["accuracy"] == 0.9
        assert state.train_history[0]["step"] == 0
        assert state.train_history[0]["epoch"] == 0
        assert state.train_loss == 0.5
    
    def test_update_metrics_eval(self):
        """Test updating evaluation metrics."""
        state = TrainingState()
        metrics = {"loss": 0.3, "f1_score": 0.85}
        
        state.update_metrics(metrics, is_eval=True)
        
        assert len(state.eval_history) == 1
        assert state.eval_history[0]["loss"] == 0.3
        assert state.eval_history[0]["f1_score"] == 0.85
        assert state.eval_loss == 0.3
    
    def test_check_improvement_minimize(self):
        """Test improvement checking for metrics to minimize."""
        state = TrainingState()
        
        # First check always improves
        improved = state.check_improvement(0.5, "loss", greater_is_better=False)
        assert improved
        assert state.best_metric == 0.5
        assert state.best_model_step == 0
        
        # Better metric improves
        state.global_step = 10
        improved = state.check_improvement(0.3, "loss", greater_is_better=False)
        assert improved
        assert state.best_metric == 0.3
        assert state.best_model_step == 10
        assert state.early_stopping_counter == 0
        
        # Worse metric doesn't improve
        state.global_step = 20
        improved = state.check_improvement(0.4, "loss", greater_is_better=False)
        assert not improved
        assert state.best_metric == 0.3  # Unchanged
        assert state.best_model_step == 10  # Unchanged
        assert state.early_stopping_counter == 1
    
    def test_check_improvement_maximize(self):
        """Test improvement checking for metrics to maximize."""
        state = TrainingState()
        
        # First check always improves
        improved = state.check_improvement(0.8, "accuracy", greater_is_better=True)
        assert improved
        assert state.best_metric == 0.8
        
        # Better metric improves
        state.global_step = 10
        improved = state.check_improvement(0.9, "accuracy", greater_is_better=True)
        assert improved
        assert state.best_metric == 0.9
        
        # Worse metric doesn't improve
        state.global_step = 20
        improved = state.check_improvement(0.85, "accuracy", greater_is_better=True)
        assert not improved
        assert state.best_metric == 0.9


class TestLearningRateSchedules:
    """Test learning rate schedule implementations."""
    
    def test_linear_schedule_warmup(self):
        """Test linear schedule during warmup."""
        schedule = LinearSchedule(base_lr=1e-3, warmup_steps=10, total_steps=100)
        
        # At step 0
        assert schedule.get_lr(0) == 0.0
        
        # Halfway through warmup
        assert schedule.get_lr(5) == 5e-4
        
        # End of warmup
        assert schedule.get_lr(10) == 1e-3
    
    def test_linear_schedule_decay(self):
        """Test linear schedule decay after warmup."""
        schedule = LinearSchedule(base_lr=1e-3, warmup_steps=10, total_steps=100)
        
        # Right after warmup
        assert schedule.get_lr(10) == 1e-3
        
        # Halfway through training
        assert abs(schedule.get_lr(55) - 5e-4) < 1e-10
        
        # End of training
        assert schedule.get_lr(100) == 0.0
    
    def test_linear_schedule_no_total_steps(self):
        """Test linear schedule without total steps."""
        schedule = LinearSchedule(base_lr=1e-3, warmup_steps=10)
        
        # Should maintain base LR after warmup
        assert schedule.get_lr(10) == 1e-3
        assert schedule.get_lr(100) == 1e-3
        assert schedule.get_lr(1000) == 1e-3
    
    def test_cosine_schedule_warmup(self):
        """Test cosine schedule during warmup."""
        schedule = CosineSchedule(base_lr=1e-3, warmup_steps=10, total_steps=100)
        
        # Warmup behavior should be same as linear
        assert schedule.get_lr(0) == 0.0
        assert schedule.get_lr(5) == 5e-4
        assert schedule.get_lr(10) == 1e-3
    
    def test_cosine_schedule_decay(self):
        """Test cosine schedule decay after warmup."""
        schedule = CosineSchedule(base_lr=1e-3, warmup_steps=0, total_steps=100)
        
        # At start
        assert schedule.get_lr(0) == 1e-3
        
        # Quarter way (cos(pi/4) = sqrt(2)/2)
        lr_25 = schedule.get_lr(25)
        expected_25 = 1e-3 * (1 + 0.7071067811865476) / 2
        assert abs(lr_25 - expected_25) < 1e-10
        
        # Halfway (cos(pi/2) = 0)
        assert abs(schedule.get_lr(50) - 5e-4) < 1e-10
        
        # End (cos(pi) = -1)
        assert abs(schedule.get_lr(100)) < 1e-10


class TestTrainingStrategies:
    """Test training strategy implementations."""
    
    def test_epoch_strategy_evaluation(self):
        """Test epoch-based evaluation strategy."""
        config = TrainingConfig(eval_strategy="epoch")
        strategy = EpochStrategy(config)
        state = TrainingState()
        
        # No evaluation at step 0
        assert not strategy.should_evaluate(state)
        
        # Evaluation after first step
        state.global_step = 1
        assert strategy.should_evaluate(state)
    
    def test_epoch_strategy_saving(self):
        """Test epoch-based saving strategy."""
        config = TrainingConfig(save_strategy="epoch")
        strategy = EpochStrategy(config)
        state = TrainingState()
        
        # No saving at step 0
        assert not strategy.should_save(state)
        
        # Saving after first step
        state.global_step = 1
        assert strategy.should_save(state)
    
    def test_epoch_strategy_logging(self):
        """Test epoch-based logging strategy."""
        config = TrainingConfig(logging_steps=10, logging_first_step=True)
        strategy = EpochStrategy(config)
        state = TrainingState()
        
        # Log first step
        state.global_step = 1
        assert strategy.should_log(state)
        
        # Don't log intermediate steps
        state.global_step = 5
        assert not strategy.should_log(state)
        
        # Log at logging interval
        state.global_step = 10
        assert strategy.should_log(state)
    
    def test_step_strategy_evaluation(self):
        """Test step-based evaluation strategy."""
        config = TrainingConfig(eval_strategy="steps", eval_steps=100)
        strategy = StepStrategy(config)
        state = TrainingState()
        
        # Evaluate at step 0
        assert strategy.should_evaluate(state)
        
        # Don't evaluate at non-interval steps
        state.global_step = 50
        assert not strategy.should_evaluate(state)
        
        # Evaluate at interval
        state.global_step = 100
        assert strategy.should_evaluate(state)
    
    def test_step_strategy_saving(self):
        """Test step-based saving strategy."""
        config = TrainingConfig(save_strategy="steps", save_steps=500)
        strategy = StepStrategy(config)
        state = TrainingState()
        
        # Save at step 0
        assert strategy.should_save(state)
        
        # Don't save at non-interval steps
        state.global_step = 250
        assert not strategy.should_save(state)
        
        # Save at interval
        state.global_step = 500
        assert strategy.should_save(state)
    
    def test_early_stopping(self):
        """Test early stopping logic."""
        config = TrainingConfig(early_stopping_patience=3)
        strategy = EpochStrategy(config)
        state = TrainingState()
        
        # Initially should not stop
        assert not strategy.should_stop_early(state)
        
        # Increment counter but below patience
        state.early_stopping_counter = 2
        assert not strategy.should_stop_early(state)
        
        # Reach patience threshold
        state.early_stopping_counter = 3
        assert strategy.should_stop_early(state)
    
    def test_no_early_stopping(self):
        """Test disabled early stopping."""
        config = TrainingConfig(early_stopping_patience=None)
        strategy = EpochStrategy(config)
        state = TrainingState()
        
        # Should never stop early
        state.early_stopping_counter = 100
        assert not strategy.should_stop_early(state)


class TestTrainingMetrics:
    """Test training metrics handling."""
    
    def test_metrics_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = TrainingMetrics(
            loss=0.5,
            gradient_norm=2.0,
            learning_rate=1e-4,
            epoch=1.5,
            samples_per_second=100.0,
            steps_per_second=10.0,
            memory_used_gb=4.5,
            task_metrics={"accuracy": 0.9, "f1": 0.85}
        )
        
        result = metrics.to_dict()
        
        assert result["loss"] == 0.5
        assert result["gradient_norm"] == 2.0
        assert result["learning_rate"] == 1e-4
        assert result["epoch"] == 1.5
        assert result["samples_per_second"] == 100.0
        assert result["steps_per_second"] == 10.0
        assert result["memory_used_gb"] == 4.5
        assert result["accuracy"] == 0.9
        assert result["f1"] == 0.85
    
    def test_metrics_partial_dict(self):
        """Test converting partial metrics to dictionary."""
        metrics = TrainingMetrics(
            loss=0.3,
            learning_rate=5e-5,
            epoch=2.0
        )
        
        result = metrics.to_dict()
        
        assert result["loss"] == 0.3
        assert result["learning_rate"] == 5e-5
        assert result["epoch"] == 2.0
        assert "gradient_norm" not in result
        assert "samples_per_second" not in result


# Mock implementation for testing abstract TrainingService
class MockTrainingService(TrainingService[List[float], Dict[str, Any], Dict[str, Any]]):
    """Mock training service for testing base functionality."""
    
    def create_optimizer(self, parameters: Any) -> Dict[str, Any]:
        """Create mock optimizer."""
        return {
            "type": self.config.optimizer_type.value,
            "lr": self.config.learning_rate,
            "parameters": parameters
        }
    
    def create_scheduler(self, optimizer: Dict[str, Any]) -> Dict[str, Any]:
        """Create mock scheduler."""
        return {
            "type": self.config.scheduler_type.value,
            "optimizer": optimizer,
            "warmup_steps": self.config.warmup_steps
        }
    
    def training_step(
        self,
        model: Any,
        batch: Dict[str, List[float]],
        optimizer: Dict[str, Any]
    ) -> TrainingMetrics:
        """Execute mock training step."""
        return TrainingMetrics(
            loss=0.5,
            gradient_norm=1.5,
            learning_rate=optimizer["lr"],
            epoch=self.state.epoch
        )
    
    def evaluation_step(
        self,
        model: Any,
        batch: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """Execute mock evaluation step."""
        return {"loss": 0.3, "accuracy": 0.9}


class TestTrainingService:
    """Test abstract training service functionality."""
    
    def test_service_initialization(self):
        """Test training service initialization."""
        config = TrainingConfig(num_epochs=5, batch_size=16)
        service = MockTrainingService(config)
        
        assert service.config.num_epochs == 5
        assert service.config.batch_size == 16
        assert service.state.epoch == 0
        assert service.state.global_step == 0
        assert isinstance(service.strategy, EpochStrategy)
    
    def test_service_with_step_strategy(self):
        """Test service with step-based strategy."""
        config = TrainingConfig(eval_strategy="steps", eval_steps=100)
        service = MockTrainingService(config)
        
        assert isinstance(service.strategy, StepStrategy)
    
    def test_service_should_methods(self):
        """Test service delegation to strategy."""
        config = TrainingConfig(
            eval_strategy="steps",
            eval_steps=10,
            save_strategy="steps",
            save_steps=20,
            logging_steps=5
        )
        service = MockTrainingService(config)
        
        # Test evaluation
        assert service.should_evaluate()  # Step 0
        service.state.global_step = 5
        assert not service.should_evaluate()
        service.state.global_step = 10
        assert service.should_evaluate()
        
        # Test saving
        service.state.global_step = 0
        assert service.should_save()
        service.state.global_step = 15
        assert not service.should_save()
        service.state.global_step = 20
        assert service.should_save()
        
        # Test logging
        service.state.global_step = 0
        assert service.should_log()
        service.state.global_step = 3
        assert not service.should_log()
        service.state.global_step = 5
        assert service.should_log()
    
    def test_service_update_state(self):
        """Test updating service state."""
        config = TrainingConfig()
        service = MockTrainingService(config)
        
        metrics = TrainingMetrics(
            loss=0.4,
            learning_rate=1e-4,
            epoch=1.0
        )
        
        # Update training metrics
        service.update_state(metrics, is_eval=False)
        
        assert service.state.global_step == 1
        assert service.state.learning_rate == 1e-4
        assert len(service.state.train_history) == 1
        assert service.state.train_history[0]["loss"] == 0.4
        
        # Update eval metrics
        eval_metrics = TrainingMetrics(
            loss=0.3,
            learning_rate=1e-4,
            epoch=1.0,
            task_metrics={"accuracy": 0.95}
        )
        service.update_state(eval_metrics, is_eval=True)
        
        assert service.state.global_step == 1  # No increment for eval
        assert len(service.state.eval_history) == 1
        assert service.state.eval_history[0]["loss"] == 0.3
        assert service.state.eval_history[0]["accuracy"] == 0.95
    
    def test_service_should_stop(self):
        """Test stopping conditions."""
        config = TrainingConfig(early_stopping_patience=2)
        service = MockTrainingService(config)
        
        # Initially should not stop
        assert not service.should_stop()
        
        # Set early stopping counter at threshold
        service.state.early_stopping_counter = 2
        assert service.should_stop()
        
        # Direct stop flag
        service.state.early_stopping_counter = 0
        service.state.should_stop = True
        assert service.should_stop()