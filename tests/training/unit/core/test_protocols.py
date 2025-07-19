"""Unit tests for training protocols."""

import pytest
import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Dict, Any, Iterator
from pathlib import Path

from training.core.protocols import (
    Model,
    DataLoader,
    Optimizer,
    LRScheduler,
    TrainingState,
    TrainingResult,
    TrainerConfig,
    Trainer,
    TrainingHook,
    MetricsCollector,
    CheckpointManager,
)


class TestProtocols:
    """Test protocol definitions and compliance."""
    
    def test_model_protocol(self):
        """Test Model protocol compliance."""
        
        class ValidModel:
            def __call__(self, x: mx.array) -> mx.array:
                return x
            
            def loss(self, batch: Dict[str, mx.array]) -> mx.array:
                return mx.array(0.5)
            
            def parameters(self) -> Dict[str, mx.array]:
                return {"weight": mx.ones((10, 10))}
            
            def trainable_parameters(self) -> Dict[str, mx.array]:
                return self.parameters()
            
            def save_weights(self, path: Path) -> None:
                pass
            
            def load_weights(self, path: Path) -> None:
                pass
        
        # Should not raise
        model: Model = ValidModel()
        assert callable(model)
        assert hasattr(model, "loss")
        assert hasattr(model, "parameters")
    
    def test_dataloader_protocol(self):
        """Test DataLoader protocol compliance."""
        
        class ValidDataLoader:
            def __iter__(self) -> Iterator[Dict[str, mx.array]]:
                for i in range(3):
                    yield {"input": mx.ones((4, 10)), "labels": mx.zeros((4,))}
            
            def __len__(self) -> int:
                return 3
        
        # Should not raise
        loader: DataLoader = ValidDataLoader()
        assert len(loader) == 3
        
        # Test iteration
        batches = list(loader)
        assert len(batches) == 3
        assert all("input" in batch for batch in batches)
    
    def test_optimizer_protocol(self):
        """Test Optimizer protocol compliance."""
        
        class ValidOptimizer:
            def __init__(self):
                self.learning_rate = 1e-3
                self.state = {}
            
            def update(self, model: Any, gradients: Dict[str, mx.array]) -> None:
                pass
            
            def state_dict(self) -> Dict[str, Any]:
                return self.state
            
            def load_state_dict(self, state: Dict[str, Any]) -> None:
                self.state = state
        
        # Should not raise
        opt: Optimizer = ValidOptimizer()
        assert opt.learning_rate == 1e-3
        assert opt.state_dict() == {}
    
    def test_lr_scheduler_protocol(self):
        """Test LRScheduler protocol compliance."""
        
        class ValidScheduler:
            def __init__(self):
                self.last_lr = 1e-3
            
            def step(self, metrics: Optional[float] = None) -> None:
                pass
            
            def get_last_lr(self) -> float:
                return self.last_lr
            
            def state_dict(self) -> Dict[str, Any]:
                return {"last_lr": self.last_lr}
            
            def load_state_dict(self, state: Dict[str, Any]) -> None:
                self.last_lr = state["last_lr"]
        
        # Should not raise
        scheduler: LRScheduler = ValidScheduler()
        assert scheduler.get_last_lr() == 1e-3
    
    def test_training_state_protocol(self):
        """Test TrainingState protocol compliance."""
        
        class ValidState:
            def __init__(self):
                self.epoch = 0
                self.global_step = 0
                self.best_metric = None
                self.metrics_history = {}
                self.early_stopping_counter = 0
            
            def to_dict(self) -> Dict[str, Any]:
                return {
                    "epoch": self.epoch,
                    "global_step": self.global_step,
                    "best_metric": self.best_metric,
                }
            
            @classmethod
            def from_dict(cls, data: Dict[str, Any]) -> "ValidState":
                state = cls()
                state.epoch = data.get("epoch", 0)
                state.global_step = data.get("global_step", 0)
                state.best_metric = data.get("best_metric")
                return state
        
        # Should not raise
        state: TrainingState = ValidState()
        assert state.epoch == 0
        assert state.global_step == 0
        
        # Test serialization
        state_dict = state.to_dict()
        restored = ValidState.from_dict(state_dict)
        assert restored.epoch == state.epoch
    
    def test_training_result_protocol(self):
        """Test TrainingResult protocol compliance."""
        
        class ValidResult:
            def __init__(self):
                self.metrics = {"loss": 0.5, "accuracy": 0.9}
                self.best_checkpoint = Path("/tmp/best")
                self.final_checkpoint = Path("/tmp/final")
                self.history = {"loss": [0.7, 0.6, 0.5]}
            
            def to_dict(self) -> Dict[str, Any]:
                return {
                    "metrics": self.metrics,
                    "best_checkpoint": str(self.best_checkpoint),
                    "final_checkpoint": str(self.final_checkpoint),
                    "history": self.history,
                }
        
        # Should not raise
        result: TrainingResult = ValidResult()
        assert result.metrics["loss"] == 0.5
        assert isinstance(result.best_checkpoint, Path)
    
    def test_trainer_config_protocol(self):
        """Test TrainerConfig protocol compliance."""
        
        class ValidConfig:
            def __init__(self):
                self.learning_rate = 1e-3
                self.num_epochs = 10
                self.batch_size = 32
                self.output_dir = Path("/tmp/output")
            
            def to_dict(self) -> Dict[str, Any]:
                return {
                    "learning_rate": self.learning_rate,
                    "num_epochs": self.num_epochs,
                    "batch_size": self.batch_size,
                    "output_dir": str(self.output_dir),
                }
            
            def validate(self) -> None:
                assert self.learning_rate > 0
                assert self.num_epochs > 0
                assert self.batch_size > 0
        
        # Should not raise
        config: TrainerConfig = ValidConfig()
        config.validate()
        assert config.learning_rate == 1e-3
    
    def test_trainer_protocol(self):
        """Test Trainer protocol compliance."""
        
        class ValidTrainer:
            def __init__(self):
                self.model = None
                self.config = None
                self.state = None
            
            def train(
                self,
                train_dataloader: DataLoader,
                val_dataloader: Optional[DataLoader] = None,
                resume_from: Optional[Path] = None,
            ) -> TrainingResult:
                return type("Result", (), {
                    "metrics": {"loss": 0.5},
                    "best_checkpoint": Path("/tmp/best"),
                    "final_checkpoint": Path("/tmp/final"),
                    "history": {},
                    "to_dict": lambda: {},
                })()
            
            def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
                return {"eval_loss": 0.4}
            
            def predict(self, dataloader: DataLoader) -> mx.array:
                return mx.zeros((10, 2))
            
            def save_checkpoint(self, path: Path) -> None:
                pass
            
            def load_checkpoint(self, path: Path) -> None:
                pass
        
        # Should not raise
        trainer: Trainer = ValidTrainer()
        assert hasattr(trainer, "train")
        assert hasattr(trainer, "evaluate")
    
    def test_training_hook_protocol(self):
        """Test TrainingHook protocol compliance."""
        
        class ValidHook:
            def on_train_begin(self, trainer: Trainer, state: TrainingState) -> None:
                pass
            
            def on_train_end(self, trainer: Trainer, state: TrainingState) -> None:
                pass
            
            def on_epoch_begin(self, trainer: Trainer, state: TrainingState) -> None:
                pass
            
            def on_epoch_end(self, trainer: Trainer, state: TrainingState) -> None:
                pass
            
            def on_step_begin(self, trainer: Trainer, state: TrainingState) -> None:
                pass
            
            def on_step_end(self, trainer: Trainer, state: TrainingState) -> None:
                pass
            
            def on_evaluate(self, trainer: Trainer, state: TrainingState, metrics: Dict[str, float]) -> None:
                pass
            
            def on_checkpoint(self, trainer: Trainer, state: TrainingState, checkpoint_path: Path) -> None:
                pass
        
        # Should not raise
        hook: TrainingHook = ValidHook()
        assert hasattr(hook, "on_train_begin")
        assert hasattr(hook, "on_step_end")
    
    def test_metrics_collector_protocol(self):
        """Test MetricsCollector protocol compliance."""
        
        class ValidCollector:
            def __init__(self):
                self.metrics = {}
            
            def add_metric(self, name: str, metric: Any) -> None:
                self.metrics[name] = metric
            
            def update(self, predictions: mx.array, targets: mx.array) -> None:
                pass
            
            def compute(self) -> Dict[str, float]:
                return {"accuracy": 0.9}
            
            def reset(self) -> None:
                self.metrics = {}
            
            def get_history(self) -> Dict[str, list]:
                return {"accuracy": [0.8, 0.85, 0.9]}
        
        # Should not raise
        collector: MetricsCollector = ValidCollector()
        collector.add_metric("test", 1.0)
        assert collector.compute() == {"accuracy": 0.9}
    
    def test_checkpoint_manager_protocol(self):
        """Test CheckpointManager protocol compliance."""
        
        class ValidManager:
            def __init__(self):
                self.checkpoints = []
            
            def save_checkpoint(
                self,
                model: Model,
                optimizer: Optimizer,
                scheduler: Optional[LRScheduler],
                state: TrainingState,
                path: Path,
                is_best: bool = False,
            ) -> None:
                self.checkpoints.append(path)
            
            def load_checkpoint(
                self,
                path: Path,
                model: Model,
                optimizer: Optional[Optimizer] = None,
                scheduler: Optional[LRScheduler] = None,
            ) -> TrainingState:
                return type("State", (), {
                    "epoch": 1,
                    "global_step": 100,
                    "best_metric": 0.9,
                    "metrics_history": {},
                    "early_stopping_counter": 0,
                    "to_dict": lambda: {},
                    "from_dict": lambda x: x,
                })()
            
            def get_latest_checkpoint(self) -> Optional[Path]:
                return self.checkpoints[-1] if self.checkpoints else None
            
            def get_best_checkpoint(self) -> Optional[Path]:
                return Path("/tmp/best") if self.checkpoints else None
            
            def cleanup_checkpoints(self, keep_last_n: int = 3) -> None:
                self.checkpoints = self.checkpoints[-keep_last_n:]
        
        # Should not raise
        manager: CheckpointManager = ValidManager()
        assert manager.get_latest_checkpoint() is None
        manager.checkpoints.append(Path("/tmp/ckpt"))
        assert manager.get_latest_checkpoint() == Path("/tmp/ckpt")


class TestProtocolInheritance:
    """Test that concrete implementations properly inherit from protocols."""
    
    def test_mlx_module_is_model(self):
        """Test that MLX nn.Module satisfies Model protocol."""
        
        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 2)
            
            def __call__(self, x):
                return self.linear(x)
            
            def loss(self, batch):
                x = batch["input"]
                y = batch["labels"]
                logits = self(x)
                return mx.mean((logits - y) ** 2)
        
        # MLX Module should satisfy Model protocol requirements
        model = TestModule()
        
        # Test required methods
        assert callable(model)
        assert hasattr(model, "loss")
        assert hasattr(model, "parameters")
        assert hasattr(model, "trainable_parameters")
        assert hasattr(model, "save_weights")
        assert hasattr(model, "load_weights")
        
        # Test method functionality
        params = model.parameters()
        assert isinstance(params, dict)
        assert len(params) > 0
        
        trainable = model.trainable_parameters()
        assert isinstance(trainable, dict)
    
    def test_protocol_typing(self):
        """Test that protocol typing works correctly."""
        
        def process_model(model: Model) -> Dict[str, mx.array]:
            """Function that accepts Model protocol."""
            return model.parameters()
        
        # Should work with any class implementing Model protocol
        class CustomModel:
            def __call__(self, x): return x
            def loss(self, batch): return mx.array(0.0)
            def parameters(self): return {"w": mx.ones((5, 5))}
            def trainable_parameters(self): return self.parameters()
            def save_weights(self, path): pass
            def load_weights(self, path): pass
        
        model = CustomModel()
        params = process_model(model)
        assert "w" in params