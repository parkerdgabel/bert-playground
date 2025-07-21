"""Unit tests for base callback functionality."""

from pathlib import Path

from training.callbacks.base import Callback, CallbackList
from training.core.protocols import TrainingResult
from training.core.state import TrainingState


class MockTrainer:
    """Mock trainer for testing callbacks."""

    def __init__(self):
        self.model = None
        self.config = None
        self.state = TrainingState()


class RecordingCallback(Callback):
    """Callback that records all events."""

    def __init__(self, name: str = "recorder", priority: int = 50):
        super().__init__()
        self.name = name
        self._priority = priority
        self.events = []

    @property
    def priority(self) -> int:
        return self._priority

    def on_train_begin(self, trainer, state):
        self.events.append(f"{self.name}:train_begin")

    def on_train_end(self, trainer, state, result):
        self.events.append(f"{self.name}:train_end")

    def on_epoch_begin(self, trainer, state):
        self.events.append(f"{self.name}:epoch_begin_{state.epoch}")

    def on_epoch_end(self, trainer, state):
        self.events.append(f"{self.name}:epoch_end_{state.epoch}")

    def on_batch_begin(self, trainer, state, batch):
        self.events.append(f"{self.name}:batch_begin_{state.global_step}")

    def on_batch_end(self, trainer, state, loss):
        self.events.append(f"{self.name}:batch_end_{state.global_step}")

    def on_evaluate_begin(self, trainer, state):
        self.events.append(f"{self.name}:evaluate_begin")

    def on_evaluate_end(self, trainer, state, metrics):
        self.events.append(f"{self.name}:evaluate_end")

    def on_checkpoint_save(self, trainer, state, checkpoint_path):
        self.events.append(f"{self.name}:checkpoint_save")

    def on_checkpoint_load(self, trainer, state, checkpoint_path):
        self.events.append(f"{self.name}:checkpoint_load")

    def on_log(self, trainer, state, logs):
        self.events.append(f"{self.name}:log")


class ErrorCallback(Callback):
    """Callback that raises errors for testing error handling."""

    def __init__(self, error_on: str = "train_begin"):
        super().__init__()
        self.error_on = error_on

    def on_train_begin(self, trainer, state):
        if self.error_on == "train_begin":
            raise RuntimeError("Error in on_train_begin")

    def on_epoch_begin(self, trainer, state):
        if self.error_on == "epoch_begin":
            raise RuntimeError("Error in on_epoch_begin")


class TestCallback:
    """Test base Callback class."""

    def test_callback_initialization(self):
        """Test callback initialization."""
        callback = Callback()
        assert callback.trainer is None
        assert hasattr(callback, "priority")
        assert callback.priority == 50  # Default priority

    def test_callback_set_trainer(self):
        """Test setting trainer on callback."""
        callback = Callback()
        trainer = MockTrainer()

        callback.set_trainer(trainer)
        assert callback.trainer == trainer

    def test_callback_methods_exist(self):
        """Test all callback methods exist."""
        callback = Callback()

        # All methods should exist and be callable
        methods = [
            "on_train_begin",
            "on_train_end",
            "on_epoch_begin",
            "on_epoch_end",
            "on_batch_begin",
            "on_batch_end",
            "on_evaluate_begin",
            "on_evaluate_end",
            "on_checkpoint_save",
            "on_checkpoint_load",
            "on_log",
        ]

        for method in methods:
            assert hasattr(callback, method)
            assert callable(getattr(callback, method))

    def test_callback_priority(self):
        """Test callback priority values."""
        # Create callbacks with different priorities
        high_priority = RecordingCallback("high", 10)  # Lower number = higher priority
        normal_priority = RecordingCallback("normal", 50)
        low_priority = RecordingCallback("low", 100)

        assert high_priority.priority < normal_priority.priority
        assert normal_priority.priority < low_priority.priority


class TestCallbackList:
    """Test CallbackList functionality."""

    def test_callback_list_initialization(self):
        """Test callback list initialization."""
        callbacks = [RecordingCallback("cb1"), RecordingCallback("cb2")]
        callback_list = CallbackList(callbacks)

        assert len(callback_list.callbacks) == 2
        assert all(isinstance(cb, Callback) for cb in callback_list.callbacks)

    def test_callback_list_empty(self):
        """Test empty callback list."""
        callback_list = CallbackList([])
        trainer = MockTrainer()

        # Should not raise any errors
        callback_list.on_train_begin(trainer, trainer.state)
        callback_list.on_train_end(trainer, trainer.state, None)

    def test_callback_list_order(self):
        """Test callback execution order based on priority."""
        # Create callbacks with different priorities
        cb_low = RecordingCallback("low", 100)
        cb_high = RecordingCallback("high", 10)
        cb_normal = RecordingCallback("normal", 50)

        # Add in random order
        callback_list = CallbackList([cb_low, cb_high, cb_normal])

        trainer = MockTrainer()
        callback_list.on_train_begin(trainer, trainer.state)

        # Check execution order (high priority first)
        all_events = cb_high.events + cb_normal.events + cb_low.events
        assert all_events[0] == "high:train_begin"
        assert all_events[1] == "normal:train_begin"
        assert all_events[2] == "low:train_begin"

    def test_callback_list_all_methods(self):
        """Test all callback methods are called."""
        callback = RecordingCallback()
        callback_list = CallbackList([callback])

        trainer = MockTrainer()
        state = trainer.state

        # Call all methods
        callback_list.on_train_begin(trainer, state)

        state.epoch = 1
        callback_list.on_epoch_begin(trainer, state)

        state.global_step = 1
        callback_list.on_batch_begin(trainer, state, {"data": "batch"})
        callback_list.on_batch_end(trainer, state, 0.5)

        callback_list.on_epoch_end(trainer, state)

        callback_list.on_evaluate_begin(trainer, state)
        callback_list.on_evaluate_end(trainer, state, {"loss": 0.4})
        callback_list.on_checkpoint_save(trainer, state, "/tmp/checkpoint")
        callback_list.on_log(trainer, state, {"loss": 0.5})

        # Create a minimal TrainingResult
        result = TrainingResult(
            final_train_loss=0.3,
            final_val_loss=0.2,
            best_val_loss=0.15,
            best_val_metric=0.9,
            final_metrics={"accuracy": 0.9},
            train_history=[],
            val_history=[],
            final_model_path=Path("/tmp/final"),
            best_model_path=Path("/tmp/best"),
            total_epochs=1,
            total_steps=100,
        )
        callback_list.on_train_end(trainer, state, result)

        # Verify all events were recorded
        expected_events = [
            "recorder:train_begin",
            "recorder:epoch_begin_1",
            "recorder:batch_begin_1",
            "recorder:batch_end_1",
            "recorder:epoch_end_1",
            "recorder:evaluate_begin",
            "recorder:evaluate_end",
            "recorder:checkpoint_save",
            "recorder:log",
            "recorder:train_end",
        ]

        assert callback.events == expected_events

    def test_callback_list_error_handling(self):
        """Test error handling in callback list."""
        error_callback = ErrorCallback(error_on="train_begin")
        normal_callback = RecordingCallback()

        callback_list = CallbackList([error_callback, normal_callback])

        trainer = MockTrainer()

        # Error in one callback should not stop others - errors are logged
        callback_list.on_train_begin(trainer, trainer.state)

        # Normal callback should have been called despite error in first callback
        assert len(normal_callback.events) == 1
        assert normal_callback.events[0] == "recorder:train_begin"

    def test_callback_list_set_trainer(self):
        """Test setting trainer on all callbacks."""
        cb1 = RecordingCallback("cb1")
        cb2 = RecordingCallback("cb2")

        callback_list = CallbackList([cb1, cb2])
        trainer = MockTrainer()

        # Set trainer on callback list
        for cb in callback_list.callbacks:
            cb.set_trainer(trainer)

        assert cb1.trainer == trainer
        assert cb2.trainer == trainer

    def test_callback_list_multiple_callbacks_same_priority(self):
        """Test multiple callbacks with same priority."""
        cb1 = RecordingCallback("cb1", 50)
        cb2 = RecordingCallback("cb2", 50)
        cb3 = RecordingCallback("cb3", 50)

        callback_list = CallbackList([cb1, cb2, cb3])

        trainer = MockTrainer()
        callback_list.on_train_begin(trainer, trainer.state)

        # All should be called
        assert len(cb1.events) == 1
        assert len(cb2.events) == 1
        assert len(cb3.events) == 1

    def test_callback_state_sharing(self):
        """Test callbacks can share state through trainer."""

        class StateWriterCallback(Callback):
            def on_train_begin(self, trainer, state):
                # Write some state
                trainer.shared_value = 42

        class StateReaderCallback(Callback):
            def __init__(self):
                super().__init__()
                self.read_value = None

            def on_epoch_begin(self, trainer, state):
                # Read the state
                self.read_value = getattr(trainer, "shared_value", None)

        writer = StateWriterCallback()
        reader = StateReaderCallback()

        callback_list = CallbackList([writer, reader])
        trainer = MockTrainer()

        callback_list.on_train_begin(trainer, trainer.state)
        callback_list.on_epoch_begin(trainer, trainer.state)

        assert reader.read_value == 42
