"""
Integration between the event system and existing k-bert components.
"""

from typing import Any, Optional

import mlx.core as mx
from loguru import logger

from domain.protocols.training import TrainingResult, TrainingState
from training.callbacks.base import Callback

from .bus import EventBus, GlobalEventBus
from .types import EventType, TrainingEvent


class EventBusCallback(Callback):
    """
    Callback that publishes training events to an event bus.
    
    This bridges the existing callback system with the new event system.
    """

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        include_batch_events: bool = True,
        custom_event_mapping: Optional[dict[str, EventType]] = None
    ):
        """
        Initialize event bus callback.
        
        Args:
            event_bus: Event bus to publish to (uses global if None)
            include_batch_events: Whether to publish batch-level events
            custom_event_mapping: Custom mapping of callback methods to event types
        """
        super().__init__()
        self.event_bus = event_bus or GlobalEventBus()
        self.include_batch_events = include_batch_events
        self.custom_event_mapping = custom_event_mapping or {}
        
        logger.debug(
            f"EventBusCallback initialized with bus '{self.event_bus.name}'"
        )

    @property
    def priority(self) -> int:
        """High priority to ensure events are published early."""
        return 10

    def on_train_begin(self, trainer: Any, state: TrainingState) -> None:
        """Publish training started event."""
        event = TrainingEvent.create(
            event_type=self.custom_event_mapping.get(
                "on_train_begin", EventType.TRAINING_STARTED
            ),
            name="training.started",
            source="EventBusCallback",
            source_type=type(self),
            training_state=state,
            data={
                "total_epochs": state.num_epochs,
                "device": str(mx.default_device()),
            }
        )
        self.event_bus.publish(event)

    def on_train_end(
        self, trainer: Any, state: TrainingState, result: TrainingResult
    ) -> None:
        """Publish training completed event."""
        event_type = EventType.TRAINING_COMPLETED
        if result.status == "failed":
            event_type = EventType.TRAINING_FAILED
        elif result.status == "cancelled":
            event_type = EventType.TRAINING_CANCELLED
            
        event = TrainingEvent.create(
            event_type=self.custom_event_mapping.get("on_train_end", event_type),
            name=f"training.{result.status}",
            source="EventBusCallback",
            source_type=type(self),
            training_state=state,
            data={
                "result": result.__dict__,
                "total_time": result.total_time,
                "best_metrics": result.best_metrics,
            }
        )
        self.event_bus.publish(event)

    def on_epoch_begin(self, trainer: Any, state: TrainingState) -> None:
        """Publish epoch started event."""
        event = TrainingEvent.create(
            event_type=self.custom_event_mapping.get(
                "on_epoch_begin", EventType.EPOCH_STARTED
            ),
            name="epoch.started",
            source="EventBusCallback",
            source_type=type(self),
            training_state=state,
            data={
                "epoch": state.epoch,
                "total_epochs": state.num_epochs,
            }
        )
        self.event_bus.publish(event)

    def on_epoch_end(self, trainer: Any, state: TrainingState) -> None:
        """Publish epoch completed event."""
        event = TrainingEvent.create(
            event_type=self.custom_event_mapping.get(
                "on_epoch_end", EventType.EPOCH_COMPLETED
            ),
            name="epoch.completed",
            source="EventBusCallback",
            source_type=type(self),
            training_state=state,
            data={
                "epoch": state.epoch,
                "train_loss": state.train_loss,
                "val_loss": state.val_loss,
                "metrics": state.metrics,
            }
        )
        self.event_bus.publish(event)

    def on_batch_begin(
        self, trainer: Any, state: TrainingState, batch: dict[str, mx.array]
    ) -> None:
        """Publish batch started event."""
        if not self.include_batch_events:
            return
            
        event = TrainingEvent.create(
            event_type=self.custom_event_mapping.get(
                "on_batch_begin", EventType.BATCH_STARTED
            ),
            name="batch.started",
            source="EventBusCallback",
            source_type=type(self),
            training_state=state,
            data={
                "batch_idx": state.batch_idx,
                "batch_size": batch["input_ids"].shape[0] if "input_ids" in batch else None,
            }
        )
        self.event_bus.publish(event)

    def on_batch_end(self, trainer: Any, state: TrainingState, loss: float) -> None:
        """Publish batch completed event."""
        if not self.include_batch_events:
            return
            
        event = TrainingEvent.create(
            event_type=self.custom_event_mapping.get(
                "on_batch_end", EventType.BATCH_COMPLETED
            ),
            name="batch.completed",
            source="EventBusCallback",
            source_type=type(self),
            training_state=state,
            data={
                "batch_idx": state.batch_idx,
                "loss": loss,
                "learning_rate": state.learning_rate,
            }
        )
        self.event_bus.publish(event)

    def on_evaluate_begin(self, trainer: Any, state: TrainingState) -> None:
        """Publish evaluation started event."""
        event = TrainingEvent.create(
            event_type=self.custom_event_mapping.get(
                "on_evaluate_begin", EventType.EVALUATION_STARTED
            ),
            name="evaluation.started",
            source="EventBusCallback",
            source_type=type(self),
            training_state=state,
            data={
                "epoch": state.epoch,
                "global_step": state.global_step,
            }
        )
        self.event_bus.publish(event)

    def on_evaluate_end(
        self, trainer: Any, state: TrainingState, metrics: dict[str, float]
    ) -> None:
        """Publish evaluation completed event."""
        event = TrainingEvent.create(
            event_type=self.custom_event_mapping.get(
                "on_evaluate_end", EventType.EVALUATION_COMPLETED
            ),
            name="evaluation.completed",
            source="EventBusCallback",
            source_type=type(self),
            training_state=state,
            data={
                "metrics": metrics,
                "epoch": state.epoch,
                "improved": self._check_improvement(state, metrics),
            }
        )
        self.event_bus.publish(event)

    def on_checkpoint_save(
        self, trainer: Any, state: TrainingState, checkpoint_path: str
    ) -> None:
        """Publish checkpoint saved event."""
        event = TrainingEvent.create(
            event_type=self.custom_event_mapping.get(
                "on_checkpoint_save", EventType.CHECKPOINT_SAVED
            ),
            name="checkpoint.saved",
            source="EventBusCallback",
            source_type=type(self),
            training_state=state,
            data={
                "checkpoint_path": checkpoint_path,
                "epoch": state.epoch,
                "global_step": state.global_step,
                "metrics": state.metrics,
            }
        )
        self.event_bus.publish(event)

    def on_checkpoint_load(
        self, trainer: Any, state: TrainingState, checkpoint_path: str
    ) -> None:
        """Publish checkpoint loaded event."""
        event = TrainingEvent.create(
            event_type=self.custom_event_mapping.get(
                "on_checkpoint_load", EventType.CHECKPOINT_LOADED
            ),
            name="checkpoint.loaded",
            source="EventBusCallback",
            source_type=type(self),
            training_state=state,
            data={
                "checkpoint_path": checkpoint_path,
                "resumed_epoch": state.epoch,
                "resumed_step": state.global_step,
            }
        )
        self.event_bus.publish(event)

    def on_log(
        self, trainer: Any, state: TrainingState, logs: dict[str, Any]
    ) -> None:
        """Publish metric logged event."""
        event = TrainingEvent.create(
            event_type=self.custom_event_mapping.get(
                "on_log", EventType.METRIC_LOGGED
            ),
            name="metric.logged",
            source="EventBusCallback",
            source_type=type(self),
            training_state=state,
            data={
                "logs": logs,
                "epoch": state.epoch,
                "step": state.global_step,
            }
        )
        self.event_bus.publish(event)

    def _check_improvement(
        self, state: TrainingState, metrics: dict[str, float]
    ) -> bool:
        """Check if metrics have improved."""
        # Simple check - can be enhanced
        if not state.best_val_loss:
            return True
            
        val_loss = metrics.get("val_loss", float("inf"))
        return val_loss < state.best_val_loss


class CallbackEventHandler:
    """
    Event handler that delegates to callback methods.
    
    This allows using the event system to trigger callback behavior.
    """

    def __init__(self, callback: Callback):
        """
        Initialize callback event handler.
        
        Args:
            callback: Callback to delegate to
        """
        self.callback = callback
        
        # Map event types to callback methods
        self.event_mapping = {
            EventType.TRAINING_STARTED: self._on_train_begin,
            EventType.TRAINING_COMPLETED: self._on_train_end,
            EventType.TRAINING_FAILED: self._on_train_end,
            EventType.EPOCH_STARTED: self._on_epoch_begin,
            EventType.EPOCH_COMPLETED: self._on_epoch_end,
            EventType.BATCH_STARTED: self._on_batch_begin,
            EventType.BATCH_COMPLETED: self._on_batch_end,
            EventType.EVALUATION_STARTED: self._on_evaluate_begin,
            EventType.EVALUATION_COMPLETED: self._on_evaluate_end,
            EventType.CHECKPOINT_SAVED: self._on_checkpoint_save,
            EventType.CHECKPOINT_LOADED: self._on_checkpoint_load,
            EventType.METRIC_LOGGED: self._on_log,
        }

    def handle(self, event: TrainingEvent) -> None:
        """Handle training event by delegating to callback."""
        handler = self.event_mapping.get(event.type)
        if handler:
            handler(event)

    def _on_train_begin(self, event: TrainingEvent) -> None:
        """Handle training started event."""
        if event.training_state and self.callback.trainer:
            self.callback.on_train_begin(
                self.callback.trainer, event.training_state
            )

    def _on_train_end(self, event: TrainingEvent) -> None:
        """Handle training end events."""
        if event.training_state and self.callback.trainer:
            result = event.get_data("result")
            if result:
                # Reconstruct TrainingResult
                training_result = TrainingResult(**result)
                self.callback.on_train_end(
                    self.callback.trainer, event.training_state, training_result
                )

    def _on_epoch_begin(self, event: TrainingEvent) -> None:
        """Handle epoch started event."""
        if event.training_state and self.callback.trainer:
            self.callback.on_epoch_begin(
                self.callback.trainer, event.training_state
            )

    def _on_epoch_end(self, event: TrainingEvent) -> None:
        """Handle epoch completed event."""
        if event.training_state and self.callback.trainer:
            self.callback.on_epoch_end(
                self.callback.trainer, event.training_state
            )

    def _on_batch_begin(self, event: TrainingEvent) -> None:
        """Handle batch started event."""
        if event.training_state and self.callback.trainer:
            batch = event.get_data("batch")
            if batch:
                self.callback.on_batch_begin(
                    self.callback.trainer, event.training_state, batch
                )

    def _on_batch_end(self, event: TrainingEvent) -> None:
        """Handle batch completed event."""
        if event.training_state and self.callback.trainer:
            loss = event.get_data("loss")
            if loss is not None:
                self.callback.on_batch_end(
                    self.callback.trainer, event.training_state, loss
                )

    def _on_evaluate_begin(self, event: TrainingEvent) -> None:
        """Handle evaluation started event."""
        if event.training_state and self.callback.trainer:
            self.callback.on_evaluate_begin(
                self.callback.trainer, event.training_state
            )

    def _on_evaluate_end(self, event: TrainingEvent) -> None:
        """Handle evaluation completed event."""
        if event.training_state and self.callback.trainer:
            metrics = event.get_data("metrics")
            if metrics:
                self.callback.on_evaluate_end(
                    self.callback.trainer, event.training_state, metrics
                )

    def _on_checkpoint_save(self, event: TrainingEvent) -> None:
        """Handle checkpoint saved event."""
        if event.training_state and self.callback.trainer:
            checkpoint_path = event.get_data("checkpoint_path")
            if checkpoint_path:
                self.callback.on_checkpoint_save(
                    self.callback.trainer, event.training_state, checkpoint_path
                )

    def _on_checkpoint_load(self, event: TrainingEvent) -> None:
        """Handle checkpoint loaded event."""
        if event.training_state and self.callback.trainer:
            checkpoint_path = event.get_data("checkpoint_path")
            if checkpoint_path:
                self.callback.on_checkpoint_load(
                    self.callback.trainer, event.training_state, checkpoint_path
                )

    def _on_log(self, event: TrainingEvent) -> None:
        """Handle metric logged event."""
        if event.training_state and self.callback.trainer:
            logs = event.get_data("logs")
            if logs:
                self.callback.on_log(
                    self.callback.trainer, event.training_state, logs
                )