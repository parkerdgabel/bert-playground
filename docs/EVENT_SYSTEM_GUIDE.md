# Event System Guide

## Overview

The Phase 2 event system provides a decoupled, asynchronous communication mechanism between components in k-bert. It enables real-time monitoring, plugin coordination, and flexible component interaction without tight coupling.

## Architecture

### Core Components

```python
from bert_playground.core.events import (
    EventBus,
    Event,
    EventHandler,
    AsyncEventHandler,
    EventSubscriber,
    EventFilter
)
```

### Event Bus
Central hub for event distribution:

```python
class EventBus:
    async def emit(self, event_name: str, data: Dict[str, Any]) -> None
    def emit_sync(self, event_name: str, data: Dict[str, Any]) -> None
    async def subscribe(self, pattern: str, handler: EventHandler) -> str
    def subscribe_sync(self, pattern: str, handler: Callable) -> str
    async def unsubscribe(self, subscription_id: str) -> bool
    def add_filter(self, event_filter: EventFilter) -> None
```

### Event Structure
```python
@dataclass
class Event:
    name: str
    data: Dict[str, Any]
    timestamp: datetime
    source: str
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

## Event Categories

### Training Events
```python
# Training lifecycle
"training.started" -> {"config": TrainingConfig, "run_id": str}
"training.epoch_started" -> {"epoch": int, "total_epochs": int}
"training.batch_completed" -> {"batch": int, "loss": float, "metrics": dict}
"training.epoch_completed" -> {"epoch": int, "metrics": dict, "duration": float}
"training.completed" -> {"final_metrics": dict, "total_duration": float}
"training.failed" -> {"error": str, "traceback": str}

# Model events
"model.forward_pass" -> {"input_shape": tuple, "output_shape": tuple}
"model.backward_pass" -> {"gradients": dict, "loss": float}
"model.checkpoint_saved" -> {"path": str, "epoch": int, "metrics": dict}
"model.checkpoint_loaded" -> {"path": str, "restored_epoch": int}
```

### Data Events
```python
# Data loading
"data.batch_loaded" -> {"batch_size": int, "load_time": float}
"data.preprocessing_started" -> {"dataset_size": int}
"data.preprocessing_completed" -> {"processed_samples": int, "duration": float}
"data.cache_hit" -> {"cache_key": str, "size_mb": float}
"data.cache_miss" -> {"cache_key": str, "reason": str}

# Data validation
"data.validation_started" -> {"validation_type": str}
"data.validation_completed" -> {"passed": bool, "issues": list}
"data.schema_mismatch" -> {"expected": dict, "actual": dict}
```

### Plugin Events
```python
# Plugin lifecycle
"plugin.loaded" -> {"plugin_name": str, "version": str}
"plugin.initialized" -> {"plugin_name": str, "config": dict}
"plugin.activated" -> {"plugin_name": str}
"plugin.deactivated" -> {"plugin_name": str}
"plugin.error" -> {"plugin_name": str, "error": str}

# Plugin communication
"plugin.message" -> {"from": str, "to": str, "message": dict}
"plugin.state_changed" -> {"plugin_name": str, "old_state": str, "new_state": str}
```

### System Events
```python
# Resource monitoring
"system.memory_usage" -> {"used_mb": float, "available_mb": float, "percentage": float}
"system.gpu_usage" -> {"utilization": float, "memory_used": float, "temperature": float}
"system.disk_usage" -> {"used_gb": float, "available_gb": float}

# Performance metrics
"system.performance_warning" -> {"component": str, "metric": str, "value": float, "threshold": float}
"system.bottleneck_detected" -> {"component": str, "description": str, "suggestions": list}
```

## Basic Usage

### Setting up Event Bus
```python
from bert_playground.core.events import EventBus, create_default_event_bus

# Create event bus
event_bus = create_default_event_bus()

# Or with custom configuration
event_bus = EventBus(
    max_subscribers=1000,
    buffer_size=10000,
    enable_persistence=True,
    persistence_path="events.log"
)
```

### Emitting Events
```python
# Async emission (recommended)
await event_bus.emit("training.epoch_completed", {
    "epoch": 5,
    "train_loss": 0.12,
    "val_loss": 0.15,
    "duration": 120.5
})

# Sync emission (for compatibility)
event_bus.emit_sync("model.checkpoint_saved", {
    "path": "/tmp/checkpoint_epoch_5.mlx",
    "epoch": 5,
    "size_mb": 150.2
})
```

### Event Subscription
```python
# Async handler
class TrainingMonitor:
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.metrics = []
    
    async def on_epoch_completed(self, event: Event):
        self.metrics.append({
            "epoch": event.data["epoch"],
            "loss": event.data["val_loss"],
            "timestamp": event.timestamp
        })
        
        # Emit derived event
        if event.data["val_loss"] < self.best_loss:
            await self.event_bus.emit("training.new_best_model", {
                "epoch": event.data["epoch"],
                "loss": event.data["val_loss"]
            })

# Subscribe to events
monitor = TrainingMonitor(event_bus)
await event_bus.subscribe("training.epoch_completed", monitor.on_epoch_completed)

# Pattern-based subscription
await event_bus.subscribe("training.*", monitor.on_training_event)
await event_bus.subscribe("model.checkpoint_*", monitor.on_checkpoint_event)
```

### Sync Event Handlers
```python
# For non-async components
def log_training_events(event: Event):
    logger.info(f"Training event: {event.name} - {event.data}")

# Subscribe sync handler
event_bus.subscribe_sync("training.*", log_training_events)
```

## Advanced Features

### Event Filtering
```python
from bert_playground.core.events import EventFilter

# Create custom filter
class ImportantEventsFilter(EventFilter):
    def should_process(self, event: Event) -> bool:
        important_events = [
            "training.completed",
            "training.failed", 
            "model.checkpoint_saved",
            "system.performance_warning"
        ]
        return event.name in important_events

# Add filter to event bus
event_bus.add_filter(ImportantEventsFilter())
```

### Event Correlation
```python
import uuid

# Create correlation ID for related events
correlation_id = str(uuid.uuid4())

# Emit correlated events
await event_bus.emit("training.started", 
                    {"config": config}, 
                    correlation_id=correlation_id)

await event_bus.emit("training.epoch_started", 
                    {"epoch": 1}, 
                    correlation_id=correlation_id)

# Query related events
related_events = await event_bus.get_correlated_events(correlation_id)
```

### Event Persistence
```python
# Enable event logging
event_bus = EventBus(
    enable_persistence=True,
    persistence_path="training_events.jsonl",
    max_file_size_mb=100
)

# Query historical events
from bert_playground.core.events import EventQuery

query = EventQuery()
query.event_name_pattern("training.*")
query.time_range(start=datetime.now() - timedelta(hours=1))
query.data_contains({"epoch": 5})

events = await event_bus.query_events(query)
```

### Event Metrics and Monitoring
```python
from bert_playground.core.events import EventMetrics

# Get event statistics
metrics = await event_bus.get_metrics()
print(f"Total events: {metrics.total_events}")
print(f"Events per second: {metrics.events_per_second}")
print(f"Average processing time: {metrics.avg_processing_time_ms}ms")

# Monitor event patterns
class EventPatternMonitor:
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.error_count = 0
        self.error_threshold = 10
    
    async def on_error_event(self, event: Event):
        self.error_count += 1
        if self.error_count >= self.error_threshold:
            await self.event_bus.emit("system.error_threshold_exceeded", {
                "error_count": self.error_count,
                "threshold": self.error_threshold,
                "component": event.data.get("component", "unknown")
            })

await event_bus.subscribe("*.error", monitor.on_error_event)
await event_bus.subscribe("*.failed", monitor.on_error_event)
```

## Integration Patterns

### Plugin Integration
```python
from bert_playground.core.ports import PluginPort
from bert_playground.core.events import EventBus

class EventAwarePlugin(PluginPort):
    def __init__(self, config: PluginConfig, event_bus: EventBus):
        super().__init__(config)
        self.event_bus = event_bus
        self.subscription_ids = []
    
    async def initialize(self):
        # Subscribe to relevant events
        sub_id = await self.event_bus.subscribe(
            "training.epoch_completed", 
            self.on_epoch_completed
        )
        self.subscription_ids.append(sub_id)
        
        # Announce plugin activation
        await self.event_bus.emit("plugin.activated", {
            "plugin_name": self.__class__.__name__,
            "version": self.version,
            "capabilities": self.capabilities
        })
    
    async def on_epoch_completed(self, event: Event):
        # Plugin-specific logic
        metrics = event.data.get("metrics", {})
        if self.should_trigger_action(metrics):
            await self.perform_action(metrics)
    
    async def cleanup(self):
        # Unsubscribe from events
        for sub_id in self.subscription_ids:
            await self.event_bus.unsubscribe(sub_id)
        
        # Announce deactivation
        await self.event_bus.emit("plugin.deactivated", {
            "plugin_name": self.__class__.__name__
        })
```

### Training Integration
```python
from bert_playground.training.core import AsyncTrainer

class EventDrivenTrainer(AsyncTrainer):
    def __init__(self, config: TrainingConfig, event_bus: EventBus):
        super().__init__(config)
        self.event_bus = event_bus
    
    async def train_async(self):
        await self.event_bus.emit("training.started", {
            "config": self.config.dict(),
            "run_id": self.run_id,
            "model_type": self.model.__class__.__name__
        })
        
        try:
            for epoch in range(self.config.num_epochs):
                await self.event_bus.emit("training.epoch_started", {
                    "epoch": epoch + 1,
                    "total_epochs": self.config.num_epochs
                })
                
                epoch_metrics = await self.train_epoch(epoch)
                
                await self.event_bus.emit("training.epoch_completed", {
                    "epoch": epoch + 1,
                    "metrics": epoch_metrics,
                    "duration": epoch_metrics["epoch_time"]
                })
                
                # Check for early stopping based on events
                if await self.should_stop_early():
                    await self.event_bus.emit("training.early_stopped", {
                        "epoch": epoch + 1,
                        "reason": "early_stopping_triggered"
                    })
                    break
            
            await self.event_bus.emit("training.completed", {
                "final_metrics": self.get_final_metrics(),
                "total_duration": self.get_training_duration()
            })
            
        except Exception as e:
            await self.event_bus.emit("training.failed", {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "epoch": getattr(self, 'current_epoch', 0)
            })
            raise
```

### Data Pipeline Integration
```python
from bert_playground.data.core import AsyncDataLoader

class EventDrivenDataLoader(AsyncDataLoader):
    def __init__(self, config: DataConfig, event_bus: EventBus):
        super().__init__(config)
        self.event_bus = event_bus
    
    async def load_batch(self) -> Dict[str, mx.array]:
        start_time = time.time()
        
        batch = await super().load_batch()
        load_time = time.time() - start_time
        
        await self.event_bus.emit("data.batch_loaded", {
            "batch_size": batch["input_ids"].shape[0],
            "load_time": load_time,
            "sequence_length": batch["input_ids"].shape[1],
            "memory_usage": self.get_memory_usage()
        })
        
        # Emit performance warnings if needed
        if load_time > self.config.load_time_threshold:
            await self.event_bus.emit("data.performance_warning", {
                "component": "data_loader",
                "metric": "load_time",
                "value": load_time,
                "threshold": self.config.load_time_threshold
            })
        
        return batch
```

## Real-time Monitoring Dashboard

### Event Stream Consumer
```python
import asyncio
from collections import deque

class RealTimeMonitor:
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.metrics_buffer = deque(maxlen=1000)
        self.active_subscriptions = []
    
    async def start_monitoring(self):
        # Subscribe to all training events
        sub_id = await self.event_bus.subscribe(
            "training.*", 
            self.on_training_event
        )
        self.active_subscriptions.append(sub_id)
        
        # Subscribe to system events
        sub_id = await self.event_bus.subscribe(
            "system.*", 
            self.on_system_event
        )
        self.active_subscriptions.append(sub_id)
        
        # Start metrics aggregation task
        asyncio.create_task(self.aggregate_metrics())
    
    async def on_training_event(self, event: Event):
        self.metrics_buffer.append({
            "timestamp": event.timestamp,
            "type": "training",
            "name": event.name,
            "data": event.data
        })
    
    async def on_system_event(self, event: Event):
        self.metrics_buffer.append({
            "timestamp": event.timestamp,
            "type": "system", 
            "name": event.name,
            "data": event.data
        })
    
    async def aggregate_metrics(self):
        while True:
            await asyncio.sleep(5)  # Aggregate every 5 seconds
            
            if not self.metrics_buffer:
                continue
            
            # Calculate aggregated metrics
            recent_metrics = list(self.metrics_buffer)[-50:]  # Last 50 events
            
            training_events = [m for m in recent_metrics if m["type"] == "training"]
            system_events = [m for m in recent_metrics if m["type"] == "system"]
            
            # Emit aggregated metrics
            await self.event_bus.emit("metrics.aggregated", {
                "training_events_count": len(training_events),
                "system_events_count": len(system_events),
                "events_per_second": len(recent_metrics) / 5.0,
                "timestamp": datetime.now()
            })

# Usage
monitor = RealTimeMonitor(event_bus)
await monitor.start_monitoring()
```

## Testing Event-Driven Components

### Event Bus Testing
```python
import pytest
from bert_playground.core.events import EventBus, Event

@pytest.fixture
async def event_bus():
    bus = EventBus()
    yield bus
    await bus.shutdown()

@pytest.mark.asyncio
async def test_event_emission_and_subscription(event_bus):
    received_events = []
    
    async def event_handler(event: Event):
        received_events.append(event)
    
    # Subscribe to events
    await event_bus.subscribe("test.*", event_handler)
    
    # Emit test event
    await event_bus.emit("test.example", {"key": "value"})
    
    # Wait for event processing
    await asyncio.sleep(0.1)
    
    # Verify event received
    assert len(received_events) == 1
    assert received_events[0].name == "test.example"
    assert received_events[0].data == {"key": "value"}

@pytest.mark.asyncio
async def test_event_filtering(event_bus):
    from bert_playground.core.events import EventFilter
    
    class TestEventFilter(EventFilter):
        def should_process(self, event: Event) -> bool:
            return event.data.get("important", False)
    
    event_bus.add_filter(TestEventFilter())
    
    received_events = []
    async def handler(event: Event):
        received_events.append(event)
    
    await event_bus.subscribe("test.*", handler)
    
    # Emit filtered out event
    await event_bus.emit("test.filtered", {"important": False})
    
    # Emit important event
    await event_bus.emit("test.important", {"important": True})
    
    await asyncio.sleep(0.1)
    
    # Only important event should be received
    assert len(received_events) == 1
    assert received_events[0].name == "test.important"
```

### Mock Event Bus
```python
from unittest.mock import AsyncMock
from bert_playground.core.events import EventBus

class MockEventBus(EventBus):
    def __init__(self):
        super().__init__()
        self.emitted_events = []
        self.emit = AsyncMock(side_effect=self._mock_emit)
    
    async def _mock_emit(self, event_name: str, data: dict, **kwargs):
        self.emitted_events.append({
            "name": event_name,
            "data": data,
            **kwargs
        })

# Usage in tests
@pytest.mark.asyncio
async def test_component_with_events():
    mock_bus = MockEventBus()
    component = MyEventDrivenComponent(mock_bus)
    
    await component.do_something()
    
    # Verify events were emitted
    assert len(mock_bus.emitted_events) == 2
    assert mock_bus.emitted_events[0]["name"] == "component.started"
    assert mock_bus.emitted_events[1]["name"] == "component.completed"
```

## Best Practices

### Event Design
1. **Consistent Naming**: Use hierarchical dot notation (e.g., `component.action.result`)
2. **Rich Data**: Include relevant context in event data
3. **Immutable Events**: Don't modify event data after emission
4. **Error Events**: Always emit error events for failures

### Performance
1. **Async First**: Use async event handlers when possible
2. **Batch Processing**: Group related events for efficiency
3. **Event Filtering**: Filter events early to reduce processing
4. **Buffer Management**: Configure appropriate buffer sizes

### Testing
1. **Mock Event Bus**: Use mocks for unit testing
2. **Event Verification**: Test that correct events are emitted
3. **Integration Testing**: Test event flows between components
4. **Performance Testing**: Measure event processing overhead

### Monitoring
1. **Event Metrics**: Track event rates and processing times
2. **Error Tracking**: Monitor event processing failures
3. **Pattern Detection**: Identify unusual event patterns
4. **Resource Usage**: Monitor event system resource consumption

## Conclusion

The event system provides a powerful foundation for building reactive, observable ML training systems. It enables loose coupling between components while providing rich monitoring and coordination capabilities.

Key benefits:
- **Decoupling**: Components interact without direct dependencies
- **Observability**: Complete visibility into system behavior
- **Extensibility**: Easy to add new functionality through event handlers
- **Testability**: Event-driven components are easier to test
- **Scalability**: Async event processing supports high throughput