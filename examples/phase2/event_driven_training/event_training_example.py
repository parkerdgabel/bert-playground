"""
Event-Driven Training Example

This example demonstrates how to use the event system to create
a flexible, extensible training pipeline with real-time monitoring
and custom event handling.
"""

import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List

import mlx.core as mx
from training.core.base import BaseTrainer
from training.core.config import BaseTrainerConfig
from training.core.events import EventBus, EventEmitter


class TrainingMetricsCollector:
    """Collects and aggregates training metrics via events."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.batch_metrics = []
        self.epoch_metrics = []
        self.alerts = []
        
        # Register event listeners
        event_bus.on('batch.completed', self.collect_batch_metrics)
        event_bus.on('epoch.completed', self.collect_epoch_metrics)
        event_bus.on('training.completed', self.finalize_metrics)
        event_bus.on('alert.high_loss', self.handle_high_loss_alert)
    
    def collect_batch_metrics(self, data: Dict[str, Any]):
        """Collect batch-level metrics."""
        self.batch_metrics.append({
            'step': data['global_step'],
            'loss': data['loss'],
            'learning_rate': data.get('learning_rate', 0),
            'timestamp': time.time()
        })
        
        # Check for high loss and emit alert
        if data['loss'] > 1.5:  # Configurable threshold
            self.event_bus.emit('alert.high_loss', {
                'step': data['global_step'],
                'loss': data['loss'],
                'severity': 'warning' if data['loss'] < 2.0 else 'critical'
            })
    
    def collect_epoch_metrics(self, data: Dict[str, Any]):
        """Collect epoch-level metrics."""
        self.epoch_metrics.append({
            'epoch': data['epoch'],
            'train_loss': data.get('train_loss', 0),
            'val_loss': data.get('val_loss', 0),
            'duration': data.get('duration', 0),
            'timestamp': time.time()
        })
    
    def handle_high_loss_alert(self, data: Dict[str, Any]):
        """Handle high loss alerts."""
        self.alerts.append({
            'type': 'high_loss',
            'step': data['step'],
            'loss': data['loss'],
            'severity': data['severity'],
            'timestamp': time.time()
        })
        
        print(f"🚨 HIGH LOSS ALERT: Step {data['step']}, Loss: {data['loss']:.4f}")
    
    def finalize_metrics(self, data: Dict[str, Any]):
        """Finalize metrics collection."""
        print("\n📊 Training Metrics Summary:")
        print(f"Total Steps: {data['total_steps']}")
        print(f"Total Epochs: {data['total_epochs']}")
        print(f"Duration: {data.get('duration', 0):.2f}s")
        print(f"Batch Metrics Collected: {len(self.batch_metrics)}")
        print(f"Epoch Metrics Collected: {len(self.epoch_metrics)}")
        print(f"Alerts Generated: {len(self.alerts)}")
        
        if self.alerts:
            print(f"Alert Summary:")
            for alert in self.alerts:
                print(f"  - {alert['severity'].upper()}: {alert['type']} at step {alert['step']}")


class AdaptiveLearningRateScheduler:
    """Adaptive learning rate scheduler based on loss events."""
    
    def __init__(self, event_bus: EventBus, initial_lr: float = 1e-4):
        self.event_bus = event_bus
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.loss_history = []
        self.patience_counter = 0
        self.patience_threshold = 5
        
        # Register for loss events
        event_bus.on('epoch.completed', self.adjust_learning_rate)
    
    def adjust_learning_rate(self, data: Dict[str, Any]):
        """Adjust learning rate based on loss trends."""
        current_loss = data.get('val_loss', data.get('train_loss', float('inf')))
        self.loss_history.append(current_loss)
        
        # Keep only recent history
        if len(self.loss_history) > 10:
            self.loss_history.pop(0)
        
        # Check if loss has plateaued
        if len(self.loss_history) >= 3:
            recent_losses = self.loss_history[-3:]
            if all(abs(recent_losses[i] - recent_losses[i-1]) < 0.001 
                   for i in range(1, len(recent_losses))):
                self.patience_counter += 1
            else:
                self.patience_counter = 0
        
        # Reduce learning rate if patience exceeded
        if self.patience_counter >= self.patience_threshold:
            old_lr = self.current_lr
            self.current_lr *= 0.5
            self.patience_counter = 0
            
            print(f"📉 Learning rate reduced: {old_lr:.6f} → {self.current_lr:.6f}")
            
            # Emit learning rate change event
            self.event_bus.emit('scheduler.lr_reduced', {
                'epoch': data['epoch'],
                'old_lr': old_lr,
                'new_lr': self.current_lr,
                'reason': 'loss_plateau'
            })


class ModelCheckpointer:
    """Event-driven model checkpointing."""
    
    def __init__(self, event_bus: EventBus, checkpoint_dir: Path):
        self.event_bus = event_bus
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_loss = float('inf')
        self.checkpoints_saved = 0
        
        # Register for evaluation events
        event_bus.on('epoch.completed', self.maybe_save_checkpoint)
        event_bus.on('training.completed', self.save_final_checkpoint)
    
    def maybe_save_checkpoint(self, data: Dict[str, Any]):
        """Save checkpoint if conditions are met."""
        current_loss = data.get('val_loss', data.get('train_loss', float('inf')))
        epoch = data.get('epoch', 0)
        
        # Save if this is the best model so far
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            checkpoint_path = self.checkpoint_dir / f"best_model_epoch_{epoch}.safetensors"
            
            print(f"💾 Saving best model checkpoint: {checkpoint_path.name}")
            self.checkpoints_saved += 1
            
            # Emit checkpoint saved event
            self.event_bus.emit('checkpoint.saved', {
                'epoch': epoch,
                'loss': current_loss,
                'path': str(checkpoint_path),
                'type': 'best'
            })
    
    def save_final_checkpoint(self, data: Dict[str, Any]):
        """Save final model checkpoint."""
        checkpoint_path = self.checkpoint_dir / "final_model.safetensors"
        print(f"💾 Saving final checkpoint: {checkpoint_path.name}")
        self.checkpoints_saved += 1
        
        # Emit checkpoint saved event
        self.event_bus.emit('checkpoint.saved', {
            'epoch': data.get('total_epochs', 0),
            'path': str(checkpoint_path),
            'type': 'final'
        })


class RealTimeMonitor:
    """Real-time training monitor with live updates."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.start_time = None
        self.batch_count = 0
        self.epoch_count = 0
        
        # Register for all relevant events
        event_bus.on('training.started', self.on_training_started)
        event_bus.on('batch.completed', self.on_batch_completed)
        event_bus.on('epoch.completed', self.on_epoch_completed)
        event_bus.on('training.completed', self.on_training_completed)
        event_bus.on('alert.high_loss', self.on_alert)
        event_bus.on('checkpoint.saved', self.on_checkpoint_saved)
    
    def on_training_started(self, data: Dict[str, Any]):
        """Handle training start."""
        self.start_time = time.time()
        print("🚀 Training Started!")
        print(f"Target Epochs: {data.get('total_epochs', 'Unknown')}")
        print("-" * 50)
    
    def on_batch_completed(self, data: Dict[str, Any]):
        """Handle batch completion with live updates."""
        self.batch_count += 1
        
        # Show progress every 10 batches
        if self.batch_count % 10 == 0:
            elapsed = time.time() - self.start_time if self.start_time else 0
            print(f"⚡ Step {data['global_step']:04d} | "
                  f"Loss: {data['loss']:.4f} | "
                  f"LR: {data.get('learning_rate', 0):.6f} | "
                  f"Time: {elapsed:.1f}s")
    
    def on_epoch_completed(self, data: Dict[str, Any]):
        """Handle epoch completion."""
        self.epoch_count += 1
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        print(f"\n📊 Epoch {data['epoch']+1} Complete:")
        print(f"  Train Loss: {data.get('train_loss', 'N/A')}")
        print(f"  Val Loss: {data.get('val_loss', 'N/A')}")
        print(f"  Duration: {elapsed:.1f}s")
        print("-" * 50)
    
    def on_alert(self, data: Dict[str, Any]):
        """Handle alerts with visual indicators."""
        severity_icon = "⚠️" if data['severity'] == 'warning' else "🚨"
        print(f"\n{severity_icon} ALERT: {data['type'].upper()}")
        print(f"  Step: {data['step']}")
        print(f"  Loss: {data['loss']:.4f}")
        print(f"  Severity: {data['severity'].upper()}")
        print("-" * 50)
    
    def on_checkpoint_saved(self, data: Dict[str, Any]):
        """Handle checkpoint saves."""
        print(f"💾 Checkpoint saved: {Path(data['path']).name} "
              f"(type: {data['type']}) at epoch {data['epoch']}")
    
    def on_training_completed(self, data: Dict[str, Any]):
        """Handle training completion."""
        total_time = time.time() - self.start_time if self.start_time else 0
        
        print(f"\n🎉 Training Completed!")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Total Steps: {data.get('total_steps', self.batch_count)}")
        print(f"Total Epochs: {data.get('total_epochs', self.epoch_count)}")
        
        if data.get('early_stopped', False):
            print("🛑 Training stopped early")
        
        print("=" * 50)


class EventDrivenCallback:
    """Callback that bridges BaseTrainer to the event system."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
    
    def on_train_begin(self, trainer, state):
        """Emit training started event."""
        self.event_bus.emit('training.started', {
            'total_epochs': trainer.config.training.num_epochs,
            'model_type': trainer.model.__class__.__name__,
            'batch_size': trainer.config.data.batch_size
        })
    
    def on_batch_end(self, trainer, state, loss):
        """Emit batch completed event."""
        loss_value = float(loss.item()) if hasattr(loss, 'item') else float(loss)
        self.event_bus.emit('batch.completed', {
            'global_step': state.global_step,
            'loss': loss_value,
            'learning_rate': trainer.optimizer.learning_rate if trainer.optimizer else 0,
            'grad_norm': getattr(state, 'grad_norm', 0)
        })
    
    def on_epoch_end(self, trainer, state):
        """Emit epoch completed event."""
        self.event_bus.emit('epoch.completed', {
            'epoch': state.epoch,
            'train_loss': state.train_loss,
            'val_loss': state.val_loss,
            'duration': getattr(state, 'epoch_duration', 0)
        })
    
    def on_train_end(self, trainer, state, result):
        """Emit training completed event."""
        self.event_bus.emit('training.completed', {
            'total_epochs': state.epoch + 1,
            'total_steps': state.global_step,
            'duration': result.total_time if hasattr(result, 'total_time') else 0,
            'early_stopped': getattr(result, 'early_stopped', False),
            'final_metrics': result.final_metrics if hasattr(result, 'final_metrics') else {}
        })


def create_mock_model():
    """Create a mock model for demonstration."""
    from unittest.mock import MagicMock
    import mlx.core as mx
    
    model = MagicMock()
    model.parameters.return_value = {
        "weight": mx.random.normal((10, 2)),
        "bias": mx.zeros((2,))
    }
    
    def forward(**kwargs):
        batch_size = kwargs.get("input_ids", mx.zeros((1, 1))).shape[0]
        # Simulate decreasing loss over time
        base_loss = 2.0 * (1.0 - min(kwargs.get("_step", 0) / 100.0, 0.8))
        noise = float(mx.random.normal(scale=0.1))
        loss = mx.array(max(0.1, base_loss + noise))
        
        return {
            "loss": loss,
            "logits": mx.random.normal((batch_size, 2))
        }
    
    model.__call__ = MagicMock(side_effect=forward)
    model.eval = MagicMock()
    model.train = MagicMock()
    
    return model


def create_mock_dataloader(num_samples: int = 200, batch_size: int = 8):
    """Create a mock dataloader."""
    from unittest.mock import MagicMock
    import mlx.core as mx
    
    num_batches = num_samples // batch_size
    batches = []
    
    for i in range(num_batches):
        batch = {
            "input_ids": mx.random.randint(0, 1000, (batch_size, 32)),
            "labels": mx.random.randint(0, 2, (batch_size,)),
            "_step": i  # Add step info for decreasing loss simulation
        }
        batches.append(batch)
    
    loader = MagicMock()
    loader.__iter__.return_value = iter(batches)
    loader.__len__.return_value = len(batches)
    
    return loader


def main():
    """Main function demonstrating event-driven training."""
    print("🎯 Event-Driven Training Example")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create event bus
        event_bus = EventBus()
        
        # Set up event-driven components
        metrics_collector = TrainingMetricsCollector(event_bus)
        lr_scheduler = AdaptiveLearningRateScheduler(event_bus, initial_lr=1e-3)
        checkpointer = ModelCheckpointer(event_bus, temp_path / "checkpoints")
        monitor = RealTimeMonitor(event_bus)
        
        # Create trainer configuration
        config = BaseTrainerConfig()
        config.environment.output_dir = temp_path / "output"
        config.training.num_epochs = 3
        config.training.logging_steps = 5
        config.training.save_strategy = "no"  # We handle this via events
        config.optimizer.learning_rate = 1e-3
        
        # Create model and data
        model = create_mock_model()
        train_loader = create_mock_dataloader(200, 8)
        val_loader = create_mock_dataloader(50, 8)
        
        # Create event-driven callback
        event_callback = EventDrivenCallback(event_bus)
        
        # Create trainer with callback
        trainer = BaseTrainer(model, config, callbacks=[event_callback])
        
        # Run training
        print("Starting event-driven training...")
        result = trainer.train(train_loader, val_loader)
        
        # Final report
        print(f"\n🎯 Event-Driven Training Complete!")
        print(f"Final Result: {result}")
        print(f"Checkpoints saved: {checkpointer.checkpoints_saved}")
        print(f"Metrics collected: {len(metrics_collector.batch_metrics)} batch metrics")
        print(f"Alerts generated: {len(metrics_collector.alerts)} alerts")


if __name__ == "__main__":
    main()