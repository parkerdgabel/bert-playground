"""Demonstration of advanced Loguru features in the training pipeline."""

import time
from pathlib import Path
import mlx.core as mx
from loguru import logger

# Import advanced logging utilities
from utils.loguru_advanced import (
    bind_training_context,
    log_timing,
    lazy_debug,
    MetricsLogger,
    ProgressTracker,
    catch_and_log,
    FrequencyLogger,
    log_mlx_info
)


def demo_structured_logging():
    """Demonstrate structured logging with context."""
    print("\n=== Structured Logging Demo ===")
    
    # Create a logger with training context
    for epoch in range(2):
        epoch_logger = bind_training_context(epoch=epoch, fold=0, model="modernbert")
        
        epoch_logger.info("Starting epoch")
        
        for step in range(5):
            step_logger = bind_training_context(
                epoch=epoch,
                step=step,
                phase="train",
                batch_size=32
            )
            
            # Simulate training step
            fake_loss = 1.0 - (step * 0.1)
            step_logger.info(f"Training step completed", loss=fake_loss)


def demo_performance_timing():
    """Demonstrate performance timing utilities."""
    print("\n=== Performance Timing Demo ===")
    
    # Time a simple operation
    with log_timing("data_preprocessing", dataset="titanic"):
        time.sleep(0.1)  # Simulate work
    
    # Time with memory tracking
    with log_timing("model_forward_pass", include_memory=True, batch_size=64):
        # Simulate model forward pass
        data = mx.random.normal((64, 256))
        time.sleep(0.05)
    
    # Nested timing
    with log_timing("full_training_step") as timer:
        with log_timing("forward_pass"):
            time.sleep(0.02)
        
        with log_timing("backward_pass"):
            time.sleep(0.03)
        
        timer.debug("Step breakdown complete")


def demo_lazy_evaluation():
    """Demonstrate lazy debug evaluation."""
    print("\n=== Lazy Evaluation Demo ===")
    
    # This expensive computation only runs if DEBUG is enabled
    def expensive_stats():
        print("Computing expensive statistics...")
        time.sleep(0.1)
        return {"mean": 0.5, "std": 0.1, "max": 1.0}
    
    # Won't execute expensive_stats unless in DEBUG mode
    lazy_debug("Model statistics", expensive_stats)
    
    # With context
    lazy_debug("Gradient norms", lambda: {"l2": 0.05, "max": 0.1}, step=100)


def demo_metrics_logging():
    """Demonstrate structured metrics logging."""
    print("\n=== Metrics Logging Demo ===")
    
    # Create metrics logger
    metrics_logger = MetricsLogger()
    
    # Log training metrics
    for epoch in range(2):
        for step in range(3):
            metrics = {
                "loss": 1.0 - (epoch * 0.3 + step * 0.1),
                "accuracy": 0.7 + (epoch * 0.1 + step * 0.05),
                "learning_rate": 0.001 * (0.9 ** epoch)
            }
            
            metrics_logger.log_metrics(
                metrics,
                step=step,
                epoch=epoch,
                phase="train"
            )


def demo_error_handling():
    """Demonstrate error handling with logging."""
    print("\n=== Error Handling Demo ===")
    
    @catch_and_log(
        ValueError,
        "Configuration validation failed",
        reraise=False,
        default=None,
        config_path="config.yaml"
    )
    def load_config(path):
        if not path.exists():
            raise ValueError(f"Config file not found: {path}")
        return {"loaded": True}
    
    # This will log the error and return None
    result = load_config(Path("nonexistent.yaml"))
    print(f"Result: {result}")
    
    # Example with retries
    @catch_and_log(Exception, "Operation failed", reraise=False)
    def flaky_operation():
        import random
        if random.random() < 0.5:
            raise RuntimeError("Random failure")
        return "Success"
    
    result = flaky_operation()
    print(f"Flaky operation result: {result}")


def demo_progress_tracking():
    """Demonstrate enhanced progress tracking."""
    print("\n=== Progress Tracking Demo ===")
    
    # Simulate training with progress tracking
    total_batches = 20
    
    with ProgressTracker(
        total_batches,
        desc="Training epoch",
        log_frequency=25,  # Log every 25%
        epoch=1
    ) as tracker:
        
        for batch_idx in range(total_batches):
            # Simulate batch processing
            time.sleep(0.05)
            
            # Update with metrics
            tracker.update(
                1,
                loss=1.0 - (batch_idx / total_batches),
                accuracy=0.8 + (batch_idx / total_batches) * 0.1
            )


def demo_frequency_logging():
    """Demonstrate frequency-based logging."""
    print("\n=== Frequency Logging Demo ===")
    
    freq_logger = FrequencyLogger(frequency=5)
    
    # This will only log every 5th occurrence
    for i in range(20):
        freq_logger.log(
            "batch_process",
            f"Processing batch",
            batch_idx=i,
            loss=1.0 - i * 0.01
        )


def demo_mlx_array_logging():
    """Demonstrate MLX array debugging."""
    print("\n=== MLX Array Logging Demo ===")
    
    # Create some test arrays
    weights = mx.random.normal((768, 256))
    log_mlx_info(weights, "model_weights")
    
    # Log problematic array
    bad_array = mx.zeros((10, 0, 5))  # Has zero dimension
    log_mlx_info(bad_array, "problematic_array")


def demo_combined_features():
    """Demonstrate combining multiple features."""
    print("\n=== Combined Features Demo ===")
    
    # Setup
    metrics_logger = MetricsLogger()
    freq_logger = FrequencyLogger(frequency=10)
    
    # Simulate a training epoch with all features
    epoch = 0
    epoch_logger = bind_training_context(epoch=epoch, model="bert-base")
    
    with log_timing("epoch_training", epoch=epoch):
        with ProgressTracker(50, "Training", log_frequency=20) as tracker:
            for step in range(50):
                # Bind step context
                step_logger = bind_training_context(
                    epoch=epoch,
                    step=step,
                    phase="train"
                )
                
                # Simulate computation
                with log_timing("batch_processing", level="DEBUG"):
                    time.sleep(0.01)
                    loss = 1.0 - step * 0.02
                    acc = 0.6 + step * 0.008
                
                # Update progress
                tracker.update(1, loss=loss, accuracy=acc)
                
                # Log metrics
                if step % 10 == 0:
                    metrics_logger.log_metrics(
                        {"loss": loss, "accuracy": acc},
                        step=step,
                        epoch=epoch
                    )
                
                # Frequency logging
                freq_logger.log(
                    "gradient_norm",
                    "Gradient norm check",
                    norm=0.1 * (1 + step * 0.01)
                )
                
                # Lazy debug
                lazy_debug(
                    "Batch statistics",
                    lambda: {"min": 0.0, "max": 1.0, "mean": 0.5}
                )


if __name__ == "__main__":
    # Configure logger for demo
    logger.remove()
    logger.add(
        lambda msg: print(msg),
        format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{extra[epoch]}</cyan>:<cyan>{extra[step]}</cyan> - <level>{message}</level>",
        level="DEBUG"
    )
    
    # Run demos
    demo_structured_logging()
    demo_performance_timing()
    demo_lazy_evaluation()
    demo_metrics_logging()
    demo_error_handling()
    demo_progress_tracking()
    demo_frequency_logging()
    demo_mlx_array_logging()
    demo_combined_features()
    
    print("\n✨ Advanced logging demo complete!")