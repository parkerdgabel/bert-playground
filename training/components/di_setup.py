"""Dependency injection setup for training components.

This module registers all training components with the DI container,
mapping protocols to their implementations.
"""

from typing import Optional
from pathlib import Path
from loguru import logger

from core.di.services import register_service, register_factory
from core.di.container import get_container
from core.protocols.training import CheckpointManager as CheckpointManagerProtocol, MetricsCollector

# Import component implementations
from .checkpoint_manager import CheckpointManager
from .evaluation_loop import EvaluationLoop
from .metrics_tracker import MetricsTracker
from .training_loop import TrainingLoop
from .training_orchestrator import TrainingOrchestrator


def register_training_components(
    checkpoint_dir: Optional[Path] = None,
    save_total_limit: int = 5,
    keep_best_only: bool = False,
    metrics_window_size: int = 100,
) -> None:
    """Register training components with the DI container.
    
    Args:
        checkpoint_dir: Optional default checkpoint directory
        save_total_limit: Default checkpoint retention limit
        keep_best_only: Default checkpoint retention policy
        metrics_window_size: Default metrics window size
    """
    container = get_container()
    
    # Register CheckpointManager
    def checkpoint_manager_factory() -> CheckpointManager:
        """Factory for CheckpointManager instances."""
        default_dir = checkpoint_dir or Path("output/checkpoints")
        return CheckpointManager(
            checkpoint_dir=default_dir,
            save_total_limit=save_total_limit,
            keep_best_only=keep_best_only,
        )
    
    register_factory(
        CheckpointManagerProtocol,
        checkpoint_manager_factory,
        singleton=False,  # Different training runs may need different configs
    )
    
    # Register MetricsTracker
    def metrics_tracker_factory() -> MetricsTracker:
        """Factory for MetricsTracker instances."""
        return MetricsTracker(
            window_size=metrics_window_size,
            output_dir=None,  # Will be set by specific trainer instances
        )
    
    register_factory(
        MetricsCollector,
        metrics_tracker_factory,
        singleton=False,
    )
    
    # Register component classes for direct instantiation
    register_service(TrainingLoop, TrainingLoop, singleton=False)
    register_service(EvaluationLoop, EvaluationLoop, singleton=False)
    register_service(TrainingOrchestrator, TrainingOrchestrator, singleton=False)
    
    logger.info("Registered training components with DI container")


def create_training_components_for_trainer(
    model,
    config,
    callbacks=None,
):
    """Create training components configured for a specific trainer.
    
    This is a convenience factory function that creates all components
    with proper configuration for a training session.
    
    Args:
        model: The model to train
        config: Training configuration
        callbacks: Optional list of callbacks
        
    Returns:
        Dict containing all configured components
    """
    # Create CheckpointManager with trainer-specific config
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=config.environment.output_dir / "checkpoints",
        save_total_limit=config.training.save_total_limit or 5,
        keep_best_only=config.training.save_best_only,
    )
    
    # Create MetricsTracker with trainer-specific config
    metrics_tracker = MetricsTracker(
        window_size=100,
        output_dir=config.environment.output_dir,
    )
    
    # Configure metrics tracking
    metrics_tracker.configure_metric(
        config.training.best_metric,
        config.training.best_metric_mode
    )
    
    # Create EvaluationLoop
    evaluation_loop = EvaluationLoop(model)
    
    return {
        "checkpoint_manager": checkpoint_manager,
        "metrics_tracker": metrics_tracker,
        "evaluation_loop": evaluation_loop,
        # TrainingLoop and TrainingOrchestrator will be created when optimizer is available
    }


# Auto-register components when module is imported
try:
    register_training_components()
except Exception as e:
    logger.warning(f"Failed to auto-register training components: {e}")
    logger.info("Training components can be registered manually using register_training_components()")