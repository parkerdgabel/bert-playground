"""Training components for decomposed trainer architecture.

This module provides focused, single-responsibility components that work together
to implement the training functionality. Each component follows SOLID principles
and is independently testable.
"""

from .checkpoint_manager import CheckpointManager
from .evaluation_loop import EvaluationLoop
from .metrics_tracker import MetricsTracker
from .training_loop import TrainingLoop
from .training_orchestrator import TrainingOrchestrator

# Import DI setup to auto-register components
from . import di_setup

__all__ = [
    "TrainingLoop",
    "EvaluationLoop", 
    "CheckpointManager",
    "MetricsTracker",
    "TrainingOrchestrator",
    "di_setup",
]