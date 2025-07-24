"""Ports for hexagonal architecture.

This package contains all port definitions for the k-bert project.
Ports define the boundaries between the application core and external systems.

- Primary ports: APIs exposed to external actors (driving ports)
- Secondary ports: Interfaces the core depends on (driven ports)
"""

# Re-export main port types for convenience
from .primary import (
    # Training API
    Trainer,
    TrainingStrategy,
    TrainingResult,
    TrainingState,
    TrainerConfig,
    # Command API
    Command,
    CommandContext,
    CommandResult,
    Pipeline,
    # Model Management API
    ModelManager,
    # ModelInfo,  # Not yet implemented
)

from .secondary import (
    # Compute
    ComputeBackend,
    # Storage
    StorageService,
    ModelStorageService,
    # Monitoring
    MonitoringService,
    ExperimentTracker,
    # Configuration
    ConfigurationProvider,
    ConfigRegistry,
    # Neural
    NeuralBackend,
    # Tokenizer
    TokenizerPort,
    # Optimization
    Optimizer,
    LRScheduler,
    # Metrics
    Metric,
    MetricsCollector,
    # Checkpointing
    CheckpointManager,
)

__all__ = [
    # Primary ports
    "Trainer",
    "TrainingStrategy",
    "TrainingResult",
    "TrainingState",
    "TrainerConfig",
    "Command",
    "CommandContext",
    "CommandResult",
    "Pipeline",
    "ModelManager",
    # "ModelInfo",  # Not yet implemented
    # Secondary ports
    "ComputeBackend",
    "StorageService",
    "ModelStorageService",
    "MonitoringService",
    "ExperimentTracker",
    "ConfigurationProvider",
    "ConfigRegistry",
    "NeuralBackend",
    "TokenizerPort",
    "Optimizer",
    "LRScheduler",
    "Metric",
    "MetricsCollector",
    "CheckpointManager",
]