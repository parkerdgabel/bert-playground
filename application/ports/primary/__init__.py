"""Primary ports for hexagonal architecture.

Primary ports (driving ports) are interfaces used by external actors to interact
with the application core. These define the APIs that the application exposes.

Examples:
- Training API: Used by CLI/UI to train models
- Prediction API: Used by external systems to get predictions
- Model Management API: Used to save/load models
"""

from .training import (
    Trainer,
    TrainingStrategy,
    TrainingResult,
    TrainingState,
    TrainerConfig,
    train_model,
    evaluate_model,
    predict_with_model,
)
from .commands import (
    Command,
    CommandContext,
    CommandResult,
    Pipeline,
    execute_command,
    create_pipeline,
)
from .model_management import (
    ModelManager,
    save_model,
    load_model,
    list_models,
    delete_model,
)

__all__ = [
    # Training API
    "Trainer",
    "TrainingStrategy", 
    "TrainingResult",
    "TrainingState",
    "TrainerConfig",
    "train_model",
    "evaluate_model",
    "predict_with_model",
    # Command API
    "Command",
    "CommandContext",
    "CommandResult",
    "Pipeline",
    "execute_command",
    "create_pipeline",
    # Model Management API
    "ModelManager",
    "save_model",
    "load_model",
    "list_models",
    "delete_model",
]