"""MLflow configuration."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MLflowConfig:
    """Configuration for MLflow monitoring."""
    
    tracking_uri: Optional[str] = None
    experiment_name: str = "k-bert-experiments"
    registry_uri: Optional[str] = None
    artifact_location: Optional[str] = None
    
    # Run configuration
    nested_runs: bool = True
    auto_log_metrics: bool = True
    log_models: bool = True
    log_input_examples: bool = False
    
    # Logging intervals
    log_every_n_steps: int = 10
    log_every_n_epochs: int = 1
    
    # Model registry
    register_models: bool = False
    model_name: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration."""
        if self.register_models and not self.model_name:
            raise ValueError("model_name must be provided when register_models is True")