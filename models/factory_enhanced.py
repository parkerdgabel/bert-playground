"""Enhanced model factory with advanced Loguru features."""

from pathlib import Path
from typing import Any, Dict, Optional, Union

from loguru import logger

from .bert import ModernBertConfig, ModernBertCore
from .bert.core import create_bert_core, create_modernbert_core
from .factory import create_model as original_create_model
from .heads import HeadConfig, create_head
from .heads.base import get_default_config_for_head_type
from .model import BertWithHead
from utils.loguru_advanced import (
    catch_and_log,
    log_timing,
    bind_context,
    lazy_debug
)


@catch_and_log(
    ValueError,
    "Model creation failed",
    reraise=True
)
def create_model_with_logging(
    model_name: str = "bert",
    model_type: str = "base",
    config: Optional[Union[ModernBertConfig, Dict[str, Any]]] = None,
    head_type: Optional[str] = None,
    head_config: Optional[Union[HeadConfig, Dict[str, Any]]] = None,
    **kwargs,
) -> Any:
    """
    Create a model instance with enhanced logging.
    
    This is a wrapper around the original create_model function that adds:
    - Performance timing
    - Structured logging with context
    - Lazy debug evaluation
    - Better error handling
    """
    # Bind context for all logs in this function
    log = bind_context(
        model_name=model_name,
        model_type=model_type,
        head_type=head_type
    )
    
    with log_timing("model_creation", model_type=model_type):
        # Log configuration details
        log.info(f"Creating model: {model_type}")
        
        # Use lazy debug for expensive config serialization
        lazy_debug(
            "Model configuration",
            lambda: {
                "config": config.__dict__ if hasattr(config, "__dict__") else config,
                "head_config": head_config.__dict__ if hasattr(head_config, "__dict__") else head_config,
                "kwargs": kwargs
            }
        )
        
        # Create model using original factory
        model = original_create_model(
            model_name=model_name,
            model_type=model_type,
            config=config,
            head_type=head_type,
            head_config=head_config,
            **kwargs
        )
        
        # Log model statistics
        if hasattr(model, "num_parameters"):
            param_count = model.num_parameters()
            log.info(f"Model created with {param_count:,} parameters")
            
            # Lazy debug for parameter breakdown
            lazy_debug(
                "Parameter breakdown",
                lambda: _get_parameter_breakdown(model)
            )
        
        return model


@catch_and_log(
    Exception,
    "Failed to load model from checkpoint",
    reraise=True
)
def create_model_from_checkpoint_with_logging(checkpoint_path: Union[str, Path]) -> Any:
    """
    Load a model from checkpoint with enhanced logging.
    
    Args:
        checkpoint_path: Path to model checkpoint
        
    Returns:
        Loaded model instance
    """
    checkpoint_path = Path(checkpoint_path)
    log = bind_context(checkpoint=str(checkpoint_path))
    
    with log_timing("checkpoint_loading", checkpoint=str(checkpoint_path)):
        log.info(f"Loading model from checkpoint: {checkpoint_path}")
        
        # Import original function
        from .factory import create_model_from_checkpoint
        
        # Load model
        model = create_model_from_checkpoint(checkpoint_path)
        
        # Log loaded model info
        if hasattr(model, "config"):
            lazy_debug(
                "Loaded model config",
                lambda: model.config.__dict__ if hasattr(model.config, "__dict__") else str(model.config)
            )
        
        log.info("Model loaded successfully from checkpoint")
        return model


def _get_parameter_breakdown(model) -> Dict[str, int]:
    """Get parameter count breakdown by component."""
    breakdown = {}
    
    if hasattr(model, "bert") and hasattr(model, "head"):
        # BertWithHead model
        if hasattr(model.bert, "num_parameters"):
            breakdown["bert"] = model.bert.num_parameters()
        if hasattr(model.head, "num_parameters"):
            breakdown["head"] = model.head.num_parameters()
    
    # Add more component breakdowns as needed
    breakdown["total"] = model.num_parameters() if hasattr(model, "num_parameters") else 0
    
    return breakdown


# Example of using FrequencyLogger for batch processing
from utils.loguru_advanced import FrequencyLogger

_batch_logger = FrequencyLogger(frequency=100)


def log_batch_processing(batch_idx: int, batch_size: int, loss: float):
    """Log batch processing info only every N batches."""
    _batch_logger.log(
        "batch_processing",
        f"Batch {batch_idx}: size={batch_size}, loss={loss:.4f}",
        batch_idx=batch_idx
    )


# Example of metrics logger usage
def create_metrics_logger(output_dir: Path) -> 'MetricsLogger':
    """Create a metrics logger for training."""
    from utils.loguru_advanced import MetricsLogger
    
    metrics_path = output_dir / "metrics.jsonl"
    return MetricsLogger(sink_path=metrics_path)


# Export enhanced functions
__all__ = [
    "create_model_with_logging",
    "create_model_from_checkpoint_with_logging",
    "log_batch_processing",
    "create_metrics_logger",
]