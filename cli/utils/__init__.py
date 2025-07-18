"""CLI utilities and helpers."""

from .console import get_console, print_error, print_success, print_warning, print_info, create_progress, create_table, format_bytes, format_timestamp
from .config import load_config, validate_config, get_default_config_path
from .validators import validate_path, validate_batch_size, validate_model_type, validate_port, validate_learning_rate, validate_epochs
from .decorators import handle_errors, track_time, require_auth, requires_project
from .base_command import BaseCommand
from .contracts import (
    ContractViolation,
    validate_contract,
    enforce_api_version,
    ParameterContract,
    validate_pydantic_contract,
    backward_compatible,
    ContractTest,
    generate_contract_from_function
)

__all__ = [
    # Console utilities
    "get_console",
    "print_error",
    "print_success", 
    "print_warning",
    "print_info",
    "create_progress",
    "create_table",
    "format_bytes",
    "format_timestamp",
    
    # Config utilities
    "load_config",
    "validate_config",
    "get_default_config_path",
    
    # Validators
    "validate_path",
    "validate_batch_size",
    "validate_model_type",
    "validate_port",
    "validate_learning_rate", 
    "validate_epochs",
    
    # Decorators
    "handle_errors",
    "track_time",
    "require_auth",
    "requires_project",
    
    # Base classes
    "BaseCommand",
    
    # Contract utilities
    "ContractViolation",
    "validate_contract",
    "enforce_api_version",
    "ParameterContract",
    "validate_pydantic_contract",
    "backward_compatible",
    "ContractTest",
    "generate_contract_from_function",
]