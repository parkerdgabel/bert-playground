"""CLI utilities and helpers."""

from .base_command import BaseCommand
from .config import get_default_config_path, load_config, validate_config
from .console import (
    create_progress,
    create_table,
    format_bytes,
    format_timestamp,
    get_console,
    print_error,
    print_info,
    print_success,
    print_warning,
)
from .contracts import (
    ContractTest,
    ContractViolation,
    ParameterContract,
    backward_compatible,
    enforce_api_version,
    generate_contract_from_function,
    validate_contract,
    validate_pydantic_contract,
)
from .decorators import handle_errors, require_auth, requires_project, track_time
from .validators import (
    validate_batch_size,
    validate_epochs,
    validate_learning_rate,
    validate_model_type,
    validate_path,
    validate_port,
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
