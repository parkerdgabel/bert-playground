"""API contract validation utilities for CLI commands.

This module provides decorators and utilities to enforce API contracts
at runtime, ensuring interfaces remain stable.
"""

import functools
import inspect
import warnings
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ValidationError


class ContractViolation(Exception):
    """Raised when an API contract is violated."""

    pass


class DeprecatedParameter:
    """Marks a parameter as deprecated."""

    def __init__(self, message: str, removed_in: str, alternative: str | None = None):
        self.message = message
        self.removed_in = removed_in
        self.alternative = alternative


def validate_contract(
    expected_params: dict[str, type],
    return_type: type | None = None,
    deprecated_params: dict[str, DeprecatedParameter] | None = None,
) -> Callable:
    """Decorator to validate function contracts at runtime.

    Args:
        expected_params: Dictionary of parameter names to expected types
        return_type: Expected return type
        deprecated_params: Dictionary of deprecated parameters

    Example:
        @validate_contract(
            expected_params={
                'model_type': str,
                'batch_size': int,
                'learning_rate': float
            },
            return_type=dict
        )
        def create_model(model_type: str, **kwargs) -> dict:
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Check deprecated parameters
            if deprecated_params:
                for param_name, deprecation in deprecated_params.items():
                    if param_name in bound_args.arguments:
                        warnings.warn(
                            f"Parameter '{param_name}' is deprecated and will be "
                            f"removed in {deprecation.removed_in}. {deprecation.message}"
                            f"{f' Use {deprecation.alternative} instead.' if deprecation.alternative else ''}",
                            DeprecationWarning,
                            stacklevel=2,
                        )

            # Validate parameter types
            for param_name, expected_type in expected_params.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if value is not None and not isinstance(value, expected_type):
                        raise ContractViolation(
                            f"Parameter '{param_name}' expected type {expected_type.__name__}, "
                            f"got {type(value).__name__}"
                        )

            # Call function
            result = func(*args, **kwargs)

            # Validate return type
            if return_type is not None and result is not None:
                if not isinstance(result, return_type):
                    raise ContractViolation(
                        f"Return value expected type {return_type.__name__}, "
                        f"got {type(result).__name__}"
                    )

            return result

        return wrapper

    return decorator


def enforce_api_version(min_version: str, max_version: str | None = None) -> Callable:
    """Decorator to enforce API version requirements.

    Args:
        min_version: Minimum supported API version
        max_version: Maximum supported API version (optional)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # In a real implementation, this would check actual API versions
            # For now, we just pass through
            return func(*args, **kwargs)

        return wrapper

    return decorator


class ParameterContract(BaseModel):
    """Base class for parameter contracts using Pydantic."""

    class Config:
        extra = "forbid"  # Don't allow extra parameters


def validate_pydantic_contract(contract_class: type[ParameterContract]) -> Callable:
    """Decorator to validate parameters using Pydantic models.

    Example:
        class TrainParams(ParameterContract):
            model_type: str
            batch_size: int = 32
            learning_rate: float = 2e-5

        @validate_pydantic_contract(TrainParams)
        def train_command(model_type: str, batch_size: int = 32, **kwargs):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Extract parameters for validation
            params_to_validate = {}
            for field_name in contract_class.__fields__:
                if field_name in bound_args.arguments:
                    params_to_validate[field_name] = bound_args.arguments[field_name]

            # Validate using Pydantic
            try:
                contract_class(**params_to_validate)
            except ValidationError as e:
                raise ContractViolation(f"Parameter validation failed: {e}")

            # Call function
            return func(*args, **kwargs)

        return wrapper

    return decorator


def backward_compatible(old_param_mapping: dict[str, str]) -> Callable:
    """Decorator to maintain backward compatibility with old parameter names.

    Args:
        old_param_mapping: Dictionary mapping old parameter names to new ones

    Example:
        @backward_compatible({
            'page_size': 'page',  # old_name: new_name
            'num_results': 'limit'
        })
        def list_items(page: int, limit: int):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Transform old parameter names to new ones
            for old_name, new_name in old_param_mapping.items():
                if old_name in kwargs and new_name not in kwargs:
                    warnings.warn(
                        f"Parameter '{old_name}' is deprecated, use '{new_name}' instead",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    kwargs[new_name] = kwargs.pop(old_name)

            return func(*args, **kwargs)

        return wrapper

    return decorator


class ContractTest:
    """Base class for contract tests."""

    def __init__(self, function: Callable, contract: dict[str, Any]):
        self.function = function
        self.contract = contract

    def test_parameters(self) -> list[str]:
        """Test that function accepts expected parameters."""
        errors = []
        sig = inspect.signature(self.function)

        for param_name, param_info in self.contract.get("parameters", {}).items():
            if param_name not in sig.parameters:
                errors.append(f"Missing parameter: {param_name}")

        return errors

    def test_return_type(self) -> list[str]:
        """Test that function returns expected type."""
        errors = []
        expected_return = self.contract.get("return_type")

        if expected_return:
            # This would need actual execution to test properly
            # For now, we just check annotations
            sig = inspect.signature(self.function)
            if sig.return_annotation != expected_return:
                errors.append(
                    f"Return type mismatch: expected {expected_return}, "
                    f"got {sig.return_annotation}"
                )

        return errors


def generate_contract_from_function(func: Callable) -> dict[str, Any]:
    """Generate a contract specification from a function signature.

    This is useful for documenting existing APIs.
    """
    sig = inspect.signature(func)

    contract = {"name": func.__name__, "parameters": {}, "return_type": None}

    # Extract parameters
    for param_name, param in sig.parameters.items():
        param_info = {
            "type": param.annotation
            if param.annotation != inspect.Parameter.empty
            else Any,
            "default": param.default
            if param.default != inspect.Parameter.empty
            else None,
            "required": param.default == inspect.Parameter.empty,
        }
        contract["parameters"][param_name] = param_info

    # Extract return type
    if sig.return_annotation != inspect.Signature.empty:
        contract["return_type"] = sig.return_annotation

    return contract


# Example contracts for common CLI patterns


class MLflowHealthContract(ParameterContract):
    """Contract for MLflow health check results."""

    status: str  # Must be "PASS" or "FAIL"
    message: str
    suggestions: list[str] | None = None


class KaggleCompetitionContract(ParameterContract):
    """Contract for Kaggle competition data."""

    id: str
    title: str
    deadline: str | Any  # Can be string or pd.Timestamp
    numTeams: int
    reward: str | None = None
    isCompleted: bool = False


class ModelOutputContract(ParameterContract):
    """Contract for model output format."""

    loss: float | None = None
    predictions: list[float] | None = None
    logits: list[list[float]] | None = None


# Export utilities
__all__ = [
    "ContractViolation",
    "validate_contract",
    "enforce_api_version",
    "ParameterContract",
    "validate_pydantic_contract",
    "backward_compatible",
    "ContractTest",
    "generate_contract_from_function",
    "MLflowHealthContract",
    "KaggleCompetitionContract",
    "ModelOutputContract",
]
