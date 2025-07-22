"""Validation middleware for CLI commands."""

import asyncio
import inspect
from typing import Any, Callable, Dict, List, Optional, Type, get_type_hints

from loguru import logger
from pydantic import BaseModel, ValidationError, create_model

from cli.middleware.base import CommandContext, Middleware, MiddlewareResult


class ValidationMiddleware(Middleware):
    """Middleware for command argument validation."""
    
    def __init__(
        self,
        name: str = "ValidationMiddleware",
        validators: Optional[Dict[str, Callable]] = None,
        schemas: Optional[Dict[str, Type[BaseModel]]] = None,
        strict: bool = True
    ):
        """Initialize validation middleware.
        
        Args:
            name: Middleware name
            validators: Custom validators by command name
            schemas: Pydantic schemas by command name
            strict: Whether to fail on validation errors
        """
        super().__init__(name)
        self.validators = validators or {}
        self.schemas = schemas or {}
        self.strict = strict
    
    def _validate_with_schema(
        self,
        context: CommandContext,
        schema: Type[BaseModel]
    ) -> Optional[ValidationError]:
        """Validate using Pydantic schema."""
        try:
            # Combine args and kwargs for validation
            data = dict(context.kwargs)
            
            # Handle positional args if schema expects them
            sig = inspect.signature(schema)
            param_names = list(sig.parameters.keys())
            
            for i, arg in enumerate(context.args):
                if i < len(param_names):
                    data[param_names[i]] = arg
            
            # Validate
            validated = schema(**data)
            
            # Update context with validated data
            context.kwargs.update(validated.dict(exclude_unset=True))
            context.set("validated_data", validated)
            
            return None
            
        except ValidationError as e:
            return e
    
    def _validate_with_function(
        self,
        context: CommandContext,
        validator: Callable
    ) -> Optional[Exception]:
        """Validate using custom function."""
        try:
            result = validator(context)
            if result is False:
                return ValueError("Validation failed")
            return None
        except Exception as e:
            return e
    
    async def process(
        self,
        context: CommandContext,
        next_handler: Callable[[CommandContext], Any]
    ) -> MiddlewareResult:
        """Process validation middleware."""
        command_name = context.command_name
        errors = []
        
        # Schema validation
        if command_name in self.schemas:
            schema = self.schemas[command_name]
            error = self._validate_with_schema(context, schema)
            if error:
                errors.append(("schema", error))
        
        # Custom validator
        if command_name in self.validators:
            validator = self.validators[command_name]
            error = self._validate_with_function(context, validator)
            if error:
                errors.append(("custom", error))
        
        # Handle validation errors
        if errors:
            error_details = []
            for error_type, error in errors:
                if isinstance(error, ValidationError):
                    error_details.extend([
                        f"{err['loc']}: {err['msg']}"
                        for err in error.errors()
                    ])
                else:
                    error_details.append(str(error))
            
            error_msg = f"Validation failed for {command_name}: " + "; ".join(error_details)
            
            if self.strict:
                logger.error(error_msg)
                return MiddlewareResult.fail(
                    ValueError(error_msg),
                    validation_errors=errors
                )
            else:
                logger.warning(error_msg)
                context.set("validation_warnings", errors)
        
        # Continue to next handler
        if asyncio.iscoroutinefunction(next_handler):
            return await next_handler(context)
        else:
            return next_handler(context)


class TypeValidationMiddleware(ValidationMiddleware):
    """Middleware for automatic type validation based on function signatures."""
    
    def __init__(
        self,
        name: str = "TypeValidationMiddleware",
        command_handlers: Optional[Dict[str, Callable]] = None,
        **kwargs
    ):
        """Initialize type validation middleware."""
        super().__init__(name=name, **kwargs)
        self.command_handlers = command_handlers or {}
        self._build_schemas()
    
    def _build_schemas(self) -> None:
        """Build schemas from function signatures."""
        for command_name, handler in self.command_handlers.items():
            if command_name not in self.schemas:
                schema = self._create_schema_from_function(handler)
                if schema:
                    self.schemas[command_name] = schema
    
    def _create_schema_from_function(self, func: Callable) -> Optional[Type[BaseModel]]:
        """Create Pydantic schema from function signature."""
        try:
            sig = inspect.signature(func)
            hints = get_type_hints(func)
            
            fields = {}
            for param_name, param in sig.parameters.items():
                if param_name in ["self", "cls"]:
                    continue
                
                # Get type hint
                param_type = hints.get(param_name, Any)
                
                # Handle default values
                if param.default is inspect.Parameter.empty:
                    fields[param_name] = (param_type, ...)
                else:
                    fields[param_name] = (param_type, param.default)
            
            if fields:
                return create_model(f"{func.__name__}_Schema", **fields)
            
        except Exception as e:
            logger.debug(f"Could not create schema for {func.__name__}: {e}")
        
        return None


class ContractValidationMiddleware(ValidationMiddleware):
    """Middleware for contract-based validation."""
    
    def __init__(
        self,
        name: str = "ContractValidationMiddleware",
        contracts: Optional[Dict[str, List[Callable]]] = None,
        **kwargs
    ):
        """Initialize contract validation middleware.
        
        Args:
            name: Middleware name
            contracts: Pre/post condition contracts by command
        """
        super().__init__(name=name, **kwargs)
        self.contracts = contracts or {}
    
    async def process(
        self,
        context: CommandContext,
        next_handler: Callable[[CommandContext], Any]
    ) -> MiddlewareResult:
        """Process with contract validation."""
        command_name = context.command_name
        
        # Check preconditions
        if command_name in self.contracts:
            for contract in self.contracts[command_name]:
                if hasattr(contract, "precondition"):
                    try:
                        if not contract.precondition(context):
                            return MiddlewareResult.fail(
                                ValueError(f"Precondition failed: {contract.__name__}")
                            )
                    except Exception as e:
                        return MiddlewareResult.fail(e)
        
        # Execute command
        result = await super().process(context, next_handler)
        
        # Check postconditions
        if result.success and command_name in self.contracts:
            for contract in self.contracts[command_name]:
                if hasattr(contract, "postcondition"):
                    try:
                        if not contract.postcondition(context, result):
                            return MiddlewareResult.fail(
                                ValueError(f"Postcondition failed: {contract.__name__}")
                            )
                    except Exception as e:
                        return MiddlewareResult.fail(e)
        
        return result