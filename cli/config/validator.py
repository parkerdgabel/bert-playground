"""Configuration validator component for k-bert CLI.

This module provides a protocol-based validator for configuration validation.
"""

from typing import Any, Dict, List, Optional, Protocol, Type
from abc import abstractmethod
from pathlib import Path

from loguru import logger

from .schemas import KBertConfig, ProjectConfig, CompetitionConfig
from .validators import (
    ConfigValidationError,
    validate_config as validate_kbert_config,
    validate_competition_config,
    validate_cli_overrides,
    validate_path,
)


class ConfigValidatorProtocol(Protocol):
    """Protocol for configuration validators."""
    
    @abstractmethod
    def validate(self, config: Any) -> List[str]:
        """Validate a configuration object.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        ...
    
    @abstractmethod
    def validate_dict(self, config_dict: Dict[str, Any], schema: Type) -> List[str]:
        """Validate a configuration dictionary against a schema.
        
        Args:
            config_dict: Configuration dictionary
            schema: Pydantic schema to validate against
            
        Returns:
            List of validation errors
        """
        ...


class ConfigValidator:
    """Handles configuration validation with extensible rules."""
    
    def __init__(self):
        """Initialize validator with default rules."""
        self._custom_validators: Dict[str, List[callable]] = {}
    
    def validate(self, config: Any) -> List[str]:
        """Validate a configuration object.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        if isinstance(config, KBertConfig):
            errors.extend(self._validate_kbert_config(config))
        elif isinstance(config, ProjectConfig):
            errors.extend(self._validate_project_config(config))
        elif isinstance(config, CompetitionConfig):
            errors.extend(validate_competition_config(config))
        else:
            errors.append(f"Unknown configuration type: {type(config).__name__}")
        
        # Apply custom validators if registered
        config_type = type(config).__name__
        if config_type in self._custom_validators:
            for validator in self._custom_validators[config_type]:
                try:
                    custom_errors = validator(config)
                    if custom_errors:
                        errors.extend(custom_errors)
                except Exception as e:
                    logger.error(f"Custom validator failed: {e}")
                    errors.append(f"Custom validation error: {str(e)}")
        
        return errors
    
    def validate_dict(self, config_dict: Dict[str, Any], schema: Type) -> List[str]:
        """Validate a configuration dictionary against a schema.
        
        Args:
            config_dict: Configuration dictionary
            schema: Pydantic schema to validate against
            
        Returns:
            List of validation errors
        """
        errors = []
        
        try:
            # Try to create schema instance
            config = schema(**config_dict)
            # Validate the created instance
            return self.validate(config)
        except Exception as e:
            errors.append(f"Schema validation failed: {str(e)}")
            return errors
    
    def add_custom_validator(self, config_type: str, validator: callable) -> None:
        """Add a custom validator for a configuration type.
        
        Args:
            config_type: Configuration type name (e.g., "KBertConfig")
            validator: Validator function that returns list of errors
        """
        if config_type not in self._custom_validators:
            self._custom_validators[config_type] = []
        
        self._custom_validators[config_type].append(validator)
        logger.debug(f"Added custom validator for {config_type}")
    
    def _validate_kbert_config(self, config: KBertConfig) -> List[str]:
        """Validate KBertConfig using existing validators.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation errors
        """
        try:
            validate_kbert_config(config)
            return []
        except ConfigValidationError as e:
            return e.errors
    
    def _validate_project_config(self, config: ProjectConfig) -> List[str]:
        """Validate project configuration.
        
        Args:
            config: Project configuration to validate
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Validate required fields
        if not config.name:
            errors.append("Project name is required")
        
        if not config.version:
            errors.append("Project version is required")
        
        # Validate paths
        if config.src_dir:
            path_error = validate_path(
                Path(config.src_dir),
                must_exist=False,  # Don't require it to exist yet
                must_be_dir=True if Path(config.src_dir).exists() else False
            )
            if path_error:
                errors.append(f"Source directory: {path_error}")
        
        # Validate nested configurations
        if config.models:
            model_errors = self._validate_model_subset(config.models.model_dump())
            errors.extend([f"models.{e}" for e in model_errors])
        
        if config.training:
            training_errors = self._validate_training_subset(config.training.model_dump())
            errors.extend([f"training.{e}" for e in training_errors])
        
        if config.data:
            data_errors = self._validate_data_subset(config.data.model_dump())
            errors.extend([f"data.{e}" for e in data_errors])
        
        return errors
    
    def _validate_model_subset(self, model_config: Dict[str, Any]) -> List[str]:
        """Validate model configuration subset.
        
        Args:
            model_config: Model configuration dictionary
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Check if default_model is specified
        if "default_model" in model_config and not model_config["default_model"]:
            errors.append("default_model cannot be empty")
        
        # Validate architecture if specified
        if "default_architecture" in model_config:
            valid_architectures = ["bert", "modernbert", "neobert"]
            if model_config["default_architecture"] not in valid_architectures:
                errors.append(
                    f"Invalid architecture: {model_config['default_architecture']}. "
                    f"Must be one of: {valid_architectures}"
                )
        
        return errors
    
    def _validate_training_subset(self, training_config: Dict[str, Any]) -> List[str]:
        """Validate training configuration subset.
        
        Args:
            training_config: Training configuration dictionary
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Validate numeric values if present
        if "batch_size" in training_config and training_config["batch_size"] < 1:
            errors.append("batch_size must be at least 1")
        
        if "epochs" in training_config and training_config["epochs"] < 1:
            errors.append("epochs must be at least 1")
        
        if "learning_rate" in training_config and training_config["learning_rate"] <= 0:
            errors.append("learning_rate must be positive")
        
        return errors
    
    def _validate_data_subset(self, data_config: Dict[str, Any]) -> List[str]:
        """Validate data configuration subset.
        
        Args:
            data_config: Data configuration dictionary
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Validate max_length if present
        if "max_length" in data_config:
            if data_config["max_length"] < 1:
                errors.append("max_length must be at least 1")
            elif data_config["max_length"] > 8192:
                errors.append("max_length cannot exceed 8192")
        
        return errors


class StrictConfigValidator(ConfigValidator):
    """Strict validator with additional production checks."""
    
    def __init__(self):
        """Initialize strict validator."""
        super().__init__()
        
        # Add strict validation rules
        self.add_custom_validator("KBertConfig", self._strict_kbert_validation)
        self.add_custom_validator("ProjectConfig", self._strict_project_validation)
    
    def _strict_kbert_validation(self, config: KBertConfig) -> List[str]:
        """Additional strict validation for KBertConfig.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Require MLflow to be configured properly in production
        if config.mlflow.auto_log and config.mlflow.tracking_uri == "file://./mlruns":
            errors.append(
                "Production configurations should use a proper MLflow tracking server, "
                "not local file storage"
            )
        
        # Warn about debug settings
        if hasattr(config, "debug") and config.debug:
            errors.append("Debug mode should not be enabled in production")
        
        return errors
    
    def _strict_project_validation(self, config: ProjectConfig) -> List[str]:
        """Additional strict validation for ProjectConfig.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Require proper versioning
        if config.version == "0.1.0":
            errors.append("Please update the project version from the default")
        
        # Require competition to be specified
        if not config.competition:
            errors.append("Competition must be specified for project configurations")
        
        return errors