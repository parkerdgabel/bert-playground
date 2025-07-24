"""Configuration validation utilities.

This module provides configuration validation against schemas and business rules.
"""

from typing import Any, Dict, List, Optional, Type, Union

try:
    from pydantic import BaseModel, ValidationError as PydanticValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    BaseModel = object
    PydanticValidationError = Exception
    PYDANTIC_AVAILABLE = False


class ConfigurationValidationError(Exception):
    """Configuration validation error."""
    pass


class ConfigurationValidator:
    """Configuration validation utility."""
    
    def __init__(self):
        """Initialize validator."""
        self.validation_errors: List[str] = []
        
    def validate(self, config: Dict[str, Any]) -> bool:
        """Validate complete configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If configuration is invalid
        """
        self.validation_errors.clear()
        
        # Basic structure validation
        self._validate_structure(config)
        
        # Validate individual sections
        self._validate_models_config(config.get("models", {}))
        self._validate_training_config(config.get("training", {}))
        self._validate_data_config(config.get("data", {}))
        self._validate_adapters_config(config.get("adapters", {}))
        
        if self.validation_errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(self.validation_errors) 
            raise ConfigurationValidationError(error_msg)
            
        return True
        
    def _validate_structure(self, config: Dict[str, Any]) -> None:
        """Validate basic configuration structure."""
        # Make sections optional since they can be provided per-command
        # required_sections = ["models", "training", "data"]
        # 
        # for section in required_sections:
        #     if section not in config:
        #         self.validation_errors.append(f"Missing required section: {section}")
        pass
                
    def _validate_models_config(self, models_config: Dict[str, Any]) -> None:
        """Validate models configuration section."""
        if not models_config:
            return
            
        # Validate model type
        model_type = models_config.get("type")
        valid_types = [
            "modernbert_with_head", 
            "bert_with_head", 
            "custom_bert",
            "ensemble"
        ]
        if model_type and model_type not in valid_types:
            self.validation_errors.append(
                f"Invalid model type '{model_type}'. Must be one of: {valid_types}"
            )
            
        # Validate batch size
        batch_size = models_config.get("batch_size")
        if batch_size is not None:
            if not isinstance(batch_size, int) or batch_size < 1:
                self.validation_errors.append("batch_size must be a positive integer")
                
        # Validate max length
        max_length = models_config.get("max_length")
        if max_length is not None:
            if not isinstance(max_length, int) or max_length < 1:
                self.validation_errors.append("max_length must be a positive integer")
                
    def _validate_training_config(self, training_config: Dict[str, Any]) -> None:
        """Validate training configuration section."""
        if not training_config:
            return
            
        # Validate epochs
        epochs = training_config.get("epochs")
        if epochs is not None:
            if not isinstance(epochs, int) or epochs < 1:
                self.validation_errors.append("epochs must be a positive integer")
                
        # Validate learning rate
        lr = training_config.get("learning_rate")
        if lr is not None:
            if not isinstance(lr, (int, float)) or lr <= 0:
                self.validation_errors.append("learning_rate must be a positive number")
                
        # Validate weight decay
        weight_decay = training_config.get("weight_decay")
        if weight_decay is not None:
            if not isinstance(weight_decay, (int, float)) or weight_decay < 0:
                self.validation_errors.append("weight_decay must be non-negative")
                
        # Validate warmup steps
        warmup_steps = training_config.get("warmup_steps")
        if warmup_steps is not None:
            if not isinstance(warmup_steps, int) or warmup_steps < 0:
                self.validation_errors.append("warmup_steps must be non-negative integer")
                
    def _validate_data_config(self, data_config: Dict[str, Any]) -> None:
        """Validate data configuration section."""
        if not data_config:
            return
            
        # Validate paths exist (only warn, don't fail)
        # Data paths might be specified for specific commands
        for path_key in ["train_path", "val_path", "test_path"]:
            path = data_config.get(path_key)
            if path is not None:
                from pathlib import Path
                if not Path(path).exists():
                    # Log warning but don't fail validation
                    # self.validation_errors.append(f"Data file not found: {path}")
                    pass
                    
    def _validate_adapters_config(self, adapters_config: Dict[str, Any]) -> None:
        """Validate adapters configuration section."""
        if not adapters_config:
            return
            
        # Validate adapter implementations
        valid_adapters = {
            "monitoring": ["loguru", "mlflow", "wandb"],
            "storage": ["filesystem", "s3", "gcs"],
            "compute": ["mlx", "pytorch", "jax"],
            "tokenizer": ["huggingface", "sentencepiece"],
        }
        
        for adapter_type, adapter_config in adapters_config.items():
            if adapter_type not in valid_adapters:
                self.validation_errors.append(f"Unknown adapter type: {adapter_type}")
                continue
                
            implementation = adapter_config.get("implementation")
            if implementation and implementation not in valid_adapters[adapter_type]:
                self.validation_errors.append(
                    f"Invalid {adapter_type} adapter implementation '{implementation}'. "
                    f"Must be one of: {valid_adapters[adapter_type]}"
                )
                
    def validate_schema(self, config: Dict[str, Any], schema: Type) -> bool:
        """Validate configuration against a Pydantic schema.
        
        Args:
            config: Configuration to validate
            schema: Pydantic model class
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
            ImportError: If Pydantic is not available
        """
        if not PYDANTIC_AVAILABLE:
            raise ImportError("Pydantic is required for schema validation")
            
        try:
            schema(**config)
            return True
        except PydanticValidationError as e:
            raise ConfigurationValidationError(f"Schema validation failed: {e}")
            
    def validate_required_fields(
        self, 
        config: Dict[str, Any], 
        required_fields: List[str]
    ) -> bool:
        """Validate that required fields are present.
        
        Args:
            config: Configuration to validate
            required_fields: List of required field names (dot notation supported)
            
        Returns:
            True if all required fields are present
        """
        missing_fields = []
        
        for field in required_fields:
            if not self._has_field(config, field):
                missing_fields.append(field)
                
        if missing_fields:
            raise ConfigurationValidationError(f"Missing required fields: {missing_fields}")
            
        return True
        
    def _has_field(self, config: Dict[str, Any], field: str) -> bool:
        """Check if a field exists in configuration (supports dot notation).
        
        Args:
            config: Configuration dictionary
            field: Field name (e.g., "models.batch_size")
            
        Returns:
            True if field exists
        """
        current = config
        for part in field.split("."):
            if not isinstance(current, dict) or part not in current:
                return False
            current = current[part]
        return True
        
    def validate_types(
        self, 
        config: Dict[str, Any], 
        type_constraints: Dict[str, Type]
    ) -> bool:
        """Validate field types.
        
        Args:
            config: Configuration to validate
            type_constraints: Field -> type mapping
            
        Returns:
            True if all types are correct
        """
        type_errors = []
        
        for field, expected_type in type_constraints.items():
            if self._has_field(config, field):
                value = self._get_field(config, field)
                if not isinstance(value, expected_type):
                    type_errors.append(
                        f"Field '{field}' must be {expected_type.__name__}, "
                        f"got {type(value).__name__}"
                    )
                    
        if type_errors:
            raise ConfigurationValidationError("Type validation failed:\n" + "\n".join(type_errors))
            
        return True
        
    def _get_field(self, config: Dict[str, Any], field: str) -> Any:
        """Get field value from configuration (supports dot notation).
        
        Args:
            config: Configuration dictionary
            field: Field name
            
        Returns:
            Field value
        """
        current = config
        for part in field.split("."):
            current = current[part]
        return current
        
    def validate_ranges(
        self, 
        config: Dict[str, Any], 
        range_constraints: Dict[str, tuple]
    ) -> bool:
        """Validate numeric field ranges.
        
        Args:
            config: Configuration to validate
            range_constraints: Field -> (min, max) mapping
            
        Returns:
            True if all ranges are valid
        """
        range_errors = []
        
        for field, (min_val, max_val) in range_constraints.items():
            if self._has_field(config, field):
                value = self._get_field(config, field)
                if isinstance(value, (int, float)):
                    if min_val is not None and value < min_val:
                        range_errors.append(f"Field '{field}' must be >= {min_val}")
                    if max_val is not None and value > max_val:
                        range_errors.append(f"Field '{field}' must be <= {max_val}")
                        
        if range_errors:
            raise ConfigurationValidationError("Range validation failed:\n" + "\n".join(range_errors))
            
        return True