"""Schema validation for data structures."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import pandas as pd
# from loguru import logger  # Domain should not depend on logging framework


class FieldType(Enum):
    """Supported field types for schema validation."""
    
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    CATEGORICAL = "categorical"
    TEXT = "text"
    NUMERIC = "numeric"  # Integer or float
    ANY = "any"


@dataclass
class SchemaField:
    """Definition of a schema field."""
    
    name: str
    field_type: FieldType
    required: bool = True
    nullable: bool = False
    unique: bool = False
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    allowed_values: Optional[List[Any]] = None
    pattern: Optional[str] = None
    description: Optional[str] = None
    
    def validate(self, value: Any) -> tuple[bool, Optional[str]]:
        """Validate a single value against this field.
        
        Args:
            value: Value to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Handle null values
        if pd.isna(value):
            if not self.nullable:
                return False, f"Field '{self.name}' cannot be null"
            return True, None
        
        # Type validation
        if self.field_type != FieldType.ANY:
            if not self._check_type(value):
                return False, f"Field '{self.name}' type mismatch: expected {self.field_type.value}"
        
        # Value constraints
        if self.allowed_values is not None:
            if value not in self.allowed_values:
                return False, f"Field '{self.name}' value not in allowed values"
        
        # Numeric constraints
        if self.field_type in [FieldType.INTEGER, FieldType.FLOAT, FieldType.NUMERIC]:
            if self.min_value is not None and value < self.min_value:
                return False, f"Field '{self.name}' value {value} below minimum {self.min_value}"
            if self.max_value is not None and value > self.max_value:
                return False, f"Field '{self.name}' value {value} above maximum {self.max_value}"
        
        # String constraints
        if self.field_type in [FieldType.STRING, FieldType.TEXT]:
            str_value = str(value)
            if self.min_length is not None and len(str_value) < self.min_length:
                return False, f"Field '{self.name}' length below minimum {self.min_length}"
            if self.max_length is not None and len(str_value) > self.max_length:
                return False, f"Field '{self.name}' length above maximum {self.max_length}"
            
            # Pattern matching
            if self.pattern is not None:
                import re
                if not re.match(self.pattern, str_value):
                    return False, f"Field '{self.name}' does not match pattern {self.pattern}"
        
        return True, None
    
    def _check_type(self, value: Any) -> bool:
        """Check if value matches expected type."""
        if self.field_type == FieldType.STRING:
            return isinstance(value, str)
        elif self.field_type == FieldType.INTEGER:
            return isinstance(value, (int, np.integer))
        elif self.field_type == FieldType.FLOAT:
            return isinstance(value, (float, np.floating))
        elif self.field_type == FieldType.NUMERIC:
            return isinstance(value, (int, float, np.integer, np.floating))
        elif self.field_type == FieldType.BOOLEAN:
            return isinstance(value, (bool, np.bool_))
        elif self.field_type == FieldType.DATETIME:
            return isinstance(value, (pd.Timestamp, datetime))
        elif self.field_type in [FieldType.CATEGORICAL, FieldType.TEXT]:
            return True  # Accept any type for categorical/text
        
        return True


@dataclass
class Schema:
    """Data schema definition."""
    
    name: str
    fields: List[SchemaField]
    description: Optional[str] = None
    strict: bool = True  # If True, reject extra fields
    
    def __post_init__(self):
        """Create field lookup."""
        self._field_map = {field.name: field for field in self.fields}
    
    def validate_row(self, row: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate a single data row.
        
        Args:
            row: Data row as dictionary
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check required fields
        for field in self.fields:
            if field.required and field.name not in row:
                errors.append(f"Required field '{field.name}' is missing")
                continue
            
            if field.name in row:
                is_valid, error = field.validate(row[field.name])
                if not is_valid:
                    errors.append(error)
        
        # Check for extra fields
        if self.strict:
            extra_fields = set(row.keys()) - set(self._field_map.keys())
            if extra_fields:
                errors.append(f"Unexpected fields: {extra_fields}")
        
        return len(errors) == 0, errors
    
    def validate_dataframe(self, df: pd.DataFrame) -> tuple[bool, Dict[str, List[str]]]:
        """Validate a DataFrame.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, column_errors)
        """
        column_errors = {}
        
        # Check required columns
        for field in self.fields:
            if field.required and field.name not in df.columns:
                column_errors[field.name] = [f"Required column '{field.name}' is missing"]
        
        # Validate each column
        for field in self.fields:
            if field.name in df.columns:
                errors = []
                
                # Check uniqueness
                if field.unique:
                    duplicates = df[field.name].duplicated().sum()
                    if duplicates > 0:
                        errors.append(f"Column '{field.name}' has {duplicates} duplicate values")
                
                # Check each value
                for idx, value in df[field.name].items():
                    is_valid, error = field.validate(value)
                    if not is_valid:
                        errors.append(f"Row {idx}: {error}")
                
                if errors:
                    column_errors[field.name] = errors
        
        # Check for extra columns
        if self.strict:
            extra_columns = set(df.columns) - set(self._field_map.keys())
            if extra_columns:
                column_errors["_extra"] = [f"Unexpected columns: {extra_columns}"]
        
        return len(column_errors) == 0, column_errors
    
    def get_field(self, name: str) -> Optional[SchemaField]:
        """Get field by name."""
        return self._field_map.get(name)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "strict": self.strict,
            "fields": [
                {
                    "name": f.name,
                    "type": f.field_type.value,
                    "required": f.required,
                    "nullable": f.nullable,
                    "unique": f.unique,
                    "constraints": {
                        k: v for k, v in {
                            "min_value": f.min_value,
                            "max_value": f.max_value,
                            "min_length": f.min_length,
                            "max_length": f.max_length,
                            "allowed_values": f.allowed_values,
                            "pattern": f.pattern,
                        }.items() if v is not None
                    },
                    "description": f.description,
                }
                for f in self.fields
            ]
        }


class SchemaValidator:
    """Validator that uses schemas for validation."""
    
    def __init__(self, schema: Schema):
        """Initialize with schema.
        
        Args:
            schema: Schema to use for validation
        """
        self.schema = schema
        # logger.debug(f"Initialized schema validator for '{schema.name}'")
    
    def validate(self, data: Union[Dict[str, Any], pd.DataFrame]) -> tuple[bool, Any]:
        """Validate data against schema.
        
        Args:
            data: Data to validate
            
        Returns:
            Tuple of (is_valid, errors)
        """
        if isinstance(data, dict):
            return self.schema.validate_row(data)
        elif isinstance(data, pd.DataFrame):
            return self.schema.validate_dataframe(data)
        else:
            return False, f"Unsupported data type: {type(data)}"
    
    def get_schema_info(self) -> Dict[str, Any]:
        """Get information about the schema."""
        return self.schema.to_dict()


# Import numpy for type checking
import numpy as np
from datetime import datetime