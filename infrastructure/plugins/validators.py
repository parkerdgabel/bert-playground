"""Plugin validation utilities.

This module provides:
- Plugin validation logic
- Dependency checking
- Requirement verification
- Conflict detection
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from loguru import logger

from .base import Plugin, PluginError, PluginMetadata


@dataclass
class ValidationResult:
    """Result of plugin validation."""
    
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)
    
    def merge(self, other: "ValidationResult") -> None:
        """Merge another validation result into this one."""
        self.is_valid = self.is_valid and other.is_valid
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)


class PluginValidator:
    """Validates plugins for correctness and compatibility."""
    
    def __init__(self):
        """Initialize plugin validator."""
        self._registered_plugins: Dict[str, PluginMetadata] = {}
        self._provided_capabilities: Dict[str, Set[str]] = {}
    
    def register_existing(self, plugin_name: str, metadata: PluginMetadata) -> None:
        """Register an existing plugin for dependency checking.
        
        Args:
            plugin_name: Plugin name
            metadata: Plugin metadata
        """
        self._registered_plugins[plugin_name] = metadata
        
        # Update capability index
        for capability in metadata.provides:
            if capability not in self._provided_capabilities:
                self._provided_capabilities[capability] = set()
            self._provided_capabilities[capability].add(plugin_name)
    
    def validate(self, plugin: Plugin) -> ValidationResult:
        """Validate a plugin.
        
        Args:
            plugin: Plugin to validate
            
        Returns:
            Validation result
        """
        result = ValidationResult(is_valid=True)
        metadata = plugin.metadata
        
        # Validate metadata
        metadata_result = self._validate_metadata(metadata)
        result.merge(metadata_result)
        
        # Check dependencies
        dependency_result = self._check_dependencies(metadata)
        result.merge(dependency_result)
        
        # Check conflicts
        conflict_result = self._check_conflicts(metadata)
        result.merge(conflict_result)
        
        # Check requirements
        requirement_result = self._check_requirements(metadata)
        result.merge(requirement_result)
        
        # Validate plugin interface
        interface_result = self._validate_interface(plugin)
        result.merge(interface_result)
        
        return result
    
    def _validate_metadata(self, metadata: PluginMetadata) -> ValidationResult:
        """Validate plugin metadata.
        
        Args:
            metadata: Plugin metadata
            
        Returns:
            Validation result
        """
        result = ValidationResult(is_valid=True)
        
        # Check required fields
        if not metadata.name:
            result.add_error("Plugin name is required")
        
        if not metadata.version:
            result.add_error("Plugin version is required")
        
        # Validate version format
        if metadata.version and not self._is_valid_version(metadata.version):
            result.add_warning(f"Invalid version format: {metadata.version}")
        
        # Check for duplicate names
        if metadata.name in self._registered_plugins:
            result.add_error(f"Plugin with name '{metadata.name}' already exists")
        
        return result
    
    def _check_dependencies(self, metadata: PluginMetadata) -> ValidationResult:
        """Check plugin dependencies.
        
        Args:
            metadata: Plugin metadata
            
        Returns:
            Validation result
        """
        result = ValidationResult(is_valid=True)
        
        # Check plugin dependencies
        for dep in metadata.depends_on:
            if dep not in self._registered_plugins:
                result.add_error(f"Missing dependency: {dep}")
        
        # Check capability dependencies
        for capability in metadata.consumes:
            if capability not in self._provided_capabilities:
                result.add_error(f"No plugin provides capability: {capability}")
            elif not self._provided_capabilities[capability]:
                result.add_error(f"No active plugin provides capability: {capability}")
        
        return result
    
    def _check_conflicts(self, metadata: PluginMetadata) -> ValidationResult:
        """Check for plugin conflicts.
        
        Args:
            metadata: Plugin metadata
            
        Returns:
            Validation result
        """
        result = ValidationResult(is_valid=True)
        
        # Check explicit conflicts
        for conflict in metadata.conflicts_with:
            if conflict in self._registered_plugins:
                result.add_error(f"Conflicts with registered plugin: {conflict}")
        
        # Check capability conflicts (multiple providers for exclusive capabilities)
        # This is a placeholder - implement based on your needs
        
        return result
    
    def _check_requirements(self, metadata: PluginMetadata) -> ValidationResult:
        """Check plugin requirements.
        
        Args:
            metadata: Plugin metadata
            
        Returns:
            Validation result
        """
        result = ValidationResult(is_valid=True)
        
        # Check Python package requirements
        for req in metadata.requirements:
            if not self._is_requirement_satisfied(req):
                result.add_warning(f"Requirement not satisfied: {req}")
        
        return result
    
    def _validate_interface(self, plugin: Plugin) -> ValidationResult:
        """Validate plugin interface implementation.
        
        Args:
            plugin: Plugin instance
            
        Returns:
            Validation result
        """
        result = ValidationResult(is_valid=True)
        
        # Check required methods
        required_methods = ["validate", "initialize", "start", "stop", "cleanup"]
        for method in required_methods:
            if not hasattr(plugin, method) or not callable(getattr(plugin, method)):
                result.add_error(f"Missing required method: {method}")
        
        # Check required properties
        required_properties = ["metadata", "state"]
        for prop in required_properties:
            if not hasattr(plugin, prop):
                result.add_error(f"Missing required property: {prop}")
        
        return result
    
    def _is_valid_version(self, version: str) -> bool:
        """Check if version string is valid.
        
        Args:
            version: Version string
            
        Returns:
            True if valid
        """
        # Simple semantic versioning check
        parts = version.split(".")
        if len(parts) not in (2, 3):
            return False
        
        for part in parts:
            try:
                int(part)
            except ValueError:
                return False
        
        return True
    
    def _is_requirement_satisfied(self, requirement: str) -> bool:
        """Check if a requirement is satisfied.
        
        Args:
            requirement: Requirement string (e.g., "numpy>=1.20")
            
        Returns:
            True if satisfied
        """
        try:
            import pkg_resources
            
            # Parse requirement
            req = pkg_resources.Requirement.parse(requirement)
            
            # Check if installed
            try:
                pkg_resources.get_distribution(req)
                return True
            except pkg_resources.DistributionNotFound:
                return False
            except pkg_resources.VersionConflict:
                return False
                
        except Exception as e:
            logger.debug(f"Could not check requirement {requirement}: {e}")
            # Assume satisfied if we can't check
            return True


def validate_plugin(plugin: Plugin, existing_plugins: Optional[Dict[str, PluginMetadata]] = None) -> ValidationResult:
    """Validate a plugin.
    
    Args:
        plugin: Plugin to validate
        existing_plugins: Existing plugins for dependency checking
        
    Returns:
        Validation result
    """
    validator = PluginValidator()
    
    # Register existing plugins if provided
    if existing_plugins:
        for name, metadata in existing_plugins.items():
            validator.register_existing(name, metadata)
    
    return validator.validate(plugin)