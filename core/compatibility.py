"""Compatibility layer for smooth migration to Phase 2 architecture.

This module provides facades, adapters, and deprecation helpers to ensure
backward compatibility while migrating to new architectural patterns.
"""

import functools
import warnings
from typing import Any, Callable, Dict, List, Optional, Type, Union
from pathlib import Path

from loguru import logger


class DeprecationHelper:
    """Helper for managing deprecations during migration."""
    
    @staticmethod
    def deprecated(replacement: str, version: str = "2.0.0"):
        """Decorator to mark functions/classes as deprecated.
        
        Args:
            replacement: What to use instead
            version: Version when this will be removed
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                warnings.warn(
                    f"{func.__name__} is deprecated and will be removed in version {version}. "
                    f"Use {replacement} instead.",
                    DeprecationWarning,
                    stacklevel=2
                )
                logger.warning(
                    f"Deprecated function called: {func.__name__}. "
                    f"Replace with: {replacement}"
                )
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    @staticmethod
    def deprecated_class(replacement: str, version: str = "2.0.0"):
        """Decorator to mark classes as deprecated."""
        def decorator(cls):
            original_init = cls.__init__
            
            def new_init(self, *args, **kwargs):
                warnings.warn(
                    f"{cls.__name__} is deprecated and will be removed in version {version}. "
                    f"Use {replacement} instead.",
                    DeprecationWarning,
                    stacklevel=2
                )
                original_init(self, *args, **kwargs)
            
            cls.__init__ = new_init
            return cls
        return decorator


class TrainerFacade:
    """Facade to maintain backward compatibility for BaseTrainer while using new components.
    
    This allows gradual migration to the decomposed trainer architecture.
    """
    
    def __init__(self, trainer_instance):
        """Initialize with either old or new trainer."""
        self._trainer = trainer_instance
        self._event_bus = None
        self._use_new_architecture = False
    
    def train(self, *args, **kwargs):
        """Train method that works with both architectures."""
        # Check if using new architecture
        if hasattr(self._trainer, 'event_bus'):
            self._use_new_architecture = True
            self._event_bus = self._trainer.event_bus
        
        # Call underlying trainer
        result = self._trainer.train(*args, **kwargs)
        
        # Emit compatibility events if using old trainer
        if not self._use_new_architecture and self._event_bus:
            self._emit_legacy_events(result)
        
        return result
    
    def _emit_legacy_events(self, result):
        """Emit events for legacy trainer to maintain compatibility."""
        if self._event_bus:
            self._event_bus.emit("training.completed.legacy", {
                "result": result,
                "trainer_type": "legacy"
            })
    
    @property
    def model(self):
        """Access model with compatibility check."""
        return self._trainer.model
    
    @property
    def config(self):
        """Access config with compatibility check."""
        return self._trainer.config
    
    def __getattr__(self, name):
        """Forward other attributes to underlying trainer."""
        return getattr(self._trainer, name)


class PluginCompatibilityAdapter:
    """Adapter to make old-style plugins work with new plugin system."""
    
    def __init__(self, legacy_plugin):
        """Wrap a legacy plugin."""
        self.legacy_plugin = legacy_plugin
        self._adapted = False
    
    def adapt(self):
        """Adapt legacy plugin to new interface."""
        if self._adapted:
            return self.legacy_plugin
        
        # Add new required methods if missing
        if not hasattr(self.legacy_plugin, 'get_metadata'):
            self._add_metadata_method()
        
        if not hasattr(self.legacy_plugin, 'validate_config'):
            self._add_validation_method()
        
        self._adapted = True
        return self.legacy_plugin
    
    def _add_metadata_method(self):
        """Add metadata method to legacy plugin."""
        def get_metadata():
            from core.protocols.plugins import PluginMetadata
            return PluginMetadata(
                name=self.legacy_plugin.__class__.__name__,
                version="1.0.0",
                description=getattr(self.legacy_plugin, '__doc__', 'Legacy plugin'),
                is_legacy=True
            )
        
        self.legacy_plugin.get_metadata = get_metadata
    
    def _add_validation_method(self):
        """Add config validation to legacy plugin."""
        def validate_config(config):
            # Basic validation for legacy plugins
            return True
        
        self.legacy_plugin.validate_config = validate_config


class ConfigMigrationHelper:
    """Helper for migrating configuration formats."""
    
    @staticmethod
    def migrate_v1_to_v2(old_config: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate v1 config format to v2 format.
        
        Args:
            old_config: Configuration in v1 format
            
        Returns:
            Configuration in v2 format
        """
        new_config = old_config.copy()
        
        # Migrate trainer config
        if 'trainer' in old_config:
            trainer_config = old_config['trainer']
            
            # Move compilation settings to new location
            if 'use_compiled' in trainer_config:
                warnings.warn(
                    "Config key 'trainer.use_compiled' is deprecated. "
                    "Use 'training.use_compilation' instead.",
                    DeprecationWarning
                )
                new_config.setdefault('training', {})
                new_config['training']['use_compilation'] = trainer_config.pop('use_compiled')
            
            # Rename old keys
            key_mapping = {
                'lr': 'learning_rate',
                'epochs': 'num_epochs',
                'eval_freq': 'eval_steps'
            }
            
            for old_key, new_key in key_mapping.items():
                if old_key in trainer_config:
                    trainer_config[new_key] = trainer_config.pop(old_key)
                    warnings.warn(
                        f"Config key '{old_key}' renamed to '{new_key}'",
                        DeprecationWarning
                    )
        
        # Migrate model config
        if 'model' in old_config:
            model_config = old_config['model']
            
            # Handle old model type names
            if model_config.get('type') == 'bert':
                model_config['model_type'] = 'bert_classifier'
                del model_config['type']
                warnings.warn(
                    "Model type 'bert' renamed to 'bert_classifier'",
                    DeprecationWarning
                )
        
        return new_config
    
    @staticmethod
    def validate_migration(old_config: Dict[str, Any], new_config: Dict[str, Any]) -> bool:
        """Validate that migration preserved essential settings.
        
        Returns:
            True if migration is valid
        """
        # Check critical settings are preserved
        critical_paths = [
            ('training.num_epochs', 'trainer.epochs'),
            ('models.model_type', 'model.type'),
            ('data.batch_size', 'data.batch_size')
        ]
        
        for new_path, old_path in critical_paths:
            old_value = ConfigMigrationHelper._get_nested(old_config, old_path)
            new_value = ConfigMigrationHelper._get_nested(new_config, new_path)
            
            if old_value is not None and new_value is None:
                logger.error(f"Migration lost critical config: {old_path}")
                return False
        
        return True
    
    @staticmethod
    def _get_nested(config: Dict[str, Any], path: str) -> Any:
        """Get value from nested dict using dot notation."""
        keys = path.split('.')
        value = config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value


class APICompatibilityLayer:
    """Provides compatibility for old API calls."""
    
    _instance = None
    _method_mappings = {
        # Old method -> (new module, new method)
        'prepare_data': ('data.augmentation', 'augment_dataset'),
        'create_bert_model': ('models.factory', 'create_model'),
        'train_model': ('training.core', 'run_training')
    }
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def register_compatibility(self, old_name: str, new_callable: Callable):
        """Register a compatibility mapping."""
        setattr(self, old_name, DeprecationHelper.deprecated(
            f"{new_callable.__module__}.{new_callable.__name__}"
        )(new_callable))
    
    def __getattr__(self, name):
        """Provide helpful error messages for removed APIs."""
        if name in self._method_mappings:
            new_module, new_method = self._method_mappings[name]
            raise AttributeError(
                f"'{name}' has been removed. "
                f"Please use '{new_method}' from '{new_module}' instead."
            )
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")


# Singleton instance
compatibility = APICompatibilityLayer()


# Migration utilities
def ensure_compatibility(func):
    """Decorator to ensure backward compatibility for functions."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Check for old-style arguments
        if 'use_compiled' in kwargs:
            warnings.warn(
                "Parameter 'use_compiled' is deprecated. Use 'use_compilation' instead.",
                DeprecationWarning
            )
            kwargs['use_compilation'] = kwargs.pop('use_compiled')
        
        # Call original function
        return func(*args, **kwargs)
    
    return wrapper


def create_migration_report(project_path: Path) -> Dict[str, Any]:
    """Create a report of what needs to be migrated in a project.
    
    Args:
        project_path: Path to the project
        
    Returns:
        Migration report with findings and recommendations
    """
    report = {
        "deprecated_usage": [],
        "config_updates_needed": [],
        "code_updates_needed": [],
        "estimated_effort": "low"  # low, medium, high
    }
    
    # Check for deprecated imports
    deprecated_imports = [
        ("from training.base import Trainer", "from training.core.base import BaseTrainer"),
        ("from data.preprocessing import", "from data.augmentation import"),
        ("from models.bert import BertModel", "from models.factory import create_model")
    ]
    
    # Scan Python files
    for py_file in project_path.rglob("*.py"):
        if ".venv" in str(py_file) or "__pycache__" in str(py_file):
            continue
            
        try:
            content = py_file.read_text()
            for old_import, new_import in deprecated_imports:
                if old_import in content:
                    report["code_updates_needed"].append({
                        "file": str(py_file),
                        "old": old_import,
                        "new": new_import,
                        "line": next(i for i, line in enumerate(content.splitlines()) 
                                   if old_import in line)
                    })
        except Exception as e:
            logger.debug(f"Could not read {py_file}: {e}")
    
    # Check config files
    for config_file in project_path.rglob("*.yaml"):
        if ".venv" in str(config_file):
            continue
            
        try:
            import yaml
            with open(config_file) as f:
                config = yaml.safe_load(f)
            
            if config and isinstance(config, dict):
                # Check for old config keys
                if 'trainer' in config and 'use_compiled' in config.get('trainer', {}):
                    report["config_updates_needed"].append({
                        "file": str(config_file),
                        "issue": "Old 'use_compiled' key",
                        "fix": "Rename to 'use_compilation' under 'training' section"
                    })
        except Exception as e:
            logger.debug(f"Could not read {config_file}: {e}")
    
    # Estimate effort
    total_issues = (
        len(report["deprecated_usage"]) +
        len(report["config_updates_needed"]) +
        len(report["code_updates_needed"])
    )
    
    if total_issues == 0:
        report["estimated_effort"] = "none"
    elif total_issues < 5:
        report["estimated_effort"] = "low"
    elif total_issues < 20:
        report["estimated_effort"] = "medium"
    else:
        report["estimated_effort"] = "high"
    
    report["summary"] = {
        "total_issues": total_issues,
        "files_affected": len(set(
            item.get("file", "") 
            for category in report.values() 
            if isinstance(category, list)
            for item in category
        ))
    }
    
    return report


# Re-export commonly used items for convenience
__all__ = [
    'DeprecationHelper',
    'TrainerFacade',
    'PluginCompatibilityAdapter',
    'ConfigMigrationHelper',
    'APICompatibilityLayer',
    'compatibility',
    'ensure_compatibility',
    'create_migration_report'
]