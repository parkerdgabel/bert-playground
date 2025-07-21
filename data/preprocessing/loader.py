"""Plugin loader for data preprocessing plugins."""

import importlib
import pkgutil
from pathlib import Path
from typing import List, Dict, Type
from loguru import logger

from .base import DataPreprocessor, PreprocessorRegistry


def discover_plugins(plugin_dir: Path = None) -> Dict[str, Type[DataPreprocessor]]:
    """Discover and load all preprocessing plugins.
    
    Args:
        plugin_dir: Directory containing plugins (defaults to plugins subdir)
        
    Returns:
        Dictionary mapping plugin names to classes
    """
    if plugin_dir is None:
        plugin_dir = Path(__file__).parent / "plugins"
    
    discovered = {}
    
    # Import the plugins package to trigger registration
    try:
        import data.preprocessing.plugins
        logger.info("Loaded preprocessing plugins package")
    except ImportError as e:
        logger.warning(f"Failed to import plugins package: {e}")
    
    # Also try dynamic discovery
    if plugin_dir.exists():
        for file_path in plugin_dir.glob("*.py"):
            if file_path.name.startswith("_"):
                continue
                
            module_name = file_path.stem
            try:
                # Import the module
                spec = importlib.util.spec_from_file_location(
                    f"data.preprocessing.plugins.{module_name}", 
                    file_path
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    logger.debug(f"Loaded plugin module: {module_name}")
            except Exception as e:
                logger.warning(f"Failed to load plugin {module_name}: {e}")
    
    # Get all registered preprocessors
    for name in PreprocessorRegistry.list_available():
        discovered[name] = PreprocessorRegistry.get(name)
    
    logger.info(f"Discovered {len(discovered)} preprocessing plugins: {list(discovered.keys())}")
    return discovered


def list_available_preprocessors() -> List[str]:
    """List all available preprocessors.
    
    Returns:
        List of preprocessor names
    """
    discover_plugins()  # Ensure plugins are loaded
    return PreprocessorRegistry.list_available()


def get_preprocessor(name: str) -> Type[DataPreprocessor]:
    """Get a preprocessor class by name.
    
    Args:
        name: Name of the preprocessor
        
    Returns:
        Preprocessor class
    """
    discover_plugins()  # Ensure plugins are loaded
    return PreprocessorRegistry.get(name)


def create_preprocessor(name: str, config: 'DataPrepConfig') -> DataPreprocessor:
    """Create a preprocessor instance.
    
    Args:
        name: Name of the preprocessor
        config: Configuration for the preprocessor
        
    Returns:
        Preprocessor instance
    """
    discover_plugins()  # Ensure plugins are loaded
    return PreprocessorRegistry.create(name, config)