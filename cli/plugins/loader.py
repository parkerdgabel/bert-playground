"""Plugin loader for k-bert projects.

This module handles the discovery and loading of plugins from project directories,
following k-bert's conventional project structure.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import sys
import os

from loguru import logger

from .registry import get_registry, ComponentRegistry


class PluginLoader:
    """Loads plugins from project directories."""
    
    # Standard plugin directories in projects
    PLUGIN_DIRS = [
        "src/heads",
        "src/augmenters", 
        "src/features",
        "src/models",
        "src/metrics",
        "src/loaders",
        "components",
        "plugins",
    ]
    
    def __init__(self, project_root: Optional[Union[str, Path]] = None):
        """Initialize plugin loader.
        
        Args:
            project_root: Project root directory (defaults to current directory)
        """
        self.project_root = Path(project_root or os.getcwd())
        self.registry = get_registry()
        self._loaded_paths: List[Path] = []
    
    def load_project_plugins(self, override: bool = False) -> Dict[str, int]:
        """Load all plugins from the project.
        
        Args:
            override: Whether to override existing components
            
        Returns:
            Dictionary of plugin directory to number of components loaded
        """
        results = {}
        
        # Add project root to Python path
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))
        
        # Load from standard directories
        for plugin_dir in self.PLUGIN_DIRS:
            dir_path = self.project_root / plugin_dir
            if dir_path.exists() and dir_path.is_dir():
                count = self._load_from_directory(dir_path, override)
                if count > 0:
                    results[plugin_dir] = count
                    logger.info(f"Loaded {count} plugins from {plugin_dir}")
        
        # Also check for k-bert.plugins entry in pyproject.toml
        pyproject_path = self.project_root / "pyproject.toml"
        if pyproject_path.exists():
            plugin_paths = self._get_plugin_paths_from_pyproject(pyproject_path)
            for path in plugin_paths:
                full_path = self.project_root / path
                if full_path.exists():
                    count = self._load_from_directory(full_path, override)
                    if count > 0:
                        results[path] = count
        
        return results
    
    def load_from_path(
        self,
        path: Union[str, Path],
        override: bool = False
    ) -> int:
        """Load plugins from a specific path.
        
        Args:
            path: Path to load from (file or directory)
            override: Whether to override existing components
            
        Returns:
            Number of components loaded
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")
        
        if path.is_file():
            return self.registry.load_from_module(path, override)
        elif path.is_dir():
            return self._load_from_directory(path, override)
        else:
            raise ValueError(f"Invalid path type: {path}")
    
    def _load_from_directory(
        self,
        directory: Path,
        override: bool = False
    ) -> int:
        """Load plugins from a directory.
        
        Args:
            directory: Directory to load from
            override: Whether to override existing components
            
        Returns:
            Number of components loaded
        """
        # Skip if already loaded
        if directory in self._loaded_paths and not override:
            return 0
        
        # Add to Python path if needed
        if str(directory.parent) not in sys.path:
            sys.path.insert(0, str(directory.parent))
        
        count = self.registry.load_from_directory(directory, recursive=True, override=override)
        self._loaded_paths.append(directory)
        
        return count
    
    def _get_plugin_paths_from_pyproject(self, pyproject_path: Path) -> List[str]:
        """Extract plugin paths from pyproject.toml.
        
        Args:
            pyproject_path: Path to pyproject.toml
            
        Returns:
            List of plugin paths
        """
        try:
            import toml
            
            with open(pyproject_path) as f:
                data = toml.load(f)
            
            # Look for k-bert plugin configuration
            k_bert_config = data.get("tool", {}).get("k-bert", {})
            plugin_paths = k_bert_config.get("plugins", [])
            
            if isinstance(plugin_paths, str):
                plugin_paths = [plugin_paths]
            
            return plugin_paths
        
        except Exception as e:
            logger.debug(f"Could not load plugin paths from pyproject.toml: {e}")
            return []
    
    def get_loaded_components(self) -> Dict[str, List[str]]:
        """Get all loaded components.
        
        Returns:
            Dictionary of component type to list of names
        """
        return self.registry.list_components()
    
    def reload_all(self, override: bool = True) -> Dict[str, int]:
        """Reload all plugins from the project.
        
        Args:
            override: Whether to override existing components
            
        Returns:
            Dictionary of plugin directory to number of components loaded
        """
        # Clear loaded paths
        self._loaded_paths.clear()
        
        # Reload
        return self.load_project_plugins(override=override)


def load_project_plugins(
    project_root: Optional[Union[str, Path]] = None,
    override: bool = False
) -> Dict[str, int]:
    """Convenience function to load plugins from a project.
    
    Args:
        project_root: Project root directory
        override: Whether to override existing components
        
    Returns:
        Dictionary of plugin directory to number of components loaded
    """
    loader = PluginLoader(project_root)
    return loader.load_project_plugins(override)


def ensure_project_plugins_loaded(project_root: Optional[Union[str, Path]] = None) -> None:
    """Ensure project plugins are loaded exactly once.
    
    This function is idempotent and can be called multiple times safely.
    
    Args:
        project_root: Project root directory
    """
    # Use a marker to track if plugins are loaded
    marker = "_k_bert_plugins_loaded"
    project_root = Path(project_root or os.getcwd())
    
    # Check if already loaded for this project
    if hasattr(ensure_project_plugins_loaded, marker):
        loaded_roots = getattr(ensure_project_plugins_loaded, marker)
        if project_root in loaded_roots:
            return
    else:
        setattr(ensure_project_plugins_loaded, marker, set())
    
    # Load plugins
    results = load_project_plugins(project_root)
    
    # Mark as loaded
    getattr(ensure_project_plugins_loaded, marker).add(project_root)
    
    # Log summary
    total = sum(results.values())
    if total > 0:
        logger.info(f"Loaded {total} plugins from project at {project_root}")