"""Configuration loader component for k-bert CLI.

This module handles loading configuration from various file formats.
"""

import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol
from abc import abstractmethod

from loguru import logger


class ConfigLoaderProtocol(Protocol):
    """Protocol for configuration loaders."""
    
    @abstractmethod
    def load_file(self, path: Path) -> Dict[str, Any]:
        """Load configuration from a file.
        
        Args:
            path: Path to configuration file
            
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is unsupported
        """
        ...
    
    @abstractmethod
    def find_config_file(self, search_paths: List[Path]) -> Optional[Path]:
        """Find first existing configuration file from search paths.
        
        Args:
            search_paths: List of paths to search
            
        Returns:
            Path to first found config file, or None
        """
        ...


class ConfigLoader:
    """Handles loading configuration from various file formats."""
    
    SUPPORTED_EXTENSIONS = {".yaml", ".yml", ".json", ".toml"}
    
    def load_file(self, path: Path) -> Dict[str, Any]:
        """Load configuration from a file.
        
        Args:
            path: Path to configuration file
            
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is unsupported
        """
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        logger.debug(f"Loading configuration from: {path}")
        
        try:
            if path.suffix in [".yaml", ".yml"]:
                return self._load_yaml(path)
            elif path.suffix == ".json":
                return self._load_json(path)
            elif path.name == "pyproject.toml" or path.suffix == ".toml":
                return self._load_toml(path)
            else:
                raise ValueError(f"Unsupported configuration format: {path.suffix}")
        except Exception as e:
            logger.error(f"Failed to load configuration from {path}: {e}")
            raise
    
    def find_config_file(self, search_paths: List[Path]) -> Optional[Path]:
        """Find first existing configuration file from search paths.
        
        Args:
            search_paths: List of paths to search
            
        Returns:
            Path to first found config file, or None
        """
        for path in search_paths:
            if path.exists():
                logger.debug(f"Found configuration file: {path}")
                return path
        
        logger.debug("No configuration file found in search paths")
        return None
    
    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load YAML configuration file.
        
        Args:
            path: Path to YAML file
            
        Returns:
            Configuration dictionary
        """
        with open(path, "r") as f:
            content = yaml.safe_load(f)
            return content if content is not None else {}
    
    def _load_json(self, path: Path) -> Dict[str, Any]:
        """Load JSON configuration file.
        
        Args:
            path: Path to JSON file
            
        Returns:
            Configuration dictionary
        """
        with open(path, "r") as f:
            return json.load(f)
    
    def _load_toml(self, path: Path) -> Dict[str, Any]:
        """Load TOML configuration file.
        
        Args:
            path: Path to TOML file
            
        Returns:
            Configuration dictionary
        """
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        
        with open(path, "rb") as f:
            data = tomllib.load(f)
        
        # Handle pyproject.toml special case
        if path.name == "pyproject.toml":
            if "tool" in data and "k-bert" in data["tool"]:
                return data["tool"]["k-bert"]
            else:
                logger.warning(f"No [tool.k-bert] section found in {path}")
                return {}
        
        return data


class CachedConfigLoader:
    """Configuration loader with caching support."""
    
    def __init__(self, loader: ConfigLoaderProtocol):
        """Initialize cached loader.
        
        Args:
            loader: Base configuration loader
        """
        self._loader = loader
        self._cache: Dict[Path, Dict[str, Any]] = {}
    
    def load_file(self, path: Path) -> Dict[str, Any]:
        """Load configuration with caching.
        
        Args:
            path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        # Resolve path to handle symlinks
        resolved_path = path.resolve()
        
        if resolved_path in self._cache:
            logger.debug(f"Using cached configuration for: {path}")
            return self._cache[resolved_path]
        
        config = self._loader.load_file(path)
        self._cache[resolved_path] = config
        return config
    
    def find_config_file(self, search_paths: List[Path]) -> Optional[Path]:
        """Find configuration file (delegates to base loader).
        
        Args:
            search_paths: List of paths to search
            
        Returns:
            Path to first found config file, or None
        """
        return self._loader.find_config_file(search_paths)
    
    def clear_cache(self) -> None:
        """Clear the configuration cache."""
        self._cache.clear()
        logger.debug("Configuration cache cleared")