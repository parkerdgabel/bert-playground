"""Simplified CLI Bootstrap for testing.

This is a temporary bootstrap that doesn't require all adapters to be present.
"""

from pathlib import Path
from typing import Optional, Dict, Any

from infrastructure.bootstrap import ApplicationBootstrap, get_bootstrap


class SimpleCLIBootstrap:
    """Simplified bootstrap for CLI testing."""
    
    def __init__(self):
        """Initialize simple bootstrap."""
        self.bootstrap: Optional[ApplicationBootstrap] = None
        self._initialized = False
        
    def initialize(
        self,
        config_path: Optional[Path] = None,
        user_config_path: Optional[Path] = None,
        project_config_path: Optional[Path] = None,
    ) -> None:
        """Initialize with minimal dependencies."""
        if self._initialized:
            return
            
        # Use the infrastructure bootstrap
        self.bootstrap = get_bootstrap(
            config_path=config_path,
            user_config_path=user_config_path,
            project_config_path=project_config_path,
        )
        
        # Don't fully initialize - just set up config
        self.bootstrap._setup_configuration()
        self._initialized = True
        
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        if not self.bootstrap:
            return default
        return self.bootstrap.get_config(key, default)
        
    def shutdown(self) -> None:
        """Shutdown."""
        self._initialized = False


# Global instance
_simple_bootstrap: Optional[SimpleCLIBootstrap] = None


def get_simple_bootstrap() -> SimpleCLIBootstrap:
    """Get simple bootstrap instance."""
    global _simple_bootstrap
    if _simple_bootstrap is None:
        _simple_bootstrap = SimpleCLIBootstrap()
    return _simple_bootstrap


def initialize_cli_simple(
    config_path: Optional[Path] = None,
    user_config_path: Optional[Path] = None,
    project_config_path: Optional[Path] = None,
) -> None:
    """Initialize CLI in simple mode."""
    bootstrap = get_simple_bootstrap()
    bootstrap.initialize(
        config_path=config_path,
        user_config_path=user_config_path,
        project_config_path=project_path,
    )


def get_config_simple(key: str, default: Any = None) -> Any:
    """Get config value."""
    bootstrap = get_simple_bootstrap()
    return bootstrap.get_config(key, default)


def shutdown_cli_simple() -> None:
    """Shutdown CLI."""
    global _simple_bootstrap
    if _simple_bootstrap:
        _simple_bootstrap.shutdown()
        _simple_bootstrap = None