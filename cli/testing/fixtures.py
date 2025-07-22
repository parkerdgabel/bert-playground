"""Test fixtures for CLI testing."""

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pytest
import yaml
from loguru import logger

from cli.config.manager import ConfigManager
from cli.testing.mocks import MockFileSystem, create_mock_config


class CLIFixture:
    """Base fixture for CLI testing."""
    
    def __init__(self, name: Optional[str] = None):
        """Initialize fixture."""
        self.name = name or self.__class__.__name__
        self._cleanup_callbacks: List[Callable] = []
    
    def setup(self) -> None:
        """Set up fixture."""
        pass
    
    def teardown(self) -> None:
        """Tear down fixture."""
        for callback in reversed(self._cleanup_callbacks):
            try:
                callback()
            except Exception as e:
                logger.warning(f"Cleanup callback failed: {e}")
        self._cleanup_callbacks.clear()
    
    def add_cleanup(self, callback: Callable) -> None:
        """Add cleanup callback."""
        self._cleanup_callbacks.append(callback)
    
    def __enter__(self):
        """Context manager entry."""
        self.setup()
        return self
    
    def __exit__(self, *args):
        """Context manager exit."""
        self.teardown()


class ConfigFixture(CLIFixture):
    """Fixture for configuration testing."""
    
    def __init__(
        self,
        config_data: Optional[Dict[str, Any]] = None,
        config_files: Optional[Dict[str, Dict[str, Any]]] = None,
        temp_dir: Optional[Path] = None
    ):
        """Initialize config fixture."""
        super().__init__()
        self.config_data = config_data or create_mock_config()
        self.config_files = config_files or {}
        self.temp_dir = temp_dir
        self.created_files: List[Path] = []
        self.manager: Optional[ConfigManager] = None
    
    def setup(self) -> None:
        """Set up config fixture."""
        if not self.temp_dir:
            self.temp_dir = Path(tempfile.mkdtemp())
            self.add_cleanup(lambda: self.temp_dir.rmdir())
        
        # Create config files
        for filename, data in self.config_files.items():
            config_path = self.temp_dir / filename
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, "w") as f:
                yaml.dump(data, f)
            
            self.created_files.append(config_path)
            self.add_cleanup(lambda p=config_path: p.unlink())
        
        # Create config manager
        self.manager = ConfigManager()
        if self.created_files:
            # Load first config file as base
            self.manager.load_file(self.created_files[0])
    
    def get_config_path(self, filename: str) -> Optional[Path]:
        """Get path to config file."""
        for path in self.created_files:
            if path.name == filename:
                return path
        return None
    
    def update_config(self, **kwargs) -> None:
        """Update configuration."""
        if self.manager:
            for key, value in kwargs.items():
                self.manager.set(key, value)


class EnvironmentFixture(CLIFixture):
    """Fixture for environment variable testing."""
    
    def __init__(
        self,
        env_vars: Optional[Dict[str, str]] = None,
        preserve_existing: bool = True
    ):
        """Initialize environment fixture."""
        super().__init__()
        self.env_vars = env_vars or {}
        self.preserve_existing = preserve_existing
        self.original_env: Dict[str, Optional[str]] = {}
    
    def setup(self) -> None:
        """Set up environment."""
        for key, value in self.env_vars.items():
            if self.preserve_existing:
                self.original_env[key] = os.environ.get(key)
            os.environ[key] = value
        
        # Add cleanup
        self.add_cleanup(self._restore_env)
    
    def _restore_env(self) -> None:
        """Restore original environment."""
        for key, original_value in self.original_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value
    
    def set(self, key: str, value: str) -> None:
        """Set environment variable."""
        if key not in self.original_env:
            self.original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    def unset(self, key: str) -> None:
        """Unset environment variable."""
        if key not in self.original_env:
            self.original_env[key] = os.environ.get(key)
        os.environ.pop(key, None)


class FileSystemFixture(CLIFixture):
    """Fixture for filesystem testing."""
    
    def __init__(
        self,
        temp_dir: Optional[Path] = None,
        files: Optional[Dict[str, str]] = None,
        directories: Optional[List[str]] = None,
        use_real_fs: bool = True
    ):
        """Initialize filesystem fixture."""
        super().__init__()
        self.temp_dir = temp_dir
        self.files = files or {}
        self.directories = directories or []
        self.use_real_fs = use_real_fs
        self.created_paths: List[Path] = []
        self.mock_fs: Optional[MockFileSystem] = None
    
    def setup(self) -> None:
        """Set up filesystem."""
        if self.use_real_fs:
            self._setup_real_fs()
        else:
            self._setup_mock_fs()
    
    def _setup_real_fs(self) -> None:
        """Set up real filesystem."""
        if not self.temp_dir:
            self.temp_dir = Path(tempfile.mkdtemp())
            self.add_cleanup(self._cleanup_temp_dir)
        
        # Create directories
        for dir_path in self.directories:
            full_path = self.temp_dir / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            self.created_paths.append(full_path)
        
        # Create files
        for file_path, content in self.files.items():
            full_path = self.temp_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
            self.created_paths.append(full_path)
    
    def _setup_mock_fs(self) -> None:
        """Set up mock filesystem."""
        self.mock_fs = MockFileSystem()
        
        # Create directories
        for dir_path in self.directories:
            self.mock_fs.directories.add(dir_path)
        
        # Create files
        for file_path, content in self.files.items():
            self.mock_fs.write_file(file_path, content)
    
    def _cleanup_temp_dir(self) -> None:
        """Clean up temporary directory."""
        import shutil
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def create_file(self, path: str, content: str) -> Path:
        """Create a file."""
        if self.use_real_fs:
            full_path = self.temp_dir / path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
            self.created_paths.append(full_path)
            return full_path
        else:
            self.mock_fs.write_file(path, content)
            return Path(path)
    
    def create_directory(self, path: str) -> Path:
        """Create a directory."""
        if self.use_real_fs:
            full_path = self.temp_dir / path
            full_path.mkdir(parents=True, exist_ok=True)
            self.created_paths.append(full_path)
            return full_path
        else:
            self.mock_fs.directories.add(path)
            return Path(path)
    
    def get_path(self, path: str) -> Path:
        """Get full path."""
        if self.use_real_fs:
            return self.temp_dir / path
        else:
            return Path(path)


# Pytest fixtures
@pytest.fixture
def cli_runner():
    """CLI runner fixture."""
    from cli.testing.helpers import CLIRunner
    return CLIRunner()


@pytest.fixture
def mock_config():
    """Mock configuration fixture."""
    return create_mock_config()


@pytest.fixture
def config_fixture():
    """Configuration fixture."""
    with ConfigFixture() as fixture:
        yield fixture


@pytest.fixture
def env_fixture():
    """Environment fixture."""
    with EnvironmentFixture() as fixture:
        yield fixture


@pytest.fixture
def fs_fixture():
    """Filesystem fixture."""
    with FileSystemFixture() as fixture:
        yield fixture


@pytest.fixture
def temp_dir():
    """Temporary directory fixture."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def isolated_fs():
    """Isolated filesystem fixture."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        try:
            yield Path(temp_dir)
        finally:
            os.chdir(original_cwd)


@contextmanager
def isolated_cli_test(
    config: Optional[Dict[str, Any]] = None,
    env: Optional[Dict[str, str]] = None,
    files: Optional[Dict[str, str]] = None
) -> Iterator[Dict[str, Any]]:
    """Complete isolated CLI test context."""
    with FileSystemFixture(files=files) as fs_fixture, \
         EnvironmentFixture(env_vars=env) as env_fixture, \
         ConfigFixture(config_data=config) as config_fixture:
        
        yield {
            "fs": fs_fixture,
            "env": env_fixture,
            "config": config_fixture,
            "temp_dir": fs_fixture.temp_dir
        }