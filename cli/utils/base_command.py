"""Base command class with common functionality."""

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import typer
from loguru import logger

from .config import load_config
from .console import get_console, print_error


class BaseCommand(ABC):
    """Base class for all CLI commands with common functionality."""

    def __init__(self):
        self.console = get_console()
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging for the command."""
        # Remove default logger
        logger.remove()

        # Add console logger based on verbosity
        if os.environ.get("BERT_CLI_VERBOSE", "0") == "1":
            logger.add(sys.stderr, level="DEBUG")
        elif os.environ.get("BERT_CLI_QUIET", "0") == "1":
            logger.add(sys.stderr, level="ERROR")
        else:
            logger.add(sys.stderr, level="INFO")

    def load_config(self, config_path: Path | None = None) -> dict[str, Any]:
        """Load configuration with error handling."""
        try:
            return load_config(config_path)
        except Exception as e:
            print_error(f"Failed to load configuration: {str(e)}")
            raise typer.Exit(1)

    def resolve_path(self, path: Path) -> Path:
        """Resolve path relative to project root."""
        if path.is_absolute():
            return path

        # Try to find project root
        current = Path.cwd()
        while current != current.parent:
            if (current / "pyproject.toml").exists() or (current / ".git").exists():
                return current / path
            current = current.parent

        # Fall back to current directory
        return Path.cwd() / path

    def handle_error(self, error: Exception, message: str = "Command failed"):
        """Handle errors consistently."""
        logger.error(f"{message}: {str(error)}")
        if os.environ.get("BERT_CLI_VERBOSE", "0") == "1":
            logger.exception(error)
        print_error(str(error), title=message)
        raise typer.Exit(1)

    @abstractmethod
    def execute(self, **kwargs):
        """Execute the command. Must be implemented by subclasses."""
        pass


import os
