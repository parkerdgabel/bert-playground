"""Base classes and utilities for CLI adapters.

This module provides common functionality for all CLI adapters,
ensuring consistency in error handling, output formatting, and
argument processing.
"""

from typing import Any, Dict, Optional, TypeVar, Generic
from abc import ABC, abstractmethod
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from pydantic import BaseModel


T = TypeVar('T', bound=BaseModel)
R = TypeVar('R', bound=BaseModel)


class CLIAdapter(ABC, Generic[T, R]):
    """Base class for CLI adapters.
    
    Provides common functionality for:
    - Configuration loading
    - Error handling
    - Output formatting
    - Progress indication
    
    Type parameters:
    - T: Request DTO type
    - R: Response DTO type
    """
    
    def __init__(self, console: Optional[Console] = None):
        """Initialize the CLI adapter.
        
        Args:
            console: Rich console for output (creates default if not provided)
        """
        self.console = console or Console()
    
    @abstractmethod
    def create_request_dto(self, **kwargs) -> T:
        """Create request DTO from CLI arguments.
        
        This method must be implemented by subclasses to convert
        CLI arguments into the appropriate request DTO.
        """
        pass
    
    @abstractmethod
    def display_results(self, response: R) -> None:
        """Display results to the user.
        
        This method must be implemented by subclasses to format
        and display the response DTO.
        """
        pass
    
    def display_error(self, error: Exception, debug: bool = False) -> None:
        """Display error message in a consistent format.
        
        Args:
            error: The exception to display
            debug: Whether to show full traceback
        """
        self.console.print(f"\n[red]Error: {error}[/red]")
        
        if debug:
            import traceback
            self.console.print("\n[dim]Traceback:[/dim]")
            traceback.print_exc()
    
    def display_config_table(self, title: str, config: Dict[str, Any]) -> None:
        """Display configuration in a formatted table.
        
        Args:
            title: Table title
            config: Configuration dictionary to display
        """
        table = Table(title=title, show_header=True)
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in config.items():
            # Format values nicely
            if isinstance(value, Path):
                value_str = str(value)
            elif isinstance(value, float):
                value_str = f"{value:.2e}" if value < 0.01 else f"{value:.4f}"
            elif value is None:
                value_str = "Not set"
            else:
                value_str = str(value)
            
            # Convert snake_case to Title Case
            key_display = key.replace('_', ' ').title()
            table.add_row(key_display, value_str)
        
        self.console.print(table)
    
    def confirm_action(self, message: str, default: bool = False) -> bool:
        """Ask user for confirmation.
        
        Args:
            message: Confirmation message
            default: Default response if user just presses Enter
            
        Returns:
            User's response
        """
        return typer.confirm(message, default=default)


class ProgressContext:
    """Context manager for displaying progress.
    
    Provides a simple way to show progress for long-running operations.
    """
    
    def __init__(self, console: Console, message: str):
        """Initialize progress context.
        
        Args:
            console: Rich console for output
            message: Progress message to display
        """
        self.console = console
        self.message = message
        self._progress = None
        self._task = None
    
    def __enter__(self):
        """Enter the progress context."""
        from rich.progress import Progress, SpinnerColumn, TextColumn
        
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True,
        )
        self._progress.__enter__()
        self._task = self._progress.add_task(self.message, total=None)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the progress context."""
        if self._progress:
            self._progress.__exit__(exc_type, exc_val, exc_tb)
    
    def update(self, message: str):
        """Update the progress message."""
        if self._progress and self._task is not None:
            self._progress.update(self._task, description=message)


def format_time(seconds: float) -> str:
    """Format time duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_size(bytes: int) -> str:
    """Format byte size in human-readable format.
    
    Args:
        bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024.0:
            return f"{bytes:.1f}{unit}"
        bytes /= 1024.0
    return f"{bytes:.1f}PB"