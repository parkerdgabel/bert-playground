"""Console utilities for rich output."""

import os
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.syntax import Syntax
from typing import Optional, Any

# Global console instance
_console: Optional[Console] = None

def get_console() -> Console:
    """Get or create the global console instance."""
    global _console
    if _console is None:
        # Check for quiet mode
        quiet = os.environ.get("BERT_CLI_QUIET", "0") == "1"
        _console = Console(quiet=quiet)
    return _console

def print_error(message: str, title: str = "Error"):
    """Print an error message."""
    console = get_console()
    console.print(Panel(
        f"[bold red]{message}[/bold red]",
        title=f"[bold red]{title}[/bold red]",
        border_style="red"
    ))

def print_success(message: str, title: str = "Success"):
    """Print a success message."""
    console = get_console()
    console.print(Panel(
        f"[bold green]{message}[/bold green]",
        title=f"[bold green]{title}[/bold green]",
        border_style="green"
    ))

def print_warning(message: str, title: str = "Warning"):
    """Print a warning message."""
    console = get_console()
    console.print(Panel(
        f"[bold yellow]{message}[/bold yellow]",
        title=f"[bold yellow]{title}[/bold yellow]",
        border_style="yellow"
    ))

def print_info(message: str, title: str = "Info"):
    """Print an info message."""
    console = get_console()
    console.print(Panel(
        f"[bold blue]{message}[/bold blue]",
        title=f"[bold blue]{title}[/bold blue]",
        border_style="blue"
    ))

def create_table(title: str, columns: list[str]) -> Table:
    """Create a formatted table."""
    table = Table(title=title, show_header=True, header_style="bold magenta")
    for col in columns:
        table.add_column(col)
    return table

def create_progress() -> Progress:
    """Create a progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=get_console()
    )

def print_code(code: str, language: str = "python", title: Optional[str] = None):
    """Print syntax-highlighted code."""
    console = get_console()
    syntax = Syntax(code, language, theme="monokai", line_numbers=True)
    if title:
        console.print(Panel(syntax, title=title))
    else:
        console.print(syntax)

def confirm(message: str, default: bool = False) -> bool:
    """Ask for user confirmation."""
    console = get_console()
    return console.input(f"{message} ({'Y/n' if default else 'y/N'}): ").lower() in (
        ['y', 'yes'] if not default else ['y', 'yes', '']
    )

def format_bytes(size: int) -> str:
    """Format byte size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"

def format_timestamp(timestamp: float) -> str:
    """Format timestamp to human-readable string."""
    from datetime import datetime
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S")