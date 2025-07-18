"""Decorators for CLI commands."""

import time
import functools
from typing import Callable, Any, Optional
import typer
from loguru import logger

from .console import get_console, print_error, print_info

def handle_errors(func: Callable) -> Callable:
    """Decorator to handle common errors in CLI commands."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            print_error("Operation cancelled by user", title="Cancelled")
            raise typer.Exit(130)  # Standard SIGINT exit code
        except FileNotFoundError as e:
            print_error(f"File not found: {e.filename}", title="File Error")
            raise typer.Exit(1)
        except PermissionError as e:
            print_error(f"Permission denied: {e.filename}", title="Permission Error")
            raise typer.Exit(1)
        except ImportError as e:
            print_error(
                f"Missing dependency: {str(e)}\n"
                f"Try: uv pip install -r requirements.txt",
                title="Import Error"
            )
            raise typer.Exit(1)
        except Exception as e:
            logger.exception("Unexpected error")
            print_error(
                f"Unexpected error: {type(e).__name__}: {str(e)}\n"
                f"Run with --verbose for full traceback",
                title="Error"
            )
            raise typer.Exit(1)
    
    return wrapper

def track_time(message: Optional[str] = None):
    """Decorator to track and display execution time."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            console = get_console()
            
            # Show starting message
            if message:
                console.print(f"[bold blue]{message}...[/bold blue]")
            else:
                console.print(f"[bold blue]Running {func.__name__}...[/bold blue]")
            
            try:
                result = func(*args, **kwargs)
                
                # Calculate elapsed time
                elapsed = time.time() - start_time
                
                # Format time
                if elapsed < 60:
                    time_str = f"{elapsed:.1f}s"
                elif elapsed < 3600:
                    minutes = int(elapsed // 60)
                    seconds = int(elapsed % 60)
                    time_str = f"{minutes}m {seconds}s"
                else:
                    hours = int(elapsed // 3600)
                    minutes = int((elapsed % 3600) // 60)
                    time_str = f"{hours}h {minutes}m"
                
                console.print(f"[bold green]✓ Completed in {time_str}[/bold green]")
                
                return result
                
            except Exception:
                elapsed = time.time() - start_time
                console.print(f"[bold red]✗ Failed after {elapsed:.1f}s[/bold red]")
                raise
        
        return wrapper
    return decorator

def require_auth(service: str = "kaggle"):
    """Decorator to check authentication before running command."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import os
            
            if service == "kaggle":
                # Check for Kaggle credentials
                kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
                if not os.path.exists(kaggle_json):
                    print_error(
                        "Kaggle credentials not found.\n"
                        "Please run: kaggle config set -n username -v YOUR_USERNAME\n"
                        "And: kaggle config set -n key -v YOUR_API_KEY\n"
                        "Or place your kaggle.json file in ~/.kaggle/",
                        title="Authentication Required"
                    )
                    raise typer.Exit(1)
            
            elif service == "mlflow":
                # Check MLflow tracking URI if needed
                tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
                if tracking_uri and tracking_uri.startswith("http"):
                    # Could check MLflow server connectivity here
                    pass
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

def confirm_action(message: str = "Are you sure you want to continue?"):
    """Decorator to confirm destructive actions."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            console = get_console()
            
            # Skip confirmation in non-interactive mode
            import sys
            if not sys.stdin.isatty():
                return func(*args, **kwargs)
            
            # Check for --yes flag in kwargs
            if kwargs.get('yes', False):
                return func(*args, **kwargs)
            
            # Ask for confirmation
            if not typer.confirm(message):
                print_info("Operation cancelled")
                raise typer.Exit(0)
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

def requires_project():
    """Decorator to ensure command is run within a BERT project."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            from pathlib import Path
            
            # Check for project markers
            markers = [
                "pyproject.toml",
                "bert.yaml",
                "bert.yml",
                ".bertrc",
                "configs/",
            ]
            
            current = Path.cwd()
            found = False
            
            while current != current.parent:
                if any((current / marker).exists() for marker in markers):
                    found = True
                    break
                current = current.parent
            
            if not found:
                print_error(
                    "This command must be run within a BERT project.\n"
                    "Run 'bert init' to create a new project.",
                    title="Not in Project"
                )
                raise typer.Exit(1)
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator