"""
Unified Rich Display Manager

This module provides a single source of truth for all Rich console displays,
preventing conflicts between multiple Live displays and Progress bars.
"""

import threading
import time
from typing import Dict, Optional, Any, Callable
from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text
from loguru import logger


class RichDisplayManager:
    """
    Unified display manager for all Rich console output.
    
    This class coordinates all Rich displays to prevent conflicts between
    multiple Live displays and Progress bars. It provides a single Live
    display that can show metrics, progress bars, and other information.
    """
    
    def __init__(self, console: Optional[Console] = None):
        """
        Initialize the display manager.
        
        Args:
            console: Rich console instance (creates new if None)
        """
        self.console = console or Console()
        self.lock = threading.Lock()
        self.live_display: Optional[Live] = None
        self.is_active = False
        
        # Display state
        self.metrics: Dict[str, Any] = {}
        self.progress_tasks: Dict[str, int] = {}
        self.progress_instance: Optional[Progress] = None
        self.status_text: str = "Ready"
        
        # Layout components
        self.layout = Layout()
        self._setup_layout()
        
        logger.info("Initialized RichDisplayManager")
    
    def _setup_layout(self):
        """Setup the display layout structure."""
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3)
        )
        
        # Split main section
        self.layout["main"].split_row(
            Layout(name="metrics", ratio=1),
            Layout(name="progress", ratio=2)
        )
    
    def start(self):
        """Start the unified display system."""
        with self.lock:
            if self.is_active:
                logger.warning("Display manager already active")
                return
            
            # Create progress instance
            self.progress_instance = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=self.console,
                transient=False,
                auto_refresh=True
            )
            
            # Update layout with initial content
            self._update_layout()
            
            # Start live display
            self.live_display = Live(
                self.layout,
                console=self.console,
                refresh_per_second=2,
                auto_refresh=True
            )
            
            self.live_display.start()
            self.is_active = True
            
            logger.info("Started RichDisplayManager")
    
    def stop(self):
        """Stop the display system and cleanup."""
        with self.lock:
            if not self.is_active:
                return
            
            try:
                if self.live_display:
                    self.live_display.stop()
                    self.live_display = None
                
                if self.progress_instance:
                    self.progress_instance.stop()
                    self.progress_instance = None
                
                self.is_active = False
                logger.info("Stopped RichDisplayManager")
                
            except Exception as e:
                logger.error(f"Error stopping display manager: {e}")
    
    def _update_layout(self):
        """Update the layout with current state."""
        # Header
        self.layout["header"].update(
            Panel(
                Text(self.status_text, style="bold blue"),
                title="Status",
                border_style="blue"
            )
        )
        
        # Metrics panel
        metrics_table = self._create_metrics_table()
        self.layout["metrics"].update(
            Panel(
                metrics_table,
                title="Metrics",
                border_style="green"
            )
        )
        
        # Progress panel
        if self.progress_instance:
            self.layout["progress"].update(
                Panel(
                    self.progress_instance,
                    title="Progress",
                    border_style="yellow"
                )
            )
        else:
            self.layout["progress"].update(
                Panel(
                    Text("No active progress", style="dim"),
                    title="Progress",
                    border_style="yellow"
                )
            )
        
        # Footer
        self.layout["footer"].update(
            Panel(
                Text(f"MLX BERT Training - {time.strftime('%H:%M:%S')}", style="dim"),
                border_style="dim"
            )
        )
    
    def _create_metrics_table(self) -> Table:
        """Create a table showing current metrics."""
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")
        
        if not self.metrics:
            table.add_row("No metrics", "Available")
        else:
            for key, value in self.metrics.items():
                # Format value based on type
                if isinstance(value, float):
                    formatted_value = f"{value:.4f}"
                elif isinstance(value, int):
                    formatted_value = str(value)
                else:
                    formatted_value = str(value)
                
                table.add_row(key, formatted_value)
        
        return table
    
    def update_metrics(self, new_metrics: Dict[str, Any]):
        """Update the metrics display."""
        with self.lock:
            self.metrics.update(new_metrics)
            if self.is_active:
                self._update_layout()
    
    def update_status(self, status: str):
        """Update the status text."""
        with self.lock:
            self.status_text = status
            if self.is_active:
                self._update_layout()
    
    def create_progress_task(
        self, 
        task_id: str, 
        description: str, 
        total: int
    ) -> Optional[int]:
        """
        Create a new progress task.
        
        Args:
            task_id: Unique identifier for the task
            description: Task description
            total: Total number of steps
            
        Returns:
            Task ID for Rich Progress (None if not active)
        """
        with self.lock:
            if not self.is_active or not self.progress_instance:
                return None
            
            # Create task in progress instance
            rich_task_id = self.progress_instance.add_task(
                description=description,
                total=total
            )
            
            # Store mapping
            self.progress_tasks[task_id] = rich_task_id
            
            # Update layout
            self._update_layout()
            
            return rich_task_id
    
    def update_progress_task(
        self, 
        task_id: str, 
        advance: int = 1, 
        description: Optional[str] = None
    ):
        """Update a progress task."""
        with self.lock:
            if not self.is_active or not self.progress_instance:
                return
            
            if task_id not in self.progress_tasks:
                return
            
            rich_task_id = self.progress_tasks[task_id]
            
            # Update progress
            self.progress_instance.update(
                rich_task_id,
                advance=advance,
                description=description
            )
            
            # Update layout to reflect progress changes
            self._update_layout()
    
    def remove_progress_task(self, task_id: str):
        """Remove a progress task."""
        with self.lock:
            if not self.is_active or not self.progress_instance:
                return
            
            if task_id in self.progress_tasks:
                rich_task_id = self.progress_tasks[task_id]
                self.progress_instance.remove_task(rich_task_id)
                del self.progress_tasks[task_id]
                
                # Update layout
                self._update_layout()
    
    def print_message(self, message: str, style: str = ""):
        """Print a message without interfering with displays."""
        with self.lock:
            if self.is_active and self.live_display:
                # Use console.print which works with Live displays
                self.console.print(message, style=style)
            else:
                # Fallback to regular print
                self.console.print(message, style=style)
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


# Singleton instance for global access
_display_manager: Optional[RichDisplayManager] = None


def get_display_manager(console: Optional[Console] = None) -> RichDisplayManager:
    """
    Get the global display manager instance.
    
    Args:
        console: Console instance (used only for initialization)
        
    Returns:
        RichDisplayManager instance
    """
    global _display_manager
    
    if _display_manager is None:
        _display_manager = RichDisplayManager(console)
    
    return _display_manager


def cleanup_display_manager():
    """Cleanup the global display manager."""
    global _display_manager
    
    if _display_manager:
        _display_manager.stop()
        _display_manager = None