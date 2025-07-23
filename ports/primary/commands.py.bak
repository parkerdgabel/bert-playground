"""Primary command port - Command execution API for external actors.

This port defines the command pattern interface that external actors use
to execute training operations. It's a driving port in hexagonal architecture.
"""

from typing import Any, Protocol, runtime_checkable

from core.protocols.models import Model
from core.protocols.data import DataLoader


@runtime_checkable
class CommandContext(Protocol):
    """Context for command execution that external actors provide."""

    # Core components
    model: Model
    state: dict[str, Any]  # Training state

    # Optional components
    train_dataloader: DataLoader | None
    val_dataloader: DataLoader | None

    # Current execution state
    batch: dict[str, Any] | None
    batch_idx: int
    outputs: dict[str, Any]
    gradients: dict[str, Any]
    loss: float | None
    metrics: dict[str, float]

    # Control flags
    should_accumulate_gradients: bool
    should_update_weights: bool
    is_training: bool

    # Configuration
    config: dict[str, Any]


@runtime_checkable
class CommandResult(Protocol):
    """Result of command execution that external actors receive."""

    success: bool
    outputs: dict[str, Any]
    error: Exception | None
    metrics: dict[str, float]
    should_continue: bool
    should_skip_remaining: bool


@runtime_checkable
class Command(Protocol):
    """Command pattern for training operations.
    
    External actors can create and execute commands to perform
    specific training operations.
    """

    @property
    def name(self) -> str:
        """Command name for logging and debugging."""
        ...

    @property
    def requires_grad(self) -> bool:
        """Whether this command requires gradient computation."""
        ...

    def can_execute(self, context: CommandContext) -> bool:
        """Check if command can be executed given current context.
        
        Args:
            context: Current execution context
            
        Returns:
            True if command can execute
        """
        ...

    def execute(self, context: CommandContext) -> CommandResult:
        """Execute the command.
        
        Args:
            context: Execution context
            
        Returns:
            Command result
        """
        ...

    def rollback(self, context: CommandContext) -> None:
        """Rollback command effects if needed.
        
        Args:
            context: Execution context
        """
        ...


@runtime_checkable
class Pipeline(Protocol):
    """Pipeline of commands that external actors can execute."""

    @property
    def commands(self) -> list[Command]:
        """List of commands in the pipeline."""
        ...

    @property
    def name(self) -> str:
        """Pipeline name."""
        ...

    @property
    def stop_on_error(self) -> bool:
        """Whether to stop on first error."""
        ...

    def execute(self, context: CommandContext) -> CommandResult:
        """Execute the pipeline.
        
        Args:
            context: Execution context
            
        Returns:
            Pipeline execution result
        """
        ...

    def add_command(self, command: Command) -> None:
        """Add a command to the pipeline.
        
        Args:
            command: Command to add
        """
        ...

    def remove_command(self, command_name: str) -> None:
        """Remove a command from the pipeline.
        
        Args:
            command_name: Name of command to remove
        """
        ...


# Convenience functions for external actors
def execute_command(
    command: Command,
    context: CommandContext,
) -> CommandResult:
    """Execute a single command.
    
    This is a convenience function that external actors can use
    to execute individual commands.
    
    Args:
        command: Command to execute
        context: Execution context
        
    Returns:
        Command result
    """
    # This would be implemented by the application core
    raise NotImplementedError("To be implemented by application core")


def create_pipeline(
    commands: list[Command],
    name: str = "pipeline",
    stop_on_error: bool = True,
) -> Pipeline:
    """Create a command pipeline.
    
    This is a convenience function that external actors can use
    to create pipelines of commands.
    
    Args:
        commands: List of commands
        name: Pipeline name
        stop_on_error: Whether to stop on first error
        
    Returns:
        Configured pipeline
    """
    # This would be implemented by the application core
    raise NotImplementedError("To be implemented by application core")