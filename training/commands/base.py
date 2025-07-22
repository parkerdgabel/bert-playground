"""Base command protocol and context for training operations.

This module defines the core abstractions for the command pattern,
abstracting away framework-specific details (MLX) behind clean interfaces.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from core.protocols.data import DataLoader
from core.protocols.models import Model
from core.protocols.training import (
    Optimizer,
    TrainingState,
    LRScheduler,
    MetricsCollector,
    CheckpointManager,
)


@dataclass
class CommandContext:
    """Context object passed to commands containing all training state.
    
    This context abstracts away MLX-specific types, using protocols instead.
    """
    
    # Core components
    model: Model
    optimizer: Optimizer
    state: TrainingState
    
    # Optional components
    train_dataloader: DataLoader | None = None
    val_dataloader: DataLoader | None = None
    lr_scheduler: LRScheduler | None = None
    metrics_collector: MetricsCollector | None = None
    checkpoint_manager: CheckpointManager | None = None
    
    # Current batch data (framework-agnostic)
    batch: dict[str, Any] | None = None
    batch_idx: int = 0
    
    # Computed values from previous commands
    outputs: dict[str, Any] = field(default_factory=dict)
    gradients: dict[str, Any] = field(default_factory=dict)
    loss: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    
    # Control flags
    should_accumulate_gradients: bool = False
    should_update_weights: bool = True
    is_training: bool = True
    
    # Configuration
    config: dict[str, Any] = field(default_factory=dict)
    
    def update(self, **kwargs) -> None:
        """Update context fields."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                # Store unknown fields in outputs
                self.outputs[key] = value


@dataclass
class CommandResult:
    """Result of a command execution."""
    
    success: bool
    outputs: dict[str, Any] = field(default_factory=dict)
    error: Exception | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    
    # Control flow hints
    should_continue: bool = True
    should_skip_remaining: bool = False
    
    def merge(self, other: "CommandResult") -> "CommandResult":
        """Merge another result into this one."""
        self.outputs.update(other.outputs)
        self.metrics.update(other.metrics)
        self.success = self.success and other.success
        self.should_continue = self.should_continue and other.should_continue
        self.should_skip_remaining = self.should_skip_remaining or other.should_skip_remaining
        if other.error and not self.error:
            self.error = other.error
        return self


@runtime_checkable
class Command(Protocol):
    """Protocol for training commands."""
    
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
            context: Current training context
            
        Returns:
            True if command can execute, False otherwise
        """
        ...
    
    def execute(self, context: CommandContext) -> CommandResult:
        """Execute the command.
        
        Args:
            context: Training context
            
        Returns:
            CommandResult with outputs and status
        """
        ...
    
    def rollback(self, context: CommandContext) -> None:
        """Rollback command effects if needed.
        
        Args:
            context: Training context
        """
        ...


class BaseCommand(ABC):
    """Base implementation of Command with common functionality."""
    
    def __init__(self, name: str | None = None):
        """Initialize command.
        
        Args:
            name: Optional command name, defaults to class name
        """
        self._name = name or self.__class__.__name__
        self._requires_grad = False
    
    @property
    def name(self) -> str:
        """Command name."""
        return self._name
    
    @property
    def requires_grad(self) -> bool:
        """Whether command requires gradients."""
        return self._requires_grad
    
    def can_execute(self, context: CommandContext) -> bool:
        """Default implementation - always executable."""
        return True
    
    @abstractmethod
    def execute(self, context: CommandContext) -> CommandResult:
        """Execute the command."""
        pass
    
    def rollback(self, context: CommandContext) -> None:
        """Default implementation - no rollback needed."""
        pass


class CompositeCommand(BaseCommand):
    """Command that executes multiple sub-commands."""
    
    def __init__(self, commands: list[Command], name: str = "CompositeCommand"):
        """Initialize composite command.
        
        Args:
            commands: List of commands to execute
            name: Command name
        """
        super().__init__(name)
        self.commands = commands
        self._requires_grad = any(cmd.requires_grad for cmd in commands)
    
    def can_execute(self, context: CommandContext) -> bool:
        """Check if all sub-commands can execute."""
        return all(cmd.can_execute(context) for cmd in self.commands)
    
    def execute(self, context: CommandContext) -> CommandResult:
        """Execute all sub-commands in order."""
        result = CommandResult(success=True)
        executed_commands = []
        
        try:
            for command in self.commands:
                if not command.can_execute(context):
                    continue
                    
                cmd_result = command.execute(context)
                result.merge(cmd_result)
                executed_commands.append(command)
                
                if not cmd_result.success or cmd_result.should_skip_remaining:
                    break
                    
            return result
            
        except Exception as e:
            # Rollback in reverse order
            for cmd in reversed(executed_commands):
                try:
                    cmd.rollback(context)
                except Exception:
                    pass  # Best effort rollback
                    
            return CommandResult(success=False, error=e)
    
    def rollback(self, context: CommandContext) -> None:
        """Rollback all sub-commands in reverse order."""
        for command in reversed(self.commands):
            command.rollback(context)