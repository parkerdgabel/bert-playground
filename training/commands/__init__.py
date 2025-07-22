"""Training commands implementing the Command pattern.

This module provides a command pattern implementation for training operations,
allowing for modular, testable, and extensible training logic.
"""

from .base import Command, CommandContext, CommandResult
from .forward import ForwardCommand
from .backward import BackwardCommand, MLXBackwardCommand
from .optimizer_step import OptimizerStepCommand, MLXOptimizerStepCommand
from .gradient_accumulation import GradientAccumulationCommand, MLXGradientAccumulationCommand
from .checkpoint import CheckpointCommand
from .evaluation import EvaluationCommand
from .logging import LoggingCommand

__all__ = [
    "Command",
    "CommandContext",
    "CommandResult",
    "ForwardCommand",
    "BackwardCommand",
    "MLXBackwardCommand",
    "OptimizerStepCommand",
    "MLXOptimizerStepCommand",
    "GradientAccumulationCommand",
    "MLXGradientAccumulationCommand",
    "CheckpointCommand",
    "EvaluationCommand",
    "LoggingCommand",
]