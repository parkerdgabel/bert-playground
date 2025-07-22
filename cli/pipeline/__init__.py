"""Command pipeline system for CLI.

Provides support for:
- Pre and post command hooks
- Command composition
- Pipeline execution
- Hook management
"""

from cli.pipeline.base import (
    CommandHook,
    CommandPipeline,
    HookPhase,
    PipelineContext,
)
from cli.pipeline.hooks import (
    ConfigurationHook,
    DependencyHook,
    EnvironmentHook,
    ResourceHook,
)
from cli.pipeline.composer import CommandComposer, CompositionStrategy

__all__ = [
    "CommandHook",
    "CommandPipeline",
    "HookPhase",
    "PipelineContext",
    "ConfigurationHook",
    "DependencyHook",
    "EnvironmentHook",
    "ResourceHook",
    "CommandComposer",
    "CompositionStrategy",
]