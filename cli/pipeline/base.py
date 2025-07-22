"""Base classes for command pipeline system."""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from loguru import logger


class HookPhase(Enum):
    """Command hook execution phases."""
    
    PRE_PARSE = "pre_parse"
    POST_PARSE = "post_parse"
    PRE_VALIDATE = "pre_validate"
    POST_VALIDATE = "post_validate"
    PRE_EXECUTE = "pre_execute"
    POST_EXECUTE = "post_execute"
    PRE_CLEANUP = "pre_cleanup"
    POST_CLEANUP = "post_cleanup"


@dataclass
class PipelineContext:
    """Context for pipeline execution."""
    
    command: str
    args: tuple
    kwargs: Dict[str, Any]
    state: Dict[str, Any] = field(default_factory=dict)
    results: List[Any] = field(default_factory=list)
    errors: List[Exception] = field(default_factory=list)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get state value."""
        return self.state.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set state value."""
        self.state[key] = value
    
    def add_result(self, result: Any) -> None:
        """Add result to context."""
        self.results.append(result)
    
    def add_error(self, error: Exception) -> None:
        """Add error to context."""
        self.errors.append(error)
    
    @property
    def has_errors(self) -> bool:
        """Check if context has errors."""
        return len(self.errors) > 0
    
    @property
    def last_result(self) -> Any:
        """Get last result."""
        return self.results[-1] if self.results else None


class CommandHook(ABC):
    """Base class for command hooks."""
    
    def __init__(
        self,
        name: Optional[str] = None,
        phases: Optional[Set[HookPhase]] = None,
        priority: int = 50
    ):
        """Initialize hook.
        
        Args:
            name: Hook name
            phases: Phases to execute in
            priority: Execution priority (lower executes first)
        """
        self.name = name or self.__class__.__name__
        self.phases = phases or {HookPhase.PRE_EXECUTE}
        self.priority = priority
    
    @abstractmethod
    async def execute(self, phase: HookPhase, context: PipelineContext) -> None:
        """Execute hook logic.
        
        Args:
            phase: Current execution phase
            context: Pipeline context
        """
        pass
    
    def should_execute(self, phase: HookPhase) -> bool:
        """Check if hook should execute in phase."""
        return phase in self.phases
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name={self.name}, phases={[p.value for p in self.phases]})"


class CommandPipeline:
    """Manages command execution pipeline with hooks."""
    
    def __init__(self, name: Optional[str] = None):
        """Initialize pipeline."""
        self.name = name or "CommandPipeline"
        self.hooks: List[CommandHook] = []
        self._hook_cache: Dict[HookPhase, List[CommandHook]] = {}
    
    def add_hook(self, hook: CommandHook) -> "CommandPipeline":
        """Add hook to pipeline."""
        self.hooks.append(hook)
        self._invalidate_cache()
        logger.debug(f"Added hook {hook.name} to pipeline")
        return self
    
    def remove_hook(self, hook: CommandHook) -> "CommandPipeline":
        """Remove hook from pipeline."""
        self.hooks.remove(hook)
        self._invalidate_cache()
        return self
    
    def _invalidate_cache(self) -> None:
        """Invalidate hook cache."""
        self._hook_cache.clear()
    
    def _get_hooks_for_phase(self, phase: HookPhase) -> List[CommandHook]:
        """Get hooks for specific phase."""
        if phase not in self._hook_cache:
            phase_hooks = [h for h in self.hooks if h.should_execute(phase)]
            phase_hooks.sort(key=lambda h: h.priority)
            self._hook_cache[phase] = phase_hooks
        return self._hook_cache[phase]
    
    async def _execute_phase(self, phase: HookPhase, context: PipelineContext) -> None:
        """Execute all hooks for a phase."""
        hooks = self._get_hooks_for_phase(phase)
        
        for hook in hooks:
            try:
                logger.debug(f"Executing hook {hook.name} in phase {phase.value}")
                await hook.execute(phase, context)
            except Exception as e:
                logger.error(f"Hook {hook.name} failed in phase {phase.value}: {e}")
                context.add_error(e)
                # Continue with other hooks
    
    async def execute(
        self,
        command: str,
        handler: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute command through pipeline.
        
        Args:
            command: Command name
            handler: Command handler function
            *args: Command arguments
            **kwargs: Command keyword arguments
            
        Returns:
            Command result
        """
        # Create context
        context = PipelineContext(
            command=command,
            args=args,
            kwargs=kwargs
        )
        
        try:
            # Pre-parse phase
            await self._execute_phase(HookPhase.PRE_PARSE, context)
            
            # Post-parse phase
            await self._execute_phase(HookPhase.POST_PARSE, context)
            
            # Pre-validate phase
            await self._execute_phase(HookPhase.PRE_VALIDATE, context)
            
            # Post-validate phase
            await self._execute_phase(HookPhase.POST_VALIDATE, context)
            
            # Check for validation errors
            if context.has_errors:
                raise ValueError(f"Validation failed: {context.errors}")
            
            # Pre-execute phase
            await self._execute_phase(HookPhase.PRE_EXECUTE, context)
            
            # Execute command
            if asyncio.iscoroutinefunction(handler):
                result = await handler(*context.args, **context.kwargs)
            else:
                result = handler(*context.args, **context.kwargs)
            
            context.add_result(result)
            
            # Post-execute phase
            await self._execute_phase(HookPhase.POST_EXECUTE, context)
            
            return result
            
        except Exception as e:
            logger.exception(f"Pipeline execution failed for {command}")
            context.add_error(e)
            raise
        finally:
            # Cleanup phases
            await self._execute_phase(HookPhase.PRE_CLEANUP, context)
            await self._execute_phase(HookPhase.POST_CLEANUP, context)
    
    def compose(self, *commands: Callable) -> Callable:
        """Compose multiple commands into a pipeline.
        
        Args:
            *commands: Commands to compose
            
        Returns:
            Composed command function
        """
        async def composed_command(*args, **kwargs):
            """Execute composed commands."""
            results = []
            
            for i, cmd in enumerate(commands):
                # Use previous result as input if available
                if i > 0 and results:
                    # Pass previous result as first argument
                    cmd_args = (results[-1], *args[1:])
                else:
                    cmd_args = args
                
                # Execute through pipeline
                result = await self.execute(
                    f"composed_{cmd.__name__}",
                    cmd,
                    *cmd_args,
                    **kwargs
                )
                results.append(result)
            
            return results[-1] if results else None
        
        return composed_command
    
    def __len__(self) -> int:
        """Get hook count."""
        return len(self.hooks)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"CommandPipeline(name={self.name}, hooks={len(self.hooks)})"