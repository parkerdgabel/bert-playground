"""Command composition utilities for pipeline."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from loguru import logger

from cli.pipeline.base import CommandPipeline, PipelineContext


class CompositionStrategy(Enum):
    """Strategies for command composition."""
    
    SEQUENTIAL = "sequential"  # Execute commands in sequence
    PARALLEL = "parallel"      # Execute commands in parallel
    CONDITIONAL = "conditional"  # Execute based on conditions
    FALLBACK = "fallback"      # Execute fallback on failure
    REDUCE = "reduce"          # Reduce results with function


class CommandComposer:
    """Composes multiple commands with different strategies."""
    
    def __init__(self, pipeline: Optional[CommandPipeline] = None):
        """Initialize composer."""
        self.pipeline = pipeline or CommandPipeline()
    
    def compose(
        self,
        *commands: Union[Callable, "ComposedCommand"],
        strategy: CompositionStrategy = CompositionStrategy.SEQUENTIAL,
        **options
    ) -> "ComposedCommand":
        """Compose commands with strategy.
        
        Args:
            *commands: Commands to compose
            strategy: Composition strategy
            **options: Strategy-specific options
            
        Returns:
            Composed command
        """
        if strategy == CompositionStrategy.SEQUENTIAL:
            return SequentialCommand(commands, self.pipeline, **options)
        elif strategy == CompositionStrategy.PARALLEL:
            return ParallelCommand(commands, self.pipeline, **options)
        elif strategy == CompositionStrategy.CONDITIONAL:
            return ConditionalCommand(commands, self.pipeline, **options)
        elif strategy == CompositionStrategy.FALLBACK:
            return FallbackCommand(commands, self.pipeline, **options)
        elif strategy == CompositionStrategy.REDUCE:
            return ReduceCommand(commands, self.pipeline, **options)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def sequential(self, *commands: Callable, **options) -> "ComposedCommand":
        """Create sequential composition."""
        return self.compose(*commands, strategy=CompositionStrategy.SEQUENTIAL, **options)
    
    def parallel(self, *commands: Callable, **options) -> "ComposedCommand":
        """Create parallel composition."""
        return self.compose(*commands, strategy=CompositionStrategy.PARALLEL, **options)
    
    def conditional(self, *commands: Callable, **options) -> "ComposedCommand":
        """Create conditional composition."""
        return self.compose(*commands, strategy=CompositionStrategy.CONDITIONAL, **options)
    
    def fallback(self, *commands: Callable, **options) -> "ComposedCommand":
        """Create fallback composition."""
        return self.compose(*commands, strategy=CompositionStrategy.FALLBACK, **options)
    
    def reduce(self, *commands: Callable, **options) -> "ComposedCommand":
        """Create reduce composition."""
        return self.compose(*commands, strategy=CompositionStrategy.REDUCE, **options)


class ComposedCommand(ABC):
    """Base class for composed commands."""
    
    def __init__(
        self,
        commands: List[Callable],
        pipeline: CommandPipeline,
        name: Optional[str] = None
    ):
        """Initialize composed command."""
        self.commands = commands
        self.pipeline = pipeline
        self.name = name or f"{self.__class__.__name__}_{id(self)}"
    
    @abstractmethod
    async def execute(self, *args, **kwargs) -> Any:
        """Execute composed command."""
        pass
    
    def __call__(self, *args, **kwargs) -> Any:
        """Make command callable."""
        import asyncio
        
        if asyncio.iscoroutinefunction(self.execute):
            # Handle async execution
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            return loop.run_until_complete(self.execute(*args, **kwargs))
        else:
            return self.execute(*args, **kwargs)
    
    def __repr__(self) -> str:
        """String representation."""
        cmd_names = [getattr(c, "__name__", str(c)) for c in self.commands]
        return f"{self.__class__.__name__}({cmd_names})"


class SequentialCommand(ComposedCommand):
    """Execute commands sequentially, piping results."""
    
    def __init__(
        self,
        commands: List[Callable],
        pipeline: CommandPipeline,
        pipe_results: bool = True,
        **kwargs
    ):
        """Initialize sequential command."""
        super().__init__(commands, pipeline, **kwargs)
        self.pipe_results = pipe_results
    
    async def execute(self, *args, **kwargs) -> Any:
        """Execute commands sequentially."""
        result = None
        current_args = args
        
        for i, command in enumerate(self.commands):
            logger.debug(f"Executing sequential command {i+1}/{len(self.commands)}: {command.__name__}")
            
            # Use previous result as input if piping
            if i > 0 and self.pipe_results and result is not None:
                current_args = (result,) + args[1:]
            
            # Execute through pipeline
            result = await self.pipeline.execute(
                f"{self.name}_step_{i}",
                command,
                *current_args,
                **kwargs
            )
        
        return result


class ParallelCommand(ComposedCommand):
    """Execute commands in parallel."""
    
    def __init__(
        self,
        commands: List[Callable],
        pipeline: CommandPipeline,
        max_concurrency: Optional[int] = None,
        return_exceptions: bool = False,
        **kwargs
    ):
        """Initialize parallel command."""
        super().__init__(commands, pipeline, **kwargs)
        self.max_concurrency = max_concurrency
        self.return_exceptions = return_exceptions
    
    async def execute(self, *args, **kwargs) -> List[Any]:
        """Execute commands in parallel."""
        import asyncio
        
        tasks = []
        for i, command in enumerate(self.commands):
            task = self.pipeline.execute(
                f"{self.name}_parallel_{i}",
                command,
                *args,
                **kwargs
            )
            tasks.append(task)
        
        # Execute with concurrency limit
        if self.max_concurrency:
            results = []
            for i in range(0, len(tasks), self.max_concurrency):
                batch = tasks[i:i + self.max_concurrency]
                batch_results = await asyncio.gather(
                    *batch,
                    return_exceptions=self.return_exceptions
                )
                results.extend(batch_results)
            return results
        else:
            return await asyncio.gather(
                *tasks,
                return_exceptions=self.return_exceptions
            )


class ConditionalCommand(ComposedCommand):
    """Execute commands based on conditions."""
    
    def __init__(
        self,
        commands: List[Callable],
        pipeline: CommandPipeline,
        conditions: Optional[List[Callable]] = None,
        else_command: Optional[Callable] = None,
        **kwargs
    ):
        """Initialize conditional command."""
        super().__init__(commands, pipeline, **kwargs)
        self.conditions = conditions or []
        self.else_command = else_command
    
    async def execute(self, *args, **kwargs) -> Any:
        """Execute based on conditions."""
        for i, (command, condition) in enumerate(zip(self.commands, self.conditions)):
            # Evaluate condition
            if callable(condition):
                should_execute = condition(*args, **kwargs)
            else:
                should_execute = bool(condition)
            
            if should_execute:
                logger.debug(f"Condition {i} met, executing: {command.__name__}")
                return await self.pipeline.execute(
                    f"{self.name}_conditional_{i}",
                    command,
                    *args,
                    **kwargs
                )
        
        # No conditions met, execute else command
        if self.else_command:
            logger.debug(f"No conditions met, executing else: {self.else_command.__name__}")
            return await self.pipeline.execute(
                f"{self.name}_else",
                self.else_command,
                *args,
                **kwargs
            )
        
        return None


class FallbackCommand(ComposedCommand):
    """Execute commands with fallback on failure."""
    
    def __init__(
        self,
        commands: List[Callable],
        pipeline: CommandPipeline,
        stop_on_success: bool = True,
        **kwargs
    ):
        """Initialize fallback command."""
        super().__init__(commands, pipeline, **kwargs)
        self.stop_on_success = stop_on_success
    
    async def execute(self, *args, **kwargs) -> Any:
        """Execute with fallback."""
        last_error = None
        
        for i, command in enumerate(self.commands):
            try:
                logger.debug(f"Trying command {i+1}/{len(self.commands)}: {command.__name__}")
                result = await self.pipeline.execute(
                    f"{self.name}_fallback_{i}",
                    command,
                    *args,
                    **kwargs
                )
                
                if self.stop_on_success:
                    return result
                
            except Exception as e:
                logger.warning(f"Command {command.__name__} failed: {e}")
                last_error = e
                continue
        
        # All commands failed
        if last_error:
            raise last_error
        return None


class ReduceCommand(ComposedCommand):
    """Execute commands and reduce results."""
    
    def __init__(
        self,
        commands: List[Callable],
        pipeline: CommandPipeline,
        reducer: Optional[Callable] = None,
        initial: Any = None,
        **kwargs
    ):
        """Initialize reduce command."""
        super().__init__(commands, pipeline, **kwargs)
        self.reducer = reducer or self._default_reducer
        self.initial = initial
    
    def _default_reducer(self, acc: Any, value: Any) -> Any:
        """Default reducer (collect in list)."""
        if acc is None:
            return [value]
        acc.append(value)
        return acc
    
    async def execute(self, *args, **kwargs) -> Any:
        """Execute and reduce results."""
        accumulator = self.initial
        
        for i, command in enumerate(self.commands):
            result = await self.pipeline.execute(
                f"{self.name}_reduce_{i}",
                command,
                *args,
                **kwargs
            )
            
            accumulator = self.reducer(accumulator, result)
        
        return accumulator