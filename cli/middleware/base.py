"""Base middleware classes and pipeline implementation."""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from loguru import logger


class MiddlewarePhase(Enum):
    """Middleware execution phases."""
    
    PRE_VALIDATION = "pre_validation"
    VALIDATION = "validation"
    PRE_EXECUTION = "pre_execution"
    EXECUTION = "execution"
    POST_EXECUTION = "post_execution"
    ERROR_HANDLING = "error_handling"
    FINALIZATION = "finalization"


@dataclass
class CommandContext:
    """Context passed through middleware pipeline."""
    
    command_name: str
    args: tuple
    kwargs: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get metadata value."""
        return self.metadata.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set metadata value."""
        self.metadata[key] = value
    
    def update(self, **kwargs) -> None:
        """Update metadata."""
        self.metadata.update(kwargs)


@dataclass
class MiddlewareResult:
    """Result of middleware execution."""
    
    success: bool
    data: Any = None
    error: Optional[Exception] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def ok(cls, data: Any = None, **metadata) -> "MiddlewareResult":
        """Create successful result."""
        return cls(success=True, data=data, metadata=metadata)
    
    @classmethod
    def fail(cls, error: Exception, **metadata) -> "MiddlewareResult":
        """Create failed result."""
        return cls(success=False, error=error, metadata=metadata)


T = TypeVar("T")


class Middleware(ABC):
    """Base middleware class."""
    
    def __init__(self, name: Optional[str] = None):
        """Initialize middleware."""
        self.name = name or self.__class__.__name__
    
    @abstractmethod
    async def process(
        self,
        context: CommandContext,
        next_handler: Callable[[CommandContext], Any]
    ) -> MiddlewareResult:
        """Process the request through middleware.
        
        Args:
            context: Command context
            next_handler: Next middleware or command handler
            
        Returns:
            Middleware result
        """
        pass
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name={self.name})"


class MiddlewarePipeline:
    """Manages middleware execution pipeline."""
    
    def __init__(self):
        """Initialize pipeline."""
        self.middleware: List[Middleware] = []
        self._is_async = False
    
    def add(self, middleware: Middleware) -> "MiddlewarePipeline":
        """Add middleware to pipeline."""
        self.middleware.append(middleware)
        logger.debug(f"Added middleware: {middleware}")
        return self
    
    def remove(self, middleware: Union[Middleware, str]) -> "MiddlewarePipeline":
        """Remove middleware from pipeline."""
        if isinstance(middleware, str):
            self.middleware = [m for m in self.middleware if m.name != middleware]
        else:
            self.middleware.remove(middleware)
        return self
    
    def clear(self) -> "MiddlewarePipeline":
        """Clear all middleware."""
        self.middleware.clear()
        return self
    
    async def execute_async(
        self,
        context: CommandContext,
        handler: Callable[[CommandContext], Any]
    ) -> MiddlewareResult:
        """Execute pipeline asynchronously."""
        
        async def create_chain(
            middleware_list: List[Middleware],
            final_handler: Callable[[CommandContext], Any]
        ) -> Callable[[CommandContext], Any]:
            """Create middleware chain."""
            
            async def wrapped_handler(ctx: CommandContext) -> MiddlewareResult:
                """Wrap sync handler."""
                try:
                    if asyncio.iscoroutinefunction(final_handler):
                        result = await final_handler(ctx)
                    else:
                        result = final_handler(ctx)
                    
                    if isinstance(result, MiddlewareResult):
                        return result
                    return MiddlewareResult.ok(result)
                except Exception as e:
                    logger.exception("Handler error")
                    return MiddlewareResult.fail(e)
            
            # Build chain from right to left
            chain = wrapped_handler
            for middleware in reversed(middleware_list):
                current_middleware = middleware
                next_in_chain = chain
                
                async def middleware_wrapper(ctx: CommandContext) -> MiddlewareResult:
                    """Wrapper to capture middleware and next handler."""
                    return await current_middleware.process(ctx, next_in_chain)
                
                chain = middleware_wrapper
            
            return chain
        
        # Create and execute chain
        chain = await create_chain(self.middleware, handler)
        return await chain(context)
    
    def execute(
        self,
        context: CommandContext,
        handler: Callable[[CommandContext], Any]
    ) -> MiddlewareResult:
        """Execute pipeline synchronously."""
        
        def create_chain(
            middleware_list: List[Middleware],
            final_handler: Callable[[CommandContext], Any]
        ) -> Callable[[CommandContext], Any]:
            """Create middleware chain."""
            
            def wrapped_handler(ctx: CommandContext) -> MiddlewareResult:
                """Wrap handler."""
                try:
                    result = final_handler(ctx)
                    if isinstance(result, MiddlewareResult):
                        return result
                    return MiddlewareResult.ok(result)
                except Exception as e:
                    logger.exception("Handler error")
                    return MiddlewareResult.fail(e)
            
            # Build chain from right to left
            chain = wrapped_handler
            for middleware in reversed(middleware_list):
                # Create closure to capture current middleware
                def make_wrapper(mw: Middleware, next_handler: Callable):
                    def wrapper(ctx: CommandContext) -> MiddlewareResult:
                        # Convert async middleware to sync
                        if asyncio.iscoroutinefunction(mw.process):
                            loop = asyncio.new_event_loop()
                            try:
                                return loop.run_until_complete(
                                    mw.process(ctx, next_handler)
                                )
                            finally:
                                loop.close()
                        else:
                            return mw.process(ctx, next_handler)
                    return wrapper
                
                chain = make_wrapper(middleware, chain)
            
            return chain
        
        # Create and execute chain
        chain = create_chain(self.middleware, handler)
        return chain(context)
    
    def __len__(self) -> int:
        """Get middleware count."""
        return len(self.middleware)
    
    def __repr__(self) -> str:
        """String representation."""
        middleware_names = [m.name for m in self.middleware]
        return f"MiddlewarePipeline({middleware_names})"