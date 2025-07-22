"""Logging middleware for CLI commands."""

import asyncio
import json
import sys
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from loguru import logger

from cli.middleware.base import CommandContext, Middleware, MiddlewareResult


class LoggingMiddleware(Middleware):
    """Middleware for enhanced command logging."""
    
    def __init__(
        self,
        name: str = "LoggingMiddleware",
        log_args: bool = True,
        log_result: bool = True,
        log_timing: bool = True,
        structured: bool = False,
        sensitive_keys: Optional[list] = None
    ):
        """Initialize logging middleware.
        
        Args:
            name: Middleware name
            log_args: Whether to log command arguments
            log_result: Whether to log command results
            log_timing: Whether to log execution time
            structured: Whether to use structured logging
            sensitive_keys: Keys to redact from logs
        """
        super().__init__(name)
        self.log_args = log_args
        self.log_result = log_result
        self.log_timing = log_timing
        self.structured = structured
        self.sensitive_keys = sensitive_keys or ["password", "token", "secret", "api_key"]
    
    def _redact_sensitive(self, data: Any) -> Any:
        """Redact sensitive information from data."""
        if isinstance(data, dict):
            return {
                k: ("***REDACTED***" if any(key in k.lower() for key in self.sensitive_keys) else self._redact_sensitive(v))
                for k, v in data.items()
            }
        elif isinstance(data, (list, tuple)):
            return [self._redact_sensitive(item) for item in data]
        return data
    
    def _format_log_entry(self, data: Dict[str, Any]) -> str:
        """Format log entry."""
        if self.structured:
            return json.dumps(data, default=str)
        
        parts = []
        if "command" in data:
            parts.append(f"Command: {data['command']}")
        if "duration" in data:
            parts.append(f"Duration: {data['duration']:.3f}s")
        if "status" in data:
            parts.append(f"Status: {data['status']}")
        
        return " | ".join(parts)
    
    async def process(
        self,
        context: CommandContext,
        next_handler: Callable[[CommandContext], Any]
    ) -> MiddlewareResult:
        """Process logging middleware."""
        start_time = datetime.now()
        
        # Log command start
        log_data = {
            "event": "command_start",
            "command": context.command_name,
            "timestamp": start_time.isoformat(),
        }
        
        if self.log_args:
            log_data["args"] = self._redact_sensitive(context.args)
            log_data["kwargs"] = self._redact_sensitive(context.kwargs)
        
        logger.info(self._format_log_entry({
            "command": context.command_name,
            "phase": "start"
        }))
        
        if self.structured:
            logger.debug(json.dumps(log_data, default=str))
        
        # Store start time in context
        context.set("start_time", start_time)
        
        try:
            # Execute next handler
            if asyncio.iscoroutinefunction(next_handler):
                result = await next_handler(context)
            else:
                result = next_handler(context)
            
            # Log command completion
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            log_data = {
                "event": "command_complete",
                "command": context.command_name,
                "timestamp": end_time.isoformat(),
                "duration": duration,
                "status": "success" if result.success else "failed",
            }
            
            if self.log_result and result.data is not None:
                log_data["result"] = self._redact_sensitive(result.data)
            
            if self.log_timing:
                context.set("execution_time", duration)
                result.metadata["execution_time"] = duration
            
            logger.info(self._format_log_entry({
                "command": context.command_name,
                "phase": "complete",
                "duration": duration,
                "status": log_data["status"]
            }))
            
            if self.structured:
                logger.debug(json.dumps(log_data, default=str))
            
            return result
            
        except Exception as e:
            # Log command error
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            log_data = {
                "event": "command_error",
                "command": context.command_name,
                "timestamp": end_time.isoformat(),
                "duration": duration,
                "error": str(e),
                "error_type": type(e).__name__,
            }
            
            logger.error(self._format_log_entry({
                "command": context.command_name,
                "phase": "error",
                "duration": duration,
                "error": str(e)
            }))
            
            if self.structured:
                logger.debug(json.dumps(log_data, default=str))
            
            # Re-raise to let error middleware handle
            raise


class StreamingLogMiddleware(LoggingMiddleware):
    """Logging middleware with streaming support."""
    
    def __init__(self, stream=None, **kwargs):
        """Initialize streaming log middleware."""
        super().__init__(**kwargs)
        self.stream = stream or sys.stderr
    
    async def process(
        self,
        context: CommandContext,
        next_handler: Callable[[CommandContext], Any]
    ) -> MiddlewareResult:
        """Process with streaming logs."""
        # Add stream handler
        handler_id = logger.add(
            self.stream,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            level="INFO",
            filter=lambda record: record["extra"].get("stream", False)
        )
        
        try:
            # Mark context for streaming
            context.set("streaming", True)
            
            # Execute with streaming
            return await super().process(context, next_handler)
            
        finally:
            # Remove stream handler
            logger.remove(handler_id)