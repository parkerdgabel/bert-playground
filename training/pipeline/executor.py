"""Pipeline executor for training operations.

This module provides high-level execution management for training pipelines.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

from loguru import logger

from training.commands.base import CommandContext, CommandResult
from .base import Pipeline


@dataclass
class ExecutionConfig:
    """Configuration for pipeline execution."""
    
    # Execution control
    max_retries: int = 3
    retry_delay: float = 1.0
    continue_on_error: bool = False
    
    # Resource management
    memory_limit_mb: int | None = None
    time_limit_seconds: int | None = None
    
    # Monitoring
    enable_profiling: bool = False
    profile_output_dir: Path | None = None
    log_level: str = "INFO"
    
    # Checkpointing
    checkpoint_on_error: bool = True
    emergency_save_path: Path | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "continue_on_error": self.continue_on_error,
            "memory_limit_mb": self.memory_limit_mb,
            "time_limit_seconds": self.time_limit_seconds,
            "enable_profiling": self.enable_profiling,
            "profile_output_dir": str(self.profile_output_dir) if self.profile_output_dir else None,
            "log_level": self.log_level,
            "checkpoint_on_error": self.checkpoint_on_error,
            "emergency_save_path": str(self.emergency_save_path) if self.emergency_save_path else None,
        }


@dataclass
class ExecutionResult:
    """Result of pipeline execution."""
    
    success: bool
    pipeline_name: str
    execution_time: float
    command_results: list[CommandResult] = field(default_factory=list)
    error: Exception | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    
    # Resource usage
    peak_memory_mb: float = 0.0
    cpu_time_seconds: float = 0.0
    
    # Profiling data
    profile_path: Path | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "pipeline_name": self.pipeline_name,
            "execution_time": self.execution_time,
            "num_commands": len(self.command_results),
            "error": str(self.error) if self.error else None,
            "metrics": self.metrics,
            "peak_memory_mb": self.peak_memory_mb,
            "cpu_time_seconds": self.cpu_time_seconds,
            "profile_path": str(self.profile_path) if self.profile_path else None,
        }


class PipelineExecutor:
    """High-level executor for training pipelines with monitoring and error handling."""
    
    def __init__(self, config: ExecutionConfig | None = None):
        """Initialize executor.
        
        Args:
            config: Execution configuration
        """
        self.config = config or ExecutionConfig()
        self._execution_history: list[ExecutionResult] = []
    
    def execute(
        self,
        pipeline: Pipeline,
        context: CommandContext,
        name: str | None = None
    ) -> ExecutionResult:
        """Execute pipeline with monitoring and error handling.
        
        Args:
            pipeline: Pipeline to execute
            context: Execution context
            name: Optional execution name for logging
            
        Returns:
            ExecutionResult with execution details
        """
        execution_name = name or f"{pipeline.name}_{int(time.time())}"
        start_time = time.time()
        
        logger.info(f"Starting pipeline execution: {execution_name}")
        
        # Initialize result
        result = ExecutionResult(
            success=False,
            pipeline_name=pipeline.name,
            execution_time=0.0
        )
        
        # Execute with monitoring and error handling
        with self._execution_context():
            try:
                # Setup profiling if enabled
                if self.config.enable_profiling:
                    result.profile_path = self._start_profiling(execution_name)
                
                # Execute pipeline with timeout
                with self._timeout_context():
                    pipeline_result = self._execute_with_retries(pipeline, context)
                
                # Update result
                result.success = pipeline_result.success
                result.command_results = [pipeline_result]  # Flatten as needed
                result.metrics = pipeline_result.metrics
                result.error = pipeline_result.error
                
            except Exception as e:
                logger.error(f"Pipeline execution failed: {e}")
                result.error = e
                
                # Emergency checkpoint if configured
                if self.config.checkpoint_on_error and self.config.emergency_save_path:
                    try:
                        self._emergency_checkpoint(context, self.config.emergency_save_path)
                    except Exception as checkpoint_error:
                        logger.error(f"Emergency checkpoint failed: {checkpoint_error}")
            
            finally:
                # Calculate execution time
                result.execution_time = time.time() - start_time
                
                # Collect resource usage
                result.peak_memory_mb, result.cpu_time_seconds = self._collect_resource_usage()
                
                # Stop profiling if enabled
                if self.config.enable_profiling:
                    self._stop_profiling()
                
                # Log completion
                status = "SUCCESS" if result.success else "FAILED"
                logger.info(
                    f"Pipeline execution {status}: {execution_name} "
                    f"({result.execution_time:.2f}s)"
                )
                
                # Add to history
                self._execution_history.append(result)
        
        return result
    
    def _execute_with_retries(
        self,
        pipeline: Pipeline,
        context: CommandContext
    ) -> CommandResult:
        """Execute pipeline with retry logic."""
        last_error = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(f"Retrying pipeline execution (attempt {attempt + 1})")
                    time.sleep(self.config.retry_delay)
                
                return pipeline.execute(context)
                
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries:
                    logger.warning(f"Pipeline execution failed, retrying: {e}")
                    continue
                else:
                    logger.error(f"Pipeline execution failed after {self.config.max_retries} retries: {e}")
                    break
        
        # All retries failed
        return CommandResult(
            success=False,
            error=last_error or Exception("Unknown error"),
            should_continue=self.config.continue_on_error
        )
    
    @contextmanager
    def _execution_context(self) -> Generator[None, None, None]:
        """Context manager for execution setup/cleanup."""
        # Setup logging level
        original_level = logger._core.min_level
        logger.remove()
        logger.add(lambda msg: None, level=self.config.log_level)
        
        try:
            yield
        finally:
            # Restore logging
            logger.remove()
            logger.add(lambda msg: None, level=original_level)
    
    @contextmanager
    def _timeout_context(self) -> Generator[None, None, None]:
        """Context manager for execution timeout."""
        if self.config.time_limit_seconds is None:
            yield
            return
        
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Pipeline execution exceeded {self.config.time_limit_seconds}s")
        
        # Set timeout
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.config.time_limit_seconds)
        
        try:
            yield
        finally:
            # Cancel timeout
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    
    def _start_profiling(self, name: str) -> Path:
        """Start execution profiling."""
        if self.config.profile_output_dir:
            profile_path = self.config.profile_output_dir / f"{name}_profile.prof"
            # Profiling implementation would go here
            logger.info(f"Started profiling: {profile_path}")
            return profile_path
        return Path("profile.prof")
    
    def _stop_profiling(self) -> None:
        """Stop execution profiling."""
        # Profiling stop implementation would go here
        logger.info("Stopped profiling")
    
    def _collect_resource_usage(self) -> tuple[float, float]:
        """Collect resource usage statistics."""
        # Resource monitoring implementation would go here
        # For now, return dummy values
        return 0.0, 0.0
    
    def _emergency_checkpoint(self, context: CommandContext, path: Path) -> None:
        """Save emergency checkpoint."""
        if context.checkpoint_manager is not None:
            logger.info(f"Saving emergency checkpoint to {path}")
            context.checkpoint_manager.save_checkpoint(
                model=context.model,
                optimizer=context.optimizer,
                state=context.state,
                metrics=context.metrics,
                is_best=False
            )
    
    def get_execution_history(self) -> list[ExecutionResult]:
        """Get history of pipeline executions."""
        return self._execution_history.copy()
    
    def get_summary_stats(self) -> dict[str, Any]:
        """Get summary statistics across all executions."""
        if not self._execution_history:
            return {}
        
        successful = [r for r in self._execution_history if r.success]
        failed = [r for r in self._execution_history if not r.success]
        
        execution_times = [r.execution_time for r in self._execution_history]
        
        return {
            "total_executions": len(self._execution_history),
            "successful_executions": len(successful),
            "failed_executions": len(failed),
            "success_rate": len(successful) / len(self._execution_history),
            "avg_execution_time": sum(execution_times) / len(execution_times),
            "min_execution_time": min(execution_times),
            "max_execution_time": max(execution_times),
            "total_execution_time": sum(execution_times),
        }