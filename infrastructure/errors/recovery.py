"""Error recovery strategies for k-bert."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

import mlx.core as mx
from loguru import logger

from .base import KBertError
from .types import DataError, ModelError, TrainingError

T = TypeVar("T")


@dataclass
class RecoveryResult:
    """Result of a recovery attempt."""

    success: bool
    value: Optional[Any] = None
    message: Optional[str] = None
    retry_after: Optional[float] = None
    modifications: Optional[Dict[str, Any]] = None


class RecoveryStrategy(ABC):
    """Abstract base class for recovery strategies."""

    @abstractmethod
    def can_recover(self, error: Exception) -> bool:
        """Check if this strategy can recover from the error."""
        pass

    @abstractmethod
    def recover(self, error: Exception, context: Dict[str, Any]) -> RecoveryResult:
        """Attempt to recover from the error."""
        pass


class RetryStrategy(RecoveryStrategy):
    """Retry with configurable backoff."""

    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        max_delay: float = 60.0,
        jitter: bool = True,
    ):
        """Initialize retry strategy."""
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay
        self.jitter = jitter
        self.attempt = 0

    def can_recover(self, error: Exception) -> bool:
        """Check if we have attempts remaining."""
        return self.attempt < self.max_attempts

    def recover(self, error: Exception, context: Dict[str, Any]) -> RecoveryResult:
        """Retry with exponential backoff."""
        self.attempt += 1
        
        if self.attempt >= self.max_attempts:
            return RecoveryResult(
                success=False,
                message=f"Max retry attempts ({self.max_attempts}) exceeded",
            )
        
        # Calculate delay with exponential backoff
        delay = min(
            self.initial_delay * (self.backoff_factor ** (self.attempt - 1)),
            self.max_delay
        )
        
        # Add jitter to prevent thundering herd
        if self.jitter:
            import random
            delay *= (0.5 + random.random())
        
        logger.info(
            f"Retry attempt {self.attempt}/{self.max_attempts} "
            f"after {delay:.1f}s delay"
        )
        
        return RecoveryResult(
            success=True,
            retry_after=delay,
            message=f"Retrying after {delay:.1f}s (attempt {self.attempt}/{self.max_attempts})",
        )


class ResourceReductionStrategy(RecoveryStrategy):
    """Reduce resource usage on OOM or resource errors."""

    def __init__(
        self,
        reduction_factor: float = 0.5,
        min_batch_size: int = 1,
        enable_gradient_accumulation: bool = True,
    ):
        """Initialize resource reduction strategy."""
        self.reduction_factor = reduction_factor
        self.min_batch_size = min_batch_size
        self.enable_gradient_accumulation = enable_gradient_accumulation

    def can_recover(self, error: Exception) -> bool:
        """Check if this is a resource-related error."""
        if isinstance(error, TrainingError):
            return error.error_code == "TRAINING_OOM"
        # Check for MLX OOM errors
        return "out of memory" in str(error).lower()

    def recover(self, error: Exception, context: Dict[str, Any]) -> RecoveryResult:
        """Reduce resource usage."""
        current_batch_size = context.get("batch_size", 32)
        new_batch_size = max(
            int(current_batch_size * self.reduction_factor),
            self.min_batch_size
        )
        
        if new_batch_size == current_batch_size:
            return RecoveryResult(
                success=False,
                message="Cannot reduce batch size further",
            )
        
        modifications = {
            "batch_size": new_batch_size,
        }
        
        if self.enable_gradient_accumulation and new_batch_size < current_batch_size:
            # Calculate accumulation steps to maintain effective batch size
            accumulation_steps = current_batch_size // new_batch_size
            modifications["gradient_accumulation_steps"] = accumulation_steps
        
        # Clear MLX memory
        mx.metal.clear_cache()
        
        return RecoveryResult(
            success=True,
            modifications=modifications,
            message=f"Reduced batch size from {current_batch_size} to {new_batch_size}",
        )


class CheckpointRecoveryStrategy(RecoveryStrategy):
    """Recover from checkpoint after failure."""

    def __init__(
        self,
        checkpoint_dir: Optional[Path] = None,
        load_best: bool = False,
        modify_config: bool = True,
    ):
        """Initialize checkpoint recovery strategy."""
        self.checkpoint_dir = checkpoint_dir
        self.load_best = load_best
        self.modify_config = modify_config

    def can_recover(self, error: Exception) -> bool:
        """Check if we can recover from checkpoint."""
        if not isinstance(error, (TrainingError, ModelError)):
            return False
        
        # Check if checkpoints exist
        if self.checkpoint_dir and self.checkpoint_dir.exists():
            checkpoints = list(self.checkpoint_dir.glob("*.safetensors"))
            return len(checkpoints) > 0
        
        return False

    def recover(self, error: Exception, context: Dict[str, Any]) -> RecoveryResult:
        """Recover from checkpoint."""
        if not self.checkpoint_dir:
            return RecoveryResult(
                success=False,
                message="No checkpoint directory specified",
            )
        
        # Find available checkpoints
        checkpoints = sorted(self.checkpoint_dir.glob("*.safetensors"))
        if not checkpoints:
            return RecoveryResult(
                success=False,
                message="No checkpoints found",
            )
        
        # Select checkpoint
        if self.load_best:
            # Look for best checkpoint based on metrics
            best_ckpt = self._find_best_checkpoint(checkpoints)
            checkpoint_path = best_ckpt or checkpoints[-1]
        else:
            # Use most recent
            checkpoint_path = checkpoints[-1]
        
        modifications = {
            "resume_from_checkpoint": str(checkpoint_path),
        }
        
        # Modify config based on error type
        if self.modify_config and isinstance(error, TrainingError):
            if error.error_code == "TRAINING_NAN_LOSS":
                # Reduce learning rate for stability
                current_lr = context.get("learning_rate", 1e-4)
                modifications["learning_rate"] = current_lr * 0.1
                modifications["gradient_clip_norm"] = 1.0
        
        return RecoveryResult(
            success=True,
            value=checkpoint_path,
            modifications=modifications,
            message=f"Resuming from checkpoint: {checkpoint_path.name}",
        )

    def _find_best_checkpoint(self, checkpoints: List[Path]) -> Optional[Path]:
        """Find best checkpoint based on metrics."""
        # This would look at checkpoint metadata
        # For now, return None to use most recent
        return None


class FallbackStrategy(RecoveryStrategy):
    """Fallback to alternative approach."""

    def __init__(
        self,
        fallbacks: Dict[str, Dict[str, Any]],
        max_fallback_depth: int = 3,
    ):
        """Initialize fallback strategy."""
        self.fallbacks = fallbacks
        self.max_fallback_depth = max_fallback_depth
        self.fallback_depth = 0

    def can_recover(self, error: Exception) -> bool:
        """Check if we have fallbacks available."""
        error_code = getattr(error, "error_code", type(error).__name__)
        return (
            error_code in self.fallbacks and
            self.fallback_depth < self.max_fallback_depth
        )

    def recover(self, error: Exception, context: Dict[str, Any]) -> RecoveryResult:
        """Apply fallback configuration."""
        error_code = getattr(error, "error_code", type(error).__name__)
        
        if error_code not in self.fallbacks:
            return RecoveryResult(
                success=False,
                message=f"No fallback for error code: {error_code}",
            )
        
        self.fallback_depth += 1
        fallback = self.fallbacks[error_code]
        
        return RecoveryResult(
            success=True,
            modifications=fallback,
            message=f"Applying fallback configuration for {error_code}",
        )


class CompositeStrategy(RecoveryStrategy):
    """Combine multiple recovery strategies."""

    def __init__(self, strategies: List[RecoveryStrategy]):
        """Initialize composite strategy."""
        self.strategies = strategies

    def can_recover(self, error: Exception) -> bool:
        """Check if any strategy can recover."""
        return any(s.can_recover(error) for s in self.strategies)

    def recover(self, error: Exception, context: Dict[str, Any]) -> RecoveryResult:
        """Try strategies in order until one succeeds."""
        for strategy in self.strategies:
            if strategy.can_recover(error):
                result = strategy.recover(error, context)
                if result.success:
                    return result
        
        return RecoveryResult(
            success=False,
            message="All recovery strategies failed",
        )


class RecoveryManager:
    """Manage error recovery with strategies."""

    def __init__(self):
        """Initialize recovery manager."""
        self.strategies: Dict[Type[Exception], List[RecoveryStrategy]] = {}
        self.global_strategies: List[RecoveryStrategy] = []
        self.recovery_history: List[Dict[str, Any]] = []

    def register_strategy(
        self,
        strategy: RecoveryStrategy,
        error_type: Optional[Type[Exception]] = None,
    ) -> None:
        """Register a recovery strategy."""
        if error_type:
            if error_type not in self.strategies:
                self.strategies[error_type] = []
            self.strategies[error_type].append(strategy)
        else:
            self.global_strategies.append(strategy)

    def attempt_recovery(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
    ) -> RecoveryResult:
        """Attempt to recover from an error."""
        context = context or {}
        
        # Record attempt
        self.recovery_history.append({
            "error": type(error).__name__,
            "message": str(error),
            "context": context,
            "timestamp": time.time(),
        })
        
        # Try type-specific strategies first
        strategies_to_try = []
        
        for error_type, strategies in self.strategies.items():
            if isinstance(error, error_type):
                strategies_to_try.extend(strategies)
        
        # Add global strategies
        strategies_to_try.extend(self.global_strategies)
        
        # Try each strategy
        for strategy in strategies_to_try:
            if strategy.can_recover(error):
                logger.info(f"Attempting recovery with {type(strategy).__name__}")
                result = strategy.recover(error, context)
                
                if result.success:
                    logger.success(f"Recovery successful: {result.message}")
                    return result
                else:
                    logger.warning(f"Recovery failed: {result.message}")
        
        return RecoveryResult(
            success=False,
            message="No recovery strategy available",
        )

    def get_recovery_suggestions(self, error: Exception) -> List[str]:
        """Get recovery suggestions for an error."""
        suggestions = []
        
        # Check available strategies
        for strategy in self.global_strategies:
            if strategy.can_recover(error):
                suggestions.append(f"Try {type(strategy).__name__}")
        
        # Add error-specific suggestions
        if isinstance(error, KBertError):
            suggestions.extend(error.context.recovery_actions)
        
        return suggestions


# Default recovery manager instance
_recovery_manager = RecoveryManager()


def register_recovery_strategy(
    strategy: RecoveryStrategy,
    error_type: Optional[Type[Exception]] = None,
) -> None:
    """Register a recovery strategy."""
    _recovery_manager.register_strategy(strategy, error_type)


def attempt_recovery(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
) -> RecoveryResult:
    """Attempt to recover from an error."""
    return _recovery_manager.attempt_recovery(error, context)


def with_recovery(
    strategies: Optional[List[RecoveryStrategy]] = None,
    max_attempts: int = 3,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to add recovery to functions."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Create composite strategy if multiple provided
            if strategies and len(strategies) > 1:
                strategy = CompositeStrategy(strategies)
            elif strategies:
                strategy = strategies[0]
            else:
                strategy = RetryStrategy(max_attempts=max_attempts)
            
            attempt = 0
            last_error = None
            context = {"args": args, "kwargs": kwargs}
            
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    
                    if strategy.can_recover(e):
                        result = strategy.recover(e, context)
                        
                        if result.success:
                            # Apply modifications
                            if result.modifications:
                                kwargs.update(result.modifications)
                            
                            # Wait if needed
                            if result.retry_after:
                                time.sleep(result.retry_after)
                            
                            attempt += 1
                            continue
                    
                    # Cannot recover
                    raise
            
            # Max attempts reached
            if last_error:
                raise last_error
            
            raise RuntimeError("Max recovery attempts reached")
        
        return wrapper
    return decorator


# Set up default recovery strategies
def setup_default_recovery() -> None:
    """Set up default recovery strategies."""
    # Retry for transient errors
    register_recovery_strategy(
        RetryStrategy(max_attempts=3, initial_delay=1.0),
    )
    
    # Resource reduction for OOM
    register_recovery_strategy(
        ResourceReductionStrategy(),
        TrainingError,
    )
    
    # Checkpoint recovery for training failures
    register_recovery_strategy(
        CheckpointRecoveryStrategy(),
        TrainingError,
    )