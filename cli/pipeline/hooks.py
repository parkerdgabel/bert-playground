"""Built-in command hooks for common functionality."""

import os
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from loguru import logger

from cli.pipeline.base import CommandHook, HookPhase, PipelineContext


class ConfigurationHook(CommandHook):
    """Hook for configuration management."""
    
    def __init__(
        self,
        config_paths: Optional[List[Path]] = None,
        required_keys: Optional[List[str]] = None,
        **kwargs
    ):
        """Initialize configuration hook."""
        super().__init__(
            phases={HookPhase.PRE_VALIDATE},
            priority=10,
            **kwargs
        )
        self.config_paths = config_paths or []
        self.required_keys = required_keys or []
    
    async def execute(self, phase: HookPhase, context: PipelineContext) -> None:
        """Load and validate configuration."""
        if phase != HookPhase.PRE_VALIDATE:
            return
        
        # Check for config in kwargs
        config = context.kwargs.get("config", {})
        
        # Load from files if needed
        if not config and self.config_paths:
            for path in self.config_paths:
                if path.exists():
                    # Import here to avoid circular dependency
                    from cli.config.loader import ConfigLoader
                    loader = ConfigLoader()
                    config = loader.load_file(path)
                    break
        
        # Validate required keys
        if self.required_keys:
            missing = [k for k in self.required_keys if k not in config]
            if missing:
                raise ValueError(f"Missing required config keys: {missing}")
        
        # Store in context
        context.set("config", config)
        context.kwargs["config"] = config


class DependencyHook(CommandHook):
    """Hook for dependency checking."""
    
    def __init__(
        self,
        required_commands: Optional[List[str]] = None,
        required_modules: Optional[List[str]] = None,
        **kwargs
    ):
        """Initialize dependency hook."""
        super().__init__(
            phases={HookPhase.PRE_VALIDATE},
            priority=20,
            **kwargs
        )
        self.required_commands = required_commands or []
        self.required_modules = required_modules or []
    
    async def execute(self, phase: HookPhase, context: PipelineContext) -> None:
        """Check dependencies."""
        if phase != HookPhase.PRE_VALIDATE:
            return
        
        errors = []
        
        # Check commands
        for cmd in self.required_commands:
            if not shutil.which(cmd):
                errors.append(f"Required command not found: {cmd}")
        
        # Check modules
        for module in self.required_modules:
            try:
                __import__(module)
            except ImportError:
                errors.append(f"Required module not found: {module}")
        
        if errors:
            raise RuntimeError("Dependency check failed: " + "; ".join(errors))


class EnvironmentHook(CommandHook):
    """Hook for environment setup."""
    
    def __init__(
        self,
        required_vars: Optional[List[str]] = None,
        set_vars: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """Initialize environment hook."""
        super().__init__(
            phases={HookPhase.PRE_EXECUTE, HookPhase.POST_CLEANUP},
            priority=15,
            **kwargs
        )
        self.required_vars = required_vars or []
        self.set_vars = set_vars or {}
        self._original_env: Dict[str, Optional[str]] = {}
    
    async def execute(self, phase: HookPhase, context: PipelineContext) -> None:
        """Manage environment variables."""
        if phase == HookPhase.PRE_EXECUTE:
            # Check required vars
            missing = [v for v in self.required_vars if v not in os.environ]
            if missing:
                raise RuntimeError(f"Missing required environment variables: {missing}")
            
            # Set environment vars
            for key, value in self.set_vars.items():
                self._original_env[key] = os.environ.get(key)
                os.environ[key] = value
                logger.debug(f"Set environment variable: {key}")
        
        elif phase == HookPhase.POST_CLEANUP:
            # Restore original environment
            for key, original_value in self._original_env.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value
            self._original_env.clear()


class ResourceHook(CommandHook):
    """Hook for resource management."""
    
    def __init__(
        self,
        temp_dirs: Optional[List[str]] = None,
        cleanup_patterns: Optional[List[str]] = None,
        **kwargs
    ):
        """Initialize resource hook."""
        super().__init__(
            phases={HookPhase.PRE_EXECUTE, HookPhase.POST_CLEANUP},
            priority=25,
            **kwargs
        )
        self.temp_dirs = temp_dirs or []
        self.cleanup_patterns = cleanup_patterns or []
        self._created_resources: List[Path] = []
    
    async def execute(self, phase: HookPhase, context: PipelineContext) -> None:
        """Manage resources."""
        if phase == HookPhase.PRE_EXECUTE:
            # Create temporary directories
            for dir_name in self.temp_dirs:
                temp_dir = Path(dir_name)
                temp_dir.mkdir(parents=True, exist_ok=True)
                self._created_resources.append(temp_dir)
                context.set(f"temp_dir_{dir_name}", temp_dir)
                logger.debug(f"Created temporary directory: {temp_dir}")
        
        elif phase == HookPhase.POST_CLEANUP:
            # Clean up resources
            for resource in self._created_resources:
                if resource.exists():
                    if resource.is_dir():
                        shutil.rmtree(resource)
                    else:
                        resource.unlink()
                    logger.debug(f"Cleaned up resource: {resource}")
            
            # Clean up by patterns
            for pattern in self.cleanup_patterns:
                for path in Path.cwd().glob(pattern):
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink()
                    logger.debug(f"Cleaned up by pattern: {path}")
            
            self._created_resources.clear()


class ValidationHook(CommandHook):
    """Hook for input validation."""
    
    def __init__(
        self,
        validators: Optional[Dict[str, Callable]] = None,
        required_args: Optional[List[str]] = None,
        **kwargs
    ):
        """Initialize validation hook."""
        super().__init__(
            phases={HookPhase.POST_PARSE},
            priority=30,
            **kwargs
        )
        self.validators = validators or {}
        self.required_args = required_args or []
    
    async def execute(self, phase: HookPhase, context: PipelineContext) -> None:
        """Validate inputs."""
        if phase != HookPhase.POST_PARSE:
            return
        
        # Check required arguments
        for arg in self.required_args:
            if arg not in context.kwargs:
                raise ValueError(f"Required argument missing: {arg}")
        
        # Run validators
        for arg_name, validator in self.validators.items():
            if arg_name in context.kwargs:
                value = context.kwargs[arg_name]
                if not validator(value):
                    raise ValueError(f"Validation failed for {arg_name}: {value}")


class CacheHook(CommandHook):
    """Hook for caching command results."""
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        cache_key_func: Optional[Callable] = None,
        ttl_seconds: int = 3600,
        **kwargs
    ):
        """Initialize cache hook."""
        super().__init__(
            phases={HookPhase.PRE_EXECUTE, HookPhase.POST_EXECUTE},
            priority=35,
            **kwargs
        )
        self.cache_dir = cache_dir or Path.home() / ".k-bert" / "cache"
        self.cache_key_func = cache_key_func or self._default_cache_key
        self.ttl_seconds = ttl_seconds
    
    def _default_cache_key(self, context: PipelineContext) -> str:
        """Generate default cache key."""
        import hashlib
        import json
        
        data = {
            "command": context.command,
            "args": context.args,
            "kwargs": {k: v for k, v in context.kwargs.items() if k != "config"}
        }
        
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()
    
    async def execute(self, phase: HookPhase, context: PipelineContext) -> None:
        """Handle caching."""
        import pickle
        import time
        
        cache_key = self.cache_key_func(context)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if phase == HookPhase.PRE_EXECUTE:
            # Check cache
            if cache_file.exists():
                # Check TTL
                age = time.time() - cache_file.stat().st_mtime
                if age < self.ttl_seconds:
                    try:
                        with open(cache_file, "rb") as f:
                            result = pickle.load(f)
                        context.set("cached_result", result)
                        context.set("skip_execution", True)
                        logger.info(f"Using cached result for {context.command}")
                    except Exception as e:
                        logger.warning(f"Failed to load cache: {e}")
        
        elif phase == HookPhase.POST_EXECUTE:
            # Save to cache
            if context.results and not context.get("skip_execution"):
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                try:
                    with open(cache_file, "wb") as f:
                        pickle.dump(context.last_result, f)
                    logger.debug(f"Cached result for {context.command}")
                except Exception as e:
                    logger.warning(f"Failed to save cache: {e}")