"""Mock objects for CLI testing."""

from typing import Any, Callable, Dict, List, Optional
from unittest.mock import MagicMock, Mock

from cli.middleware.base import CommandContext, Middleware, MiddlewareResult
from cli.pipeline.base import CommandHook, HookPhase, PipelineContext
from cli.plugins.cli_plugin import CLIPlugin


class MockContext(CommandContext):
    """Mock command context for testing."""
    
    def __init__(
        self,
        command_name: str = "test_command",
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        **metadata
    ):
        """Initialize mock context."""
        super().__init__(
            command_name=command_name,
            args=args,
            kwargs=kwargs or {}
        )
        self.metadata.update(metadata)


class MockMiddleware(Middleware):
    """Mock middleware for testing."""
    
    def __init__(
        self,
        name: str = "MockMiddleware",
        process_func: Optional[Callable] = None,
        should_fail: bool = False,
        error: Optional[Exception] = None
    ):
        """Initialize mock middleware."""
        super().__init__(name)
        self.process_func = process_func
        self.should_fail = should_fail
        self.error = error or Exception("Mock middleware error")
        self.call_count = 0
        self.call_history: List[Dict[str, Any]] = []
    
    async def process(
        self,
        context: CommandContext,
        next_handler: Callable[[CommandContext], Any]
    ) -> MiddlewareResult:
        """Process through mock middleware."""
        self.call_count += 1
        self.call_history.append({
            "context": context,
            "handler": next_handler
        })
        
        if self.process_func:
            return self.process_func(context, next_handler)
        
        if self.should_fail:
            return MiddlewareResult.fail(self.error)
        
        # Default: pass through
        return next_handler(context)
    
    def assert_called(self, times: Optional[int] = None) -> None:
        """Assert middleware was called."""
        if times is not None:
            assert self.call_count == times, f"Expected {times} calls, got {self.call_count}"
        else:
            assert self.call_count > 0, "Middleware was not called"
    
    def assert_not_called(self) -> None:
        """Assert middleware was not called."""
        assert self.call_count == 0, f"Middleware was called {self.call_count} times"


class MockHook(CommandHook):
    """Mock command hook for testing."""
    
    def __init__(
        self,
        name: str = "MockHook",
        phases: Optional[set] = None,
        execute_func: Optional[Callable] = None,
        should_fail: bool = False,
        error: Optional[Exception] = None
    ):
        """Initialize mock hook."""
        super().__init__(name=name, phases=phases)
        self.execute_func = execute_func
        self.should_fail = should_fail
        self.error = error or Exception("Mock hook error")
        self.call_count = 0
        self.call_history: List[Dict[str, Any]] = []
    
    async def execute(self, phase: HookPhase, context: PipelineContext) -> None:
        """Execute mock hook."""
        self.call_count += 1
        self.call_history.append({
            "phase": phase,
            "context": context
        })
        
        if self.execute_func:
            await self.execute_func(phase, context)
        elif self.should_fail:
            raise self.error
    
    def assert_called_in_phase(self, phase: HookPhase, times: int = 1) -> None:
        """Assert hook was called in specific phase."""
        phase_calls = [
            call for call in self.call_history
            if call["phase"] == phase
        ]
        assert len(phase_calls) == times, f"Expected {times} calls in {phase}, got {len(phase_calls)}"


class MockPlugin(CLIPlugin):
    """Mock CLI plugin for testing."""
    
    def __init__(
        self,
        commands: Optional[Dict[str, Callable]] = None,
        middleware: Optional[List[Middleware]] = None,
        hooks: Optional[List[CommandHook]] = None,
        app_extensions: Optional[Dict[str, Any]] = None
    ):
        """Initialize mock plugin."""
        super().__init__()
        self._commands = commands or {}
        self._middleware = middleware or []
        self._hooks = hooks or []
        self._app_extensions = app_extensions or {}
    
    def get_commands(self) -> Dict[str, Callable]:
        """Get mock commands."""
        return self._commands
    
    def get_middleware(self) -> List[Middleware]:
        """Get mock middleware."""
        return self._middleware
    
    def get_hooks(self) -> List[CommandHook]:
        """Get mock hooks."""
        return self._hooks
    
    def get_app_extensions(self) -> Dict[str, Any]:
        """Get mock app extensions."""
        return self._app_extensions


def create_mock_config(**kwargs) -> Dict[str, Any]:
    """Create mock configuration."""
    default_config = {
        "models": {
            "type": "modernbert_with_head",
            "num_labels": 2,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 1e-4,
            "num_epochs": 3,
            "warmup_ratio": 0.1,
        },
        "data": {
            "data_path": "data/train.csv",
            "text_column": "text",
            "label_column": "label",
            "max_length": 512,
        },
        "checkpoint": {
            "save_dir": "output/checkpoints",
            "save_interval": 1000,
        },
        "logging": {
            "log_interval": 100,
            "log_level": "INFO",
        }
    }
    
    # Deep merge with provided kwargs
    def deep_merge(base: dict, override: dict) -> dict:
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    return deep_merge(default_config, kwargs)


def create_mock_command(
    name: str = "mock_command",
    return_value: Any = None,
    side_effect: Optional[Callable] = None,
    raises: Optional[Exception] = None
) -> Mock:
    """Create mock command function."""
    mock = Mock(name=name)
    
    if raises:
        mock.side_effect = raises
    elif side_effect:
        mock.side_effect = side_effect
    else:
        mock.return_value = return_value
    
    # Add Typer metadata
    mock.__typer_meta__ = {
        "name": name,
        "help": f"Mock command {name}",
    }
    
    return mock


class MockProgressBar:
    """Mock progress bar for testing."""
    
    def __init__(self, total: int = 100):
        """Initialize mock progress bar."""
        self.total = total
        self.current = 0
        self.started = False
        self.finished = False
        self.updates: List[Dict[str, Any]] = []
    
    def start(self) -> None:
        """Start progress bar."""
        self.started = True
    
    def update(self, advance: int = 1, **kwargs) -> None:
        """Update progress bar."""
        self.current += advance
        self.updates.append({
            "advance": advance,
            "current": self.current,
            **kwargs
        })
    
    def finish(self) -> None:
        """Finish progress bar."""
        self.finished = True
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, *args):
        """Context manager exit."""
        if not self.finished:
            self.finish()


class MockFileSystem:
    """Mock filesystem for testing."""
    
    def __init__(self):
        """Initialize mock filesystem."""
        self.files: Dict[str, str] = {}
        self.directories: set = set()
    
    def write_file(self, path: str, content: str) -> None:
        """Write mock file."""
        self.files[path] = content
        # Add parent directories
        parts = path.split("/")
        for i in range(1, len(parts)):
            self.directories.add("/".join(parts[:i]))
    
    def read_file(self, path: str) -> str:
        """Read mock file."""
        if path not in self.files:
            raise FileNotFoundError(f"Mock file not found: {path}")
        return self.files[path]
    
    def exists(self, path: str) -> bool:
        """Check if path exists."""
        return path in self.files or path in self.directories
    
    def is_file(self, path: str) -> bool:
        """Check if path is file."""
        return path in self.files
    
    def is_dir(self, path: str) -> bool:
        """Check if path is directory."""
        return path in self.directories
    
    def list_dir(self, path: str) -> List[str]:
        """List directory contents."""
        if not self.is_dir(path):
            raise NotADirectoryError(f"Not a directory: {path}")
        
        path = path.rstrip("/")
        items = []
        
        for file_path in self.files:
            if file_path.startswith(path + "/"):
                relative = file_path[len(path) + 1:]
                if "/" not in relative:
                    items.append(relative)
        
        for dir_path in self.directories:
            if dir_path.startswith(path + "/"):
                relative = dir_path[len(path) + 1:]
                if "/" not in relative:
                    items.append(relative)
        
        return sorted(set(items))