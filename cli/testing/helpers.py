"""Testing helpers for CLI commands."""

import io
import os
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import typer
from typer.testing import CliRunner as TyperRunner

from cli.middleware.base import CommandContext, MiddlewarePipeline
from cli.pipeline.base import CommandPipeline


@dataclass
class CommandResult:
    """Result of command execution."""
    
    exit_code: int
    stdout: str
    stderr: str
    exception: Optional[Exception] = None
    context: Optional[CommandContext] = None
    
    @property
    def success(self) -> bool:
        """Check if command succeeded."""
        return self.exit_code == 0
    
    @property
    def output(self) -> str:
        """Get combined output."""
        return self.stdout + self.stderr


class CLIRunner:
    """Enhanced CLI runner for testing."""
    
    def __init__(
        self,
        app: Optional[typer.Typer] = None,
        env: Optional[Dict[str, str]] = None,
        mix_stderr: bool = True,
        catch_exceptions: bool = True
    ):
        """Initialize CLI runner.
        
        Args:
            app: Typer app to test
            env: Environment variables
            mix_stderr: Whether to mix stderr with stdout
            catch_exceptions: Whether to catch exceptions
        """
        self.app = app
        self.env = env or {}
        self.mix_stderr = mix_stderr
        self.catch_exceptions = catch_exceptions
        self._typer_runner = TyperRunner(mix_stderr=mix_stderr)
    
    def invoke(
        self,
        args: Union[str, List[str]],
        input: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> CommandResult:
        """Invoke command with args.
        
        Args:
            args: Command arguments
            input: Input to provide
            env: Environment overrides
            **kwargs: Additional options
            
        Returns:
            Command result
        """
        if isinstance(args, str):
            args = args.split()
        
        # Merge environment
        test_env = {**self.env, **(env or {})}
        
        # Set up environment
        with self._setup_environment(test_env):
            if self.app:
                result = self._typer_runner.invoke(
                    self.app,
                    args,
                    input=input,
                    catch_exceptions=self.catch_exceptions,
                    **kwargs
                )
                
                return CommandResult(
                    exit_code=result.exit_code,
                    stdout=result.stdout,
                    stderr=result.stderr if not self.mix_stderr else "",
                    exception=result.exception
                )
            else:
                # Direct command execution
                return self._execute_direct(args, input)
    
    def _execute_direct(self, args: List[str], input: Optional[str] = None) -> CommandResult:
        """Execute command directly without Typer."""
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        
        try:
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                # Execute command logic here
                exit_code = 0
                
        except Exception as e:
            exit_code = 1
            if not self.catch_exceptions:
                raise
            return CommandResult(
                exit_code=exit_code,
                stdout=stdout_buffer.getvalue(),
                stderr=stderr_buffer.getvalue(),
                exception=e
            )
        
        return CommandResult(
            exit_code=exit_code,
            stdout=stdout_buffer.getvalue(),
            stderr=stderr_buffer.getvalue()
        )
    
    @contextmanager
    def _setup_environment(self, env: Dict[str, str]):
        """Set up test environment."""
        original_env = {}
        
        try:
            # Save and set environment
            for key, value in env.items():
                original_env[key] = os.environ.get(key)
                os.environ[key] = value
            
            yield
            
        finally:
            # Restore environment
            for key, original_value in original_env.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value
    
    def isolated(
        self,
        temp_dir: Optional[Path] = None,
        cwd: Optional[Path] = None
    ) -> "IsolatedRunner":
        """Create isolated runner with temporary directory."""
        return IsolatedRunner(self, temp_dir, cwd)


class IsolatedRunner:
    """Runner with isolated filesystem."""
    
    def __init__(
        self,
        runner: CLIRunner,
        temp_dir: Optional[Path] = None,
        cwd: Optional[Path] = None
    ):
        """Initialize isolated runner."""
        self.runner = runner
        self.temp_dir = temp_dir
        self.cwd = cwd
    
    def __enter__(self) -> "IsolatedRunner":
        """Enter isolated context."""
        import tempfile
        
        if not self.temp_dir:
            self._temp_obj = tempfile.TemporaryDirectory()
            self.temp_dir = Path(self._temp_obj.__enter__())
        
        self._original_cwd = os.getcwd()
        os.chdir(self.cwd or self.temp_dir)
        
        return self
    
    def __exit__(self, *args):
        """Exit isolated context."""
        os.chdir(self._original_cwd)
        
        if hasattr(self, "_temp_obj"):
            self._temp_obj.__exit__(*args)
    
    def invoke(self, *args, **kwargs) -> CommandResult:
        """Invoke command in isolated context."""
        return self.runner.invoke(*args, **kwargs)


# Test assertions
def assert_success(result: CommandResult, message: Optional[str] = None) -> None:
    """Assert command succeeded."""
    if not result.success:
        error_msg = f"Command failed with exit code {result.exit_code}"
        if message:
            error_msg = f"{message}: {error_msg}"
        if result.exception:
            error_msg += f"\nException: {result.exception}"
        if result.stderr:
            error_msg += f"\nStderr: {result.stderr}"
        raise AssertionError(error_msg)


def assert_failure(
    result: CommandResult,
    exit_code: Optional[int] = None,
    message: Optional[str] = None
) -> None:
    """Assert command failed."""
    if result.success:
        error_msg = "Command succeeded unexpectedly"
        if message:
            error_msg = f"{message}: {error_msg}"
        raise AssertionError(error_msg)
    
    if exit_code is not None and result.exit_code != exit_code:
        error_msg = f"Expected exit code {exit_code}, got {result.exit_code}"
        if message:
            error_msg = f"{message}: {error_msg}"
        raise AssertionError(error_msg)


def assert_output_contains(
    result: CommandResult,
    text: str,
    in_stderr: bool = False,
    message: Optional[str] = None
) -> None:
    """Assert output contains text."""
    output = result.stderr if in_stderr else result.stdout
    
    if text not in output:
        error_msg = f"Expected '{text}' in {'stderr' if in_stderr else 'stdout'}"
        if message:
            error_msg = f"{message}: {error_msg}"
        error_msg += f"\nActual output: {output}"
        raise AssertionError(error_msg)


def assert_output_not_contains(
    result: CommandResult,
    text: str,
    in_stderr: bool = False,
    message: Optional[str] = None
) -> None:
    """Assert output does not contain text."""
    output = result.stderr if in_stderr else result.stdout
    
    if text in output:
        error_msg = f"Unexpected '{text}' in {'stderr' if in_stderr else 'stdout'}"
        if message:
            error_msg = f"{message}: {error_msg}"
        error_msg += f"\nActual output: {output}"
        raise AssertionError(error_msg)


def assert_file_exists(
    path: Union[str, Path],
    message: Optional[str] = None
) -> None:
    """Assert file exists."""
    path = Path(path)
    if not path.exists():
        error_msg = f"File does not exist: {path}"
        if message:
            error_msg = f"{message}: {error_msg}"
        raise AssertionError(error_msg)


def assert_file_contains(
    path: Union[str, Path],
    text: str,
    message: Optional[str] = None
) -> None:
    """Assert file contains text."""
    path = Path(path)
    assert_file_exists(path, message)
    
    content = path.read_text()
    if text not in content:
        error_msg = f"Expected '{text}' in file {path}"
        if message:
            error_msg = f"{message}: {error_msg}"
        raise AssertionError(error_msg)


# Middleware testing helpers
class MiddlewareTestRunner:
    """Runner for testing middleware."""
    
    def __init__(self, pipeline: Optional[MiddlewarePipeline] = None):
        """Initialize middleware test runner."""
        self.pipeline = pipeline or MiddlewarePipeline()
    
    def execute(
        self,
        command: str,
        handler: Callable,
        *args,
        **kwargs
    ) -> CommandResult:
        """Execute command through middleware."""
        context = CommandContext(
            command_name=command,
            args=args,
            kwargs=kwargs
        )
        
        try:
            result = self.pipeline.execute(context, handler)
            
            return CommandResult(
                exit_code=0 if result.success else 1,
                stdout=str(result.data) if result.data else "",
                stderr=str(result.error) if result.error else "",
                exception=result.error,
                context=context
            )
        except Exception as e:
            return CommandResult(
                exit_code=1,
                stdout="",
                stderr=str(e),
                exception=e,
                context=context
            )


# Pipeline testing helpers
class PipelineTestRunner:
    """Runner for testing command pipelines."""
    
    def __init__(self, pipeline: Optional[CommandPipeline] = None):
        """Initialize pipeline test runner."""
        self.pipeline = pipeline or CommandPipeline()
    
    async def execute(
        self,
        command: str,
        handler: Callable,
        *args,
        **kwargs
    ) -> CommandResult:
        """Execute command through pipeline."""
        try:
            result = await self.pipeline.execute(
                command,
                handler,
                *args,
                **kwargs
            )
            
            return CommandResult(
                exit_code=0,
                stdout=str(result) if result else "",
                stderr="",
            )
        except Exception as e:
            return CommandResult(
                exit_code=1,
                stdout="",
                stderr=str(e),
                exception=e
            )