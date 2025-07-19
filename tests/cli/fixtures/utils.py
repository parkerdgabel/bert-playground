"""CLI testing utilities and helpers."""

import re
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from unittest.mock import Mock, patch

from typer.testing import CliRunner
from rich.console import Console
from rich.table import Table


# ==============================================================================
# CLI Testing Helpers
# ==============================================================================


def run_command(
    runner: CliRunner,
    app: Any,
    args: List[str],
    input: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    catch_exceptions: bool = True
) -> Any:
    """Run a CLI command and return the result."""
    return runner.invoke(
        app,
        args,
        input=input,
        env=env,
        catch_exceptions=catch_exceptions
    )


def assert_success(result: Any, expected_output: Optional[str] = None):
    """Assert command executed successfully."""
    assert result.exit_code == 0, f"Command failed with exit code {result.exit_code}\nOutput: {result.stdout}\nError: {result.stderr}"
    
    if expected_output:
        assert expected_output in result.stdout, f"Expected '{expected_output}' in output, got: {result.stdout}"


def assert_failure(result: Any, expected_error: Optional[str] = None, exit_code: int = 1):
    """Assert command failed with expected error."""
    assert result.exit_code == exit_code, f"Expected exit code {exit_code}, got {result.exit_code}"
    
    if expected_error:
        error_output = result.stderr or result.stdout
        assert expected_error in error_output, f"Expected '{expected_error}' in error output, got: {error_output}"


def assert_json_output(result: Any) -> Dict[str, Any]:
    """Assert output is valid JSON and return parsed data."""
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise AssertionError(f"Output is not valid JSON: {result.stdout}") from e


def assert_table_output(result: Any, expected_columns: List[str]):
    """Assert output contains a table with expected columns."""
    output = result.stdout
    
    # Check for column headers
    for column in expected_columns:
        assert column in output, f"Expected column '{column}' not found in table output"


def extract_table_data(output: str) -> List[Dict[str, str]]:
    """Extract data from Rich table output."""
    lines = output.strip().split("\n")
    
    # Find header line (contains ┃ or │)
    header_idx = None
    for i, line in enumerate(lines):
        if "┃" in line or "│" in line:
            header_idx = i
            break
    
    if header_idx is None:
        return []
    
    # Extract headers
    header_line = lines[header_idx]
    headers = [h.strip() for h in re.split(r"[┃│]", header_line) if h.strip()]
    
    # Extract data rows
    data = []
    for line in lines[header_idx + 2:]:  # Skip header and separator
        if "┃" in line or "│" in line:
            values = [v.strip() for v in re.split(r"[┃│]", line) if v.strip()]
            if len(values) == len(headers):
                data.append(dict(zip(headers, values)))
    
    return data


# ==============================================================================
# Output Parsing Helpers
# ==============================================================================


def parse_progress_output(output: str) -> List[Dict[str, Any]]:
    """Parse progress bar output."""
    progress_updates = []
    
    # Match progress patterns like "50%|████████|"
    pattern = r"(\d+)%\|([█▓▒░ ]+)\|"
    
    for match in re.finditer(pattern, output):
        progress_updates.append({
            "percentage": int(match.group(1)),
            "bar": match.group(2)
        })
    
    return progress_updates


def parse_metric_output(output: str) -> Dict[str, float]:
    """Parse metric output from training/evaluation."""
    metrics = {}
    
    # Common patterns for metrics
    patterns = [
        r"(\w+):\s*([0-9.]+)",  # metric: 0.85
        r"(\w+)\s*=\s*([0-9.]+)",  # metric = 0.85
        r"'(\w+)':\s*([0-9.]+)",  # 'metric': 0.85
    ]
    
    for pattern in patterns:
        for match in re.finditer(pattern, output):
            metric_name = match.group(1)
            metric_value = float(match.group(2))
            metrics[metric_name] = metric_value
    
    return metrics


def parse_file_list_output(output: str) -> List[str]:
    """Parse file listing output."""
    files = []
    
    # Match common file listing patterns
    lines = output.strip().split("\n")
    
    for line in lines:
        # Skip headers and empty lines
        if not line.strip() or line.startswith("─") or "File" in line:
            continue
        
        # Extract filename (handle various formats)
        # Format 1: "filename.ext"
        # Format 2: "- filename.ext"  
        # Format 3: "│ filename.ext │"
        
        line = line.strip()
        if line.startswith("- "):
            line = line[2:]
        if "│" in line:
            parts = line.split("│")
            if len(parts) >= 2:
                line = parts[1].strip()
        
        if line and ("." in line or "/" in line):
            files.append(line)
    
    return files


# ==============================================================================
# Mock Helpers
# ==============================================================================


def mock_successful_command(return_value: Any = None):
    """Create a mock for a successful command execution."""
    mock = Mock()
    mock.return_value = return_value or {"status": "success"}
    return mock


def mock_failed_command(error_message: str = "Command failed"):
    """Create a mock for a failed command execution."""
    mock = Mock()
    mock.side_effect = Exception(error_message)
    return mock


def mock_file_operations(temp_dir: Path):
    """Mock file operations to use temporary directory."""
    def _mock_open(path, *args, **kwargs):
        # Redirect to temp directory
        if isinstance(path, str):
            path = Path(path)
        
        if not path.is_absolute():
            path = temp_dir / path
        
        return open(path, *args, **kwargs)
    
    return patch("builtins.open", side_effect=_mock_open)


def mock_console_output():
    """Mock Rich console for testing output."""
    console = Console(force_terminal=True, width=80)
    captured_output = []
    
    def capture_print(*args, **kwargs):
        captured_output.append({
            "args": args,
            "kwargs": kwargs
        })
    
    console.print = capture_print
    return console, captured_output


# ==============================================================================
# Assertion Helpers
# ==============================================================================


def assert_contains_all(text: str, expected_items: List[str]):
    """Assert text contains all expected items."""
    missing = [item for item in expected_items if item not in text]
    
    if missing:
        raise AssertionError(
            f"Expected items not found in text: {missing}\n"
            f"Text: {text[:200]}..."
        )


def assert_file_created(path: Union[str, Path], content: Optional[str] = None):
    """Assert file was created with optional content check."""
    path = Path(path)
    
    assert path.exists(), f"Expected file not created: {path}"
    
    if content is not None:
        actual_content = path.read_text()
        assert content in actual_content, f"Expected content not found in {path}"


def assert_files_created(paths: List[Union[str, Path]]):
    """Assert multiple files were created."""
    for path in paths:
        assert_file_created(path)


def assert_config_valid(config_path: Union[str, Path], expected_keys: List[str]):
    """Assert configuration file is valid and contains expected keys."""
    import yaml
    
    path = Path(config_path)
    assert path.exists(), f"Config file not found: {path}"
    
    with open(path) as f:
        if path.suffix == ".json":
            config = json.load(f)
        else:
            config = yaml.safe_load(f)
    
    missing_keys = [key for key in expected_keys if key not in config]
    
    if missing_keys:
        raise AssertionError(
            f"Missing keys in config: {missing_keys}\n"
            f"Config keys: {list(config.keys())}"
        )


# ==============================================================================
# Environment Helpers
# ==============================================================================


def with_env_vars(**env_vars):
    """Context manager for temporarily setting environment variables."""
    import os
    
    original_env = {}
    
    for key, value in env_vars.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = str(value)
    
    try:
        yield
    finally:
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def isolate_cli_test(func):
    """Decorator to isolate CLI tests from system environment."""
    def wrapper(*args, **kwargs):
        import os
        
        # Save original environment
        original_cwd = os.getcwd()
        original_env = os.environ.copy()
        
        try:
            # Run test
            return func(*args, **kwargs)
        finally:
            # Restore environment
            os.chdir(original_cwd)
            os.environ.clear()
            os.environ.update(original_env)
    
    return wrapper


# ==============================================================================
# Comparison Helpers
# ==============================================================================


def compare_dataframes(df1: Any, df2: Any, tolerance: float = 1e-6) -> bool:
    """Compare two dataframes for equality."""
    import pandas as pd
    
    if not isinstance(df1, pd.DataFrame) or not isinstance(df2, pd.DataFrame):
        return False
    
    if df1.shape != df2.shape:
        return False
    
    if list(df1.columns) != list(df2.columns):
        return False
    
    try:
        pd.testing.assert_frame_equal(df1, df2, atol=tolerance)
        return True
    except AssertionError:
        return False


def compare_configs(config1: Dict, config2: Dict, ignore_keys: List[str] = None) -> bool:
    """Compare two configuration dictionaries."""
    ignore_keys = ignore_keys or []
    
    def clean_dict(d: Dict) -> Dict:
        return {k: v for k, v in d.items() if k not in ignore_keys}
    
    return clean_dict(config1) == clean_dict(config2)


# ==============================================================================
# Performance Testing Helpers
# ==============================================================================


def measure_command_time(runner: CliRunner, app: Any, args: List[str]) -> float:
    """Measure execution time of a CLI command."""
    import time
    
    start_time = time.time()
    result = runner.invoke(app, args)
    end_time = time.time()
    
    if result.exit_code != 0:
        raise RuntimeError(f"Command failed: {result.stdout}")
    
    return end_time - start_time


def assert_performance(
    execution_time: float,
    max_time: float,
    operation: str = "Command"
):
    """Assert performance is within acceptable limits."""
    if execution_time > max_time:
        raise AssertionError(
            f"{operation} took {execution_time:.2f}s, "
            f"exceeding limit of {max_time:.2f}s"
        )