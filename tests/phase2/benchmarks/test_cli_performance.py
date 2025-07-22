"""Performance benchmarks for CLI response times in Phase 2.

This module benchmarks CLI command response times and startup performance
to ensure the CLI remains fast and responsive with new features.
"""

import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import patch

import pytest
import yaml


class CLIBenchmark:
    """Benchmark suite for CLI operations."""
    
    def __init__(self):
        self.results = {}
    
    def time_command(self, name: str, cmd_args: List[str], timeout: int = 30):
        """Time a CLI command execution."""
        start_time = time.perf_counter()
        
        try:
            result = subprocess.run(
                ["uv", "run", "python", "-m", "cli"] + cmd_args,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=Path(__file__).parent.parent.parent.parent
            )
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            self.results[name] = {
                "duration": duration,
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
            return result, duration
            
        except subprocess.TimeoutExpired:
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            self.results[name] = {
                "duration": duration,
                "success": False,
                "timeout": True
            }
            
            return None, duration
    
    def report(self) -> str:
        """Generate benchmark report."""
        lines = ["\nCLI Performance Report:", "=" * 40]
        
        for name, result in self.results.items():
            status = "✓" if result["success"] else "✗"
            lines.append(f"{status} {name}: {result['duration']:.3f}s")
            
            if not result["success"] and result.get("stderr"):
                lines.append(f"    Error: {result['stderr'][:100]}...")
        
        return "\n".join(lines)


class TestCLIPerformance:
    """Test CLI command performance."""
    
    @pytest.fixture
    def benchmark(self):
        """Create CLI benchmark instance."""
        return CLIBenchmark()
    
    @pytest.fixture
    def temp_project(self, tmp_path):
        """Create temporary project for testing."""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()
        
        # Create basic config
        config = {
            "project": {"name": "test-project", "version": "0.1.0"},
            "models": {"model_type": "bert_classifier", "num_labels": 2},
            "training": {"num_epochs": 1, "batch_size": 8},
            "data": {"train_path": "data/train.csv"}
        }
        
        with open(project_dir / "k-bert.yaml", "w") as f:
            yaml.dump(config, f)
        
        # Create minimal data
        data_dir = project_dir / "data"
        data_dir.mkdir()
        
        with open(data_dir / "train.csv", "w") as f:
            f.write("text,label\n")
            f.write("sample text 1,0\n")
            f.write("sample text 2,1\n")
        
        return project_dir
    
    def test_startup_time(self, benchmark):
        """Test CLI startup time."""
        # Test basic help command
        result, duration = benchmark.time_command("help", ["--help"])
        
        # Help should be very fast
        assert duration < 3.0  # Less than 3 seconds
        assert benchmark.results["help"]["success"]
        
        print(f"CLI startup time: {duration:.3f}s")
    
    def test_command_discovery(self, benchmark):
        """Test command discovery performance."""
        commands = [
            "config --help",
            "train --help",
            "predict --help",
            "benchmark --help",
            "info --help",
        ]
        
        for cmd in commands:
            cmd_name = cmd.replace(" --help", "")
            result, duration = benchmark.time_command(
                f"{cmd_name} help",
                cmd.split()
            )
            
            # Each help command should be fast
            assert duration < 2.0
        
        print(benchmark.report())
    
    def test_config_operations(self, benchmark, tmp_path):
        """Test configuration operation performance."""
        # Test config init
        with patch('os.getcwd', return_value=str(tmp_path)):
            result, duration = benchmark.time_command(
                "config init",
                ["config", "init", "--non-interactive"]
            )
        
        # Config operations should be fast
        assert duration < 5.0
        
        # Test config validation
        config_file = tmp_path / "k-bert.yaml"
        if config_file.exists():
            result, duration = benchmark.time_command(
                "config validate",
                ["config", "validate", str(config_file)]
            )
            assert duration < 2.0
        
        print(benchmark.report())
    
    def test_info_commands(self, benchmark):
        """Test info command performance."""
        info_commands = [
            ["info", "system"],
            ["info", "models"],
            ["info", "data"],
        ]
        
        for cmd in info_commands:
            cmd_name = " ".join(cmd)
            result, duration = benchmark.time_command(cmd_name, cmd)
            
            # Info commands should be very fast
            assert duration < 1.0
        
        print(benchmark.report())
    
    def test_plugin_discovery(self, benchmark, temp_project):
        """Test plugin discovery performance."""
        # Create a plugin in the project
        plugin_dir = temp_project / "src" / "plugins"
        plugin_dir.mkdir(parents=True)
        
        plugin_code = '''
from cli.plugins.base import BasePlugin

class TestPlugin(BasePlugin):
    """A simple test plugin."""
    pass
'''
        
        with open(plugin_dir / "test_plugin.py", "w") as f:
            f.write(plugin_code)
        
        with open(plugin_dir / "__init__.py", "w") as f:
            f.write("")
        
        # Test plugin discovery
        with patch('os.getcwd', return_value=str(temp_project)):
            result, duration = benchmark.time_command(
                "project info",
                ["info", "project"]
            )
        
        # Plugin discovery should be reasonably fast
        assert duration < 3.0
        
        print(benchmark.report())
    
    def test_model_operations(self, benchmark, temp_project):
        """Test model-related command performance."""
        # Test model list
        result, duration = benchmark.time_command(
            "model list",
            ["model", "list"]
        )
        
        # Model operations should be fast
        assert duration < 2.0
        
        # Test model info
        result, duration = benchmark.time_command(
            "model info bert",
            ["model", "info", "bert_classifier"]
        )
        
        assert duration < 1.0
        
        print(benchmark.report())
    
    def test_data_operations(self, benchmark, temp_project):
        """Test data-related command performance."""
        data_file = temp_project / "data" / "train.csv"
        
        # Test data info
        result, duration = benchmark.time_command(
            "data info",
            ["info", "data", str(data_file)]
        )
        
        # Data operations should be fast
        assert duration < 2.0
        
        print(benchmark.report())
    
    def test_config_hierarchy_resolution(self, benchmark, temp_project):
        """Test config hierarchy resolution performance."""
        # Create user config
        user_config_dir = temp_project / ".k-bert"
        user_config_dir.mkdir()
        
        user_config = {"user": {"name": "test"}}
        with open(user_config_dir / "config.yaml", "w") as f:
            yaml.dump(user_config, f)
        
        # Test config resolution
        with patch('os.getcwd', return_value=str(temp_project)):
            result, duration = benchmark.time_command(
                "config show",
                ["config", "show"]
            )
        
        # Config resolution should be fast even with hierarchy
        assert duration < 1.5
        
        print(benchmark.report())
    
    def test_error_handling_performance(self, benchmark):
        """Test error handling doesn't significantly slow down commands."""
        # Test invalid command
        result, duration = benchmark.time_command(
            "invalid command",
            ["invalid", "command"]
        )
        
        # Error handling should still be fast
        assert duration < 2.0
        
        # Test missing file
        result, duration = benchmark.time_command(
            "config validate missing",
            ["config", "validate", "/non/existent/file.yaml"]
        )
        
        assert duration < 1.0
        
        print(benchmark.report())
    
    def test_completion_performance(self, benchmark):
        """Test command completion performance."""
        # This would typically test shell completion, but we'll simulate
        # by testing command parsing speed
        
        commands_to_parse = [
            ["train", "--help"],
            ["predict", "--help"],
            ["config", "init", "--help"],
            ["model", "list", "--help"],
        ]
        
        total_time = 0
        for cmd in commands_to_parse:
            result, duration = benchmark.time_command(
                f"parse {' '.join(cmd)}",
                cmd
            )
            total_time += duration
        
        # Average parsing time should be very fast
        avg_time = total_time / len(commands_to_parse)
        assert avg_time < 1.0
        
        print(f"Average command parsing time: {avg_time:.3f}s")
    
    def test_memory_usage(self, benchmark):
        """Test CLI memory usage."""
        import psutil
        import os
        
        # Measure memory before and after CLI operations
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Run several commands
        commands = [
            ["--help"],
            ["info", "system"],
            ["model", "list"],
            ["config", "init", "--help"],
        ]
        
        for cmd in commands:
            result, duration = benchmark.time_command(
                f"memory test {' '.join(cmd)}",
                cmd
            )
        
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        # Memory increase should be reasonable
        assert memory_increase < 50  # Less than 50MB increase
        
        print(f"Memory increase: {memory_increase:.1f} MB")
    
    def test_concurrent_cli_calls(self, benchmark):
        """Test performance with concurrent CLI calls."""
        import concurrent.futures
        import threading
        
        def run_command(cmd_args):
            """Run a CLI command and return timing."""
            start = time.perf_counter()
            result = subprocess.run(
                ["uv", "run", "python", "-m", "cli"] + cmd_args,
                capture_output=True,
                text=True,
                timeout=10,
                cwd=Path(__file__).parent.parent.parent.parent
            )
            end = time.perf_counter()
            return end - start, result.returncode == 0
        
        # Commands to run concurrently
        commands = [
            ["--help"],
            ["info", "system"],
            ["model", "list"],
            ["config", "init", "--help"],
        ] * 3  # Run each command 3 times
        
        # Run concurrently
        start_time = time.perf_counter()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(run_command, cmd) for cmd in commands]
            results = [future.result() for future in futures]
        
        end_time = time.perf_counter()
        total_concurrent_time = end_time - start_time
        
        # Run sequentially for comparison
        start_time = time.perf_counter()
        sequential_results = [run_command(cmd) for cmd in commands]
        end_time = time.perf_counter()
        total_sequential_time = end_time - start_time
        
        # Concurrent execution should provide some benefit
        speedup = total_sequential_time / total_concurrent_time
        assert speedup >= 1.0  # At least no slower
        
        print(f"Concurrent CLI speedup: {speedup:.2f}x")
        
        # All commands should succeed
        assert all(result[1] for result in results)
        assert all(result[1] for result in sequential_results)