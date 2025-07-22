"""Plugin System Performance Benchmarks

Comprehensive benchmarks for the plugin system including:
- Plugin loading and initialization overhead
- Plugin execution performance
- Plugin discovery performance
- Memory usage patterns
- Import and registration overhead
"""

import gc
import importlib
import inspect
import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Type
from unittest.mock import Mock
import psutil
import statistics
import tempfile
import os

from cli.plugins.manager import PluginManager
from cli.plugins.registry import PluginRegistry
from cli.plugins.core import BasePlugin


@dataclass
class BenchmarkResult:
    """Performance benchmark result."""
    name: str
    operation_count: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    p50_time: float
    p95_time: float
    p99_time: float
    memory_peak_mb: float
    memory_delta_mb: float
    cpu_percent: float
    throughput_ops_per_sec: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class MockPlugin(BasePlugin):
    """Mock plugin for benchmarking."""
    
    def __init__(self, name: str = "mock_plugin"):
        self.name = name
        self.initialized = False
        self.call_count = 0
    
    def initialize(self) -> None:
        """Initialize the plugin."""
        self.initialized = True
        # Simulate initialization work
        time.sleep(0.001)  # 1ms
    
    def execute(self, *args, **kwargs) -> Any:
        """Execute plugin functionality."""
        self.call_count += 1
        # Simulate some work
        result = sum(range(100))  # Simple computation
        return result
    
    def cleanup(self) -> None:
        """Cleanup plugin resources."""
        self.initialized = False


class PluginSystemBenchmark:
    """Comprehensive plugin system performance benchmarks."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.results: List[BenchmarkResult] = []
        self.temp_dir = None
    
    def setup(self):
        """Setup temporary directory for plugin files."""
        self.temp_dir = tempfile.mkdtemp(prefix="plugin_bench_")
        return self.temp_dir
    
    def teardown(self):
        """Clean up temporary files."""
        if self.temp_dir:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _measure_memory_and_cpu(self, func: Callable) -> tuple:
        """Measure memory and CPU usage during function execution."""
        gc.collect()  # Clean up before measurement
        
        tracemalloc.start()
        cpu_before = self.process.cpu_percent()
        memory_before = self.process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.perf_counter()
        result = func()
        end_time = time.perf_counter()
        
        memory_after = self.process.memory_info().rss / 1024 / 1024  # MB
        cpu_after = self.process.cpu_percent()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return (
            result,
            end_time - start_time,
            peak / 1024 / 1024,  # Peak memory in MB
            memory_after - memory_before,  # Memory delta
            (cpu_before + cpu_after) / 2  # Average CPU
        )
    
    def _calculate_stats(self, times: List[float]) -> Dict[str, float]:
        """Calculate timing statistics."""
        if not times:
            return {'avg': 0, 'min': 0, 'max': 0, 'p50': 0, 'p95': 0, 'p99': 0}
        
        times.sort()
        return {
            'avg': statistics.mean(times),
            'min': min(times),
            'max': max(times),
            'p50': statistics.median(times),
            'p95': times[int(0.95 * len(times))],
            'p99': times[int(0.99 * len(times))] if len(times) >= 100 else times[-1]
        }
    
    def create_mock_plugin_file(self, plugin_name: str, complexity: str = "simple") -> str:
        """Create a mock plugin file for testing."""
        plugin_content = f'''
"""Mock plugin: {plugin_name}"""

from cli.plugins.core import BasePlugin
import time

class {plugin_name.capitalize()}Plugin(BasePlugin):
    """Mock plugin for benchmarking."""
    
    def __init__(self):
        self.name = "{plugin_name}"
        self.initialized = False
        self.call_count = 0
    
    def initialize(self) -> None:
        """Initialize the plugin."""
        self.initialized = True
        # Simulate initialization complexity
        '''
        
        if complexity == "complex":
            plugin_content += '''
        # Heavy initialization
        import json
        import hashlib
        data = {{"key": "value" for i in range(1000)}}
        hash_result = hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()
        time.sleep(0.01)  # 10ms
            '''
        else:
            plugin_content += '''
        time.sleep(0.001)  # 1ms
            '''
        
        plugin_content += f'''
    
    def execute(self, *args, **kwargs):
        """Execute plugin functionality."""
        self.call_count += 1
        '''
        
        if complexity == "complex":
            plugin_content += '''
        # Heavy computation
        result = sum(i**2 for i in range(1000))
        time.sleep(0.005)  # 5ms
            '''
        else:
            plugin_content += '''
        result = sum(range(100))
            '''
        
        plugin_content += '''
        return result
    
    def cleanup(self) -> None:
        """Cleanup plugin resources."""
        self.initialized = False

# Plugin registration
plugin_instance = {plugin_name.capitalize()}Plugin()
'''
        
        plugin_file = os.path.join(self.temp_dir, f"{plugin_name}_plugin.py")
        with open(plugin_file, 'w') as f:
            f.write(plugin_content)
        
        return plugin_file
    
    def benchmark_plugin_registration(self, plugin_count: int = 50) -> BenchmarkResult:
        """Benchmark plugin registration performance."""
        registry = PluginRegistry()
        times = []
        
        def register_plugins():
            nonlocal times
            for i in range(plugin_count):
                plugin = MockPlugin(f"plugin_{i}")
                start = time.perf_counter()
                registry.register(f"plugin_{i}", plugin)
                end = time.perf_counter()
                times.append(end - start)
        
        _, total_time, peak_memory, memory_delta, cpu_percent = self._measure_memory_and_cpu(register_plugins)
        
        stats = self._calculate_stats(times)
        
        return BenchmarkResult(
            name="plugin_registration",
            operation_count=plugin_count,
            total_time=total_time,
            avg_time=stats['avg'],
            min_time=stats['min'],
            max_time=stats['max'],
            p50_time=stats['p50'],
            p95_time=stats['p95'],
            p99_time=stats['p99'],
            memory_peak_mb=peak_memory,
            memory_delta_mb=memory_delta,
            cpu_percent=cpu_percent,
            throughput_ops_per_sec=plugin_count / total_time,
            metadata={
                "registered_plugins": len(registry.plugins),
                "registry_size_mb": sys.getsizeof(registry.plugins) / 1024 / 1024
            }
        )
    
    def benchmark_plugin_discovery(self, plugin_count: int = 20) -> BenchmarkResult:
        """Benchmark plugin discovery performance."""
        self.setup()
        
        # Create plugin files
        plugin_files = []
        for i in range(plugin_count):
            complexity = "complex" if i % 5 == 0 else "simple"
            plugin_file = self.create_mock_plugin_file(f"discoverable_{i}", complexity)
            plugin_files.append(plugin_file)
        
        manager = PluginManager()
        times = []
        
        def discover_plugins():
            nonlocal times
            start = time.perf_counter()
            # Simulate discovery by adding temp_dir to path and importing
            if self.temp_dir not in sys.path:
                sys.path.insert(0, self.temp_dir)
            
            discovered = manager.discover_plugins([self.temp_dir])
            end = time.perf_counter()
            times.append(end - start)
            return discovered
        
        discovered_plugins, total_time, peak_memory, memory_delta, cpu_percent = self._measure_memory_and_cpu(discover_plugins)
        
        stats = self._calculate_stats(times)
        
        self.teardown()
        
        return BenchmarkResult(
            name="plugin_discovery",
            operation_count=1,  # Single discovery operation
            total_time=total_time,
            avg_time=stats['avg'] if times else total_time,
            min_time=stats['min'],
            max_time=stats['max'],
            p50_time=stats['p50'],
            p95_time=stats['p95'],
            p99_time=stats['p99'],
            memory_peak_mb=peak_memory,
            memory_delta_mb=memory_delta,
            cpu_percent=cpu_percent,
            throughput_ops_per_sec=plugin_count / total_time,
            metadata={
                "plugin_files_created": len(plugin_files),
                "plugins_discovered": len(discovered_plugins) if discovered_plugins else 0,
                "discovery_paths": [self.temp_dir]
            }
        )
    
    def benchmark_plugin_initialization(self, plugin_count: int = 100) -> BenchmarkResult:
        """Benchmark plugin initialization performance."""
        plugins = [MockPlugin(f"init_plugin_{i}") for i in range(plugin_count)]
        times = []
        
        def initialize_plugins():
            nonlocal times
            for plugin in plugins:
                start = time.perf_counter()
                plugin.initialize()
                end = time.perf_counter()
                times.append(end - start)
        
        _, total_time, peak_memory, memory_delta, cpu_percent = self._measure_memory_and_cpu(initialize_plugins)
        
        stats = self._calculate_stats(times)
        
        return BenchmarkResult(
            name="plugin_initialization",
            operation_count=plugin_count,
            total_time=total_time,
            avg_time=stats['avg'],
            min_time=stats['min'],
            max_time=stats['max'],
            p50_time=stats['p50'],
            p95_time=stats['p95'],
            p99_time=stats['p99'],
            memory_peak_mb=peak_memory,
            memory_delta_mb=memory_delta,
            cpu_percent=cpu_percent,
            throughput_ops_per_sec=plugin_count / total_time,
            metadata={
                "initialized_plugins": sum(1 for p in plugins if p.initialized)
            }
        )
    
    def benchmark_plugin_execution(self, execution_count: int = 1000) -> BenchmarkResult:
        """Benchmark plugin execution performance."""
        plugin = MockPlugin("execution_test")
        plugin.initialize()
        
        times = []
        
        def execute_plugin():
            nonlocal times
            for i in range(execution_count):
                start = time.perf_counter()
                result = plugin.execute(data=f"test_data_{i}")
                end = time.perf_counter()
                times.append(end - start)
        
        _, total_time, peak_memory, memory_delta, cpu_percent = self._measure_memory_and_cpu(execute_plugin)
        
        stats = self._calculate_stats(times)
        
        return BenchmarkResult(
            name="plugin_execution",
            operation_count=execution_count,
            total_time=total_time,
            avg_time=stats['avg'],
            min_time=stats['min'],
            max_time=stats['max'],
            p50_time=stats['p50'],
            p95_time=stats['p95'],
            p99_time=stats['p99'],
            memory_peak_mb=peak_memory,
            memory_delta_mb=memory_delta,
            cpu_percent=cpu_percent,
            throughput_ops_per_sec=execution_count / total_time,
            metadata={
                "plugin_call_count": plugin.call_count
            }
        )
    
    def benchmark_plugin_manager_overhead(self, operation_count: int = 500) -> BenchmarkResult:
        """Benchmark plugin manager overhead."""
        manager = PluginManager()
        
        # Register some plugins
        plugins = []
        for i in range(10):
            plugin = MockPlugin(f"managed_plugin_{i}")
            manager.register_plugin(f"managed_plugin_{i}", plugin)
            plugins.append(plugin)
        
        times = []
        
        def manager_operations():
            nonlocal times
            for i in range(operation_count):
                start = time.perf_counter()
                
                # Mix of manager operations
                if i % 4 == 0:
                    # Get plugin
                    plugin = manager.get_plugin(f"managed_plugin_{i % 10}")
                elif i % 4 == 1:
                    # List plugins
                    plugin_list = manager.list_plugins()
                elif i % 4 == 2:
                    # Check if plugin exists
                    exists = manager.has_plugin(f"managed_plugin_{i % 10}")
                else:
                    # Execute plugin
                    plugin = manager.get_plugin(f"managed_plugin_{i % 10}")
                    if plugin:
                        result = plugin.execute()
                
                end = time.perf_counter()
                times.append(end - start)
        
        _, total_time, peak_memory, memory_delta, cpu_percent = self._measure_memory_and_cpu(manager_operations)
        
        stats = self._calculate_stats(times)
        
        return BenchmarkResult(
            name="plugin_manager_overhead",
            operation_count=operation_count,
            total_time=total_time,
            avg_time=stats['avg'],
            min_time=stats['min'],
            max_time=stats['max'],
            p50_time=stats['p50'],
            p95_time=stats['p95'],
            p99_time=stats['p99'],
            memory_peak_mb=peak_memory,
            memory_delta_mb=memory_delta,
            cpu_percent=cpu_percent,
            throughput_ops_per_sec=operation_count / total_time,
            metadata={
                "registered_plugins": len(manager.plugins),
                "total_executions": sum(p.call_count for p in plugins)
            }
        )
    
    def benchmark_plugin_import_overhead(self, import_count: int = 50) -> BenchmarkResult:
        """Benchmark plugin import overhead."""
        self.setup()
        
        # Create plugin files
        plugin_files = []
        for i in range(import_count):
            plugin_file = self.create_mock_plugin_file(f"import_test_{i}")
            plugin_files.append(plugin_file)
        
        # Add temp directory to Python path
        if self.temp_dir not in sys.path:
            sys.path.insert(0, self.temp_dir)
        
        times = []
        imported_modules = []
        
        def import_plugins():
            nonlocal times, imported_modules
            for i in range(import_count):
                module_name = f"import_test_{i}_plugin"
                start = time.perf_counter()
                try:
                    module = importlib.import_module(module_name)
                    imported_modules.append(module)
                except ImportError:
                    pass
                end = time.perf_counter()
                times.append(end - start)
        
        _, total_time, peak_memory, memory_delta, cpu_percent = self._measure_memory_and_cpu(import_plugins)
        
        stats = self._calculate_stats(times)
        
        # Cleanup
        for i in range(import_count):
            module_name = f"import_test_{i}_plugin"
            if module_name in sys.modules:
                del sys.modules[module_name]
        
        if self.temp_dir in sys.path:
            sys.path.remove(self.temp_dir)
        
        self.teardown()
        
        return BenchmarkResult(
            name="plugin_import_overhead",
            operation_count=import_count,
            total_time=total_time,
            avg_time=stats['avg'],
            min_time=stats['min'],
            max_time=stats['max'],
            p50_time=stats['p50'],
            p95_time=stats['p95'],
            p99_time=stats['p99'],
            memory_peak_mb=peak_memory,
            memory_delta_mb=memory_delta,
            cpu_percent=cpu_percent,
            throughput_ops_per_sec=import_count / total_time,
            metadata={
                "plugin_files_created": len(plugin_files),
                "successful_imports": len(imported_modules)
            }
        )
    
    def benchmark_memory_scaling(self, max_plugins: int = 200) -> BenchmarkResult:
        """Benchmark memory usage scaling with plugin count."""
        manager = PluginManager()
        memory_usage = []
        
        def measure_memory_with_plugins():
            for plugin_count in range(0, max_plugins + 1, 20):
                # Add plugins to reach target count
                current_count = len(manager.plugins)
                while current_count < plugin_count:
                    plugin = MockPlugin(f"scale_plugin_{current_count}")
                    manager.register_plugin(f"scale_plugin_{current_count}", plugin)
                    current_count += 1
                
                # Measure memory
                gc.collect()
                memory_mb = self.process.memory_info().rss / 1024 / 1024
                memory_usage.append(memory_mb)
        
        start_time = time.perf_counter()
        measure_memory_with_plugins()
        end_time = time.perf_counter()
        
        return BenchmarkResult(
            name="memory_scaling",
            operation_count=max_plugins,
            total_time=end_time - start_time,
            avg_time=(end_time - start_time) / max_plugins,
            min_time=0,
            max_time=0,
            p50_time=0,
            p95_time=0,
            p99_time=0,
            memory_peak_mb=max(memory_usage),
            memory_delta_mb=memory_usage[-1] - memory_usage[0],
            cpu_percent=0,
            throughput_ops_per_sec=max_plugins / (end_time - start_time),
            metadata={
                "memory_progression": memory_usage,
                "max_plugins": max_plugins,
                "final_plugin_count": len(manager.plugins)
            }
        )
    
    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run all plugin system benchmarks."""
        print("Running Plugin System Benchmarks...")
        
        # Plugin registration
        print("1. Plugin registration performance...")
        self.results.append(self.benchmark_plugin_registration())
        
        # Plugin discovery
        print("2. Plugin discovery performance...")
        self.results.append(self.benchmark_plugin_discovery())
        
        # Plugin initialization
        print("3. Plugin initialization performance...")
        self.results.append(self.benchmark_plugin_initialization())
        
        # Plugin execution
        print("4. Plugin execution performance...")
        self.results.append(self.benchmark_plugin_execution())
        
        # Plugin manager overhead
        print("5. Plugin manager overhead...")
        self.results.append(self.benchmark_plugin_manager_overhead())
        
        # Plugin import overhead
        print("6. Plugin import overhead...")
        self.results.append(self.benchmark_plugin_import_overhead())
        
        # Memory scaling
        print("7. Memory scaling characteristics...")
        self.results.append(self.benchmark_memory_scaling())
        
        return self.results
    
    def print_results(self) -> None:
        """Print benchmark results in a readable format."""
        print("\n" + "="*80)
        print("PLUGIN SYSTEM BENCHMARK RESULTS")
        print("="*80)
        
        for result in self.results:
            print(f"\nBenchmark: {result.name}")
            print(f"Operations: {result.operation_count:,}")
            print(f"Total Time: {result.total_time:.3f}s")
            print(f"Throughput: {result.throughput_ops_per_sec:,.0f} ops/sec")
            print(f"Average Time: {result.avg_time*1000:.3f}ms")
            print(f"P95 Time: {result.p95_time*1000:.3f}ms")
            print(f"P99 Time: {result.p99_time*1000:.3f}ms")
            print(f"Peak Memory: {result.memory_peak_mb:.2f}MB")
            print(f"Memory Delta: {result.memory_delta_mb:.2f}MB")
            if result.metadata:
                print("Metadata:", result.metadata)
            print("-" * 40)


def main():
    """Run plugin system benchmarks."""
    benchmark = PluginSystemBenchmark()
    try:
        results = benchmark.run_all_benchmarks()
        benchmark.print_results()
        
        # Save results for comparison
        import json
        results_data = []
        for result in results:
            result_dict = {
                'name': result.name,
                'operation_count': result.operation_count,
                'total_time': result.total_time,
                'throughput_ops_per_sec': result.throughput_ops_per_sec,
                'avg_time': result.avg_time,
                'p95_time': result.p95_time,
                'p99_time': result.p99_time,
                'memory_peak_mb': result.memory_peak_mb,
                'memory_delta_mb': result.memory_delta_mb,
                'cpu_percent': result.cpu_percent,
                'metadata': result.metadata
            }
            results_data.append(result_dict)
        
        with open('/tmp/plugin_system_benchmark_results.json', 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\nResults saved to: /tmp/plugin_system_benchmark_results.json")
        
    finally:
        benchmark.teardown()


if __name__ == "__main__":
    main()