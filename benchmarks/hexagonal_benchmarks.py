"""Hexagonal Architecture Performance Benchmarks

Comprehensive benchmarks for the hexagonal architecture components including:
- Adapter performance vs direct calls
- Port interface overhead
- Dependency injection overhead
- Service layer performance
- Cross-cutting concern (logging, metrics) overhead
"""

import gc
import time
import tracemalloc
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Protocol
import psutil
import statistics
from unittest.mock import Mock


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


# Sample hexagonal architecture components for benchmarking

class DataPort(Protocol):
    """Data port interface."""
    
    def read_data(self, identifier: str) -> Dict[str, Any]:
        """Read data by identifier."""
        ...
    
    def write_data(self, identifier: str, data: Dict[str, Any]) -> bool:
        """Write data."""
        ...
    
    def delete_data(self, identifier: str) -> bool:
        """Delete data."""
        ...


class ModelPort(Protocol):
    """Model port interface."""
    
    def train(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Train model."""
        ...
    
    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make predictions."""
        ...
    
    def evaluate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate model."""
        ...


class DirectDataService:
    """Direct implementation without adapter pattern."""
    
    def __init__(self):
        self.data_store = {}
        self.operation_count = 0
    
    def read_data(self, identifier: str) -> Dict[str, Any]:
        """Read data directly."""
        self.operation_count += 1
        return self.data_store.get(identifier, {})
    
    def write_data(self, identifier: str, data: Dict[str, Any]) -> bool:
        """Write data directly."""
        self.operation_count += 1
        self.data_store[identifier] = data
        return True
    
    def delete_data(self, identifier: str) -> bool:
        """Delete data directly."""
        self.operation_count += 1
        if identifier in self.data_store:
            del self.data_store[identifier]
            return True
        return False


class DataAdapter:
    """Data adapter implementing hexagonal pattern."""
    
    def __init__(self, port: DataPort):
        self.port = port
        self.operation_count = 0
    
    def read_data(self, identifier: str) -> Dict[str, Any]:
        """Read data through port."""
        self.operation_count += 1
        return self.port.read_data(identifier)
    
    def write_data(self, identifier: str, data: Dict[str, Any]) -> bool:
        """Write data through port."""
        self.operation_count += 1
        return self.port.write_data(identifier, data)
    
    def delete_data(self, identifier: str) -> bool:
        """Delete data through port."""
        self.operation_count += 1
        return self.port.delete_data(identifier)


class InMemoryDataPort:
    """In-memory implementation of data port."""
    
    def __init__(self):
        self.data_store = {}
    
    def read_data(self, identifier: str) -> Dict[str, Any]:
        """Read data from memory."""
        return self.data_store.get(identifier, {})
    
    def write_data(self, identifier: str, data: Dict[str, Any]) -> bool:
        """Write data to memory."""
        self.data_store[identifier] = data
        return True
    
    def delete_data(self, identifier: str) -> bool:
        """Delete data from memory."""
        if identifier in self.data_store:
            del self.data_store[identifier]
            return True
        return False


class ServiceWithDI:
    """Service with dependency injection."""
    
    def __init__(self, data_adapter: DataAdapter, model_port: ModelPort):
        self.data_adapter = data_adapter
        self.model_port = model_port
        self.operation_count = 0
    
    def process_data(self, identifier: str) -> Dict[str, Any]:
        """Process data using injected dependencies."""
        self.operation_count += 1
        
        # Read data
        data = self.data_adapter.read_data(identifier)
        
        # Process with model
        result = self.model_port.predict(data)
        
        # Write result back
        self.data_adapter.write_data(f"{identifier}_result", result)
        
        return result


class DirectService:
    """Direct service without dependency injection."""
    
    def __init__(self):
        self.data_service = DirectDataService()
        self.operation_count = 0
    
    def process_data(self, identifier: str) -> Dict[str, Any]:
        """Process data directly."""
        self.operation_count += 1
        
        # Read data
        data = self.data_service.read_data(identifier)
        
        # Simple processing (mock model)
        result = {"prediction": sum(data.get("features", [1, 2, 3]))}
        
        # Write result back
        self.data_service.write_data(f"{identifier}_result", result)
        
        return result


class MockModelPort:
    """Mock model port for testing."""
    
    def __init__(self):
        self.operation_count = 0
    
    def train(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock training."""
        self.operation_count += 1
        # Simulate training work
        features = data.get("features", [1, 2, 3])
        return {"loss": sum(f * 0.1 for f in features)}
    
    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock prediction."""
        self.operation_count += 1
        # Simulate prediction work
        features = data.get("features", [1, 2, 3])
        return {"prediction": sum(features)}
    
    def evaluate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock evaluation."""
        self.operation_count += 1
        # Simulate evaluation work
        features = data.get("features", [1, 2, 3])
        return {"accuracy": len(features) / 10.0}


class HexagonalBenchmark:
    """Comprehensive hexagonal architecture performance benchmarks."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.results: List[BenchmarkResult] = []
    
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
    
    def benchmark_direct_vs_adapter(self, operation_count: int = 10000) -> BenchmarkResult:
        """Benchmark direct implementation vs adapter pattern."""
        # Setup direct service
        direct_service = DirectDataService()
        
        # Setup adapter service
        port = InMemoryDataPort()
        adapter = DataAdapter(port)
        
        # Prepare test data
        test_data = {"features": [1, 2, 3, 4, 5], "label": 1}
        
        # Benchmark direct calls
        direct_times = []
        
        def run_direct_operations():
            for i in range(operation_count):
                start = time.perf_counter()
                direct_service.write_data(f"item_{i}", test_data)
                data = direct_service.read_data(f"item_{i}")
                direct_service.delete_data(f"item_{i}")
                end = time.perf_counter()
                direct_times.append(end - start)
        
        direct_result, direct_total_time, direct_peak_memory, direct_memory_delta, direct_cpu = self._measure_memory_and_cpu(run_direct_operations)
        
        # Benchmark adapter calls
        adapter_times = []
        
        def run_adapter_operations():
            for i in range(operation_count):
                start = time.perf_counter()
                adapter.write_data(f"item_{i}", test_data)
                data = adapter.read_data(f"item_{i}")
                adapter.delete_data(f"item_{i}")
                end = time.perf_counter()
                adapter_times.append(end - start)
        
        adapter_result, adapter_total_time, adapter_peak_memory, adapter_memory_delta, adapter_cpu = self._measure_memory_and_cpu(run_adapter_operations)
        
        # Calculate overhead
        direct_stats = self._calculate_stats(direct_times)
        adapter_stats = self._calculate_stats(adapter_times)
        
        overhead_percent = ((adapter_stats['avg'] - direct_stats['avg']) / direct_stats['avg']) * 100
        
        return BenchmarkResult(
            name="direct_vs_adapter",
            operation_count=operation_count,
            total_time=adapter_total_time,
            avg_time=adapter_stats['avg'],
            min_time=adapter_stats['min'],
            max_time=adapter_stats['max'],
            p50_time=adapter_stats['p50'],
            p95_time=adapter_stats['p95'],
            p99_time=adapter_stats['p99'],
            memory_peak_mb=adapter_peak_memory,
            memory_delta_mb=adapter_memory_delta,
            cpu_percent=adapter_cpu,
            throughput_ops_per_sec=operation_count / adapter_total_time,
            metadata={
                "direct_avg_time_ms": direct_stats['avg'] * 1000,
                "adapter_avg_time_ms": adapter_stats['avg'] * 1000,
                "overhead_percent": overhead_percent,
                "direct_total_time": direct_total_time,
                "adapter_total_time": adapter_total_time,
                "direct_operations": direct_service.operation_count,
                "adapter_operations": adapter.operation_count
            }
        )
    
    def benchmark_dependency_injection_overhead(self, operation_count: int = 5000) -> BenchmarkResult:
        """Benchmark dependency injection overhead."""
        # Setup services
        direct_service = DirectService()
        
        # DI service
        port = InMemoryDataPort()
        adapter = DataAdapter(port)
        model_port = MockModelPort()
        di_service = ServiceWithDI(adapter, model_port)
        
        # Prepare test data
        test_data = {"features": [1, 2, 3, 4, 5]}
        for i in range(100):  # Pre-populate some data
            direct_service.data_service.write_data(f"data_{i}", test_data)
            adapter.write_data(f"data_{i}", test_data)
        
        # Benchmark direct service
        direct_times = []
        
        def run_direct_service():
            for i in range(operation_count):
                start = time.perf_counter()
                result = direct_service.process_data(f"data_{i % 100}")
                end = time.perf_counter()
                direct_times.append(end - start)
        
        direct_result, direct_total_time, direct_peak_memory, direct_memory_delta, direct_cpu = self._measure_memory_and_cpu(run_direct_service)
        
        # Benchmark DI service
        di_times = []
        
        def run_di_service():
            for i in range(operation_count):
                start = time.perf_counter()
                result = di_service.process_data(f"data_{i % 100}")
                end = time.perf_counter()
                di_times.append(end - start)
        
        di_result, di_total_time, di_peak_memory, di_memory_delta, di_cpu = self._measure_memory_and_cpu(run_di_service)
        
        # Calculate overhead
        direct_stats = self._calculate_stats(direct_times)
        di_stats = self._calculate_stats(di_times)
        
        overhead_percent = ((di_stats['avg'] - direct_stats['avg']) / direct_stats['avg']) * 100
        
        return BenchmarkResult(
            name="dependency_injection_overhead",
            operation_count=operation_count,
            total_time=di_total_time,
            avg_time=di_stats['avg'],
            min_time=di_stats['min'],
            max_time=di_stats['max'],
            p50_time=di_stats['p50'],
            p95_time=di_stats['p95'],
            p99_time=di_stats['p99'],
            memory_peak_mb=di_peak_memory,
            memory_delta_mb=di_memory_delta,
            cpu_percent=di_cpu,
            throughput_ops_per_sec=operation_count / di_total_time,
            metadata={
                "direct_avg_time_ms": direct_stats['avg'] * 1000,
                "di_avg_time_ms": di_stats['avg'] * 1000,
                "overhead_percent": overhead_percent,
                "direct_total_time": direct_total_time,
                "di_total_time": di_total_time,
                "direct_operations": direct_service.operation_count,
                "di_operations": di_service.operation_count,
                "model_operations": model_port.operation_count
            }
        )
    
    def benchmark_port_interface_overhead(self, operation_count: int = 10000) -> BenchmarkResult:
        """Benchmark port interface call overhead."""
        port = InMemoryDataPort()
        test_data = {"features": [1, 2, 3, 4, 5]}
        
        # Direct calls to implementation
        direct_times = []
        
        def run_direct_calls():
            for i in range(operation_count):
                start = time.perf_counter()
                port.write_data(f"item_{i}", test_data)
                data = port.read_data(f"item_{i}")
                end = time.perf_counter()
                direct_times.append(end - start)
        
        direct_result, direct_total_time, direct_peak_memory, direct_memory_delta, direct_cpu = self._measure_memory_and_cpu(run_direct_calls)
        
        # Calls through protocol interface
        port_interface: DataPort = port
        interface_times = []
        
        def run_interface_calls():
            for i in range(operation_count):
                start = time.perf_counter()
                port_interface.write_data(f"item_{i}_if", test_data)
                data = port_interface.read_data(f"item_{i}_if")
                end = time.perf_counter()
                interface_times.append(end - start)
        
        interface_result, interface_total_time, interface_peak_memory, interface_memory_delta, interface_cpu = self._measure_memory_and_cpu(run_interface_calls)
        
        # Calculate overhead
        direct_stats = self._calculate_stats(direct_times)
        interface_stats = self._calculate_stats(interface_times)
        
        overhead_percent = ((interface_stats['avg'] - direct_stats['avg']) / direct_stats['avg']) * 100
        
        return BenchmarkResult(
            name="port_interface_overhead",
            operation_count=operation_count,
            total_time=interface_total_time,
            avg_time=interface_stats['avg'],
            min_time=interface_stats['min'],
            max_time=interface_stats['max'],
            p50_time=interface_stats['p50'],
            p95_time=interface_stats['p95'],
            p99_time=interface_stats['p99'],
            memory_peak_mb=interface_peak_memory,
            memory_delta_mb=interface_memory_delta,
            cpu_percent=interface_cpu,
            throughput_ops_per_sec=operation_count / interface_total_time,
            metadata={
                "direct_avg_time_ms": direct_stats['avg'] * 1000,
                "interface_avg_time_ms": interface_stats['avg'] * 1000,
                "overhead_percent": overhead_percent,
                "direct_total_time": direct_total_time,
                "interface_total_time": interface_total_time
            }
        )
    
    def benchmark_adapter_layer_depth(self, operation_count: int = 5000) -> BenchmarkResult:
        """Benchmark performance with multiple adapter layers."""
        # Create chain of adapters
        base_port = InMemoryDataPort()
        
        # Layer 1: Basic adapter
        adapter1 = DataAdapter(base_port)
        
        # Layer 2: Logging adapter
        class LoggingAdapter:
            def __init__(self, port: DataPort):
                self.port = port
                self.log_count = 0
            
            def read_data(self, identifier: str) -> Dict[str, Any]:
                self.log_count += 1
                return self.port.read_data(identifier)
            
            def write_data(self, identifier: str, data: Dict[str, Any]) -> bool:
                self.log_count += 1
                return self.port.write_data(identifier, data)
            
            def delete_data(self, identifier: str) -> bool:
                self.log_count += 1
                return self.port.delete_data(identifier)
        
        adapter2 = LoggingAdapter(adapter1)
        
        # Layer 3: Caching adapter
        class CachingAdapter:
            def __init__(self, port: DataPort):
                self.port = port
                self.cache = {}
                self.cache_hits = 0
                self.cache_misses = 0
            
            def read_data(self, identifier: str) -> Dict[str, Any]:
                if identifier in self.cache:
                    self.cache_hits += 1
                    return self.cache[identifier]
                else:
                    self.cache_misses += 1
                    data = self.port.read_data(identifier)
                    self.cache[identifier] = data
                    return data
            
            def write_data(self, identifier: str, data: Dict[str, Any]) -> bool:
                self.cache[identifier] = data
                return self.port.write_data(identifier, data)
            
            def delete_data(self, identifier: str) -> bool:
                if identifier in self.cache:
                    del self.cache[identifier]
                return self.port.delete_data(identifier)
        
        adapter3 = CachingAdapter(adapter2)
        
        test_data = {"features": [1, 2, 3, 4, 5]}
        
        # Benchmark different depths
        times_direct = []
        times_1_layer = []
        times_2_layers = []
        times_3_layers = []
        
        def run_depth_benchmark():
            # Direct calls
            for i in range(operation_count // 4):
                start = time.perf_counter()
                base_port.write_data(f"direct_{i}", test_data)
                data = base_port.read_data(f"direct_{i}")
                end = time.perf_counter()
                times_direct.append(end - start)
            
            # 1 layer
            for i in range(operation_count // 4):
                start = time.perf_counter()
                adapter1.write_data(f"layer1_{i}", test_data)
                data = adapter1.read_data(f"layer1_{i}")
                end = time.perf_counter()
                times_1_layer.append(end - start)
            
            # 2 layers
            for i in range(operation_count // 4):
                start = time.perf_counter()
                adapter2.write_data(f"layer2_{i}", test_data)
                data = adapter2.read_data(f"layer2_{i}")
                end = time.perf_counter()
                times_2_layers.append(end - start)
            
            # 3 layers
            for i in range(operation_count // 4):
                start = time.perf_counter()
                adapter3.write_data(f"layer3_{i}", test_data)
                data = adapter3.read_data(f"layer3_{i}")
                end = time.perf_counter()
                times_3_layers.append(end - start)
        
        _, total_time, peak_memory, memory_delta, cpu_percent = self._measure_memory_and_cpu(run_depth_benchmark)
        
        # Calculate stats for each depth
        direct_stats = self._calculate_stats(times_direct)
        layer1_stats = self._calculate_stats(times_1_layer)
        layer2_stats = self._calculate_stats(times_2_layers)
        layer3_stats = self._calculate_stats(times_3_layers)
        
        return BenchmarkResult(
            name="adapter_layer_depth",
            operation_count=operation_count,
            total_time=total_time,
            avg_time=layer3_stats['avg'],  # Use deepest layer as primary metric
            min_time=layer3_stats['min'],
            max_time=layer3_stats['max'],
            p50_time=layer3_stats['p50'],
            p95_time=layer3_stats['p95'],
            p99_time=layer3_stats['p99'],
            memory_peak_mb=peak_memory,
            memory_delta_mb=memory_delta,
            cpu_percent=cpu_percent,
            throughput_ops_per_sec=operation_count / total_time,
            metadata={
                "direct_avg_ms": direct_stats['avg'] * 1000,
                "layer1_avg_ms": layer1_stats['avg'] * 1000,
                "layer2_avg_ms": layer2_stats['avg'] * 1000,
                "layer3_avg_ms": layer3_stats['avg'] * 1000,
                "layer1_overhead_percent": ((layer1_stats['avg'] - direct_stats['avg']) / direct_stats['avg']) * 100,
                "layer2_overhead_percent": ((layer2_stats['avg'] - direct_stats['avg']) / direct_stats['avg']) * 100,
                "layer3_overhead_percent": ((layer3_stats['avg'] - direct_stats['avg']) / direct_stats['avg']) * 100,
                "cache_hits": adapter3.cache_hits,
                "cache_misses": adapter3.cache_misses,
                "log_operations": adapter2.log_count
            }
        )
    
    def benchmark_service_composition(self, operation_count: int = 2000) -> BenchmarkResult:
        """Benchmark service composition overhead."""
        # Setup components
        data_port = InMemoryDataPort()
        data_adapter = DataAdapter(data_port)
        model_port = MockModelPort()
        
        # Composed service
        composed_service = ServiceWithDI(data_adapter, model_port)
        
        # Simple service for comparison
        simple_service = DirectService()
        
        # Pre-populate data
        test_data = {"features": [1, 2, 3, 4, 5]}
        for i in range(100):
            data_adapter.write_data(f"data_{i}", test_data)
            simple_service.data_service.write_data(f"data_{i}", test_data)
        
        # Benchmark composed service
        composed_times = []
        
        def run_composed_service():
            for i in range(operation_count):
                start = time.perf_counter()
                result = composed_service.process_data(f"data_{i % 100}")
                end = time.perf_counter()
                composed_times.append(end - start)
        
        composed_result, composed_total_time, composed_peak_memory, composed_memory_delta, composed_cpu = self._measure_memory_and_cpu(run_composed_service)
        
        # Benchmark simple service
        simple_times = []
        
        def run_simple_service():
            for i in range(operation_count):
                start = time.perf_counter()
                result = simple_service.process_data(f"data_{i % 100}")
                end = time.perf_counter()
                simple_times.append(end - start)
        
        simple_result, simple_total_time, simple_peak_memory, simple_memory_delta, simple_cpu = self._measure_memory_and_cpu(run_simple_service)
        
        # Calculate stats
        composed_stats = self._calculate_stats(composed_times)
        simple_stats = self._calculate_stats(simple_times)
        
        overhead_percent = ((composed_stats['avg'] - simple_stats['avg']) / simple_stats['avg']) * 100
        
        return BenchmarkResult(
            name="service_composition",
            operation_count=operation_count,
            total_time=composed_total_time,
            avg_time=composed_stats['avg'],
            min_time=composed_stats['min'],
            max_time=composed_stats['max'],
            p50_time=composed_stats['p50'],
            p95_time=composed_stats['p95'],
            p99_time=composed_stats['p99'],
            memory_peak_mb=composed_peak_memory,
            memory_delta_mb=composed_memory_delta,
            cpu_percent=composed_cpu,
            throughput_ops_per_sec=operation_count / composed_total_time,
            metadata={
                "simple_avg_ms": simple_stats['avg'] * 1000,
                "composed_avg_ms": composed_stats['avg'] * 1000,
                "overhead_percent": overhead_percent,
                "simple_total_time": simple_total_time,
                "composed_total_time": composed_total_time,
                "composed_operations": composed_service.operation_count,
                "simple_operations": simple_service.operation_count,
                "model_operations": model_port.operation_count
            }
        )
    
    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run all hexagonal architecture benchmarks."""
        print("Running Hexagonal Architecture Benchmarks...")
        
        # Direct vs Adapter comparison
        print("1. Direct vs Adapter performance...")
        self.results.append(self.benchmark_direct_vs_adapter())
        
        # Dependency injection overhead
        print("2. Dependency injection overhead...")
        self.results.append(self.benchmark_dependency_injection_overhead())
        
        # Port interface overhead
        print("3. Port interface overhead...")
        self.results.append(self.benchmark_port_interface_overhead())
        
        # Adapter layer depth
        print("4. Adapter layer depth impact...")
        self.results.append(self.benchmark_adapter_layer_depth())
        
        # Service composition
        print("5. Service composition overhead...")
        self.results.append(self.benchmark_service_composition())
        
        return self.results
    
    def print_results(self) -> None:
        """Print benchmark results in a readable format."""
        print("\n" + "="*80)
        print("HEXAGONAL ARCHITECTURE BENCHMARK RESULTS")
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
    """Run hexagonal architecture benchmarks."""
    benchmark = HexagonalBenchmark()
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
    
    with open('/tmp/hexagonal_benchmark_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\nResults saved to: /tmp/hexagonal_benchmark_results.json")


if __name__ == "__main__":
    main()