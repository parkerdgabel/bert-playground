"""Event System Performance Benchmarks

Comprehensive benchmarks for the event-driven architecture components including:
- Event pub/sub performance
- Event handler execution overhead
- Memory usage patterns
- Throughput characteristics
"""

import asyncio
import gc
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor
import psutil
import statistics

from cli.core.events import EventBus, Event


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


class EventSystemBenchmark:
    """Comprehensive event system performance benchmarks."""
    
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
        times.sort()
        return {
            'avg': statistics.mean(times),
            'min': min(times),
            'max': max(times),
            'p50': statistics.median(times),
            'p95': times[int(0.95 * len(times))],
            'p99': times[int(0.99 * len(times))] if len(times) >= 100 else times[-1]
        }
    
    def benchmark_basic_pub_sub(self, event_count: int = 10000) -> BenchmarkResult:
        """Benchmark basic pub/sub performance."""
        event_bus = EventBus()
        handler_call_count = 0
        
        def simple_handler(event: Event) -> None:
            nonlocal handler_call_count
            handler_call_count += 1
        
        # Register handler
        event_bus.subscribe("test.event", simple_handler)
        
        # Benchmark publishing
        times = []
        
        def publish_events():
            nonlocal times
            for i in range(event_count):
                start = time.perf_counter()
                event_bus.publish("test.event", {"data": f"payload_{i}"})
                end = time.perf_counter()
                times.append(end - start)
        
        _, total_time, peak_memory, memory_delta, cpu_percent = self._measure_memory_and_cpu(publish_events)
        
        stats = self._calculate_stats(times)
        
        return BenchmarkResult(
            name="basic_pub_sub",
            operation_count=event_count,
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
            throughput_ops_per_sec=event_count / total_time,
            metadata={"handler_calls": handler_call_count}
        )
    
    def benchmark_multiple_handlers(self, event_count: int = 5000, handler_count: int = 10) -> BenchmarkResult:
        """Benchmark performance with multiple event handlers."""
        event_bus = EventBus()
        handler_calls = [0] * handler_count
        
        # Create multiple handlers
        for i in range(handler_count):
            def make_handler(idx):
                def handler(event: Event) -> None:
                    handler_calls[idx] += 1
                    # Simulate some work
                    time.sleep(0.000001)  # 1 microsecond
                return handler
            
            event_bus.subscribe("test.multi", make_handler(i))
        
        # Benchmark publishing
        times = []
        
        def publish_events():
            nonlocal times
            for i in range(event_count):
                start = time.perf_counter()
                event_bus.publish("test.multi", {"data": f"payload_{i}"})
                end = time.perf_counter()
                times.append(end - start)
        
        _, total_time, peak_memory, memory_delta, cpu_percent = self._measure_memory_and_cpu(publish_events)
        
        stats = self._calculate_stats(times)
        
        return BenchmarkResult(
            name="multiple_handlers",
            operation_count=event_count,
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
            throughput_ops_per_sec=event_count / total_time,
            metadata={
                "handler_count": handler_count,
                "total_handler_calls": sum(handler_calls)
            }
        )
    
    def benchmark_async_handlers(self, event_count: int = 1000) -> BenchmarkResult:
        """Benchmark asynchronous event handlers."""
        event_bus = EventBus()
        handler_call_count = 0
        
        async def async_handler(event: Event) -> None:
            nonlocal handler_call_count
            handler_call_count += 1
            await asyncio.sleep(0.001)  # Simulate async work
        
        event_bus.subscribe("test.async", async_handler)
        
        # Benchmark publishing
        times = []
        
        async def publish_events_async():
            nonlocal times
            tasks = []
            for i in range(event_count):
                start = time.perf_counter()
                # Simulate async publishing (event bus would handle this)
                task = asyncio.create_task(async_handler(Event("test.async", {"data": f"payload_{i}"})))
                tasks.append(task)
                end = time.perf_counter()
                times.append(end - start)
            
            await asyncio.gather(*tasks)
        
        def run_async_benchmark():
            return asyncio.run(publish_events_async())
        
        _, total_time, peak_memory, memory_delta, cpu_percent = self._measure_memory_and_cpu(run_async_benchmark)
        
        stats = self._calculate_stats(times)
        
        return BenchmarkResult(
            name="async_handlers",
            operation_count=event_count,
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
            throughput_ops_per_sec=event_count / total_time,
            metadata={"handler_calls": handler_call_count}
        )
    
    def benchmark_event_filtering(self, event_count: int = 10000) -> BenchmarkResult:
        """Benchmark event filtering performance."""
        event_bus = EventBus()
        matched_events = 0
        
        def filtered_handler(event: Event) -> None:
            nonlocal matched_events
            if event.data.get("important", False):
                matched_events += 1
        
        event_bus.subscribe("test.filter", filtered_handler)
        
        # Benchmark with mixed events (50% important)
        times = []
        
        def publish_filtered_events():
            nonlocal times
            for i in range(event_count):
                start = time.perf_counter()
                important = i % 2 == 0
                event_bus.publish("test.filter", {
                    "data": f"payload_{i}",
                    "important": important
                })
                end = time.perf_counter()
                times.append(end - start)
        
        _, total_time, peak_memory, memory_delta, cpu_percent = self._measure_memory_and_cpu(publish_filtered_events)
        
        stats = self._calculate_stats(times)
        
        return BenchmarkResult(
            name="event_filtering",
            operation_count=event_count,
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
            throughput_ops_per_sec=event_count / total_time,
            metadata={"matched_events": matched_events}
        )
    
    def benchmark_concurrent_publishing(self, event_count: int = 5000, thread_count: int = 4) -> BenchmarkResult:
        """Benchmark concurrent event publishing."""
        event_bus = EventBus()
        handler_call_count = 0
        lock = asyncio.Lock() if hasattr(asyncio, 'Lock') else None
        
        def thread_safe_handler(event: Event) -> None:
            nonlocal handler_call_count
            handler_call_count += 1
        
        event_bus.subscribe("test.concurrent", thread_safe_handler)
        
        # Benchmark concurrent publishing
        times = []
        
        def publish_batch(batch_size: int, thread_id: int):
            batch_times = []
            for i in range(batch_size):
                start = time.perf_counter()
                event_bus.publish("test.concurrent", {
                    "data": f"payload_{thread_id}_{i}",
                    "thread_id": thread_id
                })
                end = time.perf_counter()
                batch_times.append(end - start)
            return batch_times
        
        def concurrent_publish():
            batch_size = event_count // thread_count
            with ThreadPoolExecutor(max_workers=thread_count) as executor:
                futures = []
                for thread_id in range(thread_count):
                    future = executor.submit(publish_batch, batch_size, thread_id)
                    futures.append(future)
                
                # Collect all timing results
                for future in futures:
                    times.extend(future.result())
        
        _, total_time, peak_memory, memory_delta, cpu_percent = self._measure_memory_and_cpu(concurrent_publish)
        
        stats = self._calculate_stats(times)
        
        return BenchmarkResult(
            name="concurrent_publishing",
            operation_count=len(times),  # Actual events published
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
            throughput_ops_per_sec=len(times) / total_time,
            metadata={
                "thread_count": thread_count,
                "handler_calls": handler_call_count
            }
        )
    
    def benchmark_memory_scaling(self, max_handlers: int = 100) -> BenchmarkResult:
        """Benchmark memory usage scaling with handler count."""
        event_bus = EventBus()
        memory_usage = []
        
        def dummy_handler(event: Event) -> None:
            pass
        
        def measure_memory_with_handlers():
            for handler_count in range(0, max_handlers + 1, 10):
                # Add handlers
                current_count = len([h for handlers in event_bus.handlers.values() for h in handlers])
                while current_count < handler_count:
                    event_bus.subscribe(f"test.scale.{current_count}", dummy_handler)
                    current_count += 1
                
                # Measure memory
                gc.collect()
                memory_mb = self.process.memory_info().rss / 1024 / 1024
                memory_usage.append(memory_mb)
        
        start_time = time.perf_counter()
        measure_memory_with_handlers()
        end_time = time.perf_counter()
        
        return BenchmarkResult(
            name="memory_scaling",
            operation_count=max_handlers,
            total_time=end_time - start_time,
            avg_time=(end_time - start_time) / max_handlers,
            min_time=0,
            max_time=0,
            p50_time=0,
            p95_time=0,
            p99_time=0,
            memory_peak_mb=max(memory_usage),
            memory_delta_mb=memory_usage[-1] - memory_usage[0],
            cpu_percent=0,
            throughput_ops_per_sec=max_handlers / (end_time - start_time),
            metadata={
                "memory_progression": memory_usage,
                "max_handlers": max_handlers
            }
        )
    
    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run all event system benchmarks."""
        print("Running Event System Benchmarks...")
        
        # Basic pub/sub performance
        print("1. Basic pub/sub performance...")
        self.results.append(self.benchmark_basic_pub_sub())
        
        # Multiple handlers
        print("2. Multiple handlers performance...")
        self.results.append(self.benchmark_multiple_handlers())
        
        # Async handlers
        print("3. Async handlers performance...")
        self.results.append(self.benchmark_async_handlers())
        
        # Event filtering
        print("4. Event filtering performance...")
        self.results.append(self.benchmark_event_filtering())
        
        # Concurrent publishing
        print("5. Concurrent publishing performance...")
        self.results.append(self.benchmark_concurrent_publishing())
        
        # Memory scaling
        print("6. Memory scaling characteristics...")
        self.results.append(self.benchmark_memory_scaling())
        
        return self.results
    
    def print_results(self) -> None:
        """Print benchmark results in a readable format."""
        print("\n" + "="*80)
        print("EVENT SYSTEM BENCHMARK RESULTS")
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
    """Run event system benchmarks."""
    benchmark = EventSystemBenchmark()
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
    
    with open('/tmp/event_system_benchmark_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\nResults saved to: /tmp/event_system_benchmark_results.json")


if __name__ == "__main__":
    main()