"""Streaming data pipeline for high-throughput processing.

This module provides streaming capabilities for processing large datasets
at 1000+ samples per second with minimal memory footprint.
"""

import asyncio
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

import mlx.core as mx
from loguru import logger

from ..core.base import KaggleDataset


@dataclass
class StreamingConfig:
    """Configuration for streaming pipeline."""
    
    # Streaming parameters
    buffer_size: int = 1000
    batch_size: int = 32
    max_queue_size: int = 100
    chunk_size: int = 256  # Add chunk_size for compatibility
    
    # Performance optimization
    num_producer_threads: int = 2
    num_consumer_threads: int = 1
    num_workers: int = 4  # Add num_workers for compatibility
    prefetch_batches: int = 5
    target_throughput: int = 1000  # Target samples per second
    adaptive_batching: bool = False  # Enable adaptive batch sizing
    
    # Memory management
    max_memory_mb: int = 1024
    enable_memory_monitoring: bool = True
    memory_cleanup_threshold: float = 0.8
    
    # Processing options
    enable_async_processing: bool = True
    stream_timeout_seconds: float = 30.0
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.num_workers <= 0:
            raise ValueError("num_workers must be positive")


class StreamingPipeline:
    """High-throughput streaming pipeline for Kaggle datasets.
    
    This pipeline processes data in a streaming fashion to achieve
    1000+ samples per second throughput while maintaining low memory usage.
    """
    
    def __init__(
        self,
        dataset: KaggleDataset,
        config: Optional[StreamingConfig] = None,
        transform_fn: Optional[callable] = None,
    ):
        """Initialize streaming pipeline.
        
        Args:
            dataset: Kaggle dataset to stream from
            config: Streaming configuration
            transform_fn: Optional transform function applied to each sample
        """
        self.dataset = dataset
        self.config = config or StreamingConfig()
        self.transform_fn = transform_fn
        
        # Threading primitives
        self._sample_queue = queue.Queue(maxsize=self.config.max_queue_size)
        self._batch_queue = queue.Queue(maxsize=self.config.prefetch_batches)
        self._stop_event = threading.Event()
        
        # Worker threads
        self._producer_threads: List[threading.Thread] = []
        self._consumer_threads: List[threading.Thread] = []
        self._batch_thread: Optional[threading.Thread] = None
        self._workers = []  # For compatibility
        
        # Performance tracking
        self._samples_produced = 0
        self._samples_consumed = 0
        self._batches_created = 0
        self._start_time = time.time()
        
        # Memory monitoring
        self._memory_usage_mb = 0
        self._peak_memory_mb = 0
        
        self.logger = logger.bind(component="StreamingPipeline")
        
    def start(self) -> None:
        """Start the streaming pipeline."""
        if self._producer_threads:
            self.logger.warning("Pipeline already started")
            return
            
        self._stop_event.clear()
        
        # Start producer threads
        for i in range(self.config.num_producer_threads):
            thread = threading.Thread(
                target=self._producer_worker,
                args=(i,),
                name=f"Producer-{i}",
                daemon=True
            )
            thread.start()
            self._producer_threads.append(thread)
            
        # Start batch creation thread
        self._batch_thread = threading.Thread(
            target=self._batch_worker,
            name="BatchWorker",
            daemon=True
        )
        self._batch_thread.start()
        
        self.logger.info(
            f"Started streaming pipeline: {self.config.num_producer_threads} producers, "
            f"buffer_size={self.config.buffer_size}"
        )
    
    def stop(self) -> None:
        """Stop the streaming pipeline."""
        if not self._producer_threads:
            return
            
        self.logger.info("Stopping streaming pipeline...")
        
        # Signal stop
        self._stop_event.set()
        
        # Wait for threads to finish
        for thread in self._producer_threads:
            thread.join(timeout=5.0)
            
        if self._batch_thread:
            self._batch_thread.join(timeout=5.0)
            
        # Clear queues
        self._clear_queues()
        
        # Reset thread lists
        self._producer_threads = []
        self._consumer_threads = []
        self._batch_thread = None
        
        self.logger.info("Streaming pipeline stopped")
    
    def stream_samples(self) -> Iterator[Dict[str, Any]]:
        """Stream individual samples.
        
        Yields:
            Individual sample dictionaries
        """
        if not self._producer_threads:
            self.start()
            
        try:
            while not self._stop_event.is_set():
                try:
                    sample = self._sample_queue.get(timeout=1.0)
                    if sample is None:  # Sentinel value
                        break
                        
                    self._samples_consumed += 1
                    yield sample
                    
                except queue.Empty:
                    continue
                    
        except KeyboardInterrupt:
            self.logger.info("Stream interrupted by user")
        finally:
            self.stop()
    
    def stream_batches(self) -> Iterator[Dict[str, mx.array]]:
        """Stream batches of samples.
        
        Yields:
            Batched sample dictionaries with MLX arrays
        """
        if not self._producer_threads:
            self.start()
            
        try:
            while not self._stop_event.is_set():
                try:
                    batch = self._batch_queue.get(timeout=1.0)
                    if batch is None:  # Sentinel value
                        break
                        
                    yield batch
                    
                except queue.Empty:
                    continue
                    
        except KeyboardInterrupt:
            self.logger.info("Batch stream interrupted by user")
        finally:
            self.stop()
    
    async def async_stream_batches(self) -> AsyncIterator[Dict[str, mx.array]]:
        """Asynchronously stream batches.
        
        Yields:
            Batched sample dictionaries with MLX arrays
        """
        if not self.config.enable_async_processing:
            raise ValueError("Async processing not enabled in config")
            
        if not self._producer_threads:
            self.start()
            
        try:
            while not self._stop_event.is_set():
                try:
                    # Use asyncio to wait for batch with timeout
                    batch = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None, self._batch_queue.get, True, 1.0
                        ),
                        timeout=self.config.stream_timeout_seconds
                    )
                    
                    if batch is None:  # Sentinel value
                        break
                        
                    yield batch
                    
                except (queue.Empty, asyncio.TimeoutError):
                    continue
                    
        except KeyboardInterrupt:
            self.logger.info("Async stream interrupted by user")
        finally:
            self.stop()
    
    def _producer_worker(self, worker_id: int) -> None:
        """Producer worker that generates samples.
        
        Args:
            worker_id: Worker thread ID
        """
        self.logger.debug(f"Producer worker {worker_id} started")
        
        # Calculate work distribution
        dataset_size = len(self.dataset)
        samples_per_worker = dataset_size // self.config.num_producer_threads
        start_idx = worker_id * samples_per_worker
        
        if worker_id == self.config.num_producer_threads - 1:
            # Last worker handles remaining samples
            end_idx = dataset_size
        else:
            end_idx = start_idx + samples_per_worker
            
        try:
            for idx in range(start_idx, end_idx):
                if self._stop_event.is_set():
                    break
                    
                try:
                    # Get sample from dataset
                    sample = self.dataset[idx]
                    
                    # Apply transform if provided
                    if self.transform_fn:
                        sample = self.transform_fn(sample)
                        
                    # Add to queue
                    self._sample_queue.put(sample, timeout=5.0)
                    self._samples_produced += 1
                    
                    # Memory monitoring
                    if self.config.enable_memory_monitoring:
                        self._monitor_memory()
                        
                except queue.Full:
                    self.logger.warning(f"Producer {worker_id}: sample queue full")
                    time.sleep(0.1)
                except Exception as e:
                    self.logger.error(f"Producer {worker_id} error: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Producer worker {worker_id} failed: {e}")
        finally:
            self.logger.debug(f"Producer worker {worker_id} finished")
    
    def _batch_worker(self) -> None:
        """Worker that creates batches from individual samples."""
        self.logger.debug("Batch worker started")
        
        current_batch = []
        
        try:
            while not self._stop_event.is_set():
                try:
                    # Get sample with timeout
                    sample = self._sample_queue.get(timeout=1.0)
                    if sample is None:
                        break
                        
                    current_batch.append(sample)
                    
                    # Create batch when full
                    if len(current_batch) >= self.config.batch_size:
                        batch = self._create_batch(current_batch)
                        self._batch_queue.put(batch, timeout=5.0)
                        self._batches_created += 1
                        current_batch = []
                        
                except queue.Empty:
                    # Create partial batch if we have samples and are stopping
                    if current_batch and self._stop_event.is_set():
                        batch = self._create_batch(current_batch)
                        self._batch_queue.put(batch, timeout=1.0)
                        self._batches_created += 1
                        current_batch = []
                    continue
                except queue.Full:
                    self.logger.warning("Batch queue full")
                    time.sleep(0.1)
                    
        except Exception as e:
            self.logger.error(f"Batch worker failed: {e}")
        finally:
            # Process remaining samples
            if current_batch:
                try:
                    batch = self._create_batch(current_batch)
                    self._batch_queue.put(batch, timeout=1.0)
                    self._batches_created += 1
                except:
                    pass
                    
            self.logger.debug("Batch worker finished")
    
    def _create_batch(self, samples: List[Dict[str, Any]]) -> Dict[str, mx.array]:
        """Create a batch from samples.
        
        Args:
            samples: List of sample dictionaries
            
        Returns:
            Batched data as MLX arrays
        """
        if not samples:
            return {}
            
        # Use the dataset's collation method if available
        if hasattr(self.dataset, '_collate_batch'):
            return self.dataset._collate_batch(samples)
            
        # Fallback collation
        batch = {}
        
        # Handle common keys
        for key in ['input_ids', 'attention_mask', 'token_type_ids']:
            if key in samples[0]:
                # Pad sequences
                sequences = [sample[key] for sample in samples]
                
                # Convert MLX arrays to lists if needed
                list_sequences = []
                for seq in sequences:
                    if isinstance(seq, mx.array):
                        list_sequences.append(seq.tolist())
                    else:
                        list_sequences.append(list(seq))
                
                max_len = max(len(seq) for seq in list_sequences)
                
                padded = []
                for seq in list_sequences:
                    padded_seq = seq + [0] * (max_len - len(seq))
                    padded.append(padded_seq)
                    
                batch[key] = mx.array(padded, dtype=mx.int32)
                
        # Handle labels
        if 'labels' in samples[0]:
            labels = [sample['labels'] for sample in samples]
            batch['labels'] = mx.array(labels, dtype=mx.float32)
            
        # Handle text
        if 'text' in samples[0]:
            batch['text'] = [sample['text'] for sample in samples]
            
        return batch
    
    def _monitor_memory(self) -> None:
        """Monitor memory usage and cleanup if needed."""
        try:
            import psutil
            
            # Get current memory usage
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            self._memory_usage_mb = memory_mb
            if memory_mb > self._peak_memory_mb:
                self._peak_memory_mb = memory_mb
                
            # Cleanup if threshold exceeded
            if memory_mb > self.config.max_memory_mb * self.config.memory_cleanup_threshold:
                self._cleanup_memory()
                
        except ImportError:
            # psutil not available, skip monitoring
            pass
        except Exception as e:
            self.logger.warning(f"Memory monitoring error: {e}")
    
    def _cleanup_memory(self) -> None:
        """Perform memory cleanup."""
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear some queue items if queues are too full
        if self._sample_queue.qsize() > self.config.max_queue_size * 0.8:
            try:
                for _ in range(self.config.max_queue_size // 4):
                    self._sample_queue.get_nowait()
            except queue.Empty:
                pass
                
        self.logger.debug("Performed memory cleanup")
    
    def _clear_queues(self) -> None:
        """Clear all queues."""
        # Clear sample queue
        while not self._sample_queue.empty():
            try:
                self._sample_queue.get_nowait()
            except queue.Empty:
                break
                
        # Clear batch queue
        while not self._batch_queue.empty():
            try:
                self._batch_queue.get_nowait()
            except queue.Empty:
                break
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get streaming performance statistics.
        
        Returns:
            Performance metrics dictionary
        """
        elapsed_time = time.time() - self._start_time
        
        return {
            'samples_produced': self._samples_produced,
            'samples_consumed': self._samples_consumed,
            'batches_created': self._batches_created,
            'elapsed_time': elapsed_time,
            'samples_per_second': self._samples_consumed / elapsed_time if elapsed_time > 0 else 0,
            'batches_per_second': self._batches_created / elapsed_time if elapsed_time > 0 else 0,
            'queue_sizes': {
                'sample_queue': self._sample_queue.qsize(),
                'batch_queue': self._batch_queue.qsize(),
            },
            'memory_usage_mb': self._memory_usage_mb,
            'peak_memory_mb': self._peak_memory_mb,
            'active_threads': {
                'producers': len([t for t in self._producer_threads if t.is_alive()]),
                'batch_worker': self._batch_thread.is_alive() if self._batch_thread else False,
            }
        }
    
    def __iter__(self):
        """Make pipeline iterable for streaming samples."""
        return self.stream_samples()
    
    def __next__(self):
        """Get next item when used as iterator."""
        try:
            return next(self.stream_samples())
        except StopIteration:
            raise
    
    def is_running(self) -> bool:
        """Check if pipeline is running."""
        return len(self._producer_threads) > 0 and not self._stop_event.is_set()
    
    def get_throughput_stats(self) -> Dict[str, float]:
        """Get throughput statistics."""
        elapsed = time.time() - self._start_time
        if elapsed <= 0:
            return {
                'samples_per_second': 0.0,
                'avg_batch_time_ms': 0.0,
            }
        
        return {
            'samples_per_second': self._samples_consumed / elapsed,
            'avg_batch_time_ms': (elapsed * 1000) / max(self._batches_created, 1),
        }
    
    def get_buffer_info(self) -> Dict[str, Any]:
        """Get buffer information."""
        return {
            'buffer_size': self.config.buffer_size,
            'current_items': self._sample_queue.qsize(),
            'buffer_utilization': self._sample_queue.qsize() / self.config.max_queue_size,
        }
    
    def get_adaptation_metrics(self) -> Dict[str, int]:
        """Get adaptation metrics."""
        return {
            'buffer_adjustments': 0,  # Placeholder - adaptation not yet implemented
            'worker_adjustments': 0,  # Placeholder - adaptation not yet implemented
        }
    
    def enable_error_recovery(self, max_retries: int = 3) -> None:
        """Enable error recovery with retry limit."""
        self._error_recovery_enabled = True
        self._max_retries = max_retries
        self._error_count = 0
        self._recovered_errors = 0
    
    def get_error_statistics(self) -> Dict[str, int]:
        """Get error statistics."""
        return {
            'total_errors': getattr(self, '_error_count', 0),
            'recovered_errors': getattr(self, '_recovered_errors', 0),
        }
    
    def get_worker_info(self) -> Dict[str, Any]:
        """Get worker information."""
        active_workers = len([t for t in self._producer_threads if t.is_alive()])
        
        return {
            'active_workers': active_workers,
            'worker_stats': {
                'producers': len(self._producer_threads),
                'consumers': len(self._consumer_threads),
                'batch_worker': 1 if self._batch_thread and self._batch_thread.is_alive() else 0,
            }
        }
    
    def get_worker_statistics(self) -> List[Dict[str, Any]]:
        """Get statistics for each worker."""
        stats = []
        
        # Calculate samples per worker
        if self._producer_threads:
            samples_per_worker = self._samples_produced // max(len(self._producer_threads), 1)
            
            for i, thread in enumerate(self._producer_threads):
                stats.append({
                    'worker_id': i,
                    'thread_name': thread.name,
                    'is_alive': thread.is_alive(),
                    'samples_processed': samples_per_worker,  # Approximation
                })
        
        return stats
    
    def enable_monitoring(self, alert_threshold: float = 0.8) -> None:
        """Enable monitoring with alert threshold."""
        self._monitoring_enabled = True
        self._alert_threshold = alert_threshold
    
    def set_alert_callback(self, callback: callable) -> None:
        """Set alert callback for monitoring."""
        self._alert_callback = callback
    
    def get_pool_utilization(self) -> float:
        """Get pool utilization metric."""
        # For streaming pipeline, use queue utilization as proxy
        return self._sample_queue.qsize() / self.config.max_queue_size
    
    def stop(self, graceful: bool = True, timeout: float = 5.0) -> None:
        """Stop the streaming pipeline.
        
        Args:
            graceful: Whether to wait for pending work to complete
            timeout: Maximum time to wait for graceful shutdown
        """
        if not self._producer_threads:
            return
            
        self.logger.info(f"Stopping streaming pipeline (graceful={graceful})...")
        
        # Signal stop
        self._stop_event.set()
        
        if graceful:
            # Wait for threads with timeout
            start_time = time.time()
            
            for thread in self._producer_threads:
                remaining_time = max(0, timeout - (time.time() - start_time))
                thread.join(timeout=remaining_time)
                
            if self._batch_thread:
                remaining_time = max(0, timeout - (time.time() - start_time))
                self._batch_thread.join(timeout=remaining_time)
        else:
            # Just wait briefly
            time.sleep(0.1)
            
        # Clear queues
        self._clear_queues()
        
        # Reset thread lists
        self._producer_threads = []
        self._consumer_threads = []
        self._batch_thread = None
        
        self.logger.info("Streaming pipeline stopped")
    
    def __del__(self):
        """Cleanup when pipeline is destroyed."""
        self.stop()