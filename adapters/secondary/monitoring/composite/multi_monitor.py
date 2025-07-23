"""Composite adapter that dispatches to multiple monitoring adapters."""

from typing import Dict, Any, Optional, List
import logging
from contextlib import contextmanager

from ports.secondary.monitoring import MonitoringService
from domain.entities.metrics import TrainingMetrics, EvaluationMetrics
from domain.entities.training import TrainingSession


logger = logging.getLogger(__name__)


class MultiMonitorAdapter(MonitoringService):
    """Composite adapter that uses multiple monitoring adapters simultaneously."""
    
    def __init__(self, adapters: List[MonitoringService], fail_fast: bool = False):
        """Initialize multi-monitor adapter.
        
        Args:
            adapters: List of monitoring adapters to use
            fail_fast: If True, raise exceptions; if False, log and continue
        """
        self.adapters = adapters
        self.fail_fast = fail_fast
        self._run_ids = {}  # Map adapter to its run IDs
    
    def _call_all(self, method_name: str, *args, **kwargs) -> List[Any]:
        """Call a method on all adapters.
        
        Args:
            method_name: Name of the method to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            List of results from each adapter
        """
        results = []
        exceptions = []
        
        for adapter in self.adapters:
            try:
                method = getattr(adapter, method_name)
                result = method(*args, **kwargs)
                results.append(result)
            except Exception as e:
                adapter_name = adapter.__class__.__name__
                logger.error(f"Error in {adapter_name}.{method_name}: {e}")
                exceptions.append((adapter_name, e))
                
                if self.fail_fast:
                    raise RuntimeError(
                        f"Failed in {adapter_name}.{method_name}: {e}"
                    ) from e
                else:
                    results.append(None)
        
        # Log all exceptions if not failing fast
        if exceptions and not self.fail_fast:
            logger.warning(
                f"{len(exceptions)} adapter(s) failed in {method_name}: "
                f"{', '.join(name for name, _ in exceptions)}"
            )
        
        return results
    
    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
    ) -> None:
        """Log metrics to all adapters.
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Optional global step
            epoch: Optional epoch number
        """
        self._call_all("log_metrics", metrics, step=step, epoch=epoch)
    
    def log_training_metrics(self, metrics: TrainingMetrics) -> None:
        """Log training metrics to all adapters.
        
        Args:
            metrics: Training metrics object
        """
        self._call_all("log_training_metrics", metrics)
    
    def log_evaluation_metrics(self, metrics: EvaluationMetrics) -> None:
        """Log evaluation metrics to all adapters.
        
        Args:
            metrics: Evaluation metrics object
        """
        self._call_all("log_evaluation_metrics", metrics)
    
    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters to all adapters.
        
        Args:
            params: Dictionary of hyperparameters
        """
        self._call_all("log_hyperparameters", params)
    
    def log_artifact(
        self,
        path: str,
        artifact_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an artifact to all adapters.
        
        Args:
            path: Path to artifact
            artifact_type: Type of artifact
            metadata: Optional metadata
        """
        self._call_all("log_artifact", path, artifact_type=artifact_type, metadata=metadata)
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """Start a new monitoring run on all adapters.
        
        Args:
            run_name: Optional run name
            tags: Optional tags for the run
            
        Returns:
            Run ID (from first adapter)
        """
        run_ids = self._call_all("start_run", run_name=run_name, tags=tags)
        
        # Store run IDs for each adapter
        for adapter, run_id in zip(self.adapters, run_ids):
            if run_id:
                if adapter not in self._run_ids:
                    self._run_ids[adapter] = []
                self._run_ids[adapter].append(run_id)
        
        # Return first valid run ID
        for run_id in run_ids:
            if run_id:
                return run_id
        
        return f"multi_run_{run_name or 'unnamed'}"
    
    def end_run(self, status: Optional[str] = None) -> None:
        """End current monitoring run on all adapters.
        
        Args:
            status: Optional run status
        """
        self._call_all("end_run", status=status)
    
    def log_message(
        self,
        message: str,
        level: str = "INFO",
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a message to all adapters.
        
        Args:
            message: Message to log
            level: Log level (DEBUG, INFO, WARNING, ERROR)
            context: Optional context information
        """
        self._call_all("log_message", message, level=level, context=context)
    
    def create_progress_bar(
        self,
        total: int,
        description: str,
        unit: str = "it",
    ) -> object:
        """Create a composite progress bar.
        
        Args:
            total: Total number of items
            description: Progress bar description
            unit: Unit name
            
        Returns:
            Composite progress bar instance
        """
        progress_bars = self._call_all(
            "create_progress_bar",
            total=total,
            description=description,
            unit=unit
        )
        
        # Filter out None results
        valid_bars = [bar for bar in progress_bars if bar is not None]
        
        return CompositeProgressBar(valid_bars, self.fail_fast)
    
    def log_training_session(self, session: TrainingSession) -> None:
        """Log complete training session to all adapters.
        
        Args:
            session: Training session object
        """
        self._call_all("log_training_session", session)
    
    def get_run_metrics(
        self,
        run_id: Optional[str] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get metrics from the first adapter that has them.
        
        Args:
            run_id: Optional run ID (current run if None)
            
        Returns:
            Dictionary of metric histories
        """
        results = self._call_all("get_run_metrics", run_id=run_id)
        
        # Return first non-empty result
        for result in results:
            if result:
                return result
        
        return {}
    
    def compare_runs(
        self,
        run_ids: List[str],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Compare runs from the first adapter that has data.
        
        Args:
            run_ids: List of run IDs to compare
            metrics: Optional list of metrics to compare
            
        Returns:
            Comparison results
        """
        results = self._call_all("compare_runs", run_ids=run_ids, metrics=metrics)
        
        # Return first non-empty result
        for result in results:
            if result:
                return result
        
        return {}
    
    @contextmanager
    def batch_logging(self):
        """Context manager for batch logging operations."""
        # Start batch mode for adapters that support it
        for adapter in self.adapters:
            if hasattr(adapter, 'batch_logging'):
                adapter.batch_logging().__enter__()
        
        try:
            yield
        finally:
            # End batch mode
            for adapter in self.adapters:
                if hasattr(adapter, 'batch_logging'):
                    try:
                        adapter.batch_logging().__exit__(None, None, None)
                    except Exception as e:
                        logger.error(f"Error ending batch logging: {e}")
                        if self.fail_fast:
                            raise


class CompositeProgressBar(object):
    """Composite progress bar that updates multiple progress bars."""
    
    def __init__(self, progress_bars: List[object], fail_fast: bool = False):
        """Initialize composite progress bar.
        
        Args:
            progress_bars: List of progress bars to update
            fail_fast: If True, raise exceptions; if False, log and continue
        """
        self.progress_bars = progress_bars
        self.fail_fast = fail_fast
    
    def _call_all(self, method_name: str, *args, **kwargs) -> None:
        """Call a method on all progress bars.
        
        Args:
            method_name: Name of the method to call
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        for bar in self.progress_bars:
            try:
                method = getattr(bar, method_name)
                method(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in progress bar {method_name}: {e}")
                if self.fail_fast:
                    raise
    
    def update(self, n: int = 1) -> None:
        """Update all progress bars.
        
        Args:
            n: Number of items completed
        """
        self._call_all("update", n)
    
    def set_description(self, description: str) -> None:
        """Update description on all progress bars.
        
        Args:
            description: New description
        """
        self._call_all("set_description", description)
    
    def set_postfix(self, **kwargs: Any) -> None:
        """Set postfix values on all progress bars.
        
        Args:
            **kwargs: Key-value pairs to display
        """
        self._call_all("set_postfix", **kwargs)
    
    def close(self) -> None:
        """Close all progress bars."""
        self._call_all("close")