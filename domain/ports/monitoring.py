"""Monitoring port for tracking training progress and metrics."""

from typing import Protocol, Dict, Any, Optional, List
from domain.entities.metrics import TrainingMetrics, EvaluationMetrics
from domain.entities.training import TrainingSession


class MonitoringPort(Protocol):
    """Port for monitoring and logging operations."""
    
    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
    ) -> None:
        """Log metrics.
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Optional global step
            epoch: Optional epoch number
        """
        ...
    
    def log_training_metrics(
        self,
        metrics: TrainingMetrics,
    ) -> None:
        """Log training metrics.
        
        Args:
            metrics: Training metrics object
        """
        ...
    
    def log_evaluation_metrics(
        self,
        metrics: EvaluationMetrics,
    ) -> None:
        """Log evaluation metrics.
        
        Args:
            metrics: Evaluation metrics object
        """
        ...
    
    def log_hyperparameters(
        self,
        params: Dict[str, Any],
    ) -> None:
        """Log hyperparameters.
        
        Args:
            params: Dictionary of hyperparameters
        """
        ...
    
    def log_artifact(
        self,
        path: str,
        artifact_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an artifact (file, model, etc.).
        
        Args:
            path: Path to artifact
            artifact_type: Type of artifact
            metadata: Optional metadata
        """
        ...
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """Start a new monitoring run.
        
        Args:
            run_name: Optional run name
            tags: Optional tags for the run
            
        Returns:
            Run ID
        """
        ...
    
    def end_run(
        self,
        status: Optional[str] = None,
    ) -> None:
        """End current monitoring run.
        
        Args:
            status: Optional run status
        """
        ...
    
    def log_message(
        self,
        message: str,
        level: str = "INFO",
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a message.
        
        Args:
            message: Message to log
            level: Log level (DEBUG, INFO, WARNING, ERROR)
            context: Optional context information
        """
        ...
    
    def create_progress_bar(
        self,
        total: int,
        description: str,
        unit: str = "it",
    ) -> 'ProgressBarPort':
        """Create a progress bar.
        
        Args:
            total: Total number of items
            description: Progress bar description
            unit: Unit name
            
        Returns:
            Progress bar instance
        """
        ...
    
    def log_training_session(
        self,
        session: TrainingSession,
    ) -> None:
        """Log complete training session information.
        
        Args:
            session: Training session object
        """
        ...
    
    def get_run_metrics(
        self,
        run_id: Optional[str] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get metrics for a run.
        
        Args:
            run_id: Optional run ID (current run if None)
            
        Returns:
            Dictionary of metric histories
        """
        ...
    
    def compare_runs(
        self,
        run_ids: List[str],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Compare multiple runs.
        
        Args:
            run_ids: List of run IDs to compare
            metrics: Optional list of metrics to compare
            
        Returns:
            Comparison results
        """
        ...


class ProgressBarPort(Protocol):
    """Port for progress bar operations."""
    
    def update(
        self,
        n: int = 1,
    ) -> None:
        """Update progress.
        
        Args:
            n: Number of items completed
        """
        ...
    
    def set_description(
        self,
        description: str,
    ) -> None:
        """Update description.
        
        Args:
            description: New description
        """
        ...
    
    def set_postfix(
        self,
        **kwargs: Any,
    ) -> None:
        """Set postfix values.
        
        Args:
            **kwargs: Key-value pairs to display
        """
        ...
    
    def close(
        self,
    ) -> None:
        """Close progress bar."""
        ...