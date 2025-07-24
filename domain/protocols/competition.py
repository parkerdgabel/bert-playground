"""Domain protocols for competition-related functionality.

These protocols define the contracts that must be implemented by
the infrastructure layer for competition functionality.
"""

from abc import abstractmethod
from typing import Protocol, List, Tuple, Optional, Dict, Any
from datetime import datetime

from domain.entities.competition import Competition, CompetitionSnapshot
from domain.entities.submission import Submission, PredictionData
from domain.entities.experiment import Experiment


class CompetitionDataLoader(Protocol):
    """Protocol for loading competition data."""
    
    @abstractmethod
    def load_competition_data(
        self,
        competition_id: str,
        data_dir: str
    ) -> Tuple[Any, Any, Any]:
        """Load competition train, validation, and test data.
        
        Args:
            competition_id: Competition identifier
            data_dir: Directory containing competition data
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        ...
    
    @abstractmethod
    def load_sample_submission(
        self,
        competition_id: str,
        data_dir: str
    ) -> Dict[str, Any]:
        """Load sample submission for format reference.
        
        Args:
            competition_id: Competition identifier
            data_dir: Directory containing competition data
            
        Returns:
            Sample submission as dictionary
        """
        ...
    
    @abstractmethod
    def get_data_info(
        self,
        competition_id: str,
        data_dir: str
    ) -> Dict[str, Any]:
        """Get information about competition data.
        
        Args:
            competition_id: Competition identifier
            data_dir: Directory containing competition data
            
        Returns:
            Dictionary with data statistics
        """
        ...


class SubmissionFormatter(Protocol):
    """Protocol for formatting predictions for submission."""
    
    @abstractmethod
    def format_predictions(
        self,
        predictions: PredictionData,
        competition: Competition
    ) -> Any:
        """Format predictions according to competition requirements.
        
        Args:
            predictions: Model predictions
            competition: Competition details
            
        Returns:
            Formatted submission data (typically DataFrame)
        """
        ...
    
    @abstractmethod
    def validate_format(
        self,
        submission_data: Any,
        competition: Competition
    ) -> Tuple[bool, List[str]]:
        """Validate submission format.
        
        Args:
            submission_data: Formatted submission
            competition: Competition details
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        ...
    
    @abstractmethod
    def save_submission(
        self,
        submission_data: Any,
        file_path: str,
        competition: Competition
    ) -> None:
        """Save submission to file.
        
        Args:
            submission_data: Formatted submission
            file_path: Path to save submission
            competition: Competition details
        """
        ...


class LeaderboardClient(Protocol):
    """Protocol for interacting with competition leaderboard."""
    
    @abstractmethod
    def submit(
        self,
        submission: Submission,
        competition_id: str
    ) -> Dict[str, Any]:
        """Submit predictions to competition.
        
        Args:
            submission: Submission to upload
            competition_id: Competition identifier
            
        Returns:
            Submission result with score and rank
        """
        ...
    
    @abstractmethod
    def get_leaderboard(
        self,
        competition_id: str,
        leaderboard_type: str = "public"
    ) -> List[Dict[str, Any]]:
        """Get current leaderboard standings.
        
        Args:
            competition_id: Competition identifier
            leaderboard_type: "public" or "private"
            
        Returns:
            List of leaderboard entries
        """
        ...
    
    @abstractmethod
    def get_submission_history(
        self,
        competition_id: str
    ) -> List[Dict[str, Any]]:
        """Get submission history for competition.
        
        Args:
            competition_id: Competition identifier
            
        Returns:
            List of past submissions with scores
        """
        ...
    
    @abstractmethod
    def get_remaining_submissions(
        self,
        competition_id: str
    ) -> Dict[str, int]:
        """Get remaining submission quota.
        
        Args:
            competition_id: Competition identifier
            
        Returns:
            Dictionary with daily and total remaining submissions
        """
        ...


class CompetitionMetadataProvider(Protocol):
    """Protocol for retrieving competition metadata."""
    
    @abstractmethod
    def get_competition_info(
        self,
        competition_id: str
    ) -> Competition:
        """Get full competition information.
        
        Args:
            competition_id: Competition identifier
            
        Returns:
            Competition entity with all details
        """
        ...
    
    @abstractmethod
    def list_active_competitions(
        self,
        platform: Optional[str] = None
    ) -> List[Competition]:
        """List currently active competitions.
        
        Args:
            platform: Optional platform filter
            
        Returns:
            List of active competitions
        """
        ...
    
    @abstractmethod
    def get_competition_snapshot(
        self,
        competition_id: str
    ) -> CompetitionSnapshot:
        """Get current snapshot of competition state.
        
        Args:
            competition_id: Competition identifier
            
        Returns:
            Competition snapshot with current status
        """
        ...


class ExperimentTracker(Protocol):
    """Protocol for tracking experiments."""
    
    @abstractmethod
    def log_experiment(
        self,
        experiment: Experiment
    ) -> None:
        """Log experiment details.
        
        Args:
            experiment: Experiment to log
        """
        ...
    
    @abstractmethod
    def log_metrics(
        self,
        experiment_id: str,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ) -> None:
        """Log experiment metrics.
        
        Args:
            experiment_id: Experiment identifier
            metrics: Metrics to log
            step: Optional step/epoch number
        """
        ...
    
    @abstractmethod
    def log_artifact(
        self,
        experiment_id: str,
        artifact_path: str,
        artifact_type: str
    ) -> None:
        """Log experiment artifact.
        
        Args:
            experiment_id: Experiment identifier
            artifact_path: Path to artifact
            artifact_type: Type of artifact
        """
        ...
    
    @abstractmethod
    def get_experiment_history(
        self,
        competition_id: str
    ) -> List[Experiment]:
        """Get all experiments for a competition.
        
        Args:
            competition_id: Competition identifier
            
        Returns:
            List of experiments
        """
        ...


class PredictionAggregator(Protocol):
    """Protocol for aggregating predictions from multiple models."""
    
    @abstractmethod
    def aggregate_predictions(
        self,
        predictions: Dict[str, Any],
        method: str,
        weights: Optional[Dict[str, float]] = None
    ) -> Any:
        """Aggregate predictions from multiple models.
        
        Args:
            predictions: Dictionary of model_id -> predictions
            method: Aggregation method
            weights: Optional weights for models
            
        Returns:
            Aggregated predictions
        """
        ...
    
    @abstractmethod
    def calculate_optimal_weights(
        self,
        predictions: Dict[str, Any],
        labels: Any,
        metric: str
    ) -> Dict[str, float]:
        """Calculate optimal weights for ensemble.
        
        Args:
            predictions: Dictionary of model_id -> predictions
            labels: True labels
            metric: Optimization metric
            
        Returns:
            Optimal weights for each model
        """
        ...


class FeatureEngineer(Protocol):
    """Protocol for feature engineering."""
    
    @abstractmethod
    def extract_features(
        self,
        data: Any,
        feature_set: str
    ) -> Any:
        """Extract features from raw data.
        
        Args:
            data: Raw data
            feature_set: Which features to extract
            
        Returns:
            Feature matrix
        """
        ...
    
    @abstractmethod
    def get_feature_importance(
        self,
        model: Any,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Get feature importance scores.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            
        Returns:
            Feature importance scores
        """
        ...


class CompetitionAnalyzer(Protocol):
    """Protocol for analyzing competition dynamics."""
    
    @abstractmethod
    def analyze_leaderboard_stability(
        self,
        leaderboard_history: List[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Analyze leaderboard stability over time.
        
        Args:
            leaderboard_history: History of leaderboard snapshots
            
        Returns:
            Analysis results
        """
        ...
    
    @abstractmethod
    def predict_final_shakeup(
        self,
        public_scores: List[float],
        submission_counts: List[int]
    ) -> Dict[str, Any]:
        """Predict potential private leaderboard shakeup.
        
        Args:
            public_scores: Public leaderboard scores
            submission_counts: Number of submissions per team
            
        Returns:
            Shakeup prediction analysis
        """
        ...
    
    @abstractmethod
    def suggest_strategy(
        self,
        current_position: int,
        days_remaining: int,
        submissions_remaining: int
    ) -> List[str]:
        """Suggest competition strategy.
        
        Args:
            current_position: Current leaderboard position
            days_remaining: Days until deadline
            submissions_remaining: Remaining submissions
            
        Returns:
            Strategy suggestions
        """
        ...