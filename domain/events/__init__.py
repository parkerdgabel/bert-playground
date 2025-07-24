"""Domain events for competition lifecycle.

Domain events represent important occurrences in the domain that
other parts of the system might be interested in.
"""

from .base import DomainEvent
from .competition_events import (
    CompetitionJoined,
    CompetitionDataDownloaded,
    CompetitionDeadlineApproaching,
    CompetitionCompleted
)
from .experiment_events import (
    ExperimentStarted,
    ExperimentCompleted,
    ExperimentFailed,
    HypothesisValidated,
    InsightDiscovered
)
from .submission_events import (
    SubmissionCreated,
    SubmissionValidated,
    SubmissionSubmitted,
    SubmissionScored,
    LeaderboardUpdated,
    LeaderboardPositionChanged
)
from .ensemble_events import (
    EnsembleCreated,
    EnsembleOptimized,
    ModelAddedToEnsemble,
    ModelRemovedFromEnsemble,
    EnsemblePerformanceImproved
)

__all__ = [
    # Base
    "DomainEvent",
    
    # Competition
    "CompetitionJoined",
    "CompetitionDataDownloaded",
    "CompetitionDeadlineApproaching",
    "CompetitionCompleted",
    
    # Experiment
    "ExperimentStarted",
    "ExperimentCompleted",
    "ExperimentFailed",
    "HypothesisValidated",
    "InsightDiscovered",
    
    # Submission
    "SubmissionCreated",
    "SubmissionValidated",
    "SubmissionSubmitted",
    "SubmissionScored",
    "LeaderboardUpdated",
    "LeaderboardPositionChanged",
    
    # Ensemble
    "EnsembleCreated",
    "EnsembleOptimized",
    "ModelAddedToEnsemble",
    "ModelRemovedFromEnsemble",
    "EnsemblePerformanceImproved"
]