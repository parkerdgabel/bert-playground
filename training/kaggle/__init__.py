"""
Kaggle-specific training components optimized for competitions.
"""

from .trainer import KaggleTrainer
from .config import KaggleTrainerConfig, CompetitionProfile
from .callbacks import (
    KaggleSubmissionCallback,
    LeaderboardTracker,
    CompetitionMetrics,
)

__all__ = [
    "KaggleTrainer",
    "KaggleTrainerConfig",
    "CompetitionProfile",
    "KaggleSubmissionCallback",
    "LeaderboardTracker",
    "CompetitionMetrics",
]