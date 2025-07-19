"""
Kaggle-specific training components optimized for competitions.
"""

from .trainer import KaggleTrainer
from .config import KaggleTrainerConfig, CompetitionProfile, get_competition_config
from .callbacks import (
    KaggleSubmissionCallback,
    LeaderboardTracker,
    CompetitionMetrics,
)

__all__ = [
    "KaggleTrainer",
    "KaggleTrainerConfig",
    "CompetitionProfile",
    "get_competition_config",
    "KaggleSubmissionCallback",
    "LeaderboardTracker",
    "CompetitionMetrics",
]