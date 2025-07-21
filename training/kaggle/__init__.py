"""
Kaggle-specific training components optimized for competitions.
"""

from .callbacks import (
    CompetitionMetrics,
    KaggleSubmissionCallback,
    LeaderboardTracker,
)
from .config import CompetitionProfile, KaggleTrainerConfig, get_competition_config
from .trainer import KaggleTrainer

__all__ = [
    "KaggleTrainer",
    "KaggleTrainerConfig",
    "CompetitionProfile",
    "get_competition_config",
    "KaggleSubmissionCallback",
    "LeaderboardTracker",
    "CompetitionMetrics",
]
