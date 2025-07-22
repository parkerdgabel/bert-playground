"""Main trainer module - provides unified training interface."""

from .core.base import BaseTrainer
from .kaggle.trainer import KaggleTrainer

# Main trainer class - alias for the most commonly used trainer
Trainer = KaggleTrainer

__all__ = ["Trainer", "BaseTrainer", "KaggleTrainer"]