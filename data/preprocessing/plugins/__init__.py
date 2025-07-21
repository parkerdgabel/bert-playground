"""Data preprocessing plugins."""

# Import all plugins to register them
from .titanic import TitanicPreprocessor

__all__ = ["TitanicPreprocessor"]