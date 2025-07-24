"""Numerical operations adapters.

This module contains adapters that implement the NumericalOperations port
for various numerical computing backends.
"""

from .numpy_adapter import NumPyNumericalAdapter, LazyNumPyArray

__all__ = [
    "NumPyNumericalAdapter",
    "LazyNumPyArray",
]