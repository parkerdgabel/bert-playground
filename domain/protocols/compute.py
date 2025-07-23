"""Domain compute protocols - Core compute abstractions.

These protocols define the fundamental compute abstractions used throughout
the system. They are framework-agnostic and belong to the domain layer.
"""

from abc import abstractmethod
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class DataType(str, Enum):
    """Supported data types for arrays."""
    
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    INT32 = "int32"
    INT64 = "int64"
    BOOL = "bool"


@runtime_checkable
class Array(Protocol):
    """Protocol for array/tensor operations.
    
    This is a domain-level abstraction for multi-dimensional arrays,
    independent of any specific framework (MLX, PyTorch, etc).
    """
    
    @property
    def shape(self) -> tuple[int, ...]:
        """Get the shape of the array."""
        ...
    
    @property
    def dtype(self) -> Any:
        """Get the data type of the array."""
        ...
    
    @property
    def ndim(self) -> int:
        """Get the number of dimensions."""
        ...
    
    def __getitem__(self, key: Any) -> "Array":
        """Index or slice the array."""
        ...
    
    def __len__(self) -> int:
        """Get the length of the first dimension."""
        ...


@runtime_checkable
class Module(Protocol):
    """Protocol for neural network modules.
    
    This is a domain-level abstraction for neural network components,
    independent of any specific framework.
    """
    
    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass through the module."""
        ...
    
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """Get module parameters."""
        ...
    
    @abstractmethod
    def update(self, parameters: dict[str, Any]) -> None:
        """Update module parameters."""
        ...