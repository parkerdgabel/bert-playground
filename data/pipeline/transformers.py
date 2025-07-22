"""Transformer classes for data pipeline."""

import asyncio
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd
from loguru import logger


class Transformer(ABC):
    """Abstract base class for data transformers."""
    
    def __init__(self, name: Optional[str] = None):
        """Initialize transformer.
        
        Args:
            name: Optional name for the transformer
        """
        self.name = name or self.__class__.__name__
        self._setup_complete = False
    
    def setup(self) -> None:
        """Setup method called once before first transform."""
        if not self._setup_complete:
            self._setup()
            self._setup_complete = True
            logger.debug(f"Setup complete for transformer '{self.name}'")
    
    def _setup(self) -> None:
        """Override this for custom setup logic."""
        pass
    
    @abstractmethod
    def transform(self, data: Any) -> Any:
        """Transform the input data.
        
        Args:
            data: Input data to transform
            
        Returns:
            Transformed data
        """
        pass
    
    def __call__(self, data: Any) -> Any:
        """Make transformer callable."""
        self.setup()
        return self.transform(data)
    
    def validate_input(self, data: Any) -> bool:
        """Validate input data.
        
        Args:
            data: Data to validate
            
        Returns:
            True if data is valid
        """
        return True
    
    def validate_output(self, data: Any) -> bool:
        """Validate output data.
        
        Args:
            data: Data to validate
            
        Returns:
            True if data is valid
        """
        return True


class AsyncTransformer(Transformer):
    """Base class for asynchronous transformers."""
    
    @abstractmethod
    async def transform_async(self, data: Any) -> Any:
        """Asynchronously transform the input data.
        
        Args:
            data: Input data to transform
            
        Returns:
            Transformed data
        """
        pass
    
    def transform(self, data: Any) -> Any:
        """Synchronous wrapper for async transform."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.transform_async(data))
        finally:
            loop.close()


class ChainedTransformer(Transformer):
    """Transformer that chains multiple transformers together."""
    
    def __init__(self, transformers: List[Transformer], name: Optional[str] = None):
        """Initialize chained transformer.
        
        Args:
            transformers: List of transformers to chain
            name: Optional name for the transformer
        """
        super().__init__(name)
        self.transformers = transformers
    
    def _setup(self) -> None:
        """Setup all child transformers."""
        for transformer in self.transformers:
            transformer.setup()
    
    def transform(self, data: Any) -> Any:
        """Apply all transformers in sequence.
        
        Args:
            data: Input data
            
        Returns:
            Final transformed data
        """
        result = data
        for i, transformer in enumerate(self.transformers):
            logger.debug(f"Applying transformer {i+1}/{len(self.transformers)}: {transformer.name}")
            result = transformer(result)
        return result
    
    def add_transformer(self, transformer: Transformer) -> None:
        """Add a transformer to the chain.
        
        Args:
            transformer: Transformer to add
        """
        self.transformers.append(transformer)
        if self._setup_complete:
            transformer.setup()


class ConditionalTransformer(Transformer):
    """Transformer that applies based on a condition."""
    
    def __init__(
        self,
        transformer: Transformer,
        condition: Callable[[Any], bool],
        name: Optional[str] = None
    ):
        """Initialize conditional transformer.
        
        Args:
            transformer: Transformer to apply conditionally
            condition: Function that returns True if transformer should be applied
            name: Optional name for the transformer
        """
        super().__init__(name)
        self.transformer = transformer
        self.condition = condition
    
    def _setup(self) -> None:
        """Setup child transformer."""
        self.transformer.setup()
    
    def transform(self, data: Any) -> Any:
        """Apply transformer if condition is met.
        
        Args:
            data: Input data
            
        Returns:
            Transformed data if condition met, otherwise original data
        """
        if self.condition(data):
            logger.debug(f"Condition met, applying transformer: {self.transformer.name}")
            return self.transformer(data)
        else:
            logger.debug(f"Condition not met, skipping transformer: {self.transformer.name}")
            return data


class ParallelTransformer(Transformer):
    """Transformer that applies multiple transformers in parallel."""
    
    def __init__(
        self,
        transformers: List[Transformer],
        merge_function: Optional[Callable[[List[Any]], Any]] = None,
        max_workers: int = 4,
        name: Optional[str] = None
    ):
        """Initialize parallel transformer.
        
        Args:
            transformers: List of transformers to apply in parallel
            merge_function: Function to merge results (default: return as list)
            max_workers: Maximum number of parallel workers
            name: Optional name for the transformer
        """
        super().__init__(name)
        self.transformers = transformers
        self.merge_function = merge_function or (lambda x: x)
        self.max_workers = max_workers
    
    def _setup(self) -> None:
        """Setup all child transformers."""
        for transformer in self.transformers:
            transformer.setup()
    
    def transform(self, data: Any) -> Any:
        """Apply all transformers in parallel.
        
        Args:
            data: Input data
            
        Returns:
            Merged results from all transformers
        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for transformer in self.transformers:
                future = executor.submit(transformer, data)
                futures.append(future)
            
            results = []
            for i, future in enumerate(futures):
                result = future.result()
                results.append(result)
                logger.debug(f"Completed parallel transformer {i+1}/{len(self.transformers)}")
        
        return self.merge_function(results)


class BatchTransformer(Transformer):
    """Transformer that processes data in batches."""
    
    def __init__(
        self,
        transformer: Transformer,
        batch_size: int = 32,
        name: Optional[str] = None
    ):
        """Initialize batch transformer.
        
        Args:
            transformer: Transformer to apply to each batch
            batch_size: Size of each batch
            name: Optional name for the transformer
        """
        super().__init__(name)
        self.transformer = transformer
        self.batch_size = batch_size
    
    def _setup(self) -> None:
        """Setup child transformer."""
        self.transformer.setup()
    
    def transform(self, data: Union[List[Any], pd.DataFrame]) -> Union[List[Any], pd.DataFrame]:
        """Transform data in batches.
        
        Args:
            data: Input data (list or DataFrame)
            
        Returns:
            Transformed data in same format as input
        """
        if isinstance(data, pd.DataFrame):
            return self._transform_dataframe(data)
        elif isinstance(data, list):
            return self._transform_list(data)
        else:
            # Fall back to single transform
            return self.transformer(data)
    
    def _transform_list(self, data: List[Any]) -> List[Any]:
        """Transform list data in batches."""
        results = []
        total_batches = (len(data) + self.batch_size - 1) // self.batch_size
        
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            logger.debug(f"Processing batch {batch_num}/{total_batches}")
            
            batch_result = self.transformer(batch)
            if isinstance(batch_result, list):
                results.extend(batch_result)
            else:
                results.append(batch_result)
        
        return results
    
    def _transform_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform DataFrame in batches."""
        results = []
        total_batches = (len(data) + self.batch_size - 1) // self.batch_size
        
        for i in range(0, len(data), self.batch_size):
            batch = data.iloc[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            logger.debug(f"Processing batch {batch_num}/{total_batches}")
            
            batch_result = self.transformer(batch)
            results.append(batch_result)
        
        return pd.concat(results, ignore_index=True)


class FunctionTransformer(Transformer):
    """Simple transformer that wraps a function."""
    
    def __init__(
        self,
        func: Callable[[Any], Any],
        name: Optional[str] = None,
        validate_input: Optional[Callable[[Any], bool]] = None,
        validate_output: Optional[Callable[[Any], bool]] = None
    ):
        """Initialize function transformer.
        
        Args:
            func: Function to use for transformation
            name: Optional name for the transformer
            validate_input: Optional input validation function
            validate_output: Optional output validation function
        """
        super().__init__(name or func.__name__)
        self.func = func
        self._validate_input = validate_input
        self._validate_output = validate_output
    
    def transform(self, data: Any) -> Any:
        """Apply the function to transform data."""
        return self.func(data)
    
    def validate_input(self, data: Any) -> bool:
        """Validate input data."""
        if self._validate_input:
            return self._validate_input(data)
        return True
    
    def validate_output(self, data: Any) -> bool:
        """Validate output data."""
        if self._validate_output:
            return self._validate_output(data)
        return True