"""Pipeline builder for constructing data processing pipelines."""

from typing import Any, Callable, Dict, List, Optional, Union

from loguru import logger

from .transformers import (
    ChainedTransformer,
    ConditionalTransformer,
    Transformer,
    FunctionTransformer,
    BatchTransformer,
    ParallelTransformer,
)


class Pipeline:
    """Data processing pipeline that manages transformer execution."""
    
    def __init__(self, transformers: List[Transformer], name: str = "pipeline"):
        """Initialize pipeline.
        
        Args:
            transformers: List of transformers in the pipeline
            name: Name of the pipeline
        """
        self.name = name
        self.transformers = transformers
        self._metrics: Dict[str, Any] = {}
        self._setup_complete = False
        logger.info(f"Created pipeline '{name}' with {len(transformers)} transformers")
    
    def setup(self) -> None:
        """Setup all transformers in the pipeline."""
        if not self._setup_complete:
            logger.info(f"Setting up pipeline '{self.name}'")
            for transformer in self.transformers:
                transformer.setup()
            self._setup_complete = True
    
    def run(self, data: Any) -> Any:
        """Run the pipeline on input data.
        
        Args:
            data: Input data
            
        Returns:
            Processed data
        """
        self.setup()
        logger.info(f"Running pipeline '{self.name}'")
        
        result = data
        for i, transformer in enumerate(self.transformers):
            logger.debug(f"Stage {i+1}/{len(self.transformers)}: {transformer.name}")
            
            # Validate input
            if not transformer.validate_input(result):
                raise ValueError(f"Invalid input for transformer '{transformer.name}'")
            
            # Transform
            result = transformer(result)
            
            # Validate output
            if not transformer.validate_output(result):
                raise ValueError(f"Invalid output from transformer '{transformer.name}'")
        
        logger.info(f"Pipeline '{self.name}' completed successfully")
        return result
    
    def __call__(self, data: Any) -> Any:
        """Make pipeline callable."""
        return self.run(data)
    
    @property
    def metrics(self) -> Dict[str, Any]:
        """Get pipeline execution metrics."""
        return self._metrics.copy()


class PipelineBuilder:
    """Builder for constructing data processing pipelines."""
    
    def __init__(self, name: str = "pipeline"):
        """Initialize pipeline builder.
        
        Args:
            name: Name for the pipeline
        """
        self.name = name
        self._transformers: List[Transformer] = []
        self._current_branch: Optional[List[Transformer]] = None
        logger.debug(f"Initialized pipeline builder '{name}'")
    
    def add(self, transformer: Union[Transformer, Callable]) -> "PipelineBuilder":
        """Add a transformer to the pipeline.
        
        Args:
            transformer: Transformer instance or callable
            
        Returns:
            Self for chaining
        """
        if callable(transformer) and not isinstance(transformer, Transformer):
            transformer = FunctionTransformer(transformer)
        
        if self._current_branch is not None:
            self._current_branch.append(transformer)
        else:
            self._transformers.append(transformer)
        
        logger.debug(f"Added transformer '{transformer.name}' to pipeline")
        return self
    
    def add_function(
        self,
        func: Callable,
        name: Optional[str] = None,
        validate_input: Optional[Callable] = None,
        validate_output: Optional[Callable] = None
    ) -> "PipelineBuilder":
        """Add a function as a transformer.
        
        Args:
            func: Function to add
            name: Optional name for the transformer
            validate_input: Optional input validation function
            validate_output: Optional output validation function
            
        Returns:
            Self for chaining
        """
        transformer = FunctionTransformer(
            func,
            name=name,
            validate_input=validate_input,
            validate_output=validate_output
        )
        return self.add(transformer)
    
    def add_conditional(
        self,
        transformer: Union[Transformer, Callable],
        condition: Callable[[Any], bool],
        name: Optional[str] = None
    ) -> "PipelineBuilder":
        """Add a conditional transformer.
        
        Args:
            transformer: Transformer to apply conditionally
            condition: Condition function
            name: Optional name
            
        Returns:
            Self for chaining
        """
        if callable(transformer) and not isinstance(transformer, Transformer):
            transformer = FunctionTransformer(transformer)
        
        conditional = ConditionalTransformer(transformer, condition, name)
        return self.add(conditional)
    
    def add_batch(
        self,
        transformer: Union[Transformer, Callable],
        batch_size: int = 32,
        name: Optional[str] = None
    ) -> "PipelineBuilder":
        """Add a batch transformer.
        
        Args:
            transformer: Transformer to apply in batches
            batch_size: Size of each batch
            name: Optional name
            
        Returns:
            Self for chaining
        """
        if callable(transformer) and not isinstance(transformer, Transformer):
            transformer = FunctionTransformer(transformer)
        
        batch = BatchTransformer(transformer, batch_size, name)
        return self.add(batch)
    
    def branch(self) -> "PipelineBuilder":
        """Start a new branch for parallel processing.
        
        Returns:
            Self for chaining
        """
        if self._current_branch is not None:
            raise ValueError("Cannot nest branches")
        
        self._current_branch = []
        logger.debug("Started new branch")
        return self
    
    def merge(
        self,
        merge_function: Optional[Callable[[List[Any]], Any]] = None,
        max_workers: int = 4,
        name: Optional[str] = None
    ) -> "PipelineBuilder":
        """Merge the current branch using parallel processing.
        
        Args:
            merge_function: Function to merge results
            max_workers: Maximum parallel workers
            name: Optional name
            
        Returns:
            Self for chaining
        """
        if self._current_branch is None:
            raise ValueError("No branch to merge")
        
        if not self._current_branch:
            raise ValueError("Empty branch")
        
        if len(self._current_branch) == 1:
            # Single transformer, no need for parallel
            self._transformers.append(self._current_branch[0])
        else:
            # Create parallel transformer
            parallel = ParallelTransformer(
                self._current_branch,
                merge_function,
                max_workers,
                name
            )
            self._transformers.append(parallel)
        
        self._current_branch = None
        logger.debug("Merged branch")
        return self
    
    def chain(self, transformers: List[Union[Transformer, Callable]], name: Optional[str] = None) -> "PipelineBuilder":
        """Add a chain of transformers.
        
        Args:
            transformers: List of transformers to chain
            name: Optional name
            
        Returns:
            Self for chaining
        """
        # Convert callables to transformers
        transformer_instances = []
        for t in transformers:
            if callable(t) and not isinstance(t, Transformer):
                transformer_instances.append(FunctionTransformer(t))
            else:
                transformer_instances.append(t)
        
        chained = ChainedTransformer(transformer_instances, name)
        return self.add(chained)
    
    def build(self) -> Pipeline:
        """Build the pipeline.
        
        Returns:
            Constructed pipeline
        """
        if self._current_branch is not None:
            raise ValueError("Unclosed branch - call merge() first")
        
        if not self._transformers:
            raise ValueError("Pipeline has no transformers")
        
        pipeline = Pipeline(self._transformers, self.name)
        logger.info(f"Built pipeline '{self.name}' with {len(self._transformers)} stages")
        return pipeline
    
    @classmethod
    def create(cls, name: str = "pipeline") -> "PipelineBuilder":
        """Create a new pipeline builder.
        
        Args:
            name: Name for the pipeline
            
        Returns:
            New pipeline builder instance
        """
        return cls(name)