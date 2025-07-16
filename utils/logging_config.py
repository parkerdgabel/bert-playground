import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from loguru import logger
from rich.console import Console
from rich.logging import RichHandler
import json

# Rich console for better terminal output
console = Console()


class LoggingConfig:
    """Centralized logging configuration for the entire project."""
    
    def __init__(
        self,
        log_dir: str = "./logs",
        log_level: str = "INFO",
        log_to_file: bool = True,
        log_to_console: bool = True,
        experiment_name: Optional[str] = None
    ):
        self.log_dir = Path(log_dir)
        self.log_level = log_level
        self.log_to_file = log_to_file
        self.log_to_console = log_to_console
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logger
        self._configure_logger()
    
    def _configure_logger(self):
        """Configure loguru logger with custom settings."""
        # Remove default logger
        logger.remove()
        
        # Add console handler with rich formatting
        if self.log_to_console:
            logger.add(
                sys.stdout,
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
                level=self.log_level,
                colorize=True,
                backtrace=True,
                diagnose=True
            )
        
        # Add file handlers
        if self.log_to_file:
            # General log file
            logger.add(
                self.log_dir / f"{self.experiment_name}_general.log",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
                level=self.log_level,
                rotation="10 MB",
                retention="7 days",
                compression="zip"
            )
            
            # Error log file
            logger.add(
                self.log_dir / f"{self.experiment_name}_errors.log",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
                level="ERROR",
                rotation="10 MB",
                retention="30 days",
                compression="zip",
                backtrace=True,
                diagnose=True
            )
            
            # Training metrics log (JSON format for easy parsing)
            logger.add(
                self.log_dir / f"{self.experiment_name}_metrics.jsonl",
                format="{message}",
                level="INFO",
                filter=lambda record: "metrics" in record["extra"],
                serialize=True
            )
    
    @staticmethod
    def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics in a structured format."""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        }
        if step is not None:
            log_data["step"] = step
        
        logger.bind(metrics=True).info(json.dumps(log_data))
    
    @staticmethod
    def log_hyperparameters(params: Dict[str, Any]):
        """Log hyperparameters at the start of training."""
        logger.info("Hyperparameters:")
        for key, value in params.items():
            logger.info(f"  {key}: {value}")
    
    @staticmethod
    def log_model_info(model_info: Dict[str, Any]):
        """Log model architecture information."""
        logger.info("Model Information:")
        for key, value in model_info.items():
            logger.info(f"  {key}: {value}")
    
    @staticmethod
    def log_data_info(data_info: Dict[str, Any]):
        """Log dataset information."""
        logger.info("Dataset Information:")
        for key, value in data_info.items():
            logger.info(f"  {key}: {value}")


class ExperimentLogger:
    """Context manager for experiment logging."""
    
    def __init__(self, experiment_name: str, config: Optional[Dict[str, Any]] = None):
        self.experiment_name = experiment_name
        self.config = config or {}
        self.start_time = None
        
    def __enter__(self):
        self.start_time = datetime.now()
        logger.info(f"{'='*60}")
        logger.info(f"Starting experiment: {self.experiment_name}")
        logger.info(f"Start time: {self.start_time}")
        logger.info(f"{'='*60}")
        
        if self.config:
            LoggingConfig.log_hyperparameters(self.config)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        logger.info(f"{'='*60}")
        logger.info(f"Experiment completed: {self.experiment_name}")
        logger.info(f"End time: {end_time}")
        logger.info(f"Duration: {duration}")
        
        if exc_type is not None:
            logger.error(f"Experiment failed with error: {exc_val}")
            logger.exception("Exception details:")
        else:
            logger.success("Experiment completed successfully!")
        
        logger.info(f"{'='*60}")


# Convenience functions
def setup_logging(
    experiment_name: Optional[str] = None,
    log_level: str = "INFO",
    log_dir: str = "./logs"
) -> LoggingConfig:
    """Quick setup for logging configuration."""
    return LoggingConfig(
        log_dir=log_dir,
        log_level=log_level,
        experiment_name=experiment_name
    )


def get_logger(name: str = __name__):
    """Get a logger instance with the given name."""
    return logger.bind(name=name)


# Progress bar utilities
def log_progress(iterable, desc: str, total: Optional[int] = None):
    """Create a rich progress bar for iterations."""
    from rich.progress import track
    return track(iterable, description=desc, total=total)


# Decorators for function logging
def log_function_call(func):
    """Decorator to log function calls with arguments and return values."""
    def wrapper(*args, **kwargs):
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} returned: {result}")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {e}")
            raise
    return wrapper


def log_execution_time(func):
    """Decorator to log function execution time."""
    import time
    
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"{func.__name__} executed in {duration:.4f} seconds")
        return result
    return wrapper