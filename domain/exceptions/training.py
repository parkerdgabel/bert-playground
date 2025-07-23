"""Training-related domain exceptions."""


class TrainingError(Exception):
    """Base exception for training-related errors."""
    pass


class ModelNotInitializedError(TrainingError):
    """Raised when attempting to use an uninitialized model."""
    def __init__(self, message: str = "Model weights are not initialized"):
        super().__init__(message)


class CheckpointError(TrainingError):
    """Raised when checkpoint operations fail."""
    pass


class CheckpointNotFoundError(CheckpointError):
    """Raised when a checkpoint cannot be found."""
    def __init__(self, checkpoint_path: str):
        super().__init__(f"Checkpoint not found: {checkpoint_path}")


class CheckpointCorruptedError(CheckpointError):
    """Raised when a checkpoint is corrupted."""
    def __init__(self, checkpoint_path: str, reason: str = "Unknown"):
        super().__init__(f"Checkpoint corrupted at {checkpoint_path}: {reason}")


class InvalidConfigurationError(TrainingError):
    """Raised when configuration is invalid."""
    pass


class InvalidModelConfigError(InvalidConfigurationError):
    """Raised when model configuration is invalid."""
    def __init__(self, field: str, value: any, reason: str):
        super().__init__(f"Invalid model config - {field}: {value}. {reason}")


class InvalidTrainingConfigError(InvalidConfigurationError):
    """Raised when training configuration is invalid."""
    def __init__(self, field: str, value: any, reason: str):
        super().__init__(f"Invalid training config - {field}: {value}. {reason}")


class DataError(TrainingError):
    """Raised when data-related errors occur."""
    pass


class EmptyDatasetError(DataError):
    """Raised when dataset is empty."""
    def __init__(self, dataset_name: str):
        super().__init__(f"Dataset '{dataset_name}' is empty")


class IncompatibleDataError(DataError):
    """Raised when data is incompatible with model."""
    def __init__(self, reason: str):
        super().__init__(f"Data incompatible with model: {reason}")


class MetricsError(TrainingError):
    """Raised when metrics calculation fails."""
    pass


class InsufficientDataError(MetricsError):
    """Raised when insufficient data for metrics calculation."""
    def __init__(self, metric_name: str, required: int, actual: int):
        super().__init__(
            f"Insufficient data for {metric_name}: "
            f"requires {required}, got {actual}"
        )


class EarlyStoppingError(TrainingError):
    """Raised when early stopping conditions are met."""
    def __init__(self, epoch: int, reason: str = "No improvement"):
        super().__init__(f"Early stopping at epoch {epoch}: {reason}")
        self.epoch = epoch
        self.reason = reason


class ResourceExhaustedError(TrainingError):
    """Raised when system resources are exhausted."""
    def __init__(self, resource: str, details: str = ""):
        message = f"{resource} exhausted"
        if details:
            message += f": {details}"
        super().__init__(message)
        self.resource = resource


class TrainingInterruptedError(TrainingError):
    """Raised when training is interrupted."""
    def __init__(self, epoch: int, step: int, reason: str = "User interrupted"):
        super().__init__(
            f"Training interrupted at epoch {epoch}, step {step}: {reason}"
        )
        self.epoch = epoch
        self.step = step
        self.reason = reason