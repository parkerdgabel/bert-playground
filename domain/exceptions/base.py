"""Base domain exceptions."""


class DomainException(Exception):
    """Base exception for all domain errors."""
    pass


class ValidationException(DomainException):
    """Raised when domain validation fails."""
    
    def __init__(self, message: str, field: str = None, value: any = None):
        self.field = field
        self.value = value
        super().__init__(message)
    
    def __str__(self):
        if self.field:
            return f"Validation error for {self.field}: {self.args[0]}"
        return self.args[0]