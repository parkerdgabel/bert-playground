# Import heads to ensure they are registered when package is imported
from .custom_head import CustomBinaryHead, CustomMulticlassHead

__all__ = ["CustomBinaryHead", "CustomMulticlassHead"]