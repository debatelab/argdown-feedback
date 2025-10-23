"""
Shared exceptions for the argdown-feedback API.
"""

from typing import Any, Dict, List, Optional


class APIException(Exception):
    """Base exception for API-related errors."""
    def __init__(self, message: str, detail: Optional[Dict[str, Any]] = None):
        self.message = message
        self.detail = detail or {}
        super().__init__(message)


class VerifierNotFoundError(APIException):
    """Raised when a requested verifier is not found."""
    def __init__(self, verifier_name: str, available_verifiers: List[str]):
        super().__init__(
            f"Verifier '{verifier_name}' not found",
            {"available_verifiers": available_verifiers}
        )


class InvalidConfigError(APIException):
    """Raised when verifier configuration is invalid."""
    def __init__(self, message: str, invalid_options: Optional[List[str]] = None):
        detail = {"invalid_options": invalid_options} if invalid_options else {}
        super().__init__(message, detail)


class InvalidFilterError(APIException):
    """Raised when filter configuration is invalid."""
    def __init__(self, message: str, invalid_roles: Optional[List[str]] = None):
        detail = {"invalid_roles": invalid_roles} if invalid_roles else {}
        super().__init__(message, detail)


class VerificationError(APIException):
    """Raised when verification processing fails."""
    pass


class FilteringError(APIException):
    """Raised when code block filtering fails."""
    pass