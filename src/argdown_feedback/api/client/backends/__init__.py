"""
Backend implementations for verification client.

This package provides different backend implementations for the verification client,
allowing flexible deployment scenarios from HTTP API calls to in-process execution.
"""

from .base import VerificationBackend
from .http import HTTPBackend
from .inprocess import InProcessBackend

__all__ = [
    "VerificationBackend",
    "HTTPBackend", 
    "InProcessBackend",
]
