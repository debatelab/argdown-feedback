"""
FastAPI server for argdown-feedback verifiers.

This module provides a REST API interface for running various argdown verification handlers.
"""

from .app import app

__all__ = ["app"]

