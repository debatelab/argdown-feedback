"""
Router initialization for API routes.
"""

# Import routers
from .verification import router as verification_router
from .discovery import router as discovery_router

__all__ = ["verification_router", "discovery_router"]