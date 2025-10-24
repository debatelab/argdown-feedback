"""
FastAPI application for argdown-feedback verification service.

This module creates the main FastAPI application with all necessary middleware,
exception handlers, and route configurations.
"""


from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from typing import Dict, Any

from ..shared.exceptions import (
    VerificationError,
    VerifierNotFoundError,
    InvalidConfigError,
    InvalidFilterError,
    FilteringError
)
from .routes.verification import router as verification_router
from .routes.discovery import router as discovery_router

import nltk  # type: ignore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK punkt tokenizer data at startup
@asynccontextmanager
async def lifespan(app):
    nltk.download('punkt')
    yield

# Create FastAPI application
app = FastAPI(
    title="Argdown Feedback API",
    description="API for verifying argdown documents and annotations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handlers
@app.exception_handler(VerificationError)
async def verification_error_handler(request: Request, exc: VerificationError) -> JSONResponse:
    """Handle verification errors."""
    logger.error(f"Verification error: {exc}")
    return JSONResponse(
        status_code=400,
        content={
            "error": "verification_error",
            "message": str(exc),
            "detail": getattr(exc, 'detail', None)
        }
    )

@app.exception_handler(VerifierNotFoundError)
async def verifier_not_found_error_handler(request: Request, exc: VerifierNotFoundError) -> JSONResponse:
    """Handle verifier not found errors."""
    logger.error(f"Verifier not found error: {exc}")
    return JSONResponse(
        status_code=404,
        content={
            "error": "verifier_not_found",
            "message": str(exc),
            "detail": getattr(exc, 'detail', None)
        }
    )

@app.exception_handler(InvalidConfigError)
async def invalid_config_error_handler(request: Request, exc: InvalidConfigError) -> JSONResponse:
    """Handle invalid configuration errors."""
    logger.error(f"Invalid config error: {exc}")
    return JSONResponse(
        status_code=422,
        content={
            "error": "invalid_config",
            "message": str(exc),
            "detail": getattr(exc, 'detail', None)
        }
    )

@app.exception_handler(InvalidFilterError)
async def invalid_filter_error_handler(request: Request, exc: InvalidFilterError) -> JSONResponse:
    """Handle invalid filter errors."""
    logger.error(f"Invalid filter error: {exc}")
    return JSONResponse(
        status_code=422,
        content={
            "error": "invalid_filter",
            "message": str(exc),
            "detail": getattr(exc, 'detail', None)
        }
    )

@app.exception_handler(FilteringError)
async def filtering_error_handler(request: Request, exc: FilteringError) -> JSONResponse:
    """Handle filtering processing errors."""
    logger.error(f"Filtering error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "filtering_error",
            "message": str(exc),
            "detail": getattr(exc, 'detail', None)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle general exceptions."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred",
            "detail": None
        }
    )

# Health check endpoint
@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "argdown-feedback-api",
        "version": "1.0.0"
    }

# Root endpoint
@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint with basic API information."""
    return {
        "message": "Argdown Feedback API",
        "docs": "/docs",
        "health": "/health"
    }

# Include routers
app.include_router(verification_router, prefix="/api/v1")
app.include_router(discovery_router, prefix="/api/v1")

