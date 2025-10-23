"""
Shared Pydantic models for the argdown-feedback API.

These models are used by both the server and client to ensure type compatibility.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from dataclasses import dataclass


@dataclass
class FilterRule:
    """Single metadata filter rule for code block extraction."""
    key: str
    value: Any
    regex: bool = False


class VerificationRequest(BaseModel):
    """Request model for verification endpoints."""
    inputs: str = Field(..., description="The text containing code blocks to verify")
    source: Optional[str] = Field(None, description="Optional source text for some verifiers")
    config: Optional[Dict[str, Any]] = Field(None, description="Verifier-specific configuration options")

    class Config:
        schema_extra = {
            "example": {
                "inputs": "```argdown\n<Arg>: Test argument.\n(1) Premise\n-- {from: ['1']} --\n(2) Conclusion\n```",
                "source": "Original source text",
                "config": {
                    "from_key": "from",
                    "filters": {
                        "infreco": [
                            {"key": "version", "value": "v3", "regex": False}
                        ]
                    }
                }
            }
        }


class VerificationData(BaseModel):
    """Represents a single piece of verification data extracted from input."""
    id: str = Field(..., description="Unique identifier for this verification data")
    dtype: str = Field(..., description="Data type (argdown, xml)")
    code_snippet: Optional[str] = Field(None, description="The original code snippet")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata extracted from code block")


class VerificationResult(BaseModel):
    """Result from a single verification check."""
    verifier_id: str = Field(..., description="ID of the verifier that produced this result")
    verification_data_references: List[str] = Field(..., description="IDs of verification data this result references")
    is_valid: bool = Field(..., description="Whether the verification passed")
    message: Optional[str] = Field(None, description="Human-readable message about the result")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional details about the verification")


class VerificationResponse(BaseModel):
    """Response model for verification endpoints."""
    verifier: str = Field(..., description="Name of the verifier that was executed")
    is_valid: bool = Field(..., description="Whether all verification checks passed")
    verification_data: List[VerificationData] = Field(..., description="Data extracted and processed")
    results: List[VerificationResult] = Field(..., description="Individual verification results")
    executed_handlers: List[str] = Field(..., description="List of handlers that were executed")
    processing_time_ms: float = Field(..., description="Time taken to process the request in milliseconds")

    class Config:
        schema_extra = {
            "example": {
                "verifier": "infreco",
                "is_valid": True,
                "verification_data": [
                    {
                        "id": "argdown_block_1",
                        "dtype": "argdown",
                        "code_snippet": "```argdown\n<Arg>: Test\n```",
                        "metadata": {"version": "v3"}
                    }
                ],
                "results": [
                    {
                        "verifier_id": "InfReco.HasArgumentsHandler",
                        "verification_data_references": ["argdown_block_1"],
                        "is_valid": True,
                        "message": None,
                        "details": {}
                    }
                ],
                "executed_handlers": ["InfReco.HasArgumentsHandler"],
                "processing_time_ms": 42.5
            }
        }


class VerifierConfigOption(BaseModel):
    """Configuration option for a verifier."""
    name: str = Field(..., description="Name of the configuration option")
    type: str = Field(..., description="Type of the option (string, int, float, bool)")
    default: Any = Field(None, description="Default value")
    description: str = Field(..., description="Human-readable description")
    required: bool = Field(False, description="Whether this option is required")


class VerifierInfo(BaseModel):
    """Information about a specific verifier."""
    name: str = Field(..., description="Verifier name")
    description: str = Field(..., description="Human-readable description of what this verifier does")
    input_types: List[str] = Field(..., description="Supported input data types (argdown, xml)")
    allowed_filter_roles: List[str] = Field(..., description="Allowed filter role identifiers")
    config_options: List[VerifierConfigOption] = Field(default_factory=list, description="Available configuration options")
    is_coherence_verifier: bool = Field(False, description="Whether this is a coherence verifier")

    class Config:
        schema_extra = {
            "example": {
                "name": "infreco",
                "description": "Validates informal argument reconstruction in Argdown format",
                "input_types": ["argdown"],
                "allowed_filter_roles": ["infreco"],
                "config_options": [
                    {
                        "name": "from_key",
                        "type": "string",
                        "default": "from",
                        "description": "Key used for inference information in arguments",
                        "required": False
                    }
                ],
                "is_coherence_verifier": False
            }
        }


class VerifiersList(BaseModel):
    """List of available verifiers grouped by category."""
    core: List[VerifierInfo] = Field(default_factory=list, description="Core verifiers")
    coherence: List[VerifierInfo] = Field(default_factory=list, description="Coherence verifiers")
    content_check: List[VerifierInfo] = Field(default_factory=list, description="Content check verifiers")


class ErrorResponse(BaseModel):
    """Standard error response format."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    detail: Optional[Dict[str, Any]] = Field(None, description="Additional error details")

    class Config:
        schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid verifier name 'invalid_verifier'",
                "detail": {"available_verifiers": ["infreco", "argmap", "arganno"]}
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    version: Optional[str] = Field(None, description="API version")
    uptime_seconds: Optional[float] = Field(None, description="Service uptime in seconds")

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "uptime_seconds": 3600.5
            }
        }