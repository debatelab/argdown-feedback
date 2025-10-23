"""
Verification endpoints for the argdown-feedback API.

Provides individual verifier endpoints for processing argdown documents
and annotations with configurable filtering and validation.
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status

from ...shared.models import VerificationRequest, VerificationResponse
from ...shared.exceptions import (
    VerifierNotFoundError,
    InvalidConfigError,
    InvalidFilterError,
    VerificationError
)
from ..services.verification_service import verification_service
from ..services import verifier_registry

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/verify",
    tags=["verification"],
    responses={
        404: {"description": "Verifier not found"},
        422: {"description": "Invalid configuration or filters"},
        400: {"description": "Verification failed"},
    }
)


@router.post("/{verifier_name}")
async def verify_code(
    verifier_name: str,
    request: VerificationRequest
) -> VerificationResponse:
    """
    Verify code using the specified verifier.
    
    Args:
        verifier_name: Name of the verifier to use
        request: Verification request with inputs and configuration
        
    Returns:
        Verification response with results
        
    Raises:
        404: If verifier doesn't exist
        422: If configuration or filters are invalid
        400: If verification fails
    """
    try:
        # Validate configuration if provided
        if request.config:
            invalid_options = verifier_registry.validate_config_options(
                verifier_name, request.config
            )
            if invalid_options:
                raise InvalidConfigError(
                    f"Invalid configuration options: {invalid_options}",
                    invalid_options
                )
            
            # Validate filter roles if filters are specified
            if "filters" in request.config:
                filter_roles = list(request.config["filters"].keys())
                invalid_roles = verifier_registry.validate_filter_roles(
                    verifier_name, filter_roles
                )
                if invalid_roles:
                    raise InvalidFilterError(
                        f"Invalid filter roles for {verifier_name}: {invalid_roles}",
                        invalid_roles
                    )
        
        # Execute verification
        response = await verification_service.verify_async(verifier_name, request)
        
        logger.info(
            f"Verification completed: {verifier_name}, "
            f"valid={response.is_valid}, "
            f"time={response.processing_time_ms:.2f}ms"
        )
        
        return response
        
    except VerifierNotFoundError as e:
        logger.warning(f"Verifier not found: {verifier_name}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "verifier_not_found",
                "message": str(e),
                "verifier_name": verifier_name,
                "available_verifiers": e.detail.get("available_verifiers", [])
            }
        )
    
    except InvalidConfigError as e:
        logger.warning(f"Invalid config for {verifier_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "invalid_config",
                "message": str(e),
                "invalid_options": e.detail.get("invalid_options", [])
            }
        )
    
    except InvalidFilterError as e:
        logger.warning(f"Invalid filters for {verifier_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "invalid_filter",
                "message": str(e),
                "invalid_roles": e.detail.get("invalid_roles", [])
            }
        )
    
    except VerificationError as e:
        logger.error(f"Verification failed for {verifier_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "verification_failed",
                "message": str(e),
                "verifier_name": verifier_name
            }
        )
    
    except Exception as e:
        logger.error(f"Unexpected error in verification: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "internal_server_error",
                "message": "An unexpected error occurred"
            }
        )


@router.get("/{verifier_name}/info")
async def get_verifier_info(verifier_name: str) -> Dict[str, Any]:
    """
    Get information about a specific verifier.
    
    Args:
        verifier_name: Name of the verifier
        
    Returns:
        Verifier information including allowed config options and filter roles
        
    Raises:
        404: If verifier doesn't exist
    """
    try:
        info = verifier_registry.get_verifier_info(verifier_name)
        return info.dict()
        
    except KeyError:
        available = verifier_registry.list_verifiers()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "verifier_not_found",
                "message": f"Verifier '{verifier_name}' not found",
                "verifier_name": verifier_name,
                "available_verifiers": available
            }
        )