"""
Discovery endpoints for the argdown-feedback API.

Provides endpoints to discover available verifiers and their specifications.
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status

from ...shared.models import VerifierInfo, VerifiersList
from ..services import verifier_registry

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/verifiers",
    tags=["discovery"],
    responses={
        404: {"description": "Verifier not found"},
    }
)


@router.get("", response_model=VerifiersList)
async def list_verifiers() -> VerifiersList:
    """
    List all available verifiers grouped by category.
    
    Returns:
        List of all verifiers organized by category (core, coherence, content_check)
    """
    try:
        verifiers_info = verifier_registry.get_all_verifiers_info()
        
        logger.info(
            f"Listed verifiers: {len(verifiers_info.core)} core, "
            f"{len(verifiers_info.coherence)} coherence, "
            f"{len(verifiers_info.content_check)} content check"
        )
        
        return verifiers_info
        
    except Exception as e:
        logger.error(f"Error listing verifiers: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "internal_server_error",
                "message": "Failed to list verifiers"
            }
        )


@router.get("/{verifier_name}", response_model=VerifierInfo)
async def get_verifier_info(verifier_name: str) -> VerifierInfo:
    """
    Get detailed information about a specific verifier.
    
    Args:
        verifier_name: Name of the verifier
        
    Returns:
        Detailed verifier information including config options and filter roles
        
    Raises:
        404: If verifier doesn't exist
    """
    try:
        info = verifier_registry.get_verifier_info(verifier_name)
        
        logger.info(f"Retrieved info for verifier: {verifier_name}")
        
        return info
        
    except KeyError:
        available = verifier_registry.list_verifiers()
        logger.warning(f"Verifier not found: {verifier_name}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "verifier_not_found",
                "message": f"Verifier '{verifier_name}' not found",
                "verifier_name": verifier_name,
                "available_verifiers": available
            }
        )
    
    except Exception as e:
        logger.error(f"Error retrieving verifier info for {verifier_name}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "internal_server_error",
                "message": f"Failed to retrieve info for verifier '{verifier_name}'"
            }
        )


@router.get("/health")
async def verifiers_health() -> Dict[str, Any]:
    """
    Health check for the verifiers discovery service.
    
    Returns:
        Health status and basic statistics
    """
    try:
        all_verifiers = verifier_registry.list_verifiers()
        verifiers_info = verifier_registry.get_all_verifiers_info()
        
        return {
            "status": "healthy",
            "total_verifiers": len(all_verifiers),
            "categories": {
                "core": len(verifiers_info.core),
                "coherence": len(verifiers_info.coherence),
                "content_check": len(verifiers_info.content_check)
            },
            "available_verifiers": all_verifiers
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "internal_server_error",
                "message": "Health check failed"
            }
        )