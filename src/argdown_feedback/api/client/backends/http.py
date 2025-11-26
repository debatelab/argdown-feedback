"""
HTTP backend for verification client.

Uses httpx to communicate with a FastAPI-based verification server over HTTP/REST.
"""

import httpx

from .base import VerificationBackend
from ...shared.models import VerificationRequest, VerificationResponse, VerifierInfo, VerifiersList


class HTTPBackend(VerificationBackend):
    """
    HTTP/REST backend using httpx for API communication.
    
    This backend communicates with a FastAPI server running the verification service.
    Supports both sync and async operations with automatic connection pooling.
    """
    
    def __init__(self, base_url: str, timeout: float = 30.0):
        """
        Initialize HTTP backend.
        
        Args:
            base_url: Base URL of the API server (e.g., "http://localhost:8000")
            timeout: Request timeout in seconds (default: 30.0)
        
        Example:
            >>> backend = HTTPBackend(base_url="http://localhost:8000", timeout=60.0)
            >>> client = VerifiersClient(backend=backend)
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self._async_client = httpx.AsyncClient(timeout=timeout)
        self._sync_client = httpx.Client(timeout=timeout)
    
    async def verify_async(
        self, 
        verifier_name: str, 
        request: VerificationRequest
    ) -> VerificationResponse:
        """
        Verify code asynchronously via HTTP POST.
        
        Args:
            verifier_name: Name of the verifier to use
            request: Verification request
            
        Returns:
            Verification response with results
            
        Raises:
            httpx.HTTPStatusError: For HTTP 4xx/5xx errors
            httpx.RequestError: For network/connection errors
        """
        response = await self._async_client.post(
            f"{self.base_url}/api/v1/verify/{verifier_name}",
            json=request.dict()
        )
        response.raise_for_status()
        return VerificationResponse(**response.json())
    
    def verify_sync(
        self, 
        verifier_name: str, 
        request: VerificationRequest
    ) -> VerificationResponse:
        """
        Verify code synchronously via HTTP POST.
        
        Args:
            verifier_name: Name of the verifier to use
            request: Verification request
            
        Returns:
            Verification response with results
            
        Raises:
            httpx.HTTPStatusError: For HTTP 4xx/5xx errors
            httpx.RequestError: For network/connection errors
        """
        response = self._sync_client.post(
            f"{self.base_url}/api/v1/verify/{verifier_name}",
            json=request.dict()
        )
        response.raise_for_status()
        return VerificationResponse(**response.json())
    
    async def list_verifiers_async(self) -> VerifiersList:
        """
        List all available verifiers asynchronously via HTTP GET.
        
        Returns:
            List of verifiers grouped by category
            
        Raises:
            httpx.HTTPStatusError: For HTTP errors
            httpx.RequestError: For network errors
        """
        response = await self._async_client.get(f"{self.base_url}/api/v1/verifiers")
        response.raise_for_status()
        return VerifiersList(**response.json())
    
    def list_verifiers_sync(self) -> VerifiersList:
        """
        List all available verifiers synchronously via HTTP GET.
        
        Returns:
            List of verifiers grouped by category
            
        Raises:
            httpx.HTTPStatusError: For HTTP errors
            httpx.RequestError: For network errors
        """
        response = self._sync_client.get(f"{self.base_url}/api/v1/verifiers")
        response.raise_for_status()
        return VerifiersList(**response.json())
    
    async def get_verifier_info_async(self, verifier_name: str) -> VerifierInfo:
        """
        Get information about a specific verifier asynchronously via HTTP GET.
        
        Args:
            verifier_name: Name of the verifier
            
        Returns:
            Verifier information including supported features
            
        Raises:
            httpx.HTTPStatusError: For HTTP errors (404 if verifier not found)
            httpx.RequestError: For network errors
        """
        response = await self._async_client.get(
            f"{self.base_url}/api/v1/verifiers/{verifier_name}"
        )
        response.raise_for_status()
        return VerifierInfo(**response.json())
    
    def get_verifier_info_sync(self, verifier_name: str) -> VerifierInfo:
        """
        Get information about a specific verifier synchronously via HTTP GET.
        
        Args:
            verifier_name: Name of the verifier
            
        Returns:
            Verifier information including supported features
            
        Raises:
            httpx.HTTPStatusError: For HTTP errors (404 if verifier not found)
            httpx.RequestError: For network errors
        """
        response = self._sync_client.get(
            f"{self.base_url}/api/v1/verifiers/{verifier_name}"
        )
        response.raise_for_status()
        return VerifierInfo(**response.json())
    
    def close(self) -> None:
        """Close HTTP clients and release connections."""
        self._sync_client.close()
    
    async def aclose(self) -> None:
        """Asynchronously close HTTP clients and release connections."""
        await self._async_client.aclose()
        self._sync_client.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.aclose()
