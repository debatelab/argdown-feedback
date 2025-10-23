"""
HTTP client for the argdown-feedback API.

Provides sync and async clients for interacting with verification endpoints.
"""

import asyncio
from typing import Union
import httpx

from ..shared.models import VerificationRequest, VerificationResponse, VerifierInfo, VerifiersList


class VerifiersClient:
    """
    HTTP client for the argdown-feedback verification API.
    
    Supports both synchronous and asynchronous operation modes.
    """
    
    def __init__(self, base_url: str, async_client: bool = True, timeout: float = 30.0):
        """
        Initialize the verifiers client.
        
        Args:
            base_url: Base URL of the API server
            async_client: Whether to use async mode
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.is_async = async_client
        
        if async_client:
            self.client: Union[httpx.AsyncClient, httpx.Client] = httpx.AsyncClient(timeout=timeout)
        else:
            self.client = httpx.Client(timeout=timeout)
    
    async def verify_async(self, verifier_name: str, request: VerificationRequest) -> VerificationResponse:
        """
        Verify code asynchronously.
        
        Args:
            verifier_name: Name of the verifier to use
            request: Verification request
            
        Returns:
            Verification response
            
        Raises:
            httpx.HTTPStatusError: For HTTP errors
        """
        if not self.is_async:
            raise RuntimeError("Client was initialized in sync mode")
        
        client = self.client
        assert isinstance(client, httpx.AsyncClient)
        
        response = await client.post(
            f"{self.base_url}/api/v1/verify/{verifier_name}",
            json=request.dict()
        )
        response.raise_for_status()
        return VerificationResponse(**response.json())
    
    def verify_sync(self, verifier_name: str, request: VerificationRequest) -> VerificationResponse:
        """
        Verify code synchronously.
        
        Args:
            verifier_name: Name of the verifier to use
            request: Verification request
            
        Returns:
            Verification response
            
        Raises:
            httpx.HTTPStatusError: For HTTP errors
        """
        if self.is_async:
            raise RuntimeError("Client was initialized in async mode")
        
        client = self.client
        assert isinstance(client, httpx.Client)
        
        response = client.post(
            f"{self.base_url}/api/v1/verify/{verifier_name}",
            json=request.dict()
        )
        response.raise_for_status()
        return VerificationResponse(**response.json())
    
    def verify(self, verifier_name: str, request: VerificationRequest) -> Union[VerificationResponse, asyncio.Task]:
        """
        Verify code using the appropriate mode (sync/async).
        
        Args:
            verifier_name: Name of the verifier to use
            request: Verification request
            
        Returns:
            Verification response (sync) or Task (async)
        """
        if self.is_async:
            return asyncio.create_task(self.verify_async(verifier_name, request))
        else:
            return self.verify_sync(verifier_name, request)
    
    async def list_verifiers_async(self) -> VerifiersList:
        """
        List all available verifiers asynchronously.
        
        Returns:
            List of verifiers grouped by category
        """
        if not self.is_async:
            raise RuntimeError("Client was initialized in sync mode")
        
        client = self.client
        assert isinstance(client, httpx.AsyncClient)
        
        response = await client.get(f"{self.base_url}/api/v1/verifiers")
        response.raise_for_status()
        return VerifiersList(**response.json())
    
    def list_verifiers_sync(self) -> VerifiersList:
        """
        List all available verifiers synchronously.
        
        Returns:
            List of verifiers grouped by category
        """
        if self.is_async:
            raise RuntimeError("Client was initialized in async mode")
        
        client = self.client
        assert isinstance(client, httpx.Client)
        
        response = client.get(f"{self.base_url}/api/v1/verifiers")
        response.raise_for_status()
        return VerifiersList(**response.json())
    
    async def get_verifier_info_async(self, verifier_name: str) -> VerifierInfo:
        """
        Get information about a specific verifier asynchronously.
        
        Args:
            verifier_name: Name of the verifier
            
        Returns:
            Verifier information
        """
        if not self.is_async:
            raise RuntimeError("Client was initialized in sync mode")
        
        client = self.client
        assert isinstance(client, httpx.AsyncClient)
        
        response = await client.get(f"{self.base_url}/api/v1/verifiers/{verifier_name}")
        response.raise_for_status()
        return VerifierInfo(**response.json())
    
    def get_verifier_info_sync(self, verifier_name: str) -> VerifierInfo:
        """
        Get information about a specific verifier synchronously.
        
        Args:
            verifier_name: Name of the verifier
            
        Returns:
            Verifier information
        """
        if self.is_async:
            raise RuntimeError("Client was initialized in async mode")
        
        client = self.client
        assert isinstance(client, httpx.Client)
        
        response = client.get(f"{self.base_url}/api/v1/verifiers/{verifier_name}")
        response.raise_for_status()
        return VerifierInfo(**response.json())
    
    def close(self):
        """Close the HTTP client."""
        if isinstance(self.client, httpx.Client):
            self.client.close()
    
    async def aclose(self):
        """Close the async HTTP client."""
        if isinstance(self.client, httpx.AsyncClient):
            await self.client.aclose()
    
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