"""
In-process backend for verification client.

Directly executes verification handlers without HTTP overhead, reusing the server's
verification service and registry infrastructure for local/embedded use cases.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor

from .base import VerificationBackend
from ...shared.models import VerificationRequest, VerificationResponse, VerifierInfo, VerifiersList
from ...shared.exceptions import VerifierNotFoundError
from ...server.services.verification_service import VerificationService
from ...server.services import verifier_registry


class InProcessBackend(VerificationBackend):
    """
    In-process backend that directly calls verification handlers.
    
    This backend bypasses HTTP entirely and directly uses the verification service
    and handler registry. Ideal for:
    - Local/embedded use cases
    - Testing without starting a server
    - High-performance scenarios where HTTP overhead is undesirable
    - Library-style integration into other applications
    
    Reuses the server's VerificationService and verifier_registry infrastructure,
    ensuring identical behavior to the HTTP API.
    """
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize in-process backend.
        
        Args:
            max_workers: Maximum number of worker threads for CPU-intensive
                        verification tasks (default: 4)
        
        Example:
            >>> backend = InProcessBackend(max_workers=8)
            >>> client = VerifiersClient(backend=backend)
        """
        self.verification_service = VerificationService(max_workers=max_workers)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def verify_async(
        self, 
        verifier_name: str, 
        request: VerificationRequest
    ) -> VerificationResponse:
        """
        Verify code asynchronously by running verification in thread pool.
        
        Args:
            verifier_name: Name of the verifier to use
            request: Verification request
            
        Returns:
            Verification response with results
            
        Raises:
            VerifierNotFoundError: If verifier doesn't exist
            Exception: For verification errors
        """
        # Reuse the verification service's async implementation
        return await self.verification_service.verify_async(verifier_name, request)
    
    def verify_sync(
        self, 
        verifier_name: str, 
        request: VerificationRequest
    ) -> VerificationResponse:
        """
        Verify code synchronously using direct handler execution.
        
        Args:
            verifier_name: Name of the verifier to use
            request: Verification request
            
        Returns:
            Verification response with results
            
        Raises:
            VerifierNotFoundError: If verifier doesn't exist
            Exception: For verification errors
        """
        # Validate verifier exists
        try:
            verifier_registry.get_verifier_info(verifier_name)
        except KeyError:
            available = verifier_registry.list_verifiers()
            raise VerifierNotFoundError(verifier_name, available)
        
        # Execute verification directly (synchronously)
        return self.verification_service._verify_sync(verifier_name, request)
    
    async def list_verifiers_async(self) -> VerifiersList:
        """
        List all available verifiers asynchronously.
        
        Since verifier registry access is fast, this just wraps the sync version.
        
        Returns:
            List of verifiers grouped by category
        """
        # Registry access is fast, just wrap in async
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.list_verifiers_sync)
    
    def list_verifiers_sync(self) -> VerifiersList:
        """
        List all available verifiers synchronously.
        
        Directly accesses the verifier registry without HTTP overhead.
        
        Returns:
            List of verifiers grouped by category
        """
        return verifier_registry.get_all_verifiers_info()
    
    async def get_verifier_info_async(self, verifier_name: str) -> VerifierInfo:
        """
        Get information about a specific verifier asynchronously.
        
        Args:
            verifier_name: Name of the verifier
            
        Returns:
            Verifier information including supported features
            
        Raises:
            VerifierNotFoundError: If verifier doesn't exist
        """
        # Registry access is fast, just wrap in async
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, 
            self.get_verifier_info_sync,
            verifier_name
        )
    
    def get_verifier_info_sync(self, verifier_name: str) -> VerifierInfo:
        """
        Get information about a specific verifier synchronously.
        
        Args:
            verifier_name: Name of the verifier
            
        Returns:
            Verifier information including supported features
            
        Raises:
            VerifierNotFoundError: If verifier doesn't exist
        """
        try:
            return verifier_registry.get_verifier_info(verifier_name)
        except KeyError:
            available = verifier_registry.list_verifiers()
            raise VerifierNotFoundError(verifier_name, available)
    
    def close(self) -> None:
        """Shutdown thread pool executor."""
        self._executor.shutdown(wait=True)
    
    async def aclose(self) -> None:
        """Asynchronously shutdown thread pool executor."""
        # Shutdown is blocking, so run in executor
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._executor.shutdown, True)
    
    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
