"""
Base abstract backend for verification operations.

This module defines the interface that all verification backends must implement,
allowing the client to work with different transport mechanisms (HTTP, in-process, etc.).
"""

from abc import ABC, abstractmethod

from ...shared.models import VerificationRequest, VerificationResponse, VerifierInfo, VerifiersList


class VerificationBackend(ABC):
    """
    Abstract base class for verification backends.
    
    Backends implement the actual communication/execution mechanism for verification
    operations, whether that's HTTP calls, in-process function calls, or other transports.
    """
    
    @abstractmethod
    async def verify_async(
        self, 
        verifier_name: str, 
        request: VerificationRequest
    ) -> VerificationResponse:
        """
        Verify code asynchronously.
        
        Args:
            verifier_name: Name of the verifier to use
            request: Verification request
            
        Returns:
            Verification response with results
            
        Raises:
            Exception: Backend-specific exceptions (should be documented by implementation)
        """
        pass
    
    @abstractmethod
    def verify_sync(
        self, 
        verifier_name: str, 
        request: VerificationRequest
    ) -> VerificationResponse:
        """
        Verify code synchronously.
        
        Args:
            verifier_name: Name of the verifier to use
            request: Verification request
            
        Returns:
            Verification response with results
            
        Raises:
            Exception: Backend-specific exceptions (should be documented by implementation)
        """
        pass
    
    @abstractmethod
    async def list_verifiers_async(self) -> VerifiersList:
        """
        List all available verifiers asynchronously.
        
        Returns:
            List of verifiers grouped by category
            
        Raises:
            Exception: Backend-specific exceptions
        """
        pass
    
    @abstractmethod
    def list_verifiers_sync(self) -> VerifiersList:
        """
        List all available verifiers synchronously.
        
        Returns:
            List of verifiers grouped by category
            
        Raises:
            Exception: Backend-specific exceptions
        """
        pass
    
    @abstractmethod
    async def get_verifier_info_async(self, verifier_name: str) -> VerifierInfo:
        """
        Get information about a specific verifier asynchronously.
        
        Args:
            verifier_name: Name of the verifier
            
        Returns:
            Verifier information including supported features
            
        Raises:
            Exception: Backend-specific exceptions
        """
        pass
    
    @abstractmethod
    def get_verifier_info_sync(self, verifier_name: str) -> VerifierInfo:
        """
        Get information about a specific verifier synchronously.
        
        Args:
            verifier_name: Name of the verifier
            
        Returns:
            Verifier information including supported features
            
        Raises:
            Exception: Backend-specific exceptions
        """
        pass
    
    def close(self) -> None:
        """
        Close/cleanup backend resources (optional).
        
        Subclasses can override this to clean up connections, thread pools, etc.
        Default implementation does nothing.
        """
        pass
    
    async def aclose(self) -> None:
        """
        Asynchronously close/cleanup backend resources (optional).
        
        Subclasses can override this to clean up async resources.
        Default implementation calls close().
        """
        self.close()
